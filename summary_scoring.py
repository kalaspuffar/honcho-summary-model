#!/usr/bin/env python3
"""summary_scoring.py — the ONE scorer for summaries (PLAN §5). Every rule lives here; the
self-test fails if any of these regexes is defined elsewhere.

A chain row (PLAN §2.1) carries a fact ledger; a summary of chunk k is scored against
  - facts due in this chunk            -> fact_coverage_new
  - facts due in earlier chunks        -> fact_coverage_carry   (the incremental-merge metric)
  - changes (old -> new value)         -> latest_state          (new present, old not asserted as current)
  - distractors                        -> fabrication
  - output_words                       -> limit_ratio, over_limit
  - form                               -> bullets, meta, think_leak, narration, echo, empty

Fact matching (has_fact, 2026-09-13 rewrite after Phase 0): normalised exact substring, else the
fact's ANCHOR tokens — stopwords, one-letter fragments (the "s" of "Fenna's") and the chain's peer
names removed — must co-occur within a window of WINDOW tokens of the summary: all of them when
there are one or two, all but one when there are three or more, and every numeric anchor always.
The dialectic scorer's "first two tokens anywhere" rule flagged "Fenna's friend Vera" as fabricated
on "Fenna" + "s", and "the city of Ashvale" on "the" + "city". Number words (one..twenty, tens) are
normalised to digits so "nine slats" == "9 slats" and so a number word is a mandatory anchor.
Distractors are near-misses of true facts by construction ("seven slats per side" vs "nine slats per
side"), so fabrication uses strict=True: every anchor must be present, within WINDOW_STRICT tokens.
"""
import re

BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S", re.M)
META = re.compile(r"^\s*(here('s| is) (a |the )?(summary|recap)|summary:|in summary,|this (conversation|summary)\b|"
                  r"below is|the following (is|summary))", re.I)
THINK_LEAK = re.compile(r"</?think>|^\s*thinking process\b", re.I | re.M)
NARRATION = re.compile(r"^\s*(i('ll| will) (summarize|summarise|now)|let me (summarize|summarise|start)|"
                       r"first,? (i|let me)|okay,? (so|let))", re.I)
# the model copied prompt scaffolding into its answer (dialectic_s50 did this in 5/133 Phase 0 rows)
ECHO = re.compile(r"</?conversation>|</?previous_summary>|there is no previous summary", re.I)

STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "with", "from", "by", "is",
             "are", "was", "were", "be", "been", "her", "his", "their", "its", "my", "our", "your", "she", "he",
             "they", "it", "that", "this", "as", "per", "not", "no", "up", "out", "if", "when", "than", "so"}
TOKEN = re.compile(r"[^\W_]+(?:[.,:/][0-9]+)*")     # 1.052, 7:15, 14380, 1/15 stay one token
WINDOW = 15          # facts: paraphrase tolerance
WINDOW_STRICT = 6    # distractors: the anchors must sit together ("15 inches ... prints" is not "fifteen prints")
NUMBER_WORDS = {w: str(i) for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                                 "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                                                 "seventeen", "eighteen", "nineteen", "twenty"])}
NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90"})
_NUMBER_WORD_RX = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b")


def _norm(text: str) -> str:
    t = (text or "").lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", t)           # 14,380 -> 14380 (thousands separators)
    return _NUMBER_WORD_RX.sub(lambda m: NUMBER_WORDS[m.group(1)], t)


def _tokens(text: str):
    return TOKEN.findall(_norm(text))


def anchors(fact: str, ignore=()):
    ig = {x.lower() for x in ignore}
    return [t for t in _tokens(fact) if t not in STOPWORDS and t not in ig and (len(t) >= 2 or t.isdigit())]


def words(text: str) -> int:
    return len((text or "").split())


def has_fact(text: str, fact: str, ignore=(), strict=False) -> bool:
    """Normalised exact substring, else anchor co-occurrence within WINDOW tokens (see module doc).
    `ignore`: tokens that carry no evidence (the chain's peer names). `strict`: every anchor must be
    present (used for distractors, which differ from a true fact in one anchor by design)."""
    t = _norm(text)
    f = _norm(fact).strip()
    if not f or not t:
        return False
    if f in t:
        return True
    anc = anchors(fact, ignore)
    if not anc:
        return False
    need = len(anc) if (strict or len(anc) <= 2) else len(anc) - 1
    numeric = [a for a in anc if any(ch.isdigit() for ch in a)]
    toks = TOKEN.findall(t)
    anc_set = set(anc)
    span = WINDOW_STRICT if strict else WINDOW
    for i in range(len(toks)):
        if toks[i] not in anc_set:
            continue
        window = set(toks[i:i + span])
        if all(n in window for n in numeric) and sum(1 for a in anc if a in window) >= need:
            return True
    return False


def peer_names(chain: dict):
    return {p.get("name", "") for p in chain.get("peers", []) if p.get("name")}


def facts_due(chain: dict, k: int):
    """(new, carry): ledger facts first stated in chunk k / in chunks < k. Superseded values are
    excluded from both (they are scored by latest_state)."""
    chunks = chain["chunks"]
    lo, hi = chunks[k]["seqs"]
    superseded = {c["fact_id"] for c in chain.get("changes", []) if c["seq"] <= hi}
    new, carry = [], []
    for f in chain.get("facts", []):
        if f["id"] in superseded:
            continue
        if lo <= f["first_seq"] <= hi:
            new.append(f["text"])
        elif f["first_seq"] < lo:
            carry.append(f["text"])
    return new, carry


def score_summary(chain: dict, k: int, summary: str, output_words: int) -> dict:
    s = summary or ""
    w = words(s)
    new, carry = facts_due(chain, k)
    hi = chain["chunks"][k]["seqs"][1]
    ig = peer_names(chain)
    row = {"words": w, "limit": output_words, "limit_ratio": round(w / output_words, 3) if output_words else None,
           "over_limit": bool(output_words) and w > output_words, "empty": w == 0,
           "bullets": bool(BULLET.search(s)), "meta": bool(META.search(s)),
           "think_leak": bool(THINK_LEAK.search(s)), "narration": bool(NARRATION.search(s)), "echo": bool(ECHO.search(s))}
    if w == 0 or row["narration"]:
        # nothing usable was produced: every fact that was due is missed; a coverage with nothing due stays None
        # (an empty chunk-0 summary must not count as a dropped-carry row)
        states = [c for c in chain.get("changes", []) if c["seq"] <= hi]
        row.update(fact_coverage_new=0.0 if new else None, fact_coverage_carry=0.0 if carry else None,
                   latest_state=0.0 if states else None, fabrication=False, n_new=len(new), n_carry=len(carry))
        return row
    row["n_new"], row["n_carry"] = len(new), len(carry)
    row["fact_coverage_new"] = round(sum(has_fact(s, f, ig) for f in new) / len(new), 3) if new else None
    row["fact_coverage_carry"] = round(sum(has_fact(s, f, ig) for f in carry) / len(carry), 3) if carry else None
    # latest_state: for every change that has happened by this chunk, the new value is present and the
    # old one is not (naming the old value while stating the new one is allowed by dialectic rules; for
    # summaries the narrative may legitimately say "changed from X to Y", so old+new together is fine)
    facts = {f["id"]: f["text"] for f in chain.get("facts", [])}
    states = []
    for c in chain.get("changes", []):
        if c["seq"] <= hi:
            new_v, old_v = facts.get(c["superseded_by"], ""), facts.get(c["fact_id"], "")
            states.append(has_fact(s, new_v, ig))
    row["latest_state"] = round(sum(states) / len(states), 3) if states else None
    row["fabrication"] = any(has_fact(s, d["text"], ig, strict=True) for d in chain.get("distractors", []))
    return row


def aggregate(rows: list) -> dict:
    import statistics
    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.median(vals), 3) if vals else None
    n = len(rows)
    return {"n": n,
            "median_words": med("words"), "median_limit_ratio": med("limit_ratio"),
            "over_limit_rows": sum(1 for r in rows if r["over_limit"]),
            "median_fact_coverage_new": med("fact_coverage_new"),
            "median_fact_coverage_carry": med("fact_coverage_carry"),
            "median_latest_state": med("latest_state"),
            "fabrication_rows": sum(1 for r in rows if r.get("fabrication")),
            "bullet_rows": sum(1 for r in rows if r["bullets"]), "meta_rows": sum(1 for r in rows if r["meta"]),
            "think_leak_rows": sum(1 for r in rows if r["think_leak"]),
            "narration_rows": sum(1 for r in rows if r["narration"]), "echo_rows": sum(1 for r in rows if r.get("echo")),
            "empty_rows": sum(1 for r in rows if r["empty"])}
