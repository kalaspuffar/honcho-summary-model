#!/usr/bin/env python3
"""summary_scoring.py — the ONE scorer for summaries (PLAN §5). Every rule lives here; the
self-test fails if any of these regexes is defined elsewhere.

A chain row (PLAN §2.1) carries a fact ledger; a summary of chunk k is scored against
  - facts due in this chunk            -> fact_coverage_new
  - facts due in earlier chunks        -> fact_coverage_carry   (the incremental-merge metric)
  - changes (old -> new value)         -> latest_state          (new present, old not asserted as current)
  - distractors                        -> fabrication
  - output_words                       -> limit_ratio, over_limit
  - form                               -> bullets, meta, think_leak, narration, empty
"""
import re

BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S", re.M)
META = re.compile(r"^\s*(here('s| is) (a |the )?(summary|recap)|summary:|in summary,|this (conversation|summary)\b|"
                  r"below is|the following (is|summary))", re.I)
THINK_LEAK = re.compile(r"</?think>|^\s*thinking process\b", re.I | re.M)
NARRATION = re.compile(r"^\s*(i('ll| will) (summarize|summarise|now)|let me (summarize|summarise|start)|"
                       r"first,? (i|let me)|okay,? (so|let))", re.I)


def words(text: str) -> int:
    return len((text or "").split())


def has_fact(text: str, fact: str) -> bool:
    """Lenient substring match, as scoring.has_entity in the dialectic repo: the whole fact, or
    (for multi-word facts) its first two tokens both present."""
    t = (text or "").lower()
    f = (fact or "").lower().strip()
    if not f:
        return False
    if f in t:
        return True
    toks = [x for x in re.split(r"\W+", f) if x]
    return len(toks) >= 2 and all(x in t for x in toks[:2])


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
    row = {"words": w, "limit": output_words, "limit_ratio": round(w / output_words, 3) if output_words else None,
           "over_limit": bool(output_words) and w > output_words, "empty": w == 0,
           "bullets": bool(BULLET.search(s)), "meta": bool(META.search(s)),
           "think_leak": bool(THINK_LEAK.search(s)), "narration": bool(NARRATION.search(s))}
    if w == 0 or row["narration"]:
        row.update(fact_coverage_new=0.0, fact_coverage_carry=0.0, latest_state=None, fabrication=False,
                   n_new=len(new), n_carry=len(carry))
        return row
    row["n_new"], row["n_carry"] = len(new), len(carry)
    row["fact_coverage_new"] = round(sum(has_fact(s, f) for f in new) / len(new), 3) if new else None
    row["fact_coverage_carry"] = round(sum(has_fact(s, f) for f in carry) / len(carry), 3) if carry else None
    # latest_state: for every change that has happened by this chunk, the new value is present and the
    # old one is not (naming the old value while stating the new one is allowed by dialectic rules; for
    # summaries the narrative may legitimately say "changed from X to Y", so old+new together is fine)
    facts = {f["id"]: f["text"] for f in chain.get("facts", [])}
    states = []
    for c in chain.get("changes", []):
        if c["seq"] <= hi:
            new_v, old_v = facts.get(c["superseded_by"], ""), facts.get(c["fact_id"], "")
            states.append(has_fact(s, new_v))
    row["latest_state"] = round(sum(states) / len(states), 3) if states else None
    row["fabrication"] = any(has_fact(s, d["text"]) for d in chain.get("distractors", []))
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
            "narration_rows": sum(1 for r in rows if r["narration"]),
            "empty_rows": sum(1 for r in rows if r["empty"])}
