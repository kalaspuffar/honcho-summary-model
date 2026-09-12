#!/usr/bin/env python3
"""gen_sessions.py — stage 1: synthetic session chains with a fact ledger (PLAN §2.1).

One row = one invented conversation of 60–120 messages between 2–3 invented peers, cut into
Honcho-sized blocks of 20, plus a ledger the scorer can check a summary against:
facts (atomic values with the message they first appear in), changes (values superseded later),
distractors (plausible values that appear nowhere — fabrication probes). Nothing here comes from a
real deployment: the teacher invents names, domains and content (PLAN §2 privacy rule).

Same two execution modes and CLI shape as the dialectic project:

  python3 gen_sessions.py estimate --n 30 --model opus
  python3 gen_sessions.py run      --n 30 --model opus --out data/chains.jsonl --concurrency 4
  python3 gen_sessions.py submit   --n 30 --model opus --out data/chains.jsonl
  python3 gen_sessions.py status
  python3 gen_sessions.py fetch    [--manifest results/batches/sessions/<file>.json]

`run` is resume-safe: rows already good in --out are skipped, failed rows retried. Ids c00000..
(--start offsets). Categories interleaved (PLAN §2.1 mix) so a 30-row run covers all six.
Estimate always printed; --max-usd caps a sync run; batches never abort.

Row: {id, category, shape, domain, peers:[{name, role, bio}], messages:[{seq, peer, text}],
      facts:[{id, text, first_seq, kind}], changes:[{fact_id, superseded_by, seq}],
      distractors:[{text}], chunks:[{k, seqs}], chunks_long:[{k, seqs}]}
"""
import argparse
import json
import random
import re

import llm_backend as be
import summary_chain as sc
import summary_scoring as ss

KIND = "sessions"
OUT_TOKENS_PER_MESSAGE = 55     # estimate only: ~35 words of JSON per message + ledger
LEDGER_TOKENS = 1500

CATEGORIES = {
    "dense-facts":    "the user shares many small concrete facts (names, dates, numbers, places, product or item names) in EVERY block of 20 messages — at least 5 ledger facts per block",
    "preference":     "the user states preferences, opinions and asks questions; most ledger facts are the exact things preferred, disliked or asked about (items 2–3 of a summariser's brief)",
    "supersession":   "at least 3 facts CHANGE later in the conversation (a date moves, a number is corrected, a plan is replaced) and the later message says so explicitly; every change is listed in `changes`",
    "long-tail-merge": "most ledger facts are stated in the FIRST 20 messages; the later blocks are follow-up discussion that adds only 1–2 new facts each, so a summary has to carry the early facts forward",
    "chit-chat":      "light small talk with very little factual content — at most 1–2 ledger facts per block; pleasant, meandering, no invented density",
    "multi-peer":     "three peers; attribution matters — several facts belong to one peer and would be wrong attributed to another; make the peers disagree at least twice",
}
MIX = [("dense-facts", 5), ("preference", 3), ("supersession", 3), ("long-tail-merge", 4), ("chit-chat", 2), ("multi-peer", 3)]
SHAPES = ["user-assistant", "user-assistant", "user-assistant", "two-people"]     # multi-peer forces three-people
LENGTHS = [60, 60, 80, 80, 100, 120]

# Broad, generic theme hints so small runs spread out; the teacher picks the concrete sub-domain
# and invents everything in it. Nothing here is a real person's topic list (PLAN §2).
THEMES = ["planning a trip", "learning a language", "a weekend cooking project", "training for an amateur race",
          "renovating one room", "a hobby collection", "a book club", "a small online shop's paperwork",
          "studying for an exam", "a community garden", "board-game nights", "moving to a new flat",
          "adopting a pet", "a music practice routine", "budgeting for a purchase", "a fictional workplace project",
          "organising a family celebration", "a woodworking build", "amateur astronomy", "a podcast someone is starting",
          "restoring an old bicycle", "a chess study plan", "volunteer shifts at an invented charity", "a knitting pattern",
          "a fantasy novel someone is writing", "a home-brewing batch", "a hiking trip", "a photography assignment",
          "a school science fair", "a fictional startup's product launch", "a choir tour", "a video-game speedrun attempt"]
LETTERS = "ABCDEFGHIJKLMNOPRSTVWYZ"

SYSTEM = ("You write synthetic multi-turn conversations, with a fact ledger, as training data for a "
          "conversation summariser. Everything is invented: names, places, products, dates, numbers. "
          "Return ONLY one JSON object, complete and untruncated, no prose, no code fences.")

PROMPT = """Write ONE conversation of exactly {n} messages and its fact ledger.

Category: {cat} — {catdef}
Shape: {shape}
Theme (pick a concrete sub-topic inside it): {theme}
Names: invent first names for the peers, starting with the letters {letters}; vary cultural origin.
Assistant peers (if any) get a short product-like name (e.g. "Nimbus") and speak like a helpful assistant.

Return exactly this JSON shape:
{{
  "domain": "<2-5 word label of the concrete topic>",
  "peers": [{{"name": "<first name>", "role": "user|assistant|peer", "bio": "<1-2 invented sentences>"}}],
  "messages": [{{"seq": 1, "peer": "<name>", "text": "<1-4 sentences>"}}, ... exactly {n} items, seq 1..{n}],
  "facts": [{{"id": "f1", "text": "<1-6 words copied VERBATIM from the message at first_seq: a name, date, number, place, item, or short exact phrase>", "first_seq": 7, "kind": "name|date|number|place|item|preference|event|opinion|plan"}}, ...],
  "changes": [{{"fact_id": "<id of the old value>", "superseded_by": "<id of the new value>", "seq": <first_seq of the new value>}}],
  "distractors": [{{"text": "<a specific value/name/date/place that is plausible for this topic but appears NOWHERE in the conversation>"}}, ...]
}}

Rules:
- Exactly {n} messages. Peers take turns naturally (an assistant answers the user; three peers need not
  alternate strictly). Messages 1-4 sentences; a few short ones are fine. Real conversational texture:
  questions, follow-ups, corrections, small digressions.
- The ledger is what a perfect summary must contain. Every fact `text` is an EXACT substring of the
  message at `first_seq` (same spelling, same digits). Prefer short distinctive values ("14 March",
  "Route 9 bus", "Lina", "twelve jars", "prefers oat milk") over whole sentences. Spread facts over
  the whole conversation unless the category says otherwise. {facts_hint}
- Facts must be specific enough that a summary either has them or not; no generic words ("plan",
  "the trip"). Do not list the same value twice.
- `changes`: only when a later message explicitly replaces an earlier value (both values are facts;
  the old value must NOT be repeated as current after the change). Categories other than supersession
  may have 0-1 changes.
- 5-8 distractors: same topic, same kind of thing, but never mentioned — a name, date, number or place
  that a careless summary might invent.
- No real people, brands you would need a licence for, or real organisations; invent them.
"""

FACTS_HINT = {
    "dense-facts": "Target 5-8 facts per 20-message block.",
    "preference": "Target 3-5 facts per block, mostly preferences/opinions/questions.",
    "supersession": "Target 3-5 facts per block plus at least 3 entries in `changes`.",
    "long-tail-merge": "Target 6-10 facts in messages 1-20, then 1-2 per later block.",
    "chit-chat": "Target 1-2 facts per block; it is fine if a block has one.",
    "multi-peer": "Target 3-5 facts per block; where attribution matters make the fact text include the peer's name if the message text allows it.",
}


def category_sequence():
    slots = sorted(((i + 0.5) / w, c) for c, w in MIX for i in range(w))
    return [c for _, c in slots]


def plan(n, start, seed):
    rnd = random.Random(seed)
    cats = category_sequence()
    rows = []
    for i in range(n):
        cat = cats[i % len(cats)]
        shape = "three-people" if cat == "multi-peer" else rnd.choice(SHAPES)
        n_peers = 3 if shape == "three-people" else 2
        rows.append({"id": f"c{start + i:05d}", "category": cat, "shape": shape,
                     "n_messages": rnd.choice(LENGTHS), "theme": rnd.choice(THEMES),
                     "letters": " and ".join(rnd.sample(LETTERS, n_peers))})
    return rows


def make_job(meta):
    user = PROMPT.format(n=meta["n_messages"], cat=meta["category"], catdef=CATEGORIES[meta["category"]],
                         shape=meta["shape"], theme=meta["theme"], letters=meta["letters"],
                         facts_hint=FACTS_HINT[meta["category"]])
    return {"custom_id": meta["id"], "system": SYSTEM, "user": user,
            "max_tokens": min(32000, meta["n_messages"] * 120 + 4000)}


def out_tokens(meta):
    return meta["n_messages"] * OUT_TOKENS_PER_MESSAGE + LEDGER_TOKENS


def _earliest(messages, text):
    t = text.lower()
    for m in messages:
        if t in m["text"].lower():
            return m["seq"]
    return None


def validate(obj, meta):
    """Coerce the teacher's JSON into a chain row or return (None, reason)."""
    if not isinstance(obj, dict):
        return None, "not an object"
    peers = []
    for p in obj.get("peers") or []:
        if isinstance(p, dict) and p.get("name"):
            peers.append({"name": str(p["name"]).strip(), "role": str(p.get("role") or "peer"), "bio": str(p.get("bio") or "")})
    names = {p["name"] for p in peers}
    msgs = []
    for m in obj.get("messages") or []:
        if isinstance(m, dict) and m.get("peer") and isinstance(m.get("text"), str) and m["text"].strip():
            msgs.append({"peer": str(m["peer"]).strip(), "text": m["text"].strip()})
    if len(msgs) < sc.LONG_EVERY:
        return None, f"only {len(msgs)} messages (< {sc.LONG_EVERY})"
    n = (len(msgs) // sc.CHUNK) * sc.CHUNK
    msgs = msgs[:n]
    for i, m in enumerate(msgs, 1):
        m["seq"] = i
        if m["peer"] not in names:
            peers.append({"name": m["peer"], "role": "peer", "bio": ""}); names.add(m["peer"])
    if len(peers) < 2:
        return None, "fewer than two peers"
    facts, seen, dropped = [], set(), 0
    for f in obj.get("facts") or []:
        if not isinstance(f, dict) or not str(f.get("text") or "").strip():
            continue
        text = re.sub(r"\s+", " ", str(f["text"]).strip())
        if text.lower() in seen:
            continue
        seq = _earliest(msgs, text)
        if seq is None:                       # not verbatim anywhere -> unscoreable, drop
            dropped += 1; continue
        seen.add(text.lower())
        facts.append({"id": str(f.get("id") or f"f{len(facts) + 1}"), "text": text, "first_seq": seq,
                      "kind": str(f.get("kind") or "other")})
    ids = {f["id"]: f for f in facts}
    if len(ids) != len(facts):                # duplicate ids -> renumber
        for i, f in enumerate(facts, 1):
            f["id"] = f"f{i}"
        ids = {f["id"]: f for f in facts}
    n_chunks = n // sc.CHUNK
    min_facts = {"chit-chat": 1}.get(meta["category"], 2) * n_chunks
    if len(facts) < min_facts:
        return None, f"only {len(facts)} verifiable facts for {n_chunks} blocks (dropped {dropped} non-verbatim)"
    changes = []
    for c in obj.get("changes") or []:
        if not isinstance(c, dict):
            continue
        old, new = ids.get(str(c.get("fact_id"))), ids.get(str(c.get("superseded_by")))
        if old and new and new["first_seq"] > old["first_seq"]:
            changes.append({"fact_id": old["id"], "superseded_by": new["id"], "seq": new["first_seq"]})
    if meta["category"] == "supersession" and len(changes) < 1:
        return None, "supersession chain without a valid change"
    all_text = "\n".join(m["text"] for m in msgs)
    distractors = []
    for d in obj.get("distractors") or []:
        text = d.get("text") if isinstance(d, dict) else d
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if ss.has_fact(all_text, text) or any(ss.has_fact(text, f["text"]) for f in facts):
            continue                          # would mark a true statement as fabrication
        distractors.append({"text": text})
    if not distractors:
        return None, "no usable distractor"
    row = {"id": meta["id"], "category": meta["category"], "shape": meta["shape"],
           "domain": str(obj.get("domain") or meta["theme"]), "peers": peers, "messages": msgs,
           "facts": facts, "changes": changes, "distractors": distractors}
    return sc.with_chunks(row), ""


def to_row(meta, result):
    base = {"id": meta["id"], "category": meta["category"], "shape": meta["shape"]}
    if result.get("error") or not result.get("text"):
        return {**base, "__failed__": f"__FAILED__: {result.get('error') or 'empty response'}"}
    row, why = validate(be.extract_json(result["text"]), meta)
    if row is None:
        return {**base, "__failed__": f"__FAILED__: invalid chain ({why})", "raw": result["text"][:3000]}
    return row


def report(rows, path):
    good = [r for r in rows if not be.failed(r)]
    by_cat, n_msgs, n_facts = {}, [], []
    for r in good:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        n_msgs.append(len(r["messages"])); n_facts.append(len(r["facts"]))
    print(f"wrote {path}: {len(rows)} rows, {len(good)} good, {len(rows) - len(good)} failed")
    print("  per category:", json.dumps(by_cat, sort_keys=True))
    if good:
        steps = sum(len(sc.steps(r)) for r in good)
        print(f"  messages/chain median {sorted(n_msgs)[len(n_msgs)//2]}, facts/chain median {sorted(n_facts)[len(n_facts)//2]}, "
              f"summary steps total {steps} (short {sum(sc.n_steps(r, 'short') for r in good)}, long {sum(sc.n_steps(r, 'long') for r in good)})")
    bad = [r for r in rows if be.failed(r)]
    if bad:
        reasons = {}
        for r in bad:
            key = r["__failed__"][:60]
            reasons[key] = reasons.get(key, 0) + 1
        print("  failures:", json.dumps(reasons))
        print("  failed rows keep a __failed__ marker; re-run the same command to retry them.")


def _ordered(rows, metas):
    order = {m["id"]: i for i, m in enumerate(metas)}
    return sorted(rows, key=lambda r: (order.get(r["id"], 10**9), r["id"]))


# ------------------------------------------------------------------ commands
def cmd_estimate(a):
    spec = be.resolve_model(a.model)
    metas = plan(a.n, a.start, a.seed)
    jobs = [make_job(m) for m in metas]
    per = sum(out_tokens(m) for m in metas) // max(1, len(metas))
    for batch in ((False, True) if spec.provider == "anthropic" else (False,)):
        usd, tin, tout = be.estimate_usd(spec, jobs, out_tokens_per_job=per, batch=batch)
        print(f"{spec}  n={a.n}  {'batch' if batch else 'sync '}  in≈{tin/1e6:.2f}M out≈{tout/1e6:.2f}M  ≈ ${usd:.2f}")
    return 0


def _todo(a):
    metas = plan(a.n, a.start, a.seed)
    existing = {r["id"]: r for r in be.read_jsonl(a.out)}
    return metas, [m for m in metas if m["id"] not in existing or be.failed(existing[m["id"]])]


def cmd_run(a):
    spec = be.resolve_model(a.model)
    metas, todo = _todo(a)
    jobs = [make_job(m) for m in todo]
    per = sum(out_tokens(m) for m in todo) // max(1, len(todo))
    usd, _, _ = be.estimate_usd(spec, jobs, per)
    print(f"{spec}: {len(todo)} chains to generate ({len(metas) - len(todo)} already good in {a.out}); "
          f"estimate ≈ ${usd:.2f}" + (f" (cap ${a.max_usd:.2f})" if a.max_usd is not None else ""))
    if not todo:
        return 0
    by_id, rows = {m["id"]: m for m in todo}, []

    def on_result(cid, r):
        row = to_row(by_id[cid], r)
        rows.append(row)
        if be.failed(row):
            print(f"  [{len(rows)}/{len(todo)}] {cid} FAIL {row['__failed__'][:80]}", flush=True)
        else:
            print(f"  [{len(rows)}/{len(todo)}] {cid} ok   {row['category']:15} {len(row['messages'])} msgs "
                  f"{len(row['facts'])} facts {len(row['changes'])} changes  {row['domain'][:40]}", flush=True)
        if len(rows) % 10 == 0:
            be.write_jsonl(a.out, _ordered(be.merge_rows(a.out, rows), metas))

    be.run_concurrent(spec, jobs, concurrency=a.concurrency, effort=a.effort, on_result=on_result, max_usd=a.max_usd)
    merged = _ordered(be.merge_rows(a.out, rows), metas)
    be.write_jsonl(a.out, merged)
    report(merged, a.out)
    return 0


def cmd_submit(a):
    spec = be.resolve_model(a.model)
    metas, todo = _todo(a)
    jobs = [make_job(m) for m in todo]
    per = sum(out_tokens(m) for m in todo) // max(1, len(todo))
    usd, _, _ = be.estimate_usd(spec, jobs, per, batch=True)
    print(f"{spec} batch: {len(todo)} chains; estimate ≈ ${usd:.2f} (reference only, no cap)")
    if not todo:
        return 0
    b = be.batch_submit(spec, jobs, effort=a.effort)
    path = be.write_manifest(KIND, {"batch_id": b["id"], "model": str(spec), "model_alias": a.model,
                                    "n": len(todo), "out": a.out, "est_usd": round(usd, 3), "metas": todo})
    print(f"submitted batch {b['id']} ({len(todo)} requests) -> manifest {path}")
    print("next: python3 gen_sessions.py fetch")
    return 0


def cmd_status(a):
    rc = 0
    for p in be.list_manifests(KIND):
        m = json.load(open(p))
        try:
            b = be.batch_status(m["batch_id"])
            st = b.get("processing_status")
            print(f"{p}: {m['model']} n={m['n']} status={st} {b.get('request_counts', {})}")
            rc = rc or (0 if st == "ended" else 1)
        except Exception as e:  # noqa: BLE001
            print(f"{p}: status check failed ({type(e).__name__}: {e})"); rc = 1
    return rc


def cmd_fetch(a):
    path = be.newest_manifest(KIND, a.manifest)
    m = json.load(open(path))
    print(f"manifest {path}: batch {m['batch_id']} {m['model']} n={m['n']} -> {m['out']}")
    if a.no_wait:
        st = be.batch_status(m["batch_id"]).get("processing_status")
        if st != "ended":
            print(f"still {st}; run fetch again later (or without --no-wait)."); return 1
    else:
        be.batch_wait(m["batch_id"], a.poll_interval)
    results = be.batch_results(m["batch_id"])
    spec = be.resolve_model(m.get("model_alias") or m["model"].split(":", 1)[1])
    rows = [to_row(meta, results.get(meta["id"], {"error": "no result for this id"})) for meta in m["metas"]]
    spent = sum(be.usage_usd(spec, r.get("usage") or {}, batch=True) for r in results.values())
    merged = sorted(be.merge_rows(m["out"], rows), key=lambda r: r["id"])
    be.write_jsonl(m["out"], merged)
    report(merged, m["out"])
    print(f"usage-based spend for this batch ≈ ${spent:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common(p, needs_model=True):
        p.add_argument("--n", type=int, default=30)
        p.add_argument("--start", type=int, default=0, help="first id number (append runs)")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--out", default="data/chains.jsonl")
        if needs_model:
            p.add_argument("--model", default="opus", help=f"alias ({', '.join(sorted(be.MODELS))}), anthropic:<id> or openrouter:<vendor/model>")
            p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max", ""],
                           help="Anthropic thinking effort (ignored for OpenRouter)")

    p = sub.add_parser("estimate", help="local cost estimate, no network"); common(p); p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("run", help="concurrent sync generation (OpenRouter or Anthropic)"); common(p)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-usd", type=float, default=None, help="stop submitting once spend passes this")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("submit", help="Anthropic Message Batch (50%% price, async)"); common(p); p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("status", help="status of every submitted sessions batch"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("fetch", help="wait for a batch, write rows to its --out")
    p.add_argument("--manifest"); p.add_argument("--no-wait", action="store_true")
    p.add_argument("--poll-interval", type=int, default=60); p.set_defaults(fn=cmd_fetch)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
