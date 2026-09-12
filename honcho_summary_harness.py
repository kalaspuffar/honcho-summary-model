#!/usr/bin/env python3
"""honcho_summary_harness.py — live check (PLAN §0.1, §6): push one synthetic chain through a REAL
Honcho, wait for its summariser, read the stored short/long summaries and score them against the
chain's ledger. Read-only towards data/: results go to results/, nothing is written back.

  python3 honcho_summary_harness.py --base http://honcho:8000 --workspace summary-check \
      --chain data/chains.jsonl:c00003 --label base_t01 [--key JWT] [--session S]
  python3 honcho_summary_harness.py compare results/harness-base_t01-*.json results/harness-v1-*.json

Flow (Honcho v3 API, commit in summary_prompt.py):
  POST /v3/workspaces {id}                     get-or-create workspace
  POST /v3/workspaces/{ws}/peers {id}          one per chain peer
  POST /v3/workspaces/{ws}/sessions {id, peers}
  POST /v3/workspaces/{ws}/sessions/{s}/messages {messages:[{peer_id, content}]}  in blocks of 20
  GET  /v3/workspaces/{ws}/sessions/{s}/summaries -> {short_summary, long_summary} each {content, message_id, token_count, ...}
After every block the harness polls the summaries endpoint until the short summary's message_id moves
(or --wait seconds pass), so each stored short summary is captured before the next block overwrites it.
The stored summary for block k is scored as step (short, k); the long summary at message 60j as (long, j-1).
Point Honcho at the model under test with SUMMARY_MODEL_CONFIG__* before running (PLAN §0.1).
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

import llm_backend as be
import summary_chain as sc
import summary_prompt as sp


def api(base, method, path, body=None, key=None, timeout=120):
    url = f"{base.rstrip('/')}/v3{path}"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url} -> HTTP {e.code}: {e.read()[:400].decode(errors='replace')}")


def summaries(base, ws, session, key):
    return api(base, "GET", f"/workspaces/{ws}/sessions/{session}/summaries", key=key)


def wait_for(base, ws, session, key, kind, prev_msg_id, seconds, poll=5):
    """Poll until <kind>_summary exists with a message_id different from prev_msg_id."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        s = summaries(base, ws, session, key).get(f"{kind}_summary")
        if s and s.get("message_id") != prev_msg_id:
            return s, round(time.time() - t0, 1)
        time.sleep(poll)
    return None, round(time.time() - t0, 1)


def load_chain(spec):
    path, _, cid = spec.rpartition(":")
    if not path:
        raise SystemExit("--chain must be <chains.jsonl>:<chain id>")
    return sc.load_chains(path, {cid})[0]


def run(a):
    chain = load_chain(a.chain)
    key = a.key or os.environ.get("HONCHO_API_KEY")
    session = a.session or f"{chain['id']}-{time.strftime('%Y%m%d-%H%M%S')}"
    names = sc.peer_names(chain)
    api(a.base, "POST", "/workspaces", {"id": a.workspace}, key)
    for n in names:
        api(a.base, "POST", f"/workspaces/{a.workspace}/peers", {"id": n}, key)
    api(a.base, "POST", f"/workspaces/{a.workspace}/sessions", {"id": session, "peers": {n: {} for n in names}}, key)
    print(f"workspace {a.workspace} session {session} peers {names}; {len(chain['messages'])} messages in blocks of {sc.CHUNK}", file=sys.stderr)

    rows, last_ids = [], {"short": None, "long": None}
    mts, mtl = a.max_tokens_short, a.max_tokens_long
    prev_stored = {"short": "", "long": ""}
    for k, ch in enumerate(chain["chunks"]):
        lo, hi = ch["seqs"]
        block = [{"peer_id": m["peer"], "content": m["text"]} for m in chain["messages"] if lo <= m["seq"] <= hi]
        api(a.base, "POST", f"/workspaces/{a.workspace}/sessions/{session}/messages", {"messages": block}, key)
        due = [("short", k)] + ([("long", hi // sc.LONG_EVERY - 1)] if hi % sc.LONG_EVERY == 0 else [])
        for kind, j in due:
            s, waited = wait_for(a.base, a.workspace, session, key, kind, last_ids[kind], a.wait, a.poll)
            content = (s or {}).get("content", "") or ""
            # score exactly as Honcho computed the limit: previous = the previously stored summary of this kind
            step = sc.build_step(chain, kind, j, prev_stored[kind], mts, mtl)
            row = {"id": sc.step_id(chain["id"], kind, j), "chain": chain["id"], "kind": kind, "k": j, "messages_through": hi,
                   "stored": s is not None, "waited_s": waited, "words": len(content.split()), "output_words": step["output_words"],
                   "token_count": (s or {}).get("token_count"), "summary": content,
                   "score": sc.score_step(chain, kind, j, content, step["output_words"])}
            rows.append(row)
            if s is not None:
                last_ids[kind] = s.get("message_id"); prev_stored[kind] = content
            sc_ = row["score"]
            print(f"  block {k} ({kind} {j}) {'stored' if s else 'NOT STORED'} after {waited}s: {row['words']}w/{row['output_words']} "
                  f"new={sc_.get('fact_coverage_new')} carry={sc_.get('fact_coverage_carry')} fab={sc_['fabrication']}"
                  f"{' bullets' if sc_['bullets'] else ''}{' meta' if sc_['meta'] else ''}{' EMPTY' if sc_['empty'] else ''}", file=sys.stderr)

    def agg(kind):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            return None
        import summary_scoring as ss
        x = ss.aggregate([r["score"] for r in rs])
        x["not_stored_rows"] = sum(1 for r in rs if not r["stored"])
        x["median_wait_s"] = round(statistics.median(r["waited_s"] for r in rs), 1)
        return x
    summary = {"label": a.label, "base": a.base, "workspace": a.workspace, "session": session, "chain": chain["id"],
               "category": chain.get("category"), "n_messages": len(chain["messages"]), "short": agg("short"), "long": agg("long"),
               "ts": time.strftime("%Y%m%d-%H%M%S")}
    out = a.out or f"results/harness-{a.label}-{summary['ts']}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    print("saved", out, file=sys.stderr)


def compare(a):
    runs = [json.load(open(p)) for p in a.files]
    labels = [r["summary"].get("label", p) for r, p in zip(runs, a.files)]
    print(f"{'step':16}" + "".join(f"{l[:22]:>24}" for l in labels))
    for sid in [r["id"] for r in runs[0]["rows"]]:
        cells = []
        for r in runs:
            m = next((x for x in r["rows"] if x["id"] == sid), None)
            if m is None:
                cells.append("-"); continue
            s = m["score"]
            cells.append(f"{m['words']}w/{m['output_words']} n={s.get('fact_coverage_new')} c={s.get('fact_coverage_carry')}" + (" FAB" if s["fabrication"] else ""))
        print(f"{sid:16}" + "".join(f"{c:>24}" for c in cells))
    for kind in sc.KINDS:
        for k in ("n", "median_limit_ratio", "over_limit_rows", "median_fact_coverage_new", "median_fact_coverage_carry",
                  "fabrication_rows", "bullet_rows", "meta_rows", "empty_rows", "not_stored_rows", "median_wait_s"):
            print(f"{kind + '.' + k:34}" + "".join(f"{str((r['summary'].get(kind) or {}).get(k, '')):>24}" for r in runs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run"); p.set_defaults(fn=run)
    p.add_argument("--base", default=os.environ.get("HONCHO_BASE", "http://localhost:8000"))
    p.add_argument("--workspace", required=True, help="a scratch workspace for this check (never the production one)")
    p.add_argument("--chain", required=True, help="<chains.jsonl>:<chain id>")
    p.add_argument("--session", default=None, help="default: <chain id>-<timestamp>")
    p.add_argument("--label", required=True, help="model/config name for the record, e.g. base_t01")
    p.add_argument("--key", default=None, help="JWT if Honcho runs with AUTH_USE_AUTH (or env HONCHO_API_KEY)")
    p.add_argument("--wait", type=int, default=600, help="seconds to wait for each summary")
    p.add_argument("--poll", type=int, default=5)
    p.add_argument("--max-tokens-short", type=int, default=sp.MAX_TOKENS_SHORT_DEFAULT, help="Honcho's SUMMARY_MAX_TOKENS_SHORT (for the limit)")
    p.add_argument("--max-tokens-long", type=int, default=sp.MAX_TOKENS_LONG_DEFAULT)
    p.add_argument("--out", default=None, help="default results/harness-<label>-<ts>.json")
    p = sub.add_parser("compare"); p.set_defaults(fn=compare)
    p.add_argument("files", nargs="+")
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "compare", "-h", "--help"):
        argv = ["run"] + argv
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
