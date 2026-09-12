#!/usr/bin/env python3
"""gen_summary_chosen.py — stage 3: the teacher writes the CHOSEN summaries (PLAN §2.2).

For every step (kind, k) of every chain the teacher answers the EXACT Honcho prompt (the same user
turn the student gets, from summary_prompt.py via summary_chain.build_step). Two variants:

  clean      (id c00003-s2)   previous summary = the teacher's own chosen summary of step k-1
  base-prev  (id c00003-s2b)  previous summary = the BASE model's summary of step k-1 from
                              --rejected (stage 2); the teacher repairs what the base dropped.
                              ~--base-prev-share of the steps with k >= 1 whose base k-1 is non-empty.
                              These rows share their prompt with the rejected row -> DPO pairs.

The system prompt adds the summariser's hard rules and, unless --blind, the ledger checklist (facts
due so far, changed values, forbidden distractors) — the student never sees it; it only makes the
target denser. build_summary_dataset.py filters with the scorer afterwards regardless.

Steps depend on the previous step, so:
  run     concurrent chains, sequential steps inside a chain (Anthropic sync or OpenRouter)
  submit  one Anthropic batch per WAVE (the next step of every chain); fetch writes it; repeat
          submit/fetch until `submit` reports nothing left (--waves in fetch loops for you).

  python3 gen_summary_chosen.py estimate --chains data/chains.jsonl --model opus [--rejected data/rejected.jsonl]
  python3 gen_summary_chosen.py run      --chains data/chains.jsonl --model opus --out data/chosen.jsonl --rejected data/rejected.jsonl
  python3 gen_summary_chosen.py submit   --chains ... --model opus --out data/chosen.jsonl --rejected data/rejected.jsonl
  python3 gen_summary_chosen.py fetch    [--waves]        # fetch, then submit+fetch the next waves until done
  python3 gen_summary_chosen.py status

Output rows: {id, chain, category, kind, k, variant, previous_from, previous_summary, output_words,
max_tokens, summary, words, teacher, score}. Failed rows carry "__failed__" and are retried by a re-run.
"""
import argparse
import hashlib
import json
import os

import llm_backend as be
import summary_chain as sc
import summary_prompt as sp
import summary_scoring as ss

KIND = "chosen"
TARGET_RATIO = 0.9    # train under the limit: models overshoot (PLAN §2.2)

SYSTEM = """You are the reference summariser whose outputs train a small model to summarise conversations
for a memory system. The user message is the exact production prompt; answer it as the best possible
summariser would, and obey these hard rules on top of it:

1. NARRATIVE: one or more plain paragraphs in chronological order. No bullet points, no numbered
   lists, no headings, no markdown, no labels like "Summary:".
2. NOTHING BUT THE SUMMARY: no preamble, no meta-commentary, no closing remark, no mention of
   "the previous summary" or "the conversation above" as objects — just tell what happened.
3. LENGTH: stay at or under {ratio} of the stated hard limit; if the conversation has little
   content, be SHORT — never pad. Count words honestly.
4. GROUNDED: every name, date, number, place and item comes from the previous summary or the new
   messages. Quote exact values (do not round, rename or paraphrase them). Invent nothing.
5. MERGE: every fact in the previous summary survives unless the new messages explicitly change it;
   then state the new value as current (mentioning the change is fine). Attribute statements to the
   right speaker when speakers differ.
6. Prefer explicit facts, preferences, questions and decisions over mood and generalities. The long
   variant may additionally describe emotional state and recurring themes, briefly, after the facts.
"""

CHECKLIST = """
CHECKLIST (for you only; never mention it): these values must be present in your summary if they are
still current — copy them exactly:
{facts}
{changes}
Never state these (they were NOT said): {distractors}
"""


def checklist(chain, kind, k):
    view = sc.chain_view(chain, kind)
    new, carry = ss.facts_due(view, k)
    hi = view["chunks"][k]["seqs"][1]
    facts = {f["id"]: f["text"] for f in chain["facts"]}
    ch = [f"- {facts.get(c['fact_id'])} was replaced by {facts.get(c['superseded_by'])} (give the latter as current)"
          for c in chain.get("changes", []) if c["seq"] <= hi]
    return CHECKLIST.format(facts="\n".join(f"- {t}" for t in carry + new) or "- (none)",
                            changes=("Changed values:\n" + "\n".join(ch)) if ch else "",
                            distractors="; ".join(d["text"] for d in chain.get("distractors", [])) or "(none)")


def make_job(chain, kind, k, previous, variant, a):
    step = sc.build_step(chain, kind, k, previous, a.max_tokens_short, a.max_tokens_long)
    system = SYSTEM.format(ratio=f"{int(TARGET_RATIO * 100)}%")
    if not a.blind:
        system += checklist(chain, kind, k)
    return {"custom_id": sc.step_id(chain["id"], kind, k, variant), "system": system, "user": step["prompt"],
            "max_tokens": step["max_tokens"], "_step": step}


def _pick_base_prev(cid, kind, k, share):
    h = int(hashlib.sha1(f"{cid}-{kind}-{k}".encode()).hexdigest(), 16) % 1000
    return h < int(share * 1000)


def to_row(chain, kind, k, variant, previous_from, step, result, teacher):
    row = {"id": sc.step_id(chain["id"], kind, k, variant), "chain": chain["id"], "category": chain.get("category"),
           "kind": kind, "k": k, "variant": "base_prev" if variant else "clean", "previous_from": previous_from,
           "previous_summary": step["previous_summary"], "output_words": step["output_words"],
           "max_tokens": step["max_tokens"], "teacher": teacher}
    text = (result.get("text") or "").strip()
    if result.get("error") or not text:
        row.update(summary="", words=0, __failed__=f"__FAILED__: {result.get('error') or 'empty response'}")
        return row
    row.update(summary=text, words=ss.words(text), score=sc.score_step(chain, kind, k, text, step["output_words"]))
    return row


def load_rejected(path):
    """{(chain, kind, k): summary} of good, non-empty base summaries."""
    out = {}
    for r in be.read_jsonl(path) if path else []:
        if not be.failed(r) and (r.get("summary") or "").strip():
            out[(r["chain"], r["kind"], r["k"])] = r["summary"].strip()
    return out


def wanted_steps(chain, rejected, share):
    """[(kind, k, variant)] every row this chain should end up with."""
    out = []
    for kind in sc.KINDS:
        for k in range(sc.n_steps(chain, kind)):
            out.append((kind, k, ""))
            if k >= 1 and (chain["id"], kind, k - 1) in rejected and _pick_base_prev(chain["id"], kind, k, share):
                out.append((kind, k, "b"))
    return out


def previous_for(chain, kind, k, variant, chosen, rejected):
    """(previous_text, previous_from) or (None, None) if the dependency is not ready."""
    if k == 0:
        return "", None
    if variant == "b":
        return rejected[(chain["id"], kind, k - 1)], f"rejected:{sc.step_id(chain['id'], kind, k - 1)}"
    pid = sc.step_id(chain["id"], kind, k - 1)
    prev = chosen.get(pid)
    if prev is None or be.failed(prev):
        return None, None
    return prev["summary"], f"chosen:{pid}"


def report(rows, path):
    good = [r for r in rows if not be.failed(r)]
    print(f"wrote {path}: {len(rows)} rows, {len(good)} good, {len(rows) - len(good)} failed")
    for kind in sc.KINDS:
        rs = [r for r in good if r["kind"] == kind]
        if rs:
            agg = ss.aggregate([r["score"] for r in rs])
            agg["base_prev_rows"] = sum(1 for r in rs if r["variant"] == "base_prev")
            print(f"  {kind}: {json.dumps(agg)}")
    if len(good) < len(rows):
        print("  failed rows keep a __failed__ marker; re-run the same command to retry them.")


# ------------------------------------------------------------------ commands
def _setup(a):
    chains = sc.load_chains(a.chains)
    rejected = load_rejected(a.rejected)
    out = a.out or os.path.join(os.path.dirname(a.chains), "chosen.jsonl")
    chosen = {r["id"]: r for r in be.read_jsonl(out)}
    return chains, rejected, out, chosen


def _all_jobs_estimate(a, chains, rejected):
    jobs, tout = [], 0
    for c in chains:
        for kind, k, variant in wanted_steps(c, rejected, a.base_prev_share):
            prev = "x" * (600 if kind == "short" else 3000) * 5 if k else ""     # stand-in for a previous summary
            j = make_job(c, kind, k, prev, variant, a)
            jobs.append(j); tout += int(j["_step"]["output_words"] * TARGET_RATIO * 1.4)
    return jobs, tout


def cmd_estimate(a):
    spec = be.resolve_model(a.model)
    chains, rejected, _, _ = _setup(a)
    jobs, tout = _all_jobs_estimate(a, chains, rejected)
    per = tout // max(1, len(jobs))
    for batch in ((False, True) if spec.provider == "anthropic" else (False,)):
        usd, tin, to = be.estimate_usd(spec, jobs, out_tokens_per_job=per, batch=batch)
        print(f"{spec}  steps={len(jobs)} ({sum(1 for j in jobs if j['custom_id'].endswith('b'))} base-prev)  "
              f"{'batch' if batch else 'sync '}  in≈{tin/1e6:.2f}M out≈{to/1e6:.2f}M  ≈ ${usd:.2f}")
    if not rejected:
        print("  (no --rejected given or no usable base summaries: no base-prev variants)")
    return 0


def cmd_run(a):
    spec = be.resolve_model(a.model)
    chains, rejected, out, chosen = _setup(a)
    key = be.key_for(spec)
    todo = [c for c in chains if any(sc.step_id(c["id"], kd, k, v) not in chosen or be.failed(chosen[sc.step_id(c["id"], kd, k, v)])
                                     for kd, k, v in wanted_steps(c, rejected, a.base_prev_share))]
    jobs, tout = _all_jobs_estimate(a, todo, rejected)
    usd, _, _ = be.estimate_usd(spec, jobs, tout // max(1, len(jobs)))
    print(f"{spec}: {len(todo)} chains with missing steps ({len(jobs)} steps at most); estimate ≈ ${usd:.2f}"
          + (f" (cap ${a.max_usd:.2f})" if a.max_usd is not None else ""))
    if not todo:
        return 0
    import concurrent.futures
    import threading
    lock, spent, stop = threading.Lock(), [0.0], threading.Event()
    new_rows = []

    def work(c):
        local = {}
        rows = []
        for kind, k, variant in wanted_steps(c, rejected, a.base_prev_share):
            sid = sc.step_id(c["id"], kind, k, variant)
            have = chosen.get(sid)
            if have is not None and not be.failed(have):
                local[sid] = have
                continue
            prev, pfrom = previous_for(c, kind, k, variant, {**chosen, **local}, rejected)
            if prev is None or stop.is_set():
                continue                                    # previous step failed / cap: retried next run
            job = make_job(c, kind, k, prev, variant, a)
            r = be.complete_with_retry(spec, key, job, a.effort)
            with lock:
                spent[0] += be.usage_usd(spec, r.get("usage") or {})
                if a.max_usd is not None and spent[0] > a.max_usd:
                    stop.set()
            row = to_row(c, kind, k, variant, pfrom, job["_step"], r, a.model)
            local[sid] = row
            rows.append(row)
        return rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        for i, rows in enumerate(ex.map(work, todo), 1):
            new_rows.extend(rows)
            for r in rows:
                if be.failed(r):
                    print(f"  [{i}/{len(todo)}] {r['id']} FAIL {r['__failed__'][:70]}", flush=True)
                else:
                    s = r["score"]
                    print(f"  [{i}/{len(todo)}] {r['id']:14} {r['words']:4}w/{r['output_words']} new={s.get('fact_coverage_new')} "
                          f"carry={s.get('fact_coverage_carry')} fab={s['fabrication']}"
                          f"{' bullets' if s['bullets'] else ''}{' meta' if s['meta'] else ''}", flush=True)
            if i % 5 == 0:
                be.write_jsonl(out, sorted(be.merge_rows(out, new_rows), key=lambda r: r["id"]))
    merged = sorted(be.merge_rows(out, new_rows), key=lambda r: r["id"])
    be.write_jsonl(out, merged)
    report(merged, out)
    print(f"usage-based spend ≈ ${spent[0]:.3f}" + (" (cost cap reached; rerun to resume)" if stop.is_set() else ""))
    return 0


def _wave(a, chains, rejected, chosen):
    """Jobs for every wanted step whose dependency is satisfied and that is not yet good."""
    jobs, metas = [], []
    for c in chains:
        for kind, k, variant in wanted_steps(c, rejected, a.base_prev_share):
            sid = sc.step_id(c["id"], kind, k, variant)
            have = chosen.get(sid)
            if have is not None and not be.failed(have):
                continue
            prev, pfrom = previous_for(c, kind, k, variant, chosen, rejected)
            if prev is None:
                continue
            j = make_job(c, kind, k, prev, variant, a)
            jobs.append({k_: v for k_, v in j.items() if k_ != "_step"})
            metas.append({"id": sid, "chain": c["id"], "kind": kind, "k": k, "variant": variant, "previous_from": pfrom,
                          "previous_summary": j["_step"]["previous_summary"], "output_words": j["_step"]["output_words"],
                          "max_tokens": j["_step"]["max_tokens"]})
    return jobs, metas


def _remaining(chains, rejected, chosen, share):
    return sum(1 for c in chains for kd, k, v in wanted_steps(c, rejected, share)
               if sc.step_id(c["id"], kd, k, v) not in chosen or be.failed(chosen[sc.step_id(c["id"], kd, k, v)]))


def cmd_submit(a):
    spec = be.resolve_model(a.model)
    chains, rejected, out, chosen = _setup(a)
    jobs, metas = _wave(a, chains, rejected, chosen)
    left = _remaining(chains, rejected, chosen, a.base_prev_share)
    if not jobs:
        print(f"nothing submittable: {left} steps remain" + (" (their previous steps failed — fix/retry those)" if left else " — all done"))
        return 0
    usd, _, _ = be.estimate_usd(spec, jobs, int(sum(m["output_words"] for m in metas) / len(metas) * TARGET_RATIO * 1.4), batch=True)
    print(f"{spec} batch wave: {len(jobs)} steps now, {left - len(jobs)} wait for a later wave; estimate ≈ ${usd:.2f} (reference only)")
    b = be.batch_submit(spec, jobs, effort=a.effort)
    path = be.write_manifest(KIND, {"batch_id": b["id"], "model": str(spec), "model_alias": a.model, "n": len(jobs), "out": out,
                                    "chains": os.path.abspath(a.chains), "rejected": os.path.abspath(a.rejected) if a.rejected else None,
                                    "base_prev_share": a.base_prev_share, "blind": a.blind, "effort": a.effort,
                                    "max_tokens_short": a.max_tokens_short, "max_tokens_long": a.max_tokens_long,
                                    "est_usd": round(usd, 3), "metas": metas})
    print(f"submitted batch {b['id']} ({len(jobs)} requests) -> manifest {path}")
    print("next: python3 gen_summary_chosen.py fetch --waves   (fetch, then submit+fetch the remaining waves)")
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


def _fetch_one(path, poll, no_wait):
    m = json.load(open(path))
    print(f"manifest {path}: batch {m['batch_id']} {m['model']} n={m['n']} -> {m['out']}")
    if no_wait:
        st = be.batch_status(m["batch_id"]).get("processing_status")
        if st != "ended":
            print(f"still {st}; run fetch again later."); return None
    else:
        be.batch_wait(m["batch_id"], poll)
    chains = {c["id"]: c for c in sc.load_chains(m["chains"])}
    results = be.batch_results(m["batch_id"])
    spec = be.resolve_model(m.get("model_alias") or m["model"].split(":", 1)[1])
    rows = []
    for meta in m["metas"]:
        c = chains[meta["chain"]]
        step = {"previous_summary": meta["previous_summary"], "output_words": meta["output_words"], "max_tokens": meta["max_tokens"]}
        rows.append(to_row(c, meta["kind"], meta["k"], meta["variant"], meta["previous_from"], step,
                           results.get(meta["id"], {"error": "no result for this id"}), m.get("model_alias", m["model"])))
    spent = sum(be.usage_usd(spec, r.get("usage") or {}, batch=True) for r in results.values())
    merged = sorted(be.merge_rows(m["out"], rows), key=lambda r: r["id"])
    be.write_jsonl(m["out"], merged)
    report(merged, m["out"])
    print(f"usage-based spend for this batch ≈ ${spent:.3f}")
    return m


def cmd_fetch(a):
    path = be.newest_manifest(KIND, a.manifest)
    m = _fetch_one(path, a.poll_interval, a.no_wait)
    if m is None:
        return 1
    if not a.waves:
        return 0
    # loop: submit the next wave with the manifest's settings, wait, fetch — until nothing is left
    ns = argparse.Namespace(chains=m["chains"], rejected=m.get("rejected"), out=m["out"], model=m.get("model_alias", m["model"]),
                            effort=m.get("effort", "medium"), base_prev_share=m.get("base_prev_share", 0.3), blind=m.get("blind", False),
                            max_tokens_short=m.get("max_tokens_short", sp.MAX_TOKENS_SHORT_DEFAULT),
                            max_tokens_long=m.get("max_tokens_long", sp.MAX_TOKENS_LONG_DEFAULT))
    for wave in range(1, 20):
        before = len(be.list_manifests(KIND))
        cmd_submit(ns)
        if len(be.list_manifests(KIND)) == before:
            break
        print(f"-- wave {wave + 1}")
        if _fetch_one(be.newest_manifest(KIND), a.poll_interval, False) is None:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--chains", required=True)
        p.add_argument("--rejected", default=None, help="stage-2 rows; enables the base-previous variant")
        p.add_argument("--out", default=None, help="default: chosen.jsonl next to --chains")
        p.add_argument("--model", default="opus", help=f"alias ({', '.join(sorted(be.MODELS))}), anthropic:<id> or openrouter:<vendor/model>")
        p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max", ""])
        p.add_argument("--base-prev-share", type=float, default=0.3, help="share of k>=1 steps that also get a base-previous variant")
        p.add_argument("--blind", action="store_true", help="no ledger checklist in the system prompt")
        p.add_argument("--max-tokens-short", type=int, default=sp.MAX_TOKENS_SHORT_DEFAULT, help="Honcho SUMMARY_MAX_TOKENS_SHORT")
        p.add_argument("--max-tokens-long", type=int, default=sp.MAX_TOKENS_LONG_DEFAULT, help="Honcho SUMMARY_MAX_TOKENS_LONG")

    p = sub.add_parser("estimate"); common(p); p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("run", help="concurrent chains, sequential steps (Anthropic sync or OpenRouter)"); common(p)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-usd", type=float, default=None)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("submit", help="Anthropic Message Batch: one wave (the next ready step of every chain)"); common(p); p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("fetch"); p.add_argument("--manifest"); p.add_argument("--no-wait", action="store_true")
    p.add_argument("--poll-interval", type=int, default=60)
    p.add_argument("--waves", action="store_true", help="after fetching, keep submitting and fetching waves until every step is done")
    p.set_defaults(fn=cmd_fetch)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
