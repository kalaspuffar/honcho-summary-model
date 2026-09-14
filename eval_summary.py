#!/usr/bin/env python3
"""eval_summary.py — offline eval (PLAN §6): a model summarises held-out chains chunk by chunk through
the exact Honcho prompt via an OpenAI-compatible /v1 (Ollama or OpenRouter) and is scored with
summary_scoring.py. Previous summary = the model's OWN output for the previous step (honest setting).

  python3 eval_summary.py --chains data/chains.jsonl --ids-from data/dataset_eval.sft.jsonl \
      --model qwen3.5:9b --base http://node7.ea.org:11434/v1 --out results/eval_base.jsonl --answer-from-reasoning
  python3 eval_summary.py --chains data/chains.jsonl --ids-from data/dataset_eval.sft.jsonl \
      --model summary-v1 --out results/eval_v1.jsonl
  python3 eval_summary.py compare results/eval_base.jsonl results/eval_v1.jsonl
  python3 eval_summary.py rescore results/eval_v1.jsonl --chains data/chains.jsonl

--ids-from restricts to the chains named in a dataset file (its rows carry "chain"; use the *eval*
split — the model never saw those personas). Without it every good chain is used.
Writes <out> (one row per step, scored) and <out>.summary.json (summary_scoring.aggregate per kind plus
answered_in_thinking_rows, finish_length_rows, median latency).
"""
import argparse
import concurrent.futures
import json
import os
import statistics
import sys

import llm_backend as be
import summary_chain as sc
import summary_prompt as sp
import summary_scoring as ss


def aggregate(rows, kind):
    rs = [r for r in rows if r["kind"] == kind and not be.failed(r)]
    if not rs:
        return None
    agg = ss.aggregate([r["score"] for r in rs])
    agg.update(answered_in_thinking_rows=sum(1 for r in rs if r.get("answered_in_thinking")),
               finish_length_rows=sum(1 for r in rs if r.get("finish_reason") == "length"),
               over_limit_rate=round(agg["over_limit_rows"] / agg["n"], 3),
               median_latency_s=round(statistics.median(r.get("latency_s") or 0 for r in rs), 1),
               failed_rows=sum(1 for r in rows if r["kind"] == kind and be.failed(r)))
    return agg


def summary_path(out):
    return os.path.splitext(out)[0] + ".summary.json"


def breakdown(rows, kind):
    """median new/carry coverage and over-limit count per category and per step k (where the smoke was weak)."""
    out = {"by_category": {}, "by_k": {}}
    rs = [r for r in rows if r["kind"] == kind and not be.failed(r)]
    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 3) if xs else None
    for key, sel in (("by_category", lambda r: r.get("category")), ("by_k", lambda r: r["k"])):
        groups = {}
        for r in rs:
            groups.setdefault(sel(r), []).append(r)
        for g, grs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            out[key][str(g)] = {"n": len(grs), "new": med([r["score"].get("fact_coverage_new") for r in grs]),
                                "carry": med([r["score"].get("fact_coverage_carry") for r in grs]),
                                "over": sum(1 for r in grs if r["score"]["over_limit"]),
                                "flags": sum(1 for r in grs if any(r["score"].get(f) for f in ("bullets", "meta", "think_leak", "echo", "empty")))}
    return out


def write_summary(out, rows, meta):
    summ = {**meta, "n_rows": len(rows), "short": aggregate(rows, "short"), "long": aggregate(rows, "long"),
            "short_breakdown": breakdown(rows, "short"), "long_breakdown": breakdown(rows, "long")}
    with open(summary_path(out), "w") as f:
        json.dump(summ, f, indent=2)
    return summ


def run(a):
    ids = None
    if a.ids_from:
        ids = {r.get("chain") or r["id"] for r in be.read_jsonl(a.ids_from)}
    chains = sc.load_chains(a.chains, ids)
    if a.limit:
        chains = chains[:a.limit]
    kinds = list(sc.KINDS) if a.kind == "both" else [a.kind]
    provider, base, model_id, api_key, _spec = be.student_endpoint(a.model, a.base)
    todo = [(c, k) for c in chains for k in kinds]
    print(f"{provider} {model_id} @ {base}: {len(chains)} chains, {sum(sc.n_steps(c, k) for c, k in todo)} steps"
          + (f", temperature {a.temperature}" if a.temperature is not None else ", temperature from the server/Modelfile"), file=sys.stderr)
    xb = {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else None

    def work(item):
        c, kind = item
        return sc.walk(c, kind, lambda st: sc.summarise_step(base, model_id, st, api_key=api_key, temperature=a.temperature, extra_body=xb),
                       answer_from_reasoning=a.answer_from_reasoning, max_tokens_short=a.max_tokens_short,
                       max_tokens_long=a.max_tokens_long, stop_on_error=False)

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        for i, out in enumerate(ex.map(work, todo), 1):
            rows.extend(out)
            for r in out:
                if be.failed(r):
                    print(f"[{i}/{len(todo)}] {r['id']} FAIL {r['__failed__'][:70]}", file=sys.stderr)
                    continue
                s = r["score"]
                flags = " ".join(f for f in ("bullets", "meta", "think_leak", "narration", "empty") if s.get(f))
                print(f"[{i}/{len(todo)}] {r['id']:14} [{r['category']:15}] {r['words']:4}w/{r['output_words']} new={s.get('fact_coverage_new')} "
                      f"carry={s.get('fact_coverage_carry')} state={s.get('latest_state')} fab={s['fabrication']} "
                      f"{r['finish_reason']} {r['latency_s']}s {'in-thinking ' if r['answered_in_thinking'] else ''}{flags}", file=sys.stderr)
    for r in rows:
        r["model"] = a.model
    be.write_jsonl(a.out, rows)
    summ = write_summary(a.out, rows, {"model": a.model, "base": base, "temperature": a.temperature,
                                       "reasoning_effort": a.reasoning_effort, "scored_from_reasoning": bool(a.answer_from_reasoning),
                                       "max_tokens_short": a.max_tokens_short, "max_tokens_long": a.max_tokens_long,
                                       "chains": len(chains)})
    print("\nAGGREGATE\n" + json.dumps(summ, indent=2))


def rescore(a):
    """Re-score an existing results jsonl with the current summary_scoring.py (scorer changes must
    not require re-running the model)."""
    chains = {c["id"]: c for c in sc.load_chains(a.chains)}
    rows = be.read_jsonl(a.file)
    for r in rows:
        if be.failed(r):
            continue
        c = chains.get(r["chain"])
        if c is None:
            sys.exit(f"{r['chain']} not in {a.chains}")
        r["score"] = sc.score_step(c, r["kind"], r["k"], r["summary"], r["output_words"])
    old = json.load(open(summary_path(a.file))) if os.path.exists(summary_path(a.file)) else {}
    meta = {k: v for k, v in old.items() if k not in ("short", "long", "n_rows")}
    meta["rescored"] = True
    be.write_jsonl(a.file, rows)
    print(json.dumps(write_summary(a.file, rows, meta), indent=2))


KEYS = ["n", "median_words", "median_limit_ratio", "over_limit_rows", "over_limit_rate", "median_fact_coverage_new",
        "median_fact_coverage_carry", "median_latest_state", "fabrication_rows", "bullet_rows", "meta_rows",
        "think_leak_rows", "narration_rows", "echo_rows", "empty_rows", "answered_in_thinking_rows", "finish_length_rows",
        "failed_rows", "median_latency_s"]


def compare(a):
    tables = []
    for p in a.files:
        sp_ = summary_path(p)
        if not os.path.exists(sp_):
            sys.exit(f"missing {sp_} (produced by a run)")
        tables.append(json.load(open(sp_)))
    w = max(len(k) for k in KEYS) + 2
    for kind in sc.KINDS:
        if not any(t.get(kind) for t in tables):
            continue
        print(f"\n== {kind} ==")
        print(" " * w + "".join(f"{str(t.get('model', '?'))[:20]:>22}" for t in tables))
        print(f"{'temperature':{w}}" + "".join(f"{str(t.get('temperature', '')):>22}" for t in tables))
        print(f"{'scored_from_reasoning':{w}}" + "".join(f"{str(t.get('scored_from_reasoning', '')):>22}" for t in tables))
        for k in KEYS:
            print(f"{k:{w}}" + "".join(f"{str((t.get(kind) or {}).get(k, '')):>22}" for t in tables))
        for key, title in (("by_category", "carry / new by category"), ("by_k", "carry / new by step k")):
            groups = sorted({g for t in tables for g in (t.get(f"{kind}_breakdown") or {}).get(key, {})})
            if not groups:
                continue
            print(f"-- {title}")
            for g in groups:
                cells = []
                for t in tables:
                    b = (t.get(f"{kind}_breakdown") or {}).get(key, {}).get(g)
                    cells.append("-" if not b else f"{b['carry'] if b['carry'] is not None else '-'} / {b['new'] if b['new'] is not None else '-'} (n={b['n']})")
                print(f"{g:{w}}" + "".join(f"{c:>22}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run"); p.set_defaults(fn=run)
    p.add_argument("--chains", required=True)
    p.add_argument("--ids-from", default=None, help="dataset jsonl whose rows' `chain` ids select the eval chains")
    p.add_argument("--model", required=True)
    p.add_argument("--base", default=None, help="default: OLLAMA_BASE, or OPENROUTER_BASE for OpenRouter models")
    p.add_argument("--out", required=True)
    p.add_argument("--kind", default="both", choices=["short", "long", "both"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--temperature", type=float, default=None, help="send a temperature (default: none, Modelfile decides)")
    p.add_argument("--max-tokens-short", type=int, default=sp.MAX_TOKENS_SHORT_DEFAULT)
    p.add_argument("--max-tokens-long", type=int, default=sp.MAX_TOKENS_LONG_DEFAULT)
    p.add_argument("--reasoning-effort", default=None, help="send reasoning_effort (e.g. none) as Honcho's MODEL_CONFIG__THINKING_EFFORT would")
    p.add_argument("--answer-from-reasoning", action="store_true",
                   help="score and chain the reasoning text when content is empty (baseline column; the tuned model never gets this)")
    p = sub.add_parser("compare"); p.set_defaults(fn=compare)
    p.add_argument("files", nargs="+")
    p = sub.add_parser("rescore", help="re-score a results jsonl with the current scorer (rewrites it + its summary)")
    p.set_defaults(fn=rescore)
    p.add_argument("file")
    p.add_argument("--chains", required=True)
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "compare", "rescore", "-h", "--help"):
        argv = ["run"] + argv
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
