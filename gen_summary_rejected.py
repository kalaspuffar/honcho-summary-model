#!/usr/bin/env python3
"""gen_summary_rejected.py — stage 2 / Phase 0: the BASE model summarises every chunk of every chain
through the exact Honcho prompt (PLAN §2.3, §0).

Honest chaining: the previous summary for step k is the model's OWN output for step k-1 (short and
long chains are independent, as in Honcho). Records content, finish_reason, reasoning_chars
(answered inside <think> -> empty content), latency and the score against the ledger. This is both
the Phase 0 measurement and the DPO "rejected" side: rejected always comes from the base model,
never from a teacher (CLAUDE.md).

  python3 gen_summary_rejected.py --chains data/chains.jsonl --out data/rejected.jsonl
  python3 gen_summary_rejected.py --chains data/chains.jsonl --out data/rejected.jsonl --model dialectic_s50 --kind short
  python3 gen_summary_rejected.py --chains ... --out ... --model qwen9b --concurrency 8 [--max-usd 2]   # OpenRouter
  OLLAMA_BASE=http://localhost:11434/v1 python3 gen_summary_rejected.py ... --temperature 0.7

Resume-safe per chain-and-kind: a (chain, kind) whose steps are all good in --out is skipped;
anything else is regenerated as a whole (a chain's steps depend on each other).
Temperature is not sent unless --temperature is given (Honcho leaves it to the Modelfile).
Output rows: {id, chain, category, kind, k, previous_summary, output_words, max_tokens, summary, words,
finish_reason, answered_in_thinking, reasoning_chars, latency_s, prompt_tokens, completion_tokens, model, score}.
"""
import argparse
import concurrent.futures
import json
import threading
import time

import llm_backend as be
import summary_chain as sc
import summary_prompt as sp
import summary_scoring as ss


def estimate(spec, chains, kinds, mts, mtl):
    """Printed before any OpenRouter spend. Prompts are approximated with an empty previous summary."""
    jobs, out = [], 0
    for c in chains:
        for kind in kinds:
            for k in range(sc.n_steps(c, kind)):
                st = sc.build_step(c, kind, k, "", mts, mtl)
                jobs.append({"system": "", "user": st["prompt"]})
                out += min(st["max_tokens"], int(st["output_words"] * 1.4))
    usd = be.estimate_usd(spec, jobs, 0)[0] + out * spec.out_usd_m / 1e6
    return usd, len(jobs), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--out", default="data/rejected.jsonl")
    ap.add_argument("--base", default=None, help="OpenAI-compatible base URL (default: OLLAMA_BASE, or OPENROUTER_BASE for OpenRouter models)")
    ap.add_argument("--model", default=be.setting("OLLAMA_MODEL", be.OLLAMA_DEFAULT_MODEL),
                    help="Ollama tag (qwen3.5:9b, dialectic_s50) or OpenRouter alias/id (qwen9b)")
    ap.add_argument("--kind", default="both", choices=["short", "long", "both"])
    ap.add_argument("--only", default="", help="comma-separated chain ids")
    ap.add_argument("--limit", type=int, default=0, help="first N chains")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel chains (Ollama: 1-2; OpenRouter: 8)")
    ap.add_argument("--temperature", type=float, default=None, help="send a temperature (default: none, Modelfile decides)")
    ap.add_argument("--max-tokens-short", type=int, default=sp.MAX_TOKENS_SHORT_DEFAULT, help="Honcho SUMMARY_MAX_TOKENS_SHORT")
    ap.add_argument("--max-tokens-long", type=int, default=sp.MAX_TOKENS_LONG_DEFAULT, help="Honcho SUMMARY_MAX_TOKENS_LONG")
    ap.add_argument("--answer-from-reasoning", action="store_true",
                    help="when content is empty but reasoning came back, use and chain the reasoning text (baseline column only)")
    ap.add_argument("--max-usd", type=float, default=None, help="OpenRouter only: stop starting new chains past this spend")
    a = ap.parse_args()
    provider, base, model_id, api_key, spec = be.student_endpoint(a.model, a.base)
    kinds = list(sc.KINDS) if a.kind == "both" else [a.kind]

    chains = sc.load_chains(a.chains, set(a.only.split(",")) if a.only else None)
    if a.limit:
        chains = chains[:a.limit]
    existing = be.read_jsonl(a.out)
    done = {}
    for r in existing:
        done.setdefault((r["chain"], r["kind"]), []).append(r)

    def complete(c, kind):
        rows = done.get((c["id"], kind), [])
        return len(rows) == sc.n_steps(c, kind) and not any(be.failed(r) for r in rows)

    todo = [(c, kind) for c in chains for kind in kinds if not complete(c, kind)]
    n_steps = sum(sc.n_steps(c, kind) for c, kind in todo)
    print(f"{provider} {model_id} @ {base}: {len(todo)} chain-kinds / {n_steps} steps to summarise "
          f"({len(chains) * len(kinds) - len(todo)} already complete in {a.out}); max_tokens short={a.max_tokens_short} long={a.max_tokens_long}"
          + (f"; temperature {a.temperature}" if a.temperature is not None else "; temperature from the server/Modelfile"))
    if provider == "openrouter":
        usd, nj, tout = estimate(spec, [c for c, _ in todo], kinds, a.max_tokens_short, a.max_tokens_long)
        print(f"estimate: ~${usd:.2f} for {nj} steps (~{tout:,} out tokens at list price)"
              + (f"; live cap ${a.max_usd:.2f}" if a.max_usd is not None else "; no cap (--max-usd)"))
    if not todo:
        return 0
    spent, lock, stop = [0.0], threading.Lock(), threading.Event()

    def work(item):
        c, kind = item
        if stop.is_set():
            return [{"id": sc.step_id(c["id"], kind, 0), "chain": c["id"], "category": c.get("category"), "kind": kind, "k": 0,
                     "__failed__": "__FAILED__: cost cap"}]

        def gen(step):
            r = sc.summarise_step(base, model_id, step, api_key=api_key, temperature=a.temperature)
            if spec is not None:
                with lock:
                    spent[0] += be.usage_usd(spec, r.get("usage") or {})
                    if a.max_usd is not None and spent[0] > a.max_usd:
                        stop.set()
            return r

        rows = sc.walk(c, kind, gen, answer_from_reasoning=a.answer_from_reasoning,
                       max_tokens_short=a.max_tokens_short, max_tokens_long=a.max_tokens_long)
        for r in rows:
            r["model"] = a.model
        return rows

    new_rows, t0 = [], time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        for i, rows in enumerate(ex.map(work, todo), 1):
            new_rows.extend(rows)
            for r in rows:
                if be.failed(r):
                    print(f"  [{i}/{len(todo)}] {r['id']} FAIL {r['__failed__'][:70]}", flush=True)
                else:
                    s = r["score"]
                    flags = " ".join(f for f in ("bullets", "meta", "think_leak", "narration", "empty") if s.get(f))
                    if r["answered_in_thinking"]:
                        flags += " in-thinking"
                    print(f"  [{i}/{len(todo)}] {r['id']:14} {r['words']:4}w/{r['output_words']} "
                          f"new={s.get('fact_coverage_new')} carry={s.get('fact_coverage_carry')} fab={s['fabrication']} "
                          f"{r['finish_reason']} {r['latency_s']}s {flags}", flush=True)
            if i % 5 == 0:
                be.write_jsonl(a.out, sorted(be.merge_rows(a.out, new_rows), key=lambda r: r["id"]))
    merged = sorted(be.merge_rows(a.out, new_rows), key=lambda r: r["id"])
    be.write_jsonl(a.out, merged)
    good = [r for r in merged if not be.failed(r)]
    print(f"saved {a.out}: {len(merged)} rows, {len(new_rows)} new, {len(merged) - len(good)} failed, {time.time() - t0:.0f}s")
    for kind in kinds:
        rows = [r for r in good if r["kind"] == kind]
        if rows:
            agg = ss.aggregate([r["score"] for r in rows])
            agg["answered_in_thinking_rows"] = sum(1 for r in rows if r["answered_in_thinking"])
            agg["finish_length_rows"] = sum(1 for r in rows if r.get("finish_reason") == "length")
            print(f"  {kind}: {json.dumps(agg)}")
    if provider == "openrouter":
        print(f"openrouter usage-based spend ≈ ${spent[0]:.3f}" + (" (cost cap reached; rerun to resume)" if stop.is_set() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
