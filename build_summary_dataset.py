#!/usr/bin/env python3
"""build_summary_dataset.py — stage 4: filter the teacher's summaries with the scorer, split by
persona, emit SFT rows (PLAN §2.2 filters, §3 row shape).

  python3 build_summary_dataset.py --chains data/chains.jsonl --chosen data/chosen.jsonl \
      [--rejected data/rejected.jsonl] --out data/dataset

writes  data/dataset_train.sft.jsonl  {"id", "chain", "kind", "k", "variant", "category",
                                       "messages": [{"role":"user","content":<exact Honcho prompt>},
                                                    {"role":"assistant","content":<chosen>}]}
        data/dataset_eval.sft.jsonl   held-out chains (stratified by category; no human peer name shared with train), --eval-frac
        data/dataset_{train,eval}.dpo.jsonl  only with --rejected: {"id", "prompt": [user], "chosen", "rejected"}
             pairs share the SAME prompt: rejected step k vs the chosen base-previous variant of step k
             (its previous summary IS the base's k-1 output), plus step 0 (no previous) clean vs rejected.

Filters on a chosen row (all must hold):
  not failed / not empty
  fact_coverage_new  >= --min-coverage (0.9) when facts are due in this chunk
  fact_coverage_carry >= --min-coverage        when earlier facts are due (the merge metric)
  latest_state == 1.0 when any value has changed
  fabrication False; bullets/meta/think_leak/narration/echo False
  words <= --max-ratio (0.9) * output_words
The prompt is rebuilt from the chain + the stored previous_summary, so a row trains exactly what
Honcho will send. Rows are never truncated here; train_lora.py drops rows above --max-seq.
"""
import argparse
import json
import random
from collections import Counter

import llm_backend as be
import summary_chain as sc
import summary_scoring as ss


def keep_reason(row, min_cov, max_ratio):
    if be.failed(row) or not (row.get("summary") or "").strip():
        return "failed_or_empty"
    s = row.get("score") or {}
    if s.get("empty"):
        return "empty"
    for flag in ("bullets", "meta", "think_leak", "narration", "echo"):
        if s.get(flag):
            return flag
    if s.get("fabrication"):
        return "fabrication"
    if s.get("fact_coverage_new") is not None and s["fact_coverage_new"] < min_cov:
        return "low_coverage_new"
    if s.get("fact_coverage_carry") is not None and s["fact_coverage_carry"] < min_cov:
        return "low_coverage_carry"
    if s.get("latest_state") is not None and s["latest_state"] < 1.0:
        return "stale_value"
    if row["words"] > max_ratio * row["output_words"]:
        return f"over_{max_ratio}x_limit"
    return ""


def rescore(chains, rows):
    """Score with the current scorer (scorer fixes must not need a regenerate)."""
    for r in rows:
        c = chains.get(r["chain"])
        if c is not None and not be.failed(r) and r.get("summary"):
            r["score"] = sc.score_step(c, r["kind"], r["k"], r["summary"], r["output_words"])
            r["words"] = ss.words(r["summary"])


def prompt_for(chain, row, mts, mtl):
    step = sc.build_step(chain, row["kind"], row["k"], row.get("previous_summary") or "", mts, mtl)
    if step["output_words"] != row["output_words"]:
        raise SystemExit(f"{row['id']}: output_words {row['output_words']} in the row but {step['output_words']} rebuilt — "
                         "generate and build with the same --max-tokens-short/--max-tokens-long")
    return step["messages"]


def split_by_persona(chains, eval_frac, seed):
    """Held-out chains, stratified by category (every category appears on both sides when it has
    >= 2 chains), then any train chain sharing a human peer name with the eval set moves to eval too.
    The unit is the chain. (Smoke 2026-09-13: an unstratified draw put all 3 multi-peer chains in eval.)"""
    rnd = random.Random(seed)
    by_cat = {}
    for cid in sorted(chains):
        by_cat.setdefault(chains[cid].get("category"), []).append(cid)
    eval_ids, eval_names = set(), set()
    for cat in sorted(by_cat):
        ids = by_cat[cat]
        rnd.shuffle(ids)
        n_eval = min(len(ids) - 1, max(1, round(len(ids) * eval_frac))) if len(ids) > 1 else 0
        for cid in ids[:n_eval]:
            eval_ids.add(cid); eval_names.update(n.lower() for n in sc.peer_names(chains[cid], humans_only=True))
    if not eval_ids and len(chains) > 1 and eval_frac > 0:      # every category a singleton: still hold something out
        ids = sorted(chains); rnd.shuffle(ids)
        for cid in ids[:max(1, round(len(ids) * eval_frac))]:
            eval_ids.add(cid); eval_names.update(n.lower() for n in sc.peer_names(chains[cid], humans_only=True))
    moved = 0
    for cid in sorted(chains):
        if cid not in eval_ids and eval_names & {n.lower() for n in sc.peer_names(chains[cid], humans_only=True)}:
            eval_ids.add(cid); moved += 1
    return eval_ids, moved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--chosen", required=True)
    ap.add_argument("--rejected", default=None, help="stage-2 rows -> also write DPO pairs")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--max-ratio", type=float, default=0.9, help="chosen words / output_words ceiling")
    ap.add_argument("--eval-frac", type=float, default=0.3, help="share of chains held out (PLAN §9: 10 of 30 for the smoke)")
    ap.add_argument("--eval-chains", default=None, help="pin the eval set to the chains of this dataset file (e.g. the smoke's "
                    "dataset_eval.sft.jsonl) so two models are scored on identical chains; new chains sharing a human peer name "
                    "with them are dropped from train, not moved")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens-short", type=int, default=None, help="default: from the chosen rows")
    ap.add_argument("--max-tokens-long", type=int, default=None)
    a = ap.parse_args()

    chains = {c["id"]: c for c in sc.load_chains(a.chains)}
    chosen = [r for r in be.read_jsonl(a.chosen) if r.get("chain") in chains]
    rescore(chains, chosen)
    mts = a.max_tokens_short or next((r["max_tokens"] for r in chosen if r["kind"] == "short"), 1000)
    mtl = a.max_tokens_long or next((r["max_tokens"] for r in chosen if r["kind"] == "long"), 4000)

    kept, dropped = [], Counter()
    for r in chosen:
        why = keep_reason(r, a.min_coverage, a.max_ratio)
        if why:
            dropped[why] += 1
        else:
            kept.append(r)
    print(f"chosen rows {len(chosen)}  kept {len(kept)}  dropped {sum(dropped.values())} {json.dumps(dict(dropped))}")
    print("kept per kind/variant:", json.dumps(dict(Counter(f"{r['kind']}/{r['variant']}" for r in kept))))
    print("kept per category:", json.dumps(dict(sorted(Counter(r["category"] for r in kept).items()))))
    if kept:
        ratios = sorted(r["words"] / r["output_words"] for r in kept)
        print(f"kept limit_ratio median {ratios[len(ratios)//2]:.2f} max {ratios[-1]:.2f}; "
              f"median words short {sorted(r['words'] for r in kept if r['kind']=='short')[len([r for r in kept if r['kind']=='short'])//2] if any(r['kind']=='short' for r in kept) else '-'} "
              f"long {sorted(r['words'] for r in kept if r['kind']=='long')[len([r for r in kept if r['kind']=='long'])//2] if any(r['kind']=='long' for r in kept) else '-'}")

    if a.eval_chains:
        eval_ids = {r.get("chain") or r["id"] for r in be.read_jsonl(a.eval_chains)} & set(chains)
        eval_names = {n.lower() for cid in eval_ids for n in sc.peer_names(chains[cid], humans_only=True)}
        clash = {cid for cid in chains if cid not in eval_ids and eval_names & {n.lower() for n in sc.peer_names(chains[cid], humans_only=True)}}
        chains = {cid: c for cid, c in chains.items() if cid not in clash}
        kept = [r for r in kept if r["chain"] in chains]
        print(f"split: pinned eval set from {a.eval_chains}: {len(eval_ids)} eval chains, {len(chains) - len(eval_ids)} train chains "
              f"({len(clash)} train chains dropped for a shared peer name: {sorted(clash)})")
    else:
        eval_ids, moved = split_by_persona(chains, a.eval_frac, a.seed)
        print(f"split: {len(chains) - len(eval_ids)} train chains, {len(eval_ids)} eval chains ({moved} moved to eval for a shared peer name)")

    rejected = {}
    if a.rejected:
        for r in be.read_jsonl(a.rejected):
            if not be.failed(r) and (r.get("summary") or "").strip() and r.get("chain") in chains:
                rejected[(r["chain"], r["kind"], r["k"])] = r

    for name, pred in (("train", lambda cid: cid not in eval_ids), ("eval", lambda cid: cid in eval_ids)):
        rows = [r for r in kept if pred(r["chain"])]
        sft = [{"id": r["id"], "chain": r["chain"], "kind": r["kind"], "k": r["k"], "variant": r["variant"], "category": r["category"],
                "messages": prompt_for(chains[r["chain"]], r, mts, mtl) + [{"role": "assistant", "content": r["summary"]}]} for r in rows]
        be.write_jsonl(f"{a.out}_{name}.sft.jsonl", sft)
        line = f"{name}: {len(sft)} SFT rows ({sum(1 for r in sft if r['kind']=='short')} short, {sum(1 for r in sft if r['kind']=='long')} long) -> {a.out}_{name}.sft.jsonl"
        if a.rejected:
            dpo = []
            for r in rows:
                rej = rejected.get((r["chain"], r["kind"], r["k"]))
                same_prompt = rej is not None and (r["variant"] == "base_prev" or r["k"] == 0) and \
                    (rej.get("previous_summary") or "").strip() == (r.get("previous_summary") or "").strip()
                if not same_prompt or ss.words(rej["summary"]) == 0:
                    continue
                if rej["summary"].strip() == r["summary"].strip():
                    continue
                dpo.append({"id": r["id"], "chain": r["chain"], "kind": r["kind"], "k": r["k"], "category": r["category"],
                            "prompt": prompt_for(chains[r["chain"]], r, mts, mtl), "chosen": r["summary"], "rejected": rej["summary"]})
            be.write_jsonl(f"{a.out}_{name}.dpo.jsonl", dpo)
            line += f"; {len(dpo)} DPO pairs -> {a.out}_{name}.dpo.jsonl"
        print(line)


if __name__ == "__main__":
    main()
