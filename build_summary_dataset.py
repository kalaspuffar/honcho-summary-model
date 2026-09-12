#!/usr/bin/env python3
"""build_summary_dataset.py — stage 4: filter + persona split -> SFT rows (PLAN §2.2 filters, §3 row shape).

NOT IMPLEMENTED YET (skeleton, 2026-09-12). Planned interface:

  python3 build_summary_dataset.py --chains data/chains.jsonl --chosen data/chosen.jsonl [--rejected data/rejected.jsonl] --out data/dataset

Keeps a chosen summary only if fact_coverage_new >= .9, fact_coverage_carry >= .9, fabrication False, words <= 0.9*limit, no bullets/meta/think-leak. Row: {messages:[{role:user, content:<exact prompt>}, {role:assistant, content:<chosen>}]}; DPO rows only when --rejected is given. Prints drop reasons and word stats.

Shared pieces: llm_backend.py (models, keys, OpenRouter concurrency, Anthropic batches, resume-safe
JSONL), summary_prompt.py (the exact Honcho prompt — the ONLY place the prompt is built),
summary_scoring.py (the ONLY scorer). Stdlib only; raw HTTP; no SDKs.
"""
import sys

if __name__ == "__main__":
    sys.exit(f"build_summary_dataset.py: not implemented yet — see the module docstring and PLAN.md")
