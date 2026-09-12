#!/usr/bin/env python3
"""eval_summary.py — offline eval: a model summarises held-out chains chunk by chunk via /v1 and is scored (PLAN §6).

NOT IMPLEMENTED YET (skeleton, 2026-09-12). Planned interface:

  python3 eval_summary.py --chains data/chains.jsonl --ids-from data/dataset_eval.sft.jsonl --model <ollama-tag> --base http://node7.ea.org:11434 --out results/x.jsonl ; python3 eval_summary.py compare a.jsonl b.jsonl

Previous summary = the model's own output for the previous chunk. Reports summary_scoring.aggregate plus answered_in_thinking_rows and latency.

Shared pieces: llm_backend.py (models, keys, OpenRouter concurrency, Anthropic batches, resume-safe
JSONL), summary_prompt.py (the exact Honcho prompt — the ONLY place the prompt is built),
summary_scoring.py (the ONLY scorer). Stdlib only; raw HTTP; no SDKs.
"""
import sys

if __name__ == "__main__":
    sys.exit(f"eval_summary.py: not implemented yet — see the module docstring and PLAN.md")
