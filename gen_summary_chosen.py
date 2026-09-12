#!/usr/bin/env python3
"""gen_summary_chosen.py — stage 3: teacher summaries (PLAN §2.2).

NOT IMPLEMENTED YET (skeleton, 2026-09-12). Planned interface:

  python3 gen_summary_chosen.py estimate|run|submit|status|fetch --chains data/chains.jsonl --model opus --out data/chosen.jsonl [--rejected data/rejected.jsonl --base-prev-share 0.3]

Clean chain (previous = teacher's own chunk k-1) for 70 % of chunks; base-previous variant (previous = rejected chunk k-1, teacher repairs dropped facts) for 30 %. Anthropic batches; estimate printed first; batches never abort.

Shared pieces: llm_backend.py (models, keys, OpenRouter concurrency, Anthropic batches, resume-safe
JSONL), summary_prompt.py (the exact Honcho prompt — the ONLY place the prompt is built),
summary_scoring.py (the ONLY scorer). Stdlib only; raw HTTP; no SDKs.
"""
import sys

if __name__ == "__main__":
    sys.exit(f"gen_summary_chosen.py: not implemented yet — see the module docstring and PLAN.md")
