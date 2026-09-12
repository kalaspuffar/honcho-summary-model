#!/usr/bin/env python3
"""gen_summary_rejected.py — stage 2 / Phase 0: the BASE model summarises every chunk through the exact Honcho prompt (PLAN §2.3).

NOT IMPLEMENTED YET (skeleton, 2026-09-12). Planned interface:

  python3 gen_summary_rejected.py --chains data/chains.jsonl --model qwen3.5:9b --base http://node7.ea.org:11434 --out data/rejected.jsonl [--kind short|long|both]

Chained: chunk k's previous_summary is the model's OWN output for chunk k-1 (the honest setting). Records content, finish_reason, reasoning_chars (answered-inside-thinking), latency. Rejected always comes from the base model, never a teacher.

Shared pieces: llm_backend.py (models, keys, OpenRouter concurrency, Anthropic batches, resume-safe
JSONL), summary_prompt.py (the exact Honcho prompt — the ONLY place the prompt is built),
summary_scoring.py (the ONLY scorer). Stdlib only; raw HTTP; no SDKs.
"""
import sys

if __name__ == "__main__":
    sys.exit(f"gen_summary_rejected.py: not implemented yet — see the module docstring and PLAN.md")
