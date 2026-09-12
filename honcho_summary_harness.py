#!/usr/bin/env python3
"""honcho_summary_harness.py — live check: drive a Honcho session past 20/60 messages and read the stored summaries (PLAN §6).

NOT IMPLEMENTED YET (skeleton, 2026-09-12). Planned interface:

  python3 honcho_summary_harness.py --base http://honcho:8000 --workspace W --session S --chain data/chains.jsonl:ID [--key JWT]

Posts the chain's messages through Honcho's messages API, waits for the summariser, fetches get_session_context, scores the stored short/long summaries against the chain's ledger.

Shared pieces: llm_backend.py (models, keys, OpenRouter concurrency, Anthropic batches, resume-safe
JSONL), summary_prompt.py (the exact Honcho prompt — the ONLY place the prompt is built),
summary_scoring.py (the ONLY scorer). Stdlib only; raw HTTP; no SDKs.
"""
import sys

if __name__ == "__main__":
    sys.exit(f"honcho_summary_harness.py: not implemented yet — see the module docstring and PLAN.md")
