#!/usr/bin/env python3
"""gen_sessions.py — stage 1: session chains with a fact ledger (PLAN §2.1).

NOT IMPLEMENTED YET (skeleton, 2026-09-12). Planned interface:

  python3 gen_sessions.py estimate|run|submit|status|fetch --n N --model opus --out data/chains.jsonl

Emits one JSON row per chain: {id, domain, category, peers, messages[{seq,peer,text}], facts[{id,text,first_seq,kind}], changes[{fact_id,superseded_by,seq}], distractors[{text}], chunks[{k,seqs,limit_words_short,limit_words_long}]}. Categories interleaved per PLAN §2.1 (dense-facts 25, preference 15, supersession 15, long-tail merge 20, chit-chat 10, multi-peer 15).

Shared pieces: llm_backend.py (models, keys, OpenRouter concurrency, Anthropic batches, resume-safe
JSONL), summary_prompt.py (the exact Honcho prompt — the ONLY place the prompt is built),
summary_scoring.py (the ONLY scorer). Stdlib only; raw HTTP; no SDKs.
"""
import sys

if __name__ == "__main__":
    sys.exit(f"gen_sessions.py: not implemented yet — see the module docstring and PLAN.md")
