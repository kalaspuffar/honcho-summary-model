# CLAUDE.md

Synthetic-data + LoRA fine-tuning pipeline for a Honcho **summary** model (short/long session
summaries) from the stripped text-only Qwen3.5-9B. Sister project of `honcho-dialectic-model`
(same conventions; its PLAN.md/TRAIN.md hold the lessons this repo starts from).

`PLAN.md` is the single source of truth (proposal, gates, decision log). `TRAIN.md` is the
runbook + failure log. Append to their logs, do not rewrite. `README.md` must stay in step with the CLI.

Rules that carry over verbatim:
- Flat directory of standalone Python 3 scripts, **stdlib only** for every data script; only
  `train_lora.py` needs the Unsloth venv on the GPU host. Raw HTTP to OpenRouter/Anthropic; no SDKs.
- `python3 verify_pipeline.py` must print GO before a commit.
- **One prompt builder**: `summary_prompt.py` is a verbatim copy of Honcho's summariser prompts with
  the source commit in its header — never hand-edit; re-copy and regenerate if Honcho changes.
- **One scorer**: `summary_scoring.py` owns every regex and rule; `verify_pipeline.py` fails if
  BULLET/META/THINK_LEAK/NARRATION are defined anywhere else.
- **One backend**: `llm_backend.py` (models/prices, keys via `keys.env`, OpenRouter concurrency,
  Anthropic sync + batches, manifests, resume-safe JSONL). Generation scripts share its CLI shape
  `estimate|run|submit|status|fetch`. Estimates always printed; batches never abort.
- **Rejected comes from the base model**, never a teacher. Failure markers `__FAILED__`.
- **No real data.** Personas, domains and conversations are invented by the generator. Nothing from
  Daniel's Honcho deployment (names, topics, sessions, summaries) is used as seed, example or domain
  list. Live checks read Honcho; they never write into `data/`.
- Training gotchas: see TRAIN.md (base checkpoint, served-prompt encoding, sequence length).

Stage order (PLAN §9): Phase 0 measure → 30-chain smoke → 150 chains → combined run with the
dialectic rows so one Ollama model serves both slots.
