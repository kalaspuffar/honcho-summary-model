# honcho-summary-model

Vibe coded project to train a summary model.

Synthetic-data + LoRA pipeline for a Honcho **summary** model: short/long session summaries that are
dense, complete, within the word limit and grounded, served by Ollama through the same
OpenAI-compatible path Honcho already uses. Sister project of `honcho-dialectic-model`, which
produced the terse dialectic model and the lessons this repo starts from.

**Status (2026-09-12): skeleton.** `PLAN.md` is the proposal and single source of truth; read §0
first — Phase 0 measures the base on the real prompt before anything is generated or trained.

| file | role | state |
|---|---|---|
| `PLAN.md` | proposal, gates, decision log | written |
| `TRAIN.md` | runbook + failure log, carry-overs from the dialectic project | written |
| `summary_prompt.py` | **verbatim** Honcho summariser prompts + message formatting + the exact call parameters (commit in header) | done, tested |
| `summary_scoring.py` | the one scorer: fact coverage (new / carried), latest state, fabrication, limit ratio, bullets/meta/think-leak/narration | done, tested |
| `llm_backend.py` | models/prices, keys, OpenRouter concurrency, Anthropic sync + batches, manifests, resume-safe JSONL | copied unchanged |
| `train_lora.py` | check / strip / sft / merge / dpo / export / sample — served-prompt encoding included | copied from `train_dialectic.py`, renamed |
| `mock_or_server.py`, `mock_anth_batch.py` | test doubles for OpenRouter/Ollama and Anthropic | copied unchanged |
| `verify_pipeline.py` | self-test: parity, scorer, train prep, backend; must print GO | done |
| `gen_sessions.py` | stage 1: session chains with a fact ledger | **stub** |
| `gen_summary_rejected.py` | stage 2 / Phase 0: base model through the exact prompt | **stub** |
| `gen_summary_chosen.py` | stage 3: teacher summaries (Anthropic batches) | **stub** |
| `build_summary_dataset.py` | stage 4: filters + persona split → SFT rows | **stub** |
| `eval_summary.py` | offline eval + compare | **stub** |
| `honcho_summary_harness.py` | live check through Honcho's API | **stub** |
| `Modelfile`, `keys.env.example` | serving + secrets template | written |

```bash
python3 verify_pipeline.py            # must print GO before a commit (--quick skips the mock round trip)
python3 -c "import summary_prompt as sp; print(sp.build_messages('short', [{'peer_name':'Anna','content':'hi'}], None, 300)[0]['content'])"
```

Conventions (CLAUDE.md): stdlib-only data scripts, raw HTTP, one prompt builder, one scorer, one
backend, rejected from the base model, estimates always printed, GO before commit.
