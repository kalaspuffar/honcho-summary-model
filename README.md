# honcho-summary-model

Vibe coded project to train a summary model.

Synthetic-data + LoRA pipeline for a Honcho **summary** model: short/long session summaries that are
dense, complete, within the word limit and grounded, served by Ollama through the same
OpenAI-compatible path Honcho already uses. Sister project of `honcho-dialectic-model`, which
produced the terse dialectic model and the lessons this repo starts from.

**Status (2026-09-12): pipeline implemented, nothing generated or trained yet.** `PLAN.md` is the
proposal and single source of truth; read §0 first — Phase 0 measures the base on the real prompt
before anything is trained. `python3 verify_pipeline.py` runs the whole pipeline against mock
servers at $0 and must print GO.

| file | role | state |
|---|---|---|
| `PLAN.md` | proposal, gates, decision log | written |
| `TRAIN.md` | runbook + failure log, carry-overs from the dialectic project | written |
| `summary_prompt.py` | **verbatim** Honcho summariser prompts + message formatting + the exact call parameters (commit in header) | done, tested |
| `summary_scoring.py` | the one scorer: fact coverage (new / carried), latest state, fabrication, limit ratio, bullets/meta/think-leak/narration | done, tested |
| `summary_chain.py` | chain → Honcho steps: 20-message short chunks, 60-message long chunks, each kind chained on its own previous summary; `output_words` parity; the `/v1` student call; the honest chunk-by-chunk walker | done, tested |
| `llm_backend.py` | models/prices, keys, OpenRouter concurrency, Anthropic sync + batches, manifests, resume-safe JSONL | copied unchanged |
| `train_lora.py` | check / strip / sft / merge / dpo / export / sample — served-prompt encoding included | copied from `train_dialectic.py`, renamed |
| `mock_or_server.py`, `mock_anth_batch.py` | test doubles for OpenRouter/Ollama and Anthropic; also answer the chain-generation and summary prompts | extended |
| `verify_pipeline.py` | self-test: parity, scorer, train prep, backend, chain logic, $0 end-to-end run on the mocks; must print GO | done |
| `gen_sessions.py` | stage 1: invented session chains with a fact ledger (`estimate/run/submit/status/fetch`) | done |
| `gen_summary_rejected.py` | stage 2 / Phase 0: base model through the exact prompt, own previous summary, scored | done |
| `gen_summary_chosen.py` | stage 3: teacher summaries, clean chain + base-previous variant; sync or wave-wise batches | done |
| `build_summary_dataset.py` | stage 4: scorer filters + human-persona split → SFT rows (+ same-prompt DPO pairs) | done |
| `eval_summary.py` | offline eval `run / compare / rescore` | done |
| `honcho_summary_harness.py` | live check through Honcho's v3 API, results only under `results/` | done, untested against a live Honcho |
| `Modelfile`, `keys.env.example` | serving + secrets template | written |

## Commands

```bash
python3 verify_pipeline.py            # must print GO before a commit (--quick skips the mock runs)

# Phase 0 + smoke (PLAN §9). Estimates are printed by every stage; batches are half price.
python3 gen_sessions.py estimate --n 30 --model opus                       # ≈ $2 batch / $4.5 sync
python3 gen_sessions.py submit   --n 30 --model opus --out data/chains.jsonl && python3 gen_sessions.py fetch
python3 gen_summary_rejected.py --chains data/chains.jsonl --out data/rejected.jsonl \
        --model qwen3.5:9b --base http://node7.ea.org:11434/v1               # Phase 0 measurement; honest chaining
python3 gen_summary_rejected.py --chains data/chains.jsonl --out data/rejected_t07.jsonl --model qwen3.5:9b --temperature 0.7
python3 gen_summary_chosen.py estimate --chains data/chains.jsonl --model opus --rejected data/rejected.jsonl
python3 gen_summary_chosen.py submit   --chains data/chains.jsonl --model opus --rejected data/rejected.jsonl --out data/chosen.jsonl
python3 gen_summary_chosen.py fetch --waves                                # steps depend on the previous step: one batch per wave
python3 build_summary_dataset.py --chains data/chains.jsonl --chosen data/chosen.jsonl --rejected data/rejected.jsonl --out data/dataset
# GPU host (TRAIN.md): train_lora.py --stage check / sft / merge / export, then `ollama create`
python3 eval_summary.py --chains data/chains.jsonl --ids-from data/dataset_eval.sft.jsonl --model qwen3.5:9b --out results/eval_base.jsonl --answer-from-reasoning
python3 eval_summary.py --chains data/chains.jsonl --ids-from data/dataset_eval.sft.jsonl --model summary-v1  --out results/eval_v1.jsonl
python3 eval_summary.py compare results/eval_base.jsonl results/eval_v1.jsonl
python3 honcho_summary_harness.py --base http://honcho:8000 --workspace summary-check --chain data/chains.jsonl:c00003 --label v1
```

Every generation script takes `--max-tokens-short/--max-tokens-long` (Honcho's
`SUMMARY_MAX_TOKENS_SHORT/LONG`, defaults 1000/4000); generate, build and evaluate with the same
values, since they set the word limits in the prompt.

Conventions (CLAUDE.md): stdlib-only data scripts, raw HTTP, one prompt builder, one scorer, one
backend, rejected from the base model, estimates always printed, GO before commit, and nothing
from the real deployment in `data/`.
