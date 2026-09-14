# honcho-summary-model

Vibe coded project to train a summary model.

Synthetic-data + LoRA pipeline for a Honcho **summary** model: short/long session summaries that are
dense, complete, within the word limit and grounded, served by Ollama through the same
OpenAI-compatible path Honcho already uses. Sister project of `honcho-dialectic-model`, which
produced the terse dialectic model and the lessons this repo starts from.

**Status (2026-09-14): `summary-smoke` cleared for the production summary slot** with Honcho
`SUMMARY_MAX_TOKENS_SHORT=1500` and Modelfile temperature 0.1 (PLAN §10, TRAIN.md): 94 SFT rows from 27
chains; on the pinned 10-chain eval 0 truncations, 0 over-limit, carried-fact coverage 0.86–0.93 vs 0.49 for
the base (which returns nothing at all in the short slot), no format failures; verified live through Honcho.
Measured and not shipped: a +30-chain increment (243 rows, `summary-v2`), Qwen3-8B on the same 94 rows
(`summary-q3`: more multi-peer carry, but longer, one real fabrication) and Qwen3-8B on 243 rows
(`summary-q3-v2`: no gain). Lesson, shown on both bases: ~100 rows set the behaviour and the dataset's
length target, not the base, is the lever against Honcho's token cap. Experiments closed (TRAIN.md "Final
comparison"); the combined dialectic + summary model was dropped as not a priority.
`PLAN.md` is the single source of truth (§10 decision log); `python3 verify_pipeline.py` runs the whole
pipeline against mock servers at $0 and must print GO.

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
| `honcho_summary_harness.py` | live check through Honcho's v3 API (`run`, `compare`, `replay` a stored step against Ollama); records which message each stored summary is anchored to; `--one-by-one` mimics real traffic | done, verified against Honcho on 2026-09-13 |
| `log_proxy.py` | transparent logging proxy in front of Ollama: records every request Honcho sends the summary model; `diff` compares a logged prompt with what `summary_chain.build_step` would send | done |
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

# PLAN §11 "retention" phase: eval-only chains, on-policy pairs, DPO on top of the shipped adapter
python3 gen_sessions.py submit --n 20 --start 100 --model opus --out data/chains_eval2.jsonl && python3 gen_sessions.py fetch
python3 gen_summary_rejected.py --chains data/chains.jsonl --out data/prev_smoke.jsonl --model summary-smoke        # student on every chain
python3 gen_summary_chosen.py submit --chains data/chains.jsonl --rejected data/prev_smoke.jsonl --base-prev-share 1.0 \
        --exclude-chains-of data/dataset_eval.sft.jsonl --out data/chosen.jsonl && python3 gen_summary_chosen.py fetch --waves
python3 build_summary_dataset.py --chains data/chains.jsonl --chosen data/chosen.jsonl --rejected data/prev_smoke.jsonl \
        --eval-chains data/dataset_eval.sft.jsonl --out data/dataset_r                                              # *.dpo.jsonl = informative pairs
python3 eval_summary.py --chains data/chains.jsonl --ids-from data/dataset_eval.sft.jsonl --extra-chains data/chains_eval2.jsonl \
        --model summary-smoke --max-tokens-short 1500 --out results/eval30_smoke_1.jsonl                          # the 30-chain ruler
```

Every generation script takes `--max-tokens-short/--max-tokens-long` (Honcho's
`SUMMARY_MAX_TOKENS_SHORT/LONG`, defaults 1000/4000); generate, build and evaluate with the same
values, since they set the word limits in the prompt.

Conventions (CLAUDE.md): stdlib-only data scripts, raw HTTP, one prompt builder, one scorer, one
backend, rejected from the base model, estimates always printed, GO before commit, and nothing
from the real deployment in `data/`.
