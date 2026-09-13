# TRAIN.md — summary model runbook and failure log

Nothing trained yet (2026-09-12). Carry-overs from honcho-dialectic-model/TRAIN.md that apply here
unchanged — read them before the first run:

- Base = the stripped text-only Qwen3.5-9B (`train_lora.py --stage strip --out /data/smoke/qwen35-9b-text`).
  Never Qwen3-8B, never the Ollama tag / VL Hub repo as `--model`.
- Ollama serves the built-in qwen3.5 renderer (`<think>\n` prefill) regardless of the GGUF template
  and ignores `reasoning_effort` on 0.32.12; Honcho reaches it only via `/v1` and reads `content`.
  The model must emit `</think>` itself → `encode_example` tokenizes the served prompt and the
  `\n</think>\n\n`+turn completion separately. Gate every run with `--stage check` (trainable text
  must begin `\n</think>\n\n`) and `--stage sample --open-think` on the merge.
- SFT only, 2 epochs, lr 2e-4, r=16. DPO adds nothing measurable at 50–150 rows (dialectic
  ablation); use it only for word-limit overshoot if SFT leaves it > 5 %.
- Rows are dropped, never truncated, above `--max-seq`. Long-summary rows run 11–14k tokens:
  `--max-seq 16384 --load-bits 16` on the A6000; if it does not fit, cap long targets at 2000 words.
- Smoke first (30 chains, < 1.5 h), full eval, then more data only against a measured gap.

## Failure log
| Date | Stage | What failed | Fix / workaround |
|---|---|---|---|
| 2026-09-13 | Phase 0 (`gen_summary_rejected.py`, base `qwen3.5:9b`, node7) | Short slot 104/104 empty: `finish_reason=length`, all 1000 tokens in `<think>`, at 0.1 and at 0.7. Long slot: 14/29 truncated, 4 empty. Honcho would store nothing for shorts. | Not fixable by settings (no thinking-off switch on Ollama for Qwen3.5; 4000 tokens still truncate half the long ones). This is the failure the served-prompt SFT removes (`encode_example`, `--stage check` must show `\n</think>\n\n` first). Proceed to the smoke. |
| 2026-09-13 | Phase 0 (`dialectic_s50` in the summary slot) | Answers, but terse (161 w median of ~450) and carries only ~1/3 of earlier facts; 8 % think leaks, 5 % fabrication. | Not a candidate for the slot; its outputs feed the stage 3 base-previous variant instead (PLAN §10). |
| 2026-09-13 | scorer | Empty rows scored carry 0.0 with nothing due → base carry medians read 0. | `summary_scoring.score_summary` fixed; rescore with `eval_summary.py rescore`. |
| 2026-09-13 | scorer | All 7 dialectic "fabrication" rows were matcher artifacts (peer name + possessive "s"; "the"+"city"; two anchors 30 words apart; near-miss distractors differing in one anchor). | `has_fact` rewritten (anchors, window, number words, peer names ignored; strict for distractors) — PLAN §10. Always re-score every results file after a scorer change: `eval_summary.py rescore <file> --chains data/chains.jsonl`. |
| 2026-09-13 | stage 3 (`gen_summary_chosen.py`, smoke) | 80/98 short teacher summaries at k ≥ 1 cut mid-sentence: the teacher got Honcho's max_tokens=1000 as its own budget, shared with its thinking tokens. New-fact coverage collapsed from k=2 while carry looked perfect; build kept 56/155. | Teacher budget decoupled (8000/16000), stop_reason recorded, truncated rows failed + dependants stale; re-run `submit` then `fetch --waves`. Always read one k ≥ 2 chosen row to the end before `build_summary_dataset.py`. |
| 2026-09-13 | stage 3 (second pass) | Complete summaries, coverage 1.0, but 85/155 over 0.9× the limit and 44 over the hard limit (750 words ≈ 1000 tokens: would be cut by Honcho's max_tokens). Teacher overshoots a percentage ask by 6–13 points. | Numeric word budget (0.75 × limit) in the system prompt; over-budget rows = failed generations, retried up to 3 times with dependants; `report` prints the given-up count. |
| 2026-09-13 | stage 3 pilot (`run --only`, 3 chains) | Budget 0.75 → median 0.79, coverage 1.0; 3/9 shorts at 0.91–0.95. | Over-budget drafts are now compressed (low effort) instead of rewritten; pilot a few chains with `--only` before every full batch. |
| 2026-09-13 | stage 1 (`gen_sessions.py fetch`) | 3/30 chains failed: c00009 truncated reply parsed as a peer object; c00028/c00029 "credit balance too low" from the Anthropic API mid-batch. | Parse/diagnosis fix + bigger budget; credits are an account matter — re-run `submit`+`fetch` to retry. Check `gen_summary_chosen.py status` after every submit: a batch with errored requests still "ends". |

## Runbook — data side (2026-09-12, pipeline implemented, nothing run)

Order (PLAN §9), every step resume-safe and preceded by its estimate:
1. `gen_sessions.py submit --n 30 --model opus` → `fetch` → `data/chains.jsonl` (30 chains, 6 categories).
2. Phase 0: `gen_summary_rejected.py --chains data/chains.jsonl --out data/rejected.jsonl --model qwen3.5:9b`
   (honest chaining, Honcho's max_tokens), once more with `--temperature 0.7` into another file, and with
   `--model dialectic_s50`. Read the per-kind aggregate: `over_limit_rows`, `median_fact_coverage_carry`,
   `fabrication_rows`, `answered_in_thinking_rows`, `finish_length_rows`, `empty_rows`. PLAN §0.4 gate decides.
3. `gen_summary_chosen.py submit … --rejected data/rejected.jsonl` → `fetch --waves` (one batch per step level).
4. `build_summary_dataset.py … --eval-frac 0.33` → `data/dataset_{train,eval}.sft.jsonl` (+ `.dpo.jsonl`).
5. GPU host: `train_lora.py --stage check --model /data/smoke/qwen35-9b-text --data data/dataset_train.sft.jsonl --max-seq 16384`
   (trainable tail must begin `\n</think>\n\n`; long rows 11–14k tokens must not be dropped), then
   `--stage sft --epochs 2 --lr 2e-4 --max-seq 16384 --load-bits 16 --eval-data data/dataset_eval.sft.jsonl`,
   `--stage merge`, `--stage sample --open-think`, `--stage export`, `ollama create summary-v1 -f Modelfile`.
6. `eval_summary.py` base (with `--answer-from-reasoning`) vs `summary-v1`, `compare`; then
   `honcho_summary_harness.py` on one eval chain against a live Honcho with `SUMMARY_MODEL_CONFIG__*` swapped.
