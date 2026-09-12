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
