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
