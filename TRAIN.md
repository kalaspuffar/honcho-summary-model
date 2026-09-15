# TRAIN.md — summary model runbook and failure log

Nothing trained yet (2026-09-12). Carry-overs from honcho-dialectic-model/TRAIN.md that apply here
unchanged — read them before the first run:

- Base = the stripped text-only Qwen3.5-9B (`train_lora.py --stage strip --out runs/base-qwen35-9b-text`).
  Never Qwen3-8B, never the Ollama tag / VL Hub repo as `--model`.
- Ollama serves the built-in qwen3.5 renderer (`<think>\n` prefill) regardless of the GGUF template
  and ignores `reasoning_effort` on 0.32.12; Honcho reaches it only via `/v1` and reads `content`.
  The model must emit `</think>` itself → `encode_example` tokenizes the served prompt and the
  `\n</think>\n\n`+turn completion separately. Gate every run with `--stage check` (trainable text
  must begin `\n</think>\n\n`) and `--stage sample --open-think` on the merge.
- SFT only, 2 epochs, lr 2e-4, r=16. DPO adds nothing measurable at 50–150 rows (dialectic
  ablation); use it only for word-limit overshoot if SFT leaves it > 5 %.
- Rows are dropped, never truncated, above `--max-seq`. Long-summary rows run 11–14k tokens:
  `--max-seq 16384 --load-bits 16` on a 48 GB card; if it does not fit, cap long targets at 2000 words.
- Smoke first (30 chains, < 1.5 h), full eval, then more data only against a measured gap.

## Failure log
| Date | Stage | What failed | Fix / workaround |
|---|---|---|---|
| 2026-09-13 | Phase 0 (`gen_summary_rejected.py`, base `qwen3.5:9b`, Ollama host) | Short slot 104/104 empty: `finish_reason=length`, all 1000 tokens in `<think>`, at 0.1 and at 0.7. Long slot: 14/29 truncated, 4 empty. Honcho would store nothing for shorts. | Not fixable by settings (no thinking-off switch on Ollama for Qwen3.5; 4000 tokens still truncate half the long ones). This is the failure the served-prompt SFT removes (`encode_example`, `--stage check` must show `\n</think>\n\n` first). Proceed to the smoke. |
| 2026-09-13 | Phase 0 (`dialectic_s50` in the summary slot) | Answers, but terse (161 w median of ~450) and carries only ~1/3 of earlier facts; 8 % think leaks, 5 % fabrication. | Not a candidate for the slot; its outputs feed the stage 3 base-previous variant instead (PLAN §10). |
| 2026-09-13 | scorer | Empty rows scored carry 0.0 with nothing due → base carry medians read 0. | `summary_scoring.score_summary` fixed; rescore with `eval_summary.py rescore`. |
| 2026-09-13 | scorer | All 7 dialectic "fabrication" rows were matcher artifacts (peer name + possessive "s"; "the"+"city"; two anchors 30 words apart; near-miss distractors differing in one anchor). | `has_fact` rewritten (anchors, window, number words, peer names ignored; strict for distractors) — PLAN §10. Always re-score every results file after a scorer change: `eval_summary.py rescore <file> --chains data/chains.jsonl`. |
| 2026-09-13 | stage 3 (`gen_summary_chosen.py`, smoke) | 80/98 short teacher summaries at k ≥ 1 cut mid-sentence: the teacher got Honcho's max_tokens=1000 as its own budget, shared with its thinking tokens. New-fact coverage collapsed from k=2 while carry looked perfect; build kept 56/155. | Teacher budget decoupled (8000/16000), stop_reason recorded, truncated rows failed + dependants stale; re-run `submit` then `fetch --waves`. Always read one k ≥ 2 chosen row to the end before `build_summary_dataset.py`. |
| 2026-09-13 | stage 3 (second pass) | Complete summaries, coverage 1.0, but 85/155 over 0.9× the limit and 44 over the hard limit (750 words ≈ 1000 tokens: would be cut by Honcho's max_tokens). Teacher overshoots a percentage ask by 6–13 points. | Numeric word budget (0.75 × limit) in the system prompt; over-budget rows = failed generations, retried up to 3 times with dependants; `report` prints the given-up count. |
| 2026-09-13 | stage 3 pilot (`run --only`, 3 chains) | Budget 0.75 → median 0.79, coverage 1.0; 3/9 shorts at 0.91–0.95. | Over-budget drafts are now compressed (low effort) instead of rewritten; pilot a few chains with `--only` before every full batch. |
| 2026-09-13 | stage 3 (`fetch --waves`) | One step resubmitted every wave (17×): the on-disk row had no failure marker, so `merge_rows` kept it over its failing retry. | `write_out` merges over the invalidated view. If `--waves` ever prints the same "1 steps now" twice, stop it and look at that id. |
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
5. GPU host: `train_lora.py --stage check --model runs/base-qwen35-9b-text --data data/dataset_train.sft.jsonl --max-seq 8192`
   (trainable tail must begin `\n</think>\n\n`; the smoke rows are ≤ 6k tokens by estimate — long teacher summaries
   come out near 800 words, not 3000 — so 8192 fits and nothing may be dropped), then
   `--stage sft --epochs 2 --lr 2e-4 --max-seq 8192 --load-bits 16 --eval-data data/dataset_eval.sft.jsonl`,
   `--stage merge`, `--stage sample --open-think`, `--stage export`, `ollama create summary-v1 -f Modelfile`.
6. `eval_summary.py` base (with `--answer-from-reasoning`) vs `summary-v1`, `compare`; then
   `honcho_summary_harness.py` on one eval chain against a live Honcho with `SUMMARY_MODEL_CONFIG__*` swapped.

## Smoke run — 2026-09-13 (PASSED the PLAN §6 gate)

| item | value |
|---|---|
| data | 94 train rows (74 short, 20 long) from 17 chains; 56 eval rows from 10 held-out chains, stratified by category, human peer names disjoint |
| base | `runs/base-qwen35-9b-text` (from `--stage strip`), `--load-bits 16 --max-seq 8192` (rows 890–5400 tokens, median 2022, none dropped) |
| run | SFT 2 epochs, lr 2e-4, r=16, 48 steps, 8 min on a 48 GB card (6–10 s/step) |
| eval loss | ep1 **0.4953**, ep2 0.5015 → epoch 1 (`checkpoint-24`) merged; epoch 2 already overfits slightly at 94 rows |
| sample `--open-think` | raw output begins `\n</think>\n\n` + summary; no reasoning text |
| export | `runs/v1-gguf` Q4_K_M → `ollama create summary-smoke -f Modelfile-smoke` |

Offline eval, 10 eval chains / 50 steps, honest chaining, Honcho max_tokens 1000/4000, Ollama host
(base column scored from its reasoning text — the base's `content` is empty on every short step):

| short (40 steps) | base qwen3.5:9b | summary-smoke |
|---|---|---|
| over-limit rows | 10 (25 %) | **0** |
| median limit ratio | 0.82 | 0.72 |
| new-fact coverage (median) | 0.92 | **1.00** |
| carried-fact coverage (median) | 0.49 | **0.85** |
| fabrication rows | 0 | 0 |
| bullets / think leaks / echoes | 40 / 39 / 33 | **0 / 0 / 0** |
| answered in thinking / truncated | 40 / 39 | **0 / 0** |
| median latency | 18.9 s | 13.4 s |

| long (10 steps) | base | summary-smoke |
|---|---|---|
| new-fact coverage | 0.89 | **0.97** |
| bullets / meta / think leaks / truncated | 3 / 1 / 3 / 8 | **0 / 0 / 0 / 0** |
| median words | 1067 | 774 |
| median latency | 72 s | 23.6 s |

Weak spots to feed the 150-chain run: carry falls with k (c00006-s4 0.77, c00017-s4 0.64, c00022-s2 0.59);
multi-peer chains are the weakest (new 0.67–0.92, carry 0.59–0.83; only 4 multi-peer rows in training);
`latest_state` 0.0/0.5 on two long summaries (c00022-l0, c00018-l0) — superseded values in the long slot.

## Live check — 2026-09-13 (local Honcho, summary slot = `summary-smoke`, scratch workspace `summary-check`)

`honcho_summary_harness.py` first contact with a real Honcho v3: workspace/peers/session creation, 20-message
blocks, queued summariser, `GET .../summaries` all worked unchanged; every summary was stored (waits 10–85 s,
long summary ready together with the third short one).

| chain | short steps | limit ratio (median) | new / carry (median) | format flags | long |
|---|---|---|---|---|---|
| c00006 dense-facts, 100 msgs | 5 | 0.80, 0 over | 1.0 / 0.98 | none | 755 w, new 1.0 |
| c00022 multi-peer, 60 msgs | 3 | 0.67, 0 over | 0.92 / 0.24 | none | 793 w, new 1.0 |

c00022-s1 stored 368 words with **0 new facts** and carry 0.28 (offline: 0.73 / 0.61). The stored text is a
confabulation ("second segment of his conversation", invented numbers) that never mentions block 1's content.
`harness replay` of the same step against Ollama with the stored s0 as previous: new 0.82 / 0.91 / 0.82,
carry 0.83 / 0.72 / 0.89 (3 samples) → **the model is fine; Honcho sent that step a different prompt.**
Honcho's `create_messages` assigns `seq_in_session` per message and `get_messages_by_seq_range` filters by it,
so the range logic reads correctly; what went over the wire is unknown → `log_proxy.py` between Honcho and
Ollama, re-run the chain in batch mode and `--one-by-one`, `log_proxy.py diff` the s1 request. The harness now
records which message each stored summary is anchored to (`anchor_ok`).

**Resolved 2026-09-14.** Re-ran c00022 live twice with `log_proxy.py` in front of Ollama: batch delivery and
`--one-by-one` both anchored every summary at the block's last message; `log_proxy.py diff` on the s1 request:
20/20 lines of block 1 present, one user turn, previous = stored s0, `max_tokens` 1000, no temperature, no
extra fields (Honcho's limit 576 words vs the harness's token-estimate 587). s1 scored 1.0/0.83 (batch) and
0.91/0.61 (single). The 2026-09-13 collapse was a one-off generation at temperature 0.1 on the weakest
category — 1 bad in 5 attempts at that step. Plumbing and prompt parity are verified end to end.
| 2026-09-14 | stage 3 cost reporting | `submit` printed the estimate for the first wave only ($0.53); the remaining waves cost $3.16 more — "waves are free" was wrong as stated. The whole job costs one generation per step plus the compress retries; the smoke's teacher pass came to $3.68 for 121 steps (~3 ¢/step). | `submit` now prints WHOLE JOB ≈ (all remaining steps × this wave's per-step cost × 1.3 for retries); `fetch --waves` prints a running total and a final sum. |

## Increment run v2 — 2026-09-14 (+30 chains, pinned eval; verdict: NOT a measurable improvement)

Data: 57 chains → 378 teacher rows (315 good; 33 steps given up at the 0.8 ceiling, all deep steps of dense
chains) → **243 train rows** (187 short, 56 long; 43 chains) vs the smoke's 94; eval pinned to the smoke's 10
chains (40 short + 10 long steps); 4 new chains dropped for a shared peer name. Teacher pass $10.16 (estimate
$8.36 → factor 1.3 → 1.6), chains $4.73. SFT 2 epochs, 122 steps, 23 min, eval loss ep2 0.5023.

Two eval runs per model on the identical 10 chains (temperature 0.1):

| short, 40 steps | smoke #1 | smoke #2 | v2 #1 | v2 #2 |
|---|---|---|---|---|
| carry (median) | 0.92 | 0.93 | 0.985 | 0.94 |
| new (median) | 1.0 | 1.0 | 1.0 | 1.0 |
| multi-peer carry | 0.76 | 0.76 | 0.945 | 0.70 |
| k=4 carry | 0.76 | 0.94 | 0.83 | 0.82 |
| limit ratio (median) | 0.80 | 0.74 | 0.84 | 0.79 |
| over limit / cut at max_tokens | 1 / 1 | 1 / 2 | 2 / 5 | 4 / 4 |
| long new (10 steps) | 0.946 | 0.952 | 1.0 | 1.0 |

Reading: carry +0.03–0.06 lies inside the run-to-run spread (smoke 0.01, v2 0.04); multi-peer swings 0.70–0.95
between two v2 runs (c00022-s1 collapsed again in v2 #2: 275 words, carry 0.33 — the same step as the live
one-off, now 2 collapses in 7 attempts across models); k ≥ 3 unchanged. Long new-coverage +5 points is the
only consistent change and its medians sit at the ceiling. **Length got worse, not better**: v2 writes longer
(median ratio 0.79–0.84 vs 0.74–0.80) and is cut at the 1000-token cap on 4–5/40 short steps vs 1–2/40,
although its training rows were capped at 0.8 (median 0.76). The model does not count words; its length
follows the content it covers, and 750 words is Honcho's cap in tokens. Conclusion as in the dialectic
project: ~100 rows set the behaviour; 2.6× the rows moved nothing outside noise. Do not scale to 150.

Open defect for BOTH models: truncation at `max_tokens=1000` on dense deep steps (2–12 %), each losing the
newest chunk. Candidates, cheapest first: (1) operator setting `SUMMARY_MAX_TOKENS_SHORT=1500` — the word
limit rises to ≤ 1125 but these chunks are content-limited at 500–700 words, so the cap stops binding; test
offline for free with `eval_summary.py --max-tokens-short 1500`; (2) the one DPO experiment PLAN §7 reserved:
prompt-identical pairs teacher-within-budget vs the model's own output already exist (base_prev rows ↔
`prev_smoke.jsonl`), no generation cost; (3) accept and ship the smoke model.

**Cap test 2026-09-14, `--max-tokens-short 1500` (prompt limit rises to ≤ 1125 words), one run per model, same 10 chains:**

| short, 40 steps | smoke @1000 (2 runs) | smoke @1500 | v2 @1000 (2 runs) | v2 @1500 |
|---|---|---|---|---|
| cut at max_tokens | 1, 2 | **0** | 5, 4 | 1 (c00016-s3 wrote 1142 words) |
| over limit | 1, 1 | **0** | 2, 4 | 3 |
| median words / ratio | 506 / 0.80, 490 / 0.74 | 488 / **0.70** | 558 / 0.84, 508 / 0.79 | 520 / 0.81 (deep steps 940–1140 w) |
| carry (median) | 0.92, 0.93 | 0.86 | 0.985, 0.94 | 0.905 |

The smoke model does **not** chase the higher limit (same word count, ratio falls to 0.70): the cap stops
binding and truncation disappears. v2 does chase it at deep steps (941, 970, 1042, 1142 words) and still
overshoots; it also produced a 22-word collapse at c00015-s0 (new 0.04) that poisoned the chain. The smoke's
carry 0.86 in this run is one chain-level cascade (c00016: s1 dropped to 0.7 and every later step inherits it;
0.77–1.0 in the two @1000 runs) — chain variance, not attributable to the cap without a repeat. Known variance
for both models: an early slip cascades through the chain; c00022 (multi-peer) collapses in ~1 of 3 runs.

**Repeat + greedy, 2026-09-14 (smoke, 10 eval chains, short slot):**

| | @1000 t0.1 (2 runs) | @1500 t0.1 (2 runs) | @1500 **t0 (greedy)** |
|---|---|---|---|
| cut at max_tokens / over limit | 1–2 / 1 | **0 / 0** | 2 / 3 |
| median words / ratio | 490–506 / 0.74–0.80 | 464–488 / **0.70–0.71** | 631 / 0.85 (deep steps 1050–1130 of 1125; chit-chat c00025-s2 padded to 1127) |
| carry (median) | 0.92, 0.93 | 0.86, 0.89 | **1.00** (multi-peer 0.98, every k 1.0) |
| collapses | c00022 (1 of 2) | c00016-l0 new 0.46 (1 of 2) | none |
| latency | 11 s | 11 s | 17 s |

Greedy decoding removes the chain collapses and carries everything, but fills the stated limit — it pads a
chit-chat chunk to the limit and is cut again at deep steps. Temperature 0.1 stays concise (does not chase
the limit) at the price of ~0.04 carry versus @1000 and an occasional early slip that cascades. The hard
requirement is "never cut" (a cut loses the newest chunk outright), so **ship: smoke, temperature 0.1,
`SUMMARY_MAX_TOKENS_SHORT=1500`.** Greedy-with-padding is the one behaviour a small length-preference (DPO §7)
could target later: pairs concise-vs-padded for the same prompt exist for free in these eval files.

## Experiment: Qwen3-8B on the smoke dataset (planned 2026-09-14)

Rationale: text conversion, not tool calling; Qwen3-8B is dense (no linear-attention layers, Ollama fast path)
and has a real thinking-off switch. Same 94 rows, pinned eval, so the only variable is the base.
1. `python3 train_lora.py --stage check --model Qwen/Qwen3-8B --data data/dataset_train.sft.jsonl --max-seq 8192`
   — Qwen3's template does not prefill `<think>\n`, so the generic label path is used: the trainable text must
   start `<think>\n\n</think>\n\n` followed by the summary. If it starts with the summary directly, stop: the
   model would then open its own think block at inference and Ollama would file the answer as reasoning.
2. `--stage sft … --out runs/q3-sft --max-seq 8192 --load-bits 16` (same epochs/lr), `--stage sample --model runs/q3-sft/merged --data data/dataset_eval.sft.jsonl --max-new 200`
   (no `--open-think`): raw output must begin with the empty think block, then prose.
3. `--stage export --out runs/q3-gguf`; Modelfile FROM that GGUF, temperature 0.1; `ollama create summary-q3`.
4. `eval_summary.py … --model summary-q3 --max-tokens-short 1500` twice; `compare` against
   `results/eval_summary-smoke_mt1500.jsonl` and `_2`. Watch `answered_in_thinking_rows` and `think_leak_rows`
   first (template/parser issues show there), then carry / cut / over-limit.

**Qwen3-8B result (2026-09-14)**: check/sample/template all clean (0 answered-in-thinking, 0 think leaks in
80 steps — the generic label path and Ollama's Qwen3 parser agree). Same 94 rows, pinned eval @1500, two runs:

| short, 40 steps | smoke (Qwen3.5-9B) ×2 | q3 (Qwen3-8B) ×2 |
|---|---|---|
| carry (median) | 0.86, 0.89 | 0.92, 0.91 |
| multi-peer carry | 0.81, 0.80 | **0.91, 0.95** (no c00022 collapse in either run) |
| median words / ratio | 464–488 / 0.70–0.71 | 593–607 / 0.76 (deep steps 850–930 of 1125) |
| over limit / cut | 0 / 0, 0 / 0 | 0 / 0, **3** / 0 (k=0 steps with small limits: 416/387, 251/247, 726/620) |
| fabrication rows | 0, 0 | 0, **2** (c00011-s2/s3 — to verify against the text) |
| long new (10) | 0.98, 0.96 | 0.96, 0.91 |
| latency | 10.9 s | 12.4–15.3 s (≈ 25 % more words; no per-word speed win) |

Reading: q3 carries a little more, most visibly on multi-peer, and pays for it with ~25 % longer summaries, three
small-limit overshoots in one run, two fabrication flags (unverified) and slightly weaker long summaries.
Neither the carry gain (≈ noise) nor the losses are decisive; at the default 1000 cap q3's length would make it
the more truncation-prone of the two. Decision: **smoke stays the ship candidate** (4 runs, 0 over-limit,
0 fabrication, shortest output); q3 kept as the alternative if multi-peer robustness turns out to matter in
production. Base-model choice is not where the remaining variance lives — chain cascades are.

**q3 fabrication rows verified (2026-09-14): real.** c00011-s2/s3: the chain has "ten episodes" superseded by
"eight episodes"; q3 wrote *"she originally planned 12 episodes, not eight (as she had said earlier), but then
backed down again — no, she would stick with eight"* — the distractor value asserted, wrapped in a
self-correcting narration the smoke model never produces. Carried into s3. First genuine fabrication in the
project's evals (all earlier flags were matcher artifacts). Counts against q3 for the slot.

**Qwen3-8B on the 243-row v2 set (2026-09-14): more data did not help here either.** Two runs @1500:
carry 0.84 / 0.88 (q3 on 94 rows: 0.92 / 0.91; smoke 0.86 / 0.89); multi-peer back to 0.81 / 0.78 (q3-94: 0.91 / 0.95);
shorts shortened to ratio 0.70 (the v2 rows' 0.8 ceiling), longs lengthened to 892–1083 words (the v2 long rows'
957 median) with new-coverage 1.0 / 1.0; 0 over-limit, 0 fabrication, 1 think leak; fastest short latency (9.6 s).
The model copies the dataset's length distribution faithfully and its coverage does not move with row count.

### Final comparison — all four models, pinned 10-chain eval, short slot @1500, two runs each

| | rows / base | carry | multi-peer carry | ratio | over / cut / fab | long new |
|---|---|---|---|---|---|---|
| **summary-smoke** (ship) | 94 / Qwen3.5-9B | 0.86, 0.89 | 0.81, 0.80 | 0.70 | 0 / 0 / 0 | 0.98, 0.96 |
| summary-v2 | 243 / Qwen3.5-9B | 0.905 (1 run) | 0.67 | 0.81 (chases the limit) | 3 / 1 / 0 | 1.0 |
| summary-q3 | 94 / Qwen3-8B | 0.92, 0.91 | 0.91, 0.95 | 0.76 | 0–3 / 0 / 0–2 (real) | 0.96, 0.91 |
| summary-q3-v2 | 243 / Qwen3-8B | 0.84, 0.88 | 0.81, 0.78 | 0.70 | 0 / 0 / 0 | 1.0, 1.0 |

Three lessons, each now shown twice: (1) ~100 rows set the behaviour, 2.6× the rows changes nothing outside
noise on either base; (2) the model copies the training rows' length distribution, so the data's length target
is the lever for the token cap, not the base; (3) the remaining variance is chain cascades (one early slip
inherited downstream) and rare collapses — not fixable by more of the same data. The one base-related
difference (q3 carries more on multi-peer at 94 rows) came with a real fabrication and longer output.

### Phase 0 addendum — untrained `qwen3:8b` (2026-09-14, pinned 10-chain eval, short cap 1500)

Unlike untrained Qwen3.5-9B (which returned nothing in the short slot), untrained Qwen3-8B produces real
summaries: its thinking finishes inside 1500 tokens on most steps (`answered_in_thinking_rows` 0) and its prose
has no bullets, meta lines or think leaks. `--reasoning-effort none` IS honoured by Ollama for Qwen3 (latency
10 s, no reasoning) — but without thinking it stops merging.

| short, 40 steps | qwen3:8b thinking (as Honcho calls it) | qwen3:8b no-think (`reasoning_effort=none`) | summary-smoke |
|---|---|---|---|
| carry (median) | 0.80 | 0.60 (dense 0.35, k ≥ 3 0.28–0.37) | 0.86, 0.89 |
| new (median) | 0.905 | 1.0 | 1.0 |
| cut at 1500 / over limit | 5 / 0 | 1 / 2 | 0 / 0 |
| median words | 354 | 391 | 464–488 |
| format flags | 0 | 0 | 0 |
| latency short / long | 15.6 s / 36.6 s | 10.0 s / 22.4 s | 10.9 s / 16.1 s |
| long new (10) | 0.857 | 0.816 | 0.96, 0.98 |

Verdict: **the smoke model beats untrained Qwen3-8B on every measured axis** — carried facts +0.06–0.09, new
facts +0.1, long-summary coverage +0.1, zero truncation vs 12 %, 1.4× faster on short and 2.3× on long — but the
margin is moderate, not the night-and-day gap against Qwen3.5. At Honcho's default 1000 cap the thinking base
would be cut far more often (its thinking already costs 5/40 at 1500). If the fine-tune were ever unavailable,
`qwen3:8b` with the 1500 cap is the fallback to configure, not `qwen3.5:9b`.

## Retention phase, step 1–2 — 2026-09-15 (PLAN §11)

**Ruler:** +20 eval-only chains (`data/chains_eval2.jsonl`, $2.91) → 30 chains / 175 steps. The shipped model
on it, two runs: short carry **0.889 / 0.889**, new 1.0, 0 over-limit, 0–1 cut, by step k: 0.98/0.94 → 0.88/0.89
→ 0.79/0.88 → 0.78/0.85 → 0.88/0.77; multi-peer 0.71/0.83 (still the noisiest cell); long carry 0.78/1.0 (n=5).
This is the baseline every later model is compared against.

**Pairs:** student (smoke) on all 57 chains (`prev_smoke.jsonl`, 324 steps: 7 over-limit, 21 cut at the *1000*
cap it was run with, carry 0.90); teacher for every student prompt, eval chains excluded ($4.85 over 3 waves).
Build: 356 SFT rows, **142 DPO pairs** (133 dropped-facts, 8 padded, 1 format; 93 student-not-worse discarded).

**DPO run:** `--lr 5e-6`, 1 epoch = 18 steps (batch 8 — I had sized the lr for batch 4). Loss 0.693 → 0.68,
margins ±0.2, pair accuracy ~55 %: **the preference was not learned** — lr × steps ≈ 9e-5, a third of the
≈ 2.5e-4 that saturated the dialectic DPO. Eval, two runs: carry 0.873 / 0.851 (vs 0.889 / 0.889), over-limit
3 / 1, deep steps unchanged, long carry 0.84 / 0.91. Differences are inside the per-cell spread; the model is
the smoke model plus noise. **Inconclusive by under-training, not a negative result.**

Next (GPU only, $0): rerun with `--lr 1.5e-5` (lr × steps ≈ 2.7e-4) and let the early stop decide; the log must
show loss falling well below 0.5 and margins > 1 before the eval is worth running. If it saturates and the
30-chain eval still does not move carry at k ≥ 3, the DPO conclusion is negative as in the dialectic project.

**DPO converged (2026-09-15):** `dpo2` (lr 1.5e-5, 1 epoch): loss 0.69 → 0.58, margins ≤ 0.6. `dpo3` (3 epochs,
54 steps): loss → 0.03, margins 3–8, pair accuracy 100 % from step 20 — fully learned. Signature: rejected
log-probs pushed to −30…−70, chosen mostly *also* down (−5…−15) — the ordering was learned by suppressing the
rejected texts, not by raising the chosen ones, so sampled outputs need not change. They didn't:

| short, 30 chains × 2 runs | smoke | dpo2 | dpo3 |
|---|---|---|---|
| carry (median) | 0.889, 0.889 | 0.875, 0.900 | 0.859, 0.900 |
| carry k=3 / k=4 | 0.79–0.88 / 0.78–0.85 | 0.84–0.85 / 0.81–0.85 | 0.83–0.88 / 0.81 |
| multi-peer carry | 0.71, 0.83 | 0.80, 0.83 | 0.79, 0.79 |
| ratio / over limit | 0.71 / 0, 0 | 0.72–0.74 / 1, 3 | 0.74 / 1, 3 |
| long carry (n=5) | 0.78, 1.0 | 0.96, 0.90 | 0.90, 0.94 |

Every difference sits inside the two-run spread; the only consistent movement is length (+3 points of ratio,
1–3 over-limit rows) — the wrong direction. **Verdict: fact-retention DPO on 142 on-policy pairs does not move
deep-step carry.** Same conclusion as the dialectic project's DPO ablation, now on a second task with a
converged run and a 30-chain ruler. §11 step 3 (the Qwen3 base under the same recipe) is moot.

The cheap plan (§11) is closed: $7.76 spent, one durable asset (the 30-chain ruler with a stable baseline) and
one durable negative (SFT + DPO on this data distribution is at its ceiling). The shipped model stays.
