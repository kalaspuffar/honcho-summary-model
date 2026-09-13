# SUMMARY_PLAN.md — a summary model for Honcho (proposal, 2026-09-12)

Status: **proposal, nothing built.** Written to move into its own repository. Everything below that
says "as in the dialectic project" refers to this repo's `PLAN.md` / `TRAIN.md` and the lessons
recorded there on 2026-09-11/12.

## 0. Decide whether to train at all — measure first

The dialectic project spent two days on failures that were not about the model (wrong base,
thinking-token mismatch, scorer blind spots). Do not start generating data for summaries before
the base has been measured on the real prompt. Phase 0 is one afternoon and may end the project.

Phase 0 (no training):
1. Point Honcho's summary slot at the base and at the dialectic model
   (`SUMMARY_MODEL_CONFIG__TRANSPORT=openai`, `…__MODEL=qwen3.5:9b` / `dialectic_s50`,
   `…__OVERRIDES__BASE_URL=http://node7.ea.org:11434/v1`), drive a session past 20 and 60
   messages, read the stored short and long summaries (`GET /v3/workspaces/{ws}/sessions/{id}/summaries`).
   This is a plumbing check only (limits, chaining, empties); nothing read here enters the dataset.
2. Run the offline eval of §5 on ~30 synthetic sessions with the base.
3. Look for the failure classes that would justify training:
   - **empty / answered-inside-thinking** — the base at temperature 0.1 wrote its dialectic
     answers inside `<think>` and stopped (TRAIN.md §7); the summary prompt also gets Ollama's
     `<think>\n` prefill, so the same thing can happen. If this is the *only* failure, the fix
     is a temperature/Modelfile change or the same encoding trick, not a dataset.
   - **word-limit violations** — the prompt says "Hard limit: N words maximum"; local models
     routinely overshoot. Measure the distribution of `words / limit`.
   - **dropped facts** — from the new messages, and, separately, from the previous summary
     (the incremental merge is the hard part: "ALWAYS make your new summary inclusive of both").
   - **fabrication** — facts not in the messages or the previous summary.
   - **format** — bullet lists (prompt asks for narrative), meta-commentary ("Here is a
     summary of…"), leaked `<think>` text, wrong language.
4. Gate: train only if at least one of word-limit, dropped-facts or fabrication is a real rate
   (say > 10 % of summaries) on the base. Empty/format failures alone have cheaper fixes.

## 1. What Honcho sends (parity — copy, never paraphrase)

From `src/utils/summarizer.py` (Honcho v3.1.2):

| item | value |
|---|---|
| call shape | `honcho_llm_call(prompt=…)` → **one user message**, no system prompt, no tools |
| short prompt | `short_summary_prompt(formatted_messages, output_words, previous_summary_text)` |
| long prompt | `long_summary_prompt(…)` — adds "emotional state", "themes and patterns", "exhaustive" |
| messages | `"\n".join(f"{peer_name}: {content}")` |
| previous summary | the stored summary text, or the literal sentence *"There is no previous summary -- the messages are the beginning of the conversation."* |
| output_words | short: `int(min(input_tokens, MAX_TOKENS_SHORT) * 0.75)`; long: `int(MAX_TOKENS_LONG * 0.75)` |
| max_tokens | `SUMMARY_MAX_TOKENS_SHORT` = 1000, `SUMMARY_MAX_TOKENS_LONG` = 4000 (defaults) |
| cadence | short every `MESSAGES_PER_SHORT_SUMMARY` = 20 messages, long every 60 |
| transport | same OpenAI-compatible path as the dialectic (`/v1/chat/completions`, `content` only) |
| temperature | not set by Honcho → the Ollama Modelfile's value (our dialectic Modelfile: 0.1) |

Parity file: `summary_prompt.py` = verbatim copy of the two prompt functions + `_format_messages`
+ the no-previous-summary sentence, with a header naming the Honcho commit. Never hand-edit; if
Honcho changes it, re-copy and regenerate (same rule as `honcho_prompt.py` here).

Note what the prompt asks for: **chronological narrative, not bullets; as many explicit facts as
possible; within a hard word limit; inclusive of the previous summary.** That is the target
behaviour. "Terse" is *not* the goal this time — a short summary may legitimately be 700 words.
The goal is *dense, complete, within limit, grounded*.

## 2. What to generate

**Privacy rule (Daniel, 2026-09-12): the training and eval data contain nothing from the real
deployment.** No real names, sessions, topics, summaries or interests — not as seeds, not as
examples, not as domain lists. The generator invents personas and domains itself (fiction,
hobbies, work life, travel, cooking, sports, small-business admin, study, health-of-invented-people,
etc.). Real Honcho data is touched only by the *live* checks (§0.1, §6), read-only, and never
written into `data/`. The model learns *how to summarise*, not *what* anyone talked about.

Everything is synthetic and ledger-based so it can be scored automatically, as in the dialectic
project (required/forbidden facts). Three stages, same CLI shape as here (`estimate/run/submit/
status/fetch`, Anthropic batches for the teacher, `llm_backend.py` reused as is).

### 2.1 Sessions with a fact ledger (stage 1, teacher)

One row = one **session chain**: a conversation of 60–120 messages between 2–3 peers (user +
assistant is the common Honcho case; include some peer–peer sessions), cut into Honcho-sized
chunks (20 messages), with a **ledger** of atomic facts:

```
{ id, domain, peers:[{name, bio}],
  messages:[{seq, peer, text}],                    # 60–120
  facts:[{id, text, first_seq, kind}],             # atomic, verifiable substrings/paraphrase keys
  changes:[{fact_id, superseded_by, seq}],         # values that change mid-session (state must be latest)
  distractors:[{text}],                            # plausible facts NOT in the session (fabrication probes)
  chunks:[{k, seqs:[a,b], limit_words_short, limit_words_long}] }
```

Category mix (interleaved like `gen_contexts.MIX`, so small runs cover everything):

| category | what it stresses | share |
|---|---|---|
| dense-facts | many small facts (names, dates, numbers) per chunk | 25 % |
| preference/opinion | user preferences and questions (prompt items 2–3) | 15 % |
| supersession | a value changes mid-session; summary must carry the latest | 15 % |
| long-tail merge | facts only in the previous summary, absent from the new chunk (must survive) | 20 % |
| chit-chat | little factual content; the summary must be SHORT, not padded to the limit | 10 % |
| multi-peer | 3 peers, attribution matters ("Anna said… Ben disagreed…") | 15 % |

Sizes: the dialectic lesson is that **50–150 rows set a behaviour**; more rows did nothing
measurable. Plan for **150 chains for training, 40 for eval**, persona-disjoint split. Each chain
yields 3–6 short-summary examples and 1–2 long ones, so ~600 short + ~200 long training rows.
Start with 30 chains (Phase 0 and the first smoke) — do not generate more until the smoke passes.

### 2.2 Chosen summaries (stage 3, teacher = Opus, batches)

For each chunk k, the teacher writes the summary **from the exact Honcho prompt** with the
previous summary = **the teacher's own chosen summary of chunk k-1** (a clean chain). Then filter
with the scorer (§5): ledger coverage ≥ 0.9 of the facts due so far, 0 distractors, words ≤ 0.9 ×
limit (train under the limit, models overshoot), narrative form, no meta line. Drop what fails;
report the drop reasons as `build_dataset.py` does.

**Exposure bias.** At inference the previous summary is the *student's* own output, not the
teacher's. Add a second variant for ~30 % of chunks: previous summary = the **base model's**
summary of chunk k-1 (from stage 2). The chosen summary must still contain every ledger fact due,
i.e. the teacher repairs what the base dropped. This teaches the merge to recover, not just to copy.

### 2.3 Rejected summaries (stage 2, base model, optional)

Base `qwen3.5:9b` on the exact prompt, same chunks, via Ollama `/v1` — its real failures (too
long, bullets, dropped previous facts, empty). This is the Phase 0 measurement *and* the DPO
rejected side if DPO is ever used. Keep the invariant: rejected comes from the base, never from
a teacher asked to "write a bad one".

## 3. Training recipe

**SFT only, first.** The dialectic ablation (TRAIN.md §12) showed SFT+DPO ≡ SFT at 50–150 rows;
DPO moved the rejected log-probs down and changed no output at temperature 0.1. Do not budget DPO
hours until SFT-only fails a measurable target that a preference signal could plausibly fix
(word-limit overshoot is the one candidate: the pair "same content, within limit" vs "over
limit" is a clean preference).

- Rows: `{"messages": [{"role":"user","content": <exact Honcho prompt>}, {"role":"assistant","content": <chosen>}]}` — no system turn, no tools. `train_dialectic.py` already handles this shape (no tool turns found → answer turn only).
- Encoding: the **served-prompt tokenization** fix (`encode_example`, TRAIN.md §12) applies unchanged — Ollama prefills `<think>\n`, the model must emit `\n</think>\n\n` itself. Verify with `--stage check` and `--stage sample --open-think` before the first run, as always.
- Base: the stripped text-only Qwen3.5-9B (`/data/smoke/qwen35-9b-text`), same as the dialectic.
- Epochs 2, lr 2e-4, r=16, `--tool-turns none` is implicit.
- **Sequence length is the new constraint.** A long-summary row = prompt (~200 tokens) + previous
  summary (≤ 3000 words ≈ 4k tokens) + 60 messages (≈ 3–6k) + target (≤ 3000 words ≈ 4k) ≈
  **11–14k tokens**. Short rows ≈ 3–5k. `--max-seq 16384 --load-bits 16` on the A6000 with
  gradient checkpointing; expect ~3× the dialectic step time on long rows. Train short and long
  together (one model serves both slots); if 16k does not fit, cap the long target at 2000 words
  in the data and set `SUMMARY_MAX_TOKENS_LONG` accordingly — Honcho's limit is operator-set.
- Sizes/time (from the dialectic measurements, 24 s per 4-sample step at ~4k tokens): 600 short
  rows ≈ 1.5 h/epoch; 200 long rows at ~12k tokens ≈ 1.5–2 h/epoch. **Smoke first: 30 chains
  (~120 short + 40 long rows), 2 epochs, under 1.5 h**, then eval, then decide on the 150 chains.

**One model or two?** Train the summary adapter **separately first** so effects are attributable.
Then a combined run (dialectic rows + summary rows in one SFT) to get **one Ollama model for both
slots** — one GPU-resident model is a real operational win, and the two prompts are far apart so
interference is unlikely; verify with both evals (the dialectic 302-row eval must not move).

## 4. Serving

Same export path (`--stage export` patches the template; Ollama uses its built-in renderer anyway
— the model closes `<think>` itself). One Modelfile; temperature: summaries are long generations,
so **test 0.1 vs 0.7** in Phase 0 — the dialectic Modelfile's 0.1 was chosen for terse factual
answers and may make 700-word narratives repetitive. `SUMMARY_MODEL_CONFIG__*` env lines point
Honcho at it; rollback is one line.

## 5. Scoring (one scorer module, as here)

`summary_scoring.py` owns every rule; the self-test fails if a regex is defined elsewhere.

| metric | rule |
|---|---|
| `fact_coverage_new` | fraction of ledger facts due in *this* chunk found in the summary (substring / two-token match like `scoring.has_entity`) |
| `fact_coverage_carry` | fraction of facts due in *earlier* chunks found — the merge metric |
| `latest_state` | for each change, the newest value present and the old one not asserted as current |
| `fabrication` | any distractor asserted |
| `limit_ratio` | words / output_words; violation if > 1.0; report the distribution |
| `format` | bullets (`^\s*[-*•]`), numbered lists, meta-commentary (`^(here is|summary:|this conversation)`), leaked `<think>`, empty |
| `narration` | reuse the dialectic NARRATION idea for any "I will summarize…" leak |

Aggregate per model: medians of the coverages, violation rates, fabrication rows, format rows,
empty rows, `answered_in_thinking_rows` (from the `/v1` reasoning field, as in `eval_model.py`).

## 6. Evaluation and gates

- **Offline**: 40 held-out chains, chunk by chunk, previous summary = the model's *own* previous
  output (the honest setting), via Ollama `/v1` with the exact prompt. Compare base vs tuned in
  one table (`compare` subcommand as here).
- **Live**: Honcho with `SUMMARY_MODEL_CONFIG` swapped; drive 3 real sessions past 60 messages;
  read the summaries; run the offline scorer where a ledger exists (seed those sessions from
  synthetic chains so it does).
- Gate to deploy: limit violations ≤ 5 %, `fact_coverage_carry` not below the base, fabrication
  0, no bullets/meta, no empties. Gate to combine with the dialectic model: dialectic eval
  unchanged within noise.

## 7. Risks and what to do about them

| risk | mitigation |
|---|---|
| counting words is hard for a 9B; overshoot persists after SFT | train targets at ≤ 0.9 × limit; if violations stay > 5 %, that is the one DPO experiment worth running (pairs: within-limit vs over-limit, same facts) |
| chain drift: facts lost once are lost forever | the `long-tail merge` category + the base-previous-summary variant (§2.2) train recovery; `fact_coverage_carry` measures it |
| long rows do not fit 16k | cap long targets at 2000 words and set Honcho's limit to match; or train long summaries on a 24k-capable run later |
| the base is already fine (Phase 0) | then stop: point the slot at the dialectic model and move on |
| Honcho prompt changes | parity file with commit hash; regenerate chosen (cheap: batches) and retrain (~2 h) |
| teacher cost | Opus long summaries ≈ 4k output tokens each; 200 long + 600 short ≈ 1.3M output tokens — estimate with `llm_backend.estimate_usd` before `submit`, batches are half price; the 30-chain smoke first |

## 8. Repository layout for the new repo

Reuse unchanged: `llm_backend.py`, `train_dialectic.py` (rename `train_lora.py`; it is already
task-agnostic), `mock_or_server.py`, `mock_anth_batch.py`, the `verify_pipeline.py` pattern.
New: `summary_prompt.py` (parity), `gen_sessions.py` (stage 1), `gen_summary_rejected.py`
(stage 2 / Phase 0), `gen_summary_chosen.py` (stage 3), `build_summary_dataset.py`,
`summary_scoring.py`, `eval_summary.py`, `honcho_summary_harness.py` (live check).
Keep `PLAN.md` + `TRAIN.md` as the decision and failure logs from day one.

## 9. Order of work

1. Phase 0 (one afternoon): parity file, 30 synthetic chains, base + dialectic model measured
   offline and live. Decide.
2. Smoke: teacher summaries for the 30 chains → filter → SFT 2 epochs (< 1.5 h) → offline eval →
   live check. Pass = the §6 gate, using 10 of the 30 chains as the eval.
3. Only then 150 chains, one run, eval, deploy to the summary slot.
4. Combined dialectic + summary run; both evals; one model for both slots.

## 10. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-09-12 | Short and long summaries are **independent chains**: short k's previous = stored short k-1, long j's previous = stored long j-1 (never the other kind). | Honcho `summarizer.py` at the parity commit calls `get_summary(db, ws, session, summary_type)`; both are created concurrently when message 60 lands. `summary_chain.steps()` mirrors this. |
| 2026-09-12 | The short word limit includes the previous summary's tokens: `output_words = int(min(tokens(messages) + tokens(previous), 1000) * 0.75)`; so chunk 0 gets ~500–650 words and every later chunk the 750 cap. Long is always `0.75 × MAX_TOKENS_LONG`. Limits are computed live per step (from the actual previous summary), not stored in the chain row — the `chunks[].limit_words_*` fields of §2.1 are dropped. | Parity with Honcho's `input_tokens = messages_tokens + previous_summary_tokens`. Token counts approximated with `len/4` (Honcho uses per-message tiktoken counts); the scorer takes `output_words` explicitly so nothing else depends on the approximation. |
| 2026-09-12 | Honcho's `max_tokens` (1000/4000) is sent as is in Phase 0. Qwen's `<think>` tokens count against it, so "truncated inside thinking → empty content, finish_reason=length" is an expected Phase 0 failure class; rows record `finish_reason`, `reasoning_chars`, `answered_in_thinking`. | Measure the real behaviour first (§0.3). Operator caps are flags everywhere (`--max-tokens-short/long`). |
| 2026-09-12 | Chains are 60/80/100/120 messages (multiples of 20; the teacher's output is cut to the last full block; < 60 fails). Facts are validated as **verbatim substrings** of a message; `first_seq` is the earliest message containing the text; distractors that occur in the text are dropped. | The scorer is substring-based, so an unverifiable fact would only produce noise. |
| 2026-09-12 | Teacher (stage 3) gets the exact Honcho prompt as the user turn plus a system prompt with the hard rules and a **ledger checklist** (facts due, changed values, forbidden distractors); `--blind` omits it. | Denser targets, fewer drops. The student never sees the checklist; the filter in stage 4 still decides. |
| 2026-09-12 | Chosen ids: `c00003-s2` (clean), `c00003-s2b` (previous = base's s1). Batches run in **waves** (one batch per dependency level; `fetch --waves` loops). | A step needs the previous step's summary; sync `run` does it per chain in a thread. |
| 2026-09-12 | DPO pairs (if ever used) share the prompt exactly: rejected step k vs the `-b` variant of step k (its previous *is* the base's k-1 output), plus step 0 vs clean. Pairs with a different previous summary are not pairs. | §3: DPO only for word-limit overshoot, and only with a clean prompt-identical preference. |
| 2026-09-12 | Persona split counts **human peers only**; an assistant's product name recurs across chains (as it does in production) and must not glue chains together. | `build_summary_dataset.split_by_persona`. Mock chains all share "Nimbus"; with the assistant counted everything landed in eval. |
| 2026-09-12 | Eval `--answer-from-reasoning` is a baseline-only fallback (score and chain the reasoning text when content is empty); the tuned model never gets it. Temperature is not sent unless asked (Honcho leaves it to the Modelfile). | Same policy as the dialectic eval (TRAIN.md there, §7). |
| 2026-09-12 | Cost, local estimate: stage 1 ≈ $2.2 (batch) / $4.4 (sync) for 30 chains on Opus, ≈ $11.7 / $23.3 for 150. Stage 3 estimate is printed once chains exist (depends on chain lengths). | `gen_sessions.py estimate`. |
| 2026-09-13 | **Phase 0 result (27 good chains of 30, 133 steps per model, node7 Ollama, Honcho max_tokens 1000/4000). Gate: TRAIN.** Base `qwen3.5:9b` short slot: 104/104 empty — every request hit `finish_reason=length` with all 1000 tokens spent inside `<think>`; identical at temperature 0.1 and 0.7. Base long slot (4000 tokens): 14–17/29 truncated in thinking, 4/29 empty; when it does answer it is decent (coverage_new median 0.9, no bullets, no overshoot: median 869 words of 3000). `dialectic_s50` in the summary slot answers (101/104) but is terse (median 161 words of ~450; long 290 of 3000) and **drops two thirds of the earlier facts** (carry median 0.32 short / 0.26 long, falling with k), 5/104 fabrication rows, 8/104 think leaks. | Empty-in-thinking is not the *only* failure and has no cheap fix: Qwen3.5 on Ollama has no thinking-off switch and even 4000 tokens truncate half the long summaries, so the fix *is* the served-prompt SFT. Dropped facts on the one model that produces output is far above the 10 % gate. Word-limit overshoot does not exist (both under-produce), so DPO stays off the table. Temperature 0.1 is kept (0.7 was slightly worse: more truncation, 3 vs 2 fabrications, one meta line). |
| 2026-09-13 | Scorer fix: an empty or narration-only summary scores 0.0 only for coverages that had facts due; a coverage with nothing due stays `None` (before, an empty chunk-0 row counted as carry 0.0 and dragged the base's carry medians to 0). Re-score existing files with `eval_summary.py rescore <file> --chains data/chains.jsonl`. | Found while reading the Phase 0 aggregates. |
| 2026-09-13 | Stage 3's base-previous variant will read `data/rejected_dialectic.jsonl` (`--rejected`), not the base file: the base's short summaries are all empty, so no variant could be built from them. The dialectic model's terse, fact-dropping summaries are exactly the "weak previous summary to repair" the variant is meant to teach. The `.dpo.jsonl` this yields is dialectic-vs-teacher and stays unused. | PLAN §2.2 exposure bias; rejected/previous still comes from a student model, never a teacher. |
| 2026-09-13 | Watch item: `c00014` was flagged for fabrication by every model — check its distractors before trusting fabrication counts. Serving risk: 750 words ≈ 1000 tokens, so a dense short summary can be cut by `SUMMARY_MAX_TOKENS_SHORT=1000` even after the fine-tune; measure `finish_length_rows` on the tuned model before touching the operator setting. | |
