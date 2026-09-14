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
| 2026-09-13 | **Scorer rewrite after reading the Phase 0 rows** (`summary_scoring.has_fact`). Old rule (dialectic `has_entity`: first two tokens present anywhere) flagged "Fenna's friend Vera" as fabricated on "fenna"+"s", "the city of Ashvale" on "the"+"city", "Batch nine" on "batch"…"nine" 30 words apart — every one of the 7 dialectic "fabrications" was an artifact. New rule: normalised exact substring (curly quotes, thousands separators, number words → digits), else the fact's *anchor* tokens (stopwords, one-letter fragments and the chain's peer names removed) co-occur within 15 tokens: all of them for 1–2 anchors, all but one for ≥ 3, numeric anchors always. **Fabrication is strict**: every anchor within 6 tokens — distractors are near-misses of true facts by construction ("seven slats per side" vs "nine"), so the differing anchor must be present. New `echo` flag (`</conversation>`, `<previous_summary>`, the no-previous sentence in the answer). Re-scored Phase 0: dialectic short new 0.88 / carry 0.22 (k=1 0.38 → k=5 0.07), **fabrication 0/104** (was 5), think-leak 8, echo 2; long new 0.74 / carry 0.18, echo 3. Base long new 0.88, fabrication 1/29 (a "15 inches … prints" near-miss, accepted as noise). Gate decision unchanged. | Matching leniency was the largest error source in the Phase 0 numbers; the dataset filter (stage 4) uses the same function, so the fix also decides what gets trained on. |
| 2026-09-13 | Stage 1 fixes from the Phase 0 chains: (a) `c00009` failed as "0 messages" because a truncated reply made `extract_json` fall back to the first inner object (a peer) — `parse_chain_json` now requires the whole reply to parse and reports "truncated? N chars" plus `stop_reason` (now exposed by `llm_backend.complete`/`batch_results`); output budget raised to 160 tokens/message + 6000; (b) a peer that never speaks (`Lubna_placeholder`, c00008) is dropped; (c) themes cycle a shuffled list instead of random choice (3 speedruns and 3 podcasts in 27 chains). `c00028`/`c00029` failed with *"Your credit balance is too low"* — an account issue, retried by re-running `submit`/`fetch` after topping up. | |
| 2026-09-13 | **Smoke stage 3 was truncated.** The teacher was sent Honcho's `max_tokens` (1000 short / 4000 long) as its *own* output budget; with thinking on (`effort=medium`) those tokens are shared, and 80/98 short summaries at k ≥ 1 were cut mid-sentence — always losing the newest chunk, since the narrative is chronological (new-fact coverage 1.0 at k=0, 0.75 at k=1, ≈ 0.2 from k=2; carry stayed 1.0). Stage 4 then kept 56/155. Fix: `TEACHER_MAX_TOKENS` 8000/16000 independent of Honcho's caps (the word limit is in the prompt and enforced by the filter); `stop_reason` recorded; a row cut at the budget is failed and every later step built on it is marked stale and regenerated wave by wave; legacy rows without `stop_reason` are failed when they end mid-sentence. Teacher is now asked for ≤ 85 % of the limit (filter stays 0.9; 21/155 smoke rows sat in 0.9–1.0). | The bug was invisible in the mock (the mock never truncates) and in the aggregate (carry looked perfect). Lesson for TRAIN.md: read a k ≥ 2 row end to end before building a dataset. Base-prev rows (previous = dialectic output) are unaffected and kept. |
| 2026-09-13 | **Smoke stage 3, second pass: content solved, length not.** With the teacher's own token budget every summary is complete: new and carried coverage 1.0 at every step (k=0 … 5), 2 low-coverage rows in 155. But the teacher overshoots any percentage ask by 6–13 points (asked 85 %, wrote median 0.91 at k=0, 0.98 at k=2, 1.05 at k=3), and from k=2 the fixed 750-word short limit meets 26–59 accumulated ledger facts: 44/155 rows exceed the *hard* limit, 85 exceed 0.9×. 800 words ≈ 1070 tokens > Honcho's `max_tokens` 1000 — such a summary would be cut in production, so the 0.9 filter stays. Fix: the teacher gets a **numeric budget** (`0.75 × output_words` words, stated in the system prompt with what to cut first: connective prose, assistant commentary, repeated attributions; a checklist value last); a row over `0.9 ×` is a *failed generation*, retried with its dependants up to `MAX_ATTEMPTS = 3`, then given up and reported. Scorer: `1/15` is one token (a "fifteen prints" false positive on a shutter speed). | Whether 59 facts fit in 675 words is the open question for dense chains at k ≥ 3; if the give-up count is high, the answer is that Honcho's short slot genuinely cannot hold a 100-message ledger and the filter should accept carry ≥ 0.8 there (PLAN §7 "counting words is hard" was about the student; the teacher has the same problem). Redo on the smoke file: 95 short rows, 60 kept. |
| 2026-09-13 | **Pilot (3 hardest chains, sync, $0.70): the numeric budget works.** Good short rows median ratio 0.79 (range 0.64–0.83), coverage 1.0 new and carry, long rows 0.34. 3/9 short steps overshot marginally (0.91, 0.91, 0.95). Retry strategy changed: an over-budget draft is not rewritten from scratch but **compressed** — the teacher gets its own draft plus the checklist, `effort=low`, "rewrite to at most N words" — cheap and convergent; `MAX_ATTEMPTS` 3 (fresh, compress, compress). `run` now loops passes in one invocation. `llm_backend._anth_params` honours a per-job `effort`. | Regenerating with thinking to land under 0.9 by luck was the wasteful part; the pilot bounded it before the full batch. |
| 2026-09-13 | **Pilot, second run: compress works (5/5 passes landed, median ratio 0.82, coverage 1.0); the residue is physics.** c00006-s4 and c00013-s4 (dense chains, 43–59 facts due, limit 750) compress to 0.94–0.95 and no lower while every checklist value is kept — 1 step given up in 21. Decision: the **last** compress attempt may drop up to a tenth of the checklist values (least consequential; never a changed value, name, date or decision). This is Honcho's own instruction ("drop lower-priority detail") and stays inside the stage-4 filter (coverage ≥ 0.9). A step that still fails is given up: a real deployment's short slot cannot hold a 100-message ledger, and the model should learn prioritisation there, not padding past the limit. Cleared for the full batch. | Pilot cost $1.46 total for the answer; a blind full batch would have burnt three attempts on every dense k ≥ 4 step. |
| 2026-09-13 | **Full stage 3 done: 151/155 good, 4 given up (all dense chains at k ≥ 4: c00006-s4, c00007-s4, c00013-s5, c00019-s2), median ratio 0.82, coverage 1.0/1.0, 75 rows reached the budget through a compress pass.** Two bugs surfaced: (1) `write_out` merged over the raw disk rows, so an over-budget row from the second run (no marker on disk) beat its own still-failing compress retry every wave — the fetch loop resubmitted c00027-s2 seventeen times ($0.02 each); now merges over the invalidated view. (2) "fifteen prints" matched "prints at 10 by 15 inches": a two-anchor distractor now needs its anchors adjacent. Dataset: 93 train / 55 eval SFT rows (short 74/44, long 19/11) after 7 drops. | Smoke dataset is ready for `train_lora.py --stage check`. |
| 2026-09-13 | **Smoke SFT passed the §6 gate** (TRAIN.md "Smoke run"): 0 over-limit, carry 0.85 vs 0.49 (base scored from its reasoning), 0 fabrication, 0 bullets/meta/echo/think-leak, 0 empties, 0 answered-in-thinking, 0 truncations; long slot 0.97 new coverage, 3× faster than the base. 94 training rows were enough to set the behaviour — as in the dialectic project. Epoch 2 already overfits slightly. Next per §9: live check through Honcho (harness untested), then the 150-chain run aimed at the measured weak spots (multi-peer, carry at k ≥ 4, superseded values in long summaries), then the combined dialectic+summary run. | The base's short slot produces nothing in production; the smoke model is a strict improvement and could take the slot now with a one-line rollback. |
| 2026-09-14 | **Live check closed.** The c00022-s1 confabulation did not reproduce (batch or one-by-one); the proxy log shows Honcho's request equals the parity prompt in shape and content. Verdict: a one-off model failure on the weakest category (1/5 attempts), not plumbing. **150-chain run (§9 step 3) retargeted**: mix dense 20 / preference 10 / supersession 20 / long-tail 20 / chit-chat 5 / multi-peer 25 %, lengths 80–120 (deep carry). The weak-previous-summary variant will use the **smoke model's own outputs** on the new chains (`gen_summary_rejected.py --model summary-smoke`) — the exposure-bias target §2.2 describes, now that a student exists; the dialectic outputs served that role only for the smoke. Estimate: chains ≈ $12 batch. | Honcho's own word limit (576) is ~2 % below our token-estimate (587): Honcho counts tokens with tiktoken, we use len/4. Harmless, noted. |
