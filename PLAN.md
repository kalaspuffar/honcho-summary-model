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
| 2026-09-14 | **§9 step 3 becomes an increment experiment (Daniel): +30 chains first, measure, then decide on 5–6×.** Smoke so far: $31 on Opus for 30 chains → 155 teacher summaries → 150 kept → 94 train / 56 eval rows (≈ $7 of it lost to the truncation bug). Design so the answer is measurable: (1) the eval set is **pinned** to the smoke's 10 eval chains (`build_summary_dataset.py --eval-chains data/dataset_eval.sft.jsonl`; new chains sharing a human peer name are dropped from train, never moved into eval); (2) the 30 new chains are drawn only from the weak categories (`--only-categories multi-peer,supersession,long-tail-merge,dense-facts`, lengths 80–120) so any gain lands where the smoke was measured weak; (3) `eval_summary.py compare` now breaks carry/new down by category and by step k, and each model is evaluated **twice** on the same chains to see the noise floor (the c00022-s1 one-off shows a single run can swing a category). Success = multi-peer and k ≥ 3 carry medians move by more than the two-run spread; otherwise 94 rows were enough and the smoke model ships. Generator stays Opus for this experiment so the only variable is data quantity. Cost ≈ $2.7 chains + ≈ $8 teacher (batch). | |
| 2026-09-14 | **Length target 0.9 → 0.8 of the limit (teacher asked 0.7).** The smoke model on the 30 new chains (`data/prev_smoke.jsonl`, a free eval on hard chains): short carry 0.91 median, new 1.0, but **17/152 short steps cut at Honcho's 1000-token cap** (`finish_reason=length`), each losing the newest chunk (new coverage 0.0–0.1 on those rows). Cause: Honcho's limit is 0.75 × max_tokens = 750 words ≈ 1000 tokens, the teacher rows at deep steps sat at 0.85–0.9, the model imitated them (650–780 words). Cut rows began at ~620 words → ceiling 0.8. 95 of the 151 smoke rows fall in (0.8, 0.9] and are regenerated as compress passes (~$3) so the v2 dataset is consistent. **Confound acknowledged**: v2 = more data *and* a tighter length target; `finish_length_rows` in the eval isolates the second effect (should go to 0), the per-category/per-k carry breakdown the first. | Also seen on the new chains: 2 meta lines, 1 bullet row, 1 think leak in 191 steps; the base-prev variant source is the smoke model's own output (PLAN §10 2026-09-14, above). |
| 2026-09-14 | **Increment experiment result: +30 chains (243 vs 94 rows) is not a measurable improvement** (TRAIN.md "Increment run v2"): carry +0.03–0.06 inside the two-run spread, multi-peer 0.70–0.95 between runs, deep steps unchanged, long new-coverage +5 points at the ceiling; length and truncation slightly worse. §9 step 3 (150 chains) is **cancelled**; the dialectic lesson holds for summaries too. Spent on the increment: $4.73 chains + $10.16 teacher. Remaining defect for both models: cut at `max_tokens=1000` on 2–12 % of dense deep short steps. Next, in cost order: free offline test of `SUMMARY_MAX_TOKENS_SHORT=1500`; then, if needed, the §7 length-DPO on the prompt-identical pairs that already exist; ship whichever of smoke/v2 truncates least under the chosen setting. | Whole-job cost factor 1.3 → 1.6 (actual +21 % over estimate). |
| 2026-09-14 | **Cap test**: with `SUMMARY_MAX_TOKENS_SHORT=1500` the smoke model has 0 truncations and 0 over-limit rows at unchanged length (median 488 words, ratio 0.70) — it does not write toward the larger limit. v2 does (deep steps 940–1140 words), still overshoots and showed a near-empty collapse. **Decision: ship the smoke model, not v2; recommend `SUMMARY_MAX_TOKENS_SHORT=1500` in Honcho.** Length-DPO (§7) is not needed for the smoke under this setting. Before flipping the production slot: one free repeat of the smoke @1500 (carry read 0.86 from one chain's cascade vs 0.92/0.93; confirm it is noise) and one run at temperature 0 (greedy) to see whether the chain collapses are sampling variance. v2's 243 rows stay on disk for the combined dialectic+summary run (§9 step 4). | Both models show chain-level cascades: one early slip is inherited by every later step; the eval's 10 chains make one cascade move the median by ~0.05. |
| 2026-09-14 | **Ship decision (final for this phase): `summary-smoke`, Modelfile temperature 0.1, Honcho `SUMMARY_MAX_TOKENS_SHORT=1500`.** Two runs @1500: 0 cut, 0 over limit, carry 0.86/0.89, length unchanged (ratio 0.70). Greedy (t=0) @1500 carries 1.0 with no collapses but pads to the limit (chit-chat to 1127 words) and is cut on 2/40 — rejected for the slot; noted as the target of a possible later length-DPO. v2 not shipped. Remaining §9 item: step 4, the combined dialectic + summary model (one Ollama model for both slots), using the v2 dataset (243 rows, superset) plus the dialectic rows; gate: both evals unchanged within noise. | Total project spend to here ≈ $47 on Anthropic ($31 smoke incl. bugs, $15 increment) + node7/A6000 time. |
| 2026-09-14 | **`SUMMARY_MAX_TOKENS_SHORT=1500` side effects checked in Honcho's source** (config.py, summarizer.py at the parity commit): the setting is used in exactly four places — the model call's `max_tokens` (short/long) and the prompt's `output_words` (short: `min(input_tokens, MAX)*0.75`, long: `MAX*0.75`). Validated `0 < x <= 10_000`. Summaries are stored as text in `session.internal_metadata` (no size limit, no truncation on store). The only downstream consumer of summary *size* is `get_session_context(tokens=…)`: it reserves 40 % of the caller's token budget for the summary and drops the summary (warning, `summary: None`) if it does not fit — so a summary that grew would eat into the messages returned or vanish from small-budget context calls. Our model does not grow at 1500 (median 464–488 words either way), so nothing changes downstream; `MESSAGES_PER_*` untouched. Safe. | If a future model *did* chase the limit (as v2 and greedy do), 1500 would make its summaries up to 1125 words and small `tokens=` context calls would drop them — keep the "does not chase the limit" check in every eval. |
| 2026-09-14 | §9 step 4 (combined dialectic + summary model) **dropped** — not a priority for this project (Daniel). Next experiment instead: same 94-row smoke dataset on **Qwen3-8B** (dense, text-only, a proper thinking-off switch; Daniel's deriver experience favoured it for text tasks). Zero generation cost; the dataset is model-agnostic. Compare on the pinned eval @1500, two runs, against `summary-smoke`. | `train_lora.py` already accepts `Qwen/Qwen3-8B` (documented fallback); its template lacks the served `<think>\n` prefill, so `encode_example` takes the generic path and the trainable text must begin `<think>\n\n</think>\n\n` — read `--stage check`, then `--stage sample --open-think` is not the right probe for Qwen3 (no prefill): use plain `--stage sample`. |
| 2026-09-14 | **Qwen3-8B on the same 94 rows** (TRAIN.md): a valid alternative, not a better model — carry +0.03–0.06 (multi-peer +0.1–0.15, consistent), but 25 % longer output, 3 over-limit rows in one run, 2 fabrication flags to verify, slightly weaker long summaries, no speed advantage. Ship decision unchanged (smoke). Both GGUFs kept. | Two bases, same data, same eval: the data and the length behaviour decide more than the base does. |
| 2026-09-14 | q3's two fabrication flags are **real** (invented "12 episodes" where the chain says ten → eight, plus a "no, she would stick with…" self-correction). Smoke: 0 in 4 runs. Ship decision unchanged. Open question (Daniel): would Qwen3-8B gain from the 243-row v2 set where Qwen3.5 did not? Free to test (GPU + two evals); expectation from the Qwen3.5 increment is no measurable change, with the v2 set's 0.8 ceiling possibly shortening q3's output. Success = carry ≥ smoke, 0 over-limit, 0 fabrication, ratio ≤ 0.76 on the pinned eval, two runs. | |
| 2026-09-14 | **Qwen3-8B × 243 rows: no gain** (carry 0.84/0.88, multi-peer edge gone, shorts shorter/longs longer — it copies the v2 length distribution). Four models, one eval (TRAIN.md "Final comparison"). **Project ships `summary-smoke`**; experiments closed. Data lesson stands on both bases: ~100 rows set the behaviour; the dataset's length target is the lever, not the base. Follow-ups if ever needed, in order: length-preference DPO on existing pairs (greedy pads; v2/q3 chase the limit), a chain-cascade study (why an early slip is never recovered — the base-prev variant was meant to teach exactly this and got 41 rows), periodic live harness runs on the production slot. | Total Anthropic spend ≈ $47. |
| 2026-09-14 | **Untrained Qwen3-8B measured** (TRAIN.md Phase 0 addendum): a working baseline (carry 0.80, new 0.905, 5/40 cut at 1500, 2× slower on long) — the smoke model is better on every axis but by a moderate margin; without thinking Qwen3-8B stops merging (carry 0.60). Fallback recommendation if the fine-tune is ever off: `qwen3:8b` + cap 1500, never `qwen3.5:9b`. The §0.4 gate would still have said "train" for Qwen3-8B (dropped facts 20 %, truncation 12 %), but with the smoke shipping it is moot. | Answers Daniel's question "is the smoke better than a pure qwen3:8b?" — yes, measured, on carry, new, long coverage, truncation and latency. |

## 11. Next phase proposal — "retention" (written 2026-09-14, not started)

**Where the remaining loss is.** On the pinned eval the shipped model keeps 93–97 % of earlier facts at each
step, and that compounds: carry 0.93–0.97 at step 1 falls to 0.80–0.85 by step 4. A fact dropped at step k is
absent from every later prompt (the prompt holds only the previous summary and 20 new messages), so it can
never be recovered — the "cascade" is arithmetic, not a bug. The only lever is per-step retention. SFT has
reached its ceiling here on both bases and at 94 and 243 rows: it shows the model good summaries but never
shows it *which* of its own habits loses facts. That is exactly what a preference signal expresses.

**Step 1 — fact-retention DPO (the one DPO case worth running; ~$5, 1 GPU-hour).**
Pairs with an identical prompt: chosen = the teacher's summary that keeps every fact; rejected = the shipped
model's own summary for the same prompt that dropped some (or padded). 46 such pairs already exist
(`base_prev` rows ↔ `prev_smoke.jsonl`); generating the teacher side for *every* step the smoke model has
produced (191 steps in `prev_smoke.jsonl`) adds ~150 more for ≈ $5 at batch rate. Train on top of the smoke
adapter (`train_lora.py --stage dpo`, lr sized for ~200 pairs per TRAIN.md), same pinned eval, two runs.
Success: per-step carry at k ≥ 3 up by more than the run spread, 0 over-limit, 0 fabrication, length ratio
≤ 0.75 (the same pairs also penalise padding: greedy showed the model *can* carry 1.0 when it writes to the
limit; the preference teaches it to carry without padding).

**Step 2 — a bigger ruler (≈ $3).** 10 eval chains resolve changes of ~0.05; step 1's expected gain is that
size. 20 more eval-only chains (no teacher summaries needed) bring resolution to ~0.03. Do this before step 1's
eval so the result is a result.

**Step 3 — base choice under the new recipe (GPU only).** Qwen3-8B carried more on multi-peer at 94 rows but
fabricated once and wrote longer; the retention DPO addresses both failure modes. Run step 1 on both bases;
ship the better. (Qwen3-8B also has the better untrained fallback behaviour — Phase 0 addendum.)

**Step 4 — production-informed categories (read-only, $0 generation).** Periodic `honcho_summary_harness.py`
runs against the live slot on synthetic chains catch drift; a read-only look at *where* real sessions lose facts
(never their content — PLAN §2 privacy rule) can add a synthetic category the mix lacks (e.g. very long
assistant turns, code/tables in messages, language switches).

**Not proposed:** more SFT rows (measured twice, both bases: no gain); a larger base (nothing >9B fits the
serving budget on node7 alongside the dialectic model); the combined dialectic+summary model (dropped).

Budget for steps 1–3: ≈ $8–10 Anthropic, ~4 GPU-hours, two evenings. Expected outcome: carried coverage at deep
steps from ~0.82 to ~0.9, and a length behaviour that stays put when the operator changes the cap.

## 12. Scaled workflow — $200–300 (written 2026-09-14, proposal)

Principle from §10: more rows of the *same* distribution bought nothing twice. Money is spent only on
(a) measuring failure classes the current eval cannot see, (b) on-policy preference data at scale,
(c) session shapes the current mix never produced — each behind a gate that says "measured gap" first.

| phase | what | cost | gate to the next phase |
|---|---|---|---|
| **A. Widen the ruler** | 40 eval-only chains (no teacher summaries) on dimensions the mix lacks: **language** (Swedish, mixed sv/en — the prompt never names a language; an English-only model may answer in English or degrade), **length** (200–300 messages → 10–15 short steps and 3–5 long steps, so *long-summary carry* is finally measured: today 3 rows), **4–5 peers**, **message shape** (code blocks, tables, URLs, very long assistant turns, one-word replies), **topic switches**. Run the shipped model on them, two runs, per-dimension breakdown. | ≈ $12 chains | Any dimension with carry < 0.8, new < 0.9, format flags, or cuts is a *measured* gap → phase C generates for it. Dimensions that pass get no training data. |
| **B. Retention DPO at scale** | On-policy pairs: run the student on every training chain (own previous), have the teacher write the ideal summary for *the student's exact prompt* at every step (not 30 %), filter with the scorer; pair chosen vs the student's own output where the student dropped facts or padded. ~600 pairs from the existing 57 chains; +~600 from phase C's chains later. Train on top of the smoke adapter; two evals on the phase-A ruler. Iterate once (student₂ → new pairs → student₃): "expert iteration". | ≈ $35 per round (teacher at 3 ¢/step), 2 rounds | Deep-step carry up by > the run spread with 0 over-limit / 0 fabrication → keep; else stop DPO. |
| **C. Targeted SFT for measured gaps** | 60–90 chains only in the dimensions phase A flagged (e.g. Swedish, 240-message, 5-peer), teacher summaries, best-of-3 teacher samples on dense deep steps (scorer picks; replaces "given up" steps), the 0.8 ceiling. Merge with the 94 smoke rows (not the 243: same lesson) and retrain; then phase B's second round on top. | ≈ $25 chains + ≈ $60 teacher (+50 % for best-of-3 on ~20 % of steps) | Per-dimension gain on the phase-A ruler; other dimensions unchanged within spread. |
| **D. Base under the final recipe** | Qwen3.5-9B vs Qwen3-8B with C+B. Optional: Qwen3-14B QLoRA feasibility (A6000 fits 4-bit; serving next to the dialectic model on node7 is the open question). | GPU only | Ship the better on the full ruler; 14B only if it fits node7 and beats 9B by more than the spread. |
| **E. Live loop** | Monthly harness run on the production slot with a rotating eval chain; read-only look at *where* real sessions lose facts (never content — §2) to propose new synthetic dimensions. | $0 | — |

**Totals:** A $12 + B $70 + C $85 ≈ **$170**, headroom to $250 for a second C round if phase A finds more than
two failing dimensions. GPU: ~15 hours over the phases. Calendar: two to three weeks part-time.

**Tooling to build first (no API cost; ~a day):** `gen_sessions.py --language sv|mixed`, `--peers 4-5`,
`--shape code|tables|long-turns|terse`, `LENGTHS` up to 300 (steps then 15 short / 5 long — `summary_chain`
already generic); `gen_summary_chosen.py --pairs-from <student.jsonl> --share 1.0` (teacher for every student
prompt) and `--best-of 3` (scorer picks); `build_summary_dataset.py` DPO output already pairs identical prompts;
`eval_summary.py` breakdown by the new dimensions (language, n_peers, shape) — chains carry the tags.
`train_lora.py --stage dpo` exists (dialectic recipe; lr sized per TRAIN.md carry-over).

**What this does not buy:** a guarantee. The honest expectation from §10 is that B moves deep-step carry from
~0.82 toward ~0.9 and that C fixes whatever A finds (a Swedish gap, if there is one, would be the single
largest win available). The ruler (A) is the part that makes the rest reportable to a manager.
| 2026-09-15 | §11 steps 1–2 run ($7.76): 30-chain ruler in place (smoke carry 0.889 on both runs — a stable baseline); 142 informative pairs built. First DPO at lr 5e-6 × 18 steps did not converge (loss 0.68, margins ~0.1) — mis-sized for batch 8. Eval unchanged within spread. **Not yet a verdict on DPO**; rerun at 1.5e-5, $0. | TRAIN.md "Retention phase". |
| 2026-09-15 | **§11 closed — negative.** A fully converged retention DPO (loss 0.03, margins 3–8) on 142 informative on-policy pairs leaves deep-step carry where the smoke had it (0.86–0.90 vs 0.889/0.889 on 30 chains) and adds a little length. The ordering was learned by suppressing rejected texts; chosen log-probs did not rise; sampled outputs did not change. Step 3 (Qwen3 base) cancelled. **`summary-smoke` remains the shipped model.** What is left in the plan is §12 phases A/C only: new *dimensions* (language, session length, message shape) — measurement first, generation only for a measured gap. | Total project spend ≈ $55 Anthropic. |
