#!/usr/bin/env python3
"""gen_summary_chosen.py — stage 3: the teacher writes the CHOSEN summaries (PLAN §2.2).

For every step (kind, k) of every chain the teacher answers the EXACT Honcho prompt (the same user
turn the student gets, from summary_prompt.py via summary_chain.build_step). Two variants:

  clean      (id c00003-s2)   previous summary = the teacher's own chosen summary of step k-1
  base-prev  (id c00003-s2b)  previous summary = the BASE model's summary of step k-1 from
                              --rejected (stage 2); the teacher repairs what the base dropped.
                              ~--base-prev-share of the steps with k >= 1 whose base k-1 is non-empty.
                              These rows share their prompt with the rejected row -> DPO pairs.

The system prompt adds the summariser's hard rules and, unless --blind, the ledger checklist (facts
due so far, changed values, forbidden distractors) — the student never sees it; it only makes the
target denser. build_summary_dataset.py filters with the scorer afterwards regardless.

Steps depend on the previous step, so:
  run     concurrent chains, sequential steps inside a chain (Anthropic sync or OpenRouter)
  submit  one Anthropic batch per WAVE (the next step of every chain); fetch writes it; repeat
          submit/fetch until `submit` reports nothing left (--waves in fetch loops for you).

  python3 gen_summary_chosen.py estimate --chains data/chains.jsonl --model opus [--rejected data/rejected.jsonl]
  python3 gen_summary_chosen.py run      --chains data/chains.jsonl --model opus --out data/chosen.jsonl --rejected data/rejected.jsonl
  python3 gen_summary_chosen.py submit   --chains ... --model opus --out data/chosen.jsonl --rejected data/rejected.jsonl
  python3 gen_summary_chosen.py fetch    [--waves]        # fetch, then submit+fetch the next waves until done
  python3 gen_summary_chosen.py status
  python3 gen_summary_chosen.py run --only c00000,c00001 ...      # pilot a few chains synchronously before a batch

Output rows: {id, chain, category, kind, k, variant, previous_from, previous_summary, output_words,
max_tokens (Honcho's, for the record), stop_reason, summary, words, attempts, teacher, score}. Failed rows
carry "__failed__" and are retried by a re-run; a row cut at the teacher's token budget or longer than
TARGET_RATIO x the limit is a failed generation and every later step built on it is marked stale. An
over-budget draft is retried as a cheap COMPRESS pass (rewrite the draft to the budget, low effort) rather
than a fresh write; MAX_ATTEMPTS in total, then the step is given up. `run` loops passes until nothing
changes, so one invocation completes the chains it can.
"""
import argparse
import hashlib
import json
import os

import llm_backend as be
import summary_chain as sc
import summary_prompt as sp
import summary_scoring as ss

KIND = "chosen"
# 2026-09-14: 0.9 -> 0.8. Honcho's word limit is 0.75 x max_tokens, i.e. 750 words in a 1000-token cap, and a word
# is ~1.3 tokens: the smoke model, trained on rows near 0.9, wrote 650-780 words at deep steps and 17/152 short
# steps on the 30 new chains were cut at max_tokens — always losing the newest chunk. Cut rows started at ~620
# words, so the safe ceiling is ~0.8 of the limit; the teacher is asked for 0.7 (it overshoots by 6-13 points).
TARGET_RATIO = 0.8    # train under the limit: models overshoot (PLAN §2.2), and 0.8 x limit ≈ 0.93 x max_tokens
PROMPT_RATIO = 0.7    # the numeric budget the teacher is given
MAX_ATTEMPTS = 3      # fresh write, then up to two COMPRESS passes of the over-budget draft; then given up (reported, not retried)
COMPRESS_EFFORT = "low"   # a rewrite-to-length needs no deep thinking; the pilot's failures were 0.91–0.95 of the limit
# The teacher's own output budget. NOT Honcho's max_tokens: on Claude 5 the thinking tokens count against
# max_tokens, and the 2026-09-13 smoke sent 1000 for short steps — 60 % of the k>=1 short summaries were cut
# mid-sentence, always losing the newest chunk (chronological order puts it last). The word limit lives in
# the prompt and is enforced by the stage-4 filter; the token budget only has to be big enough.
TEACHER_MAX_TOKENS = {"short": 8000, "long": 16000}

SYSTEM = """You are the reference summariser whose outputs train a small model to summarise conversations
for a memory system. The user message is the exact production prompt; answer it as the best possible
summariser would, and obey these hard rules on top of it:

1. NARRATIVE: one or more plain paragraphs in chronological order. No bullet points, no numbered
   lists, no headings, no markdown, no labels like "Summary:".
2. NOTHING BUT THE SUMMARY: no preamble, no meta-commentary, no closing remark, no mention of
   "the previous summary" or "the conversation above" as objects — just tell what happened.
3. LENGTH: your budget for this answer is {budget} words — deliberately below the prompt's hard
   limit of {limit}, because the model trained on your output overshoots. Every checklist value must
   fit inside the budget: cut connective prose, the assistant's advice and commentary, repeated
   attributions and adjectives first; cut a checklist value only when nothing else is left. If the
   conversation has little content, stay far below the budget — never pad. Count your words.
4. GROUNDED: every name, date, number, place and item comes from the previous summary or the new
   messages. Quote exact values (do not round, rename or paraphrase them). Invent nothing.
5. MERGE: every fact in the previous summary survives unless the new messages explicitly change it;
   then state the new value as current (mentioning the change is fine). Attribute statements to the
   right speaker when speakers differ.
6. Prefer explicit facts, preferences, questions and decisions over mood and generalities. The long
   variant may additionally describe emotional state and recurring themes, briefly, after the facts.
"""

COMPRESS_SYSTEM = """You are the reference summariser for a fine-tuning dataset. COMPRESS: the draft below is a
correct summary that is too long. Rewrite it to at most {budget} words (the production hard limit is
{limit}; the trained model overshoots, so the budget is deliberately lower). Keep every fact, value,
name, date, number, preference, question and decision — the checklist values verbatim. Remove connective
prose, the assistant's commentary and advice, repeated attributions, adjectives and restatements. Plain
chronological paragraphs, no lists, no headings, no preamble, nothing but the rewritten summary.
"""
# The physical limit (pilot, 2026-09-13): a dense 100-message chain at step 4 carries 43–59 ledger facts and
# compresses to ~0.94 of the limit, never lower, while every value is kept. Honcho's prompt itself says
# "drop lower-priority detail to stay within the limit", and the stage-4 filter accepts coverage >= 0.9,
# so the LAST attempt may drop up to a tenth of the checklist — the least consequential values.
COMPRESS_LAST_PASS = """
LAST PASS: the budget could not be met while keeping every value. You may now DROP up to {n_drop} checklist
values — choose the least consequential (the assistant's advice, minor quantities, incidental items),
never a changed value, a name, a date or a decision. Everything else stays. The budget is binding.
"""

CHECKLIST = """
CHECKLIST (for you only; never mention it): these values must be present in your summary if they are
still current — copy them exactly:
{facts}
{changes}
Never state these (they were NOT said): {distractors}
"""


def checklist(chain, kind, k):
    view = sc.chain_view(chain, kind)
    new, carry = ss.facts_due(view, k)
    hi = view["chunks"][k]["seqs"][1]
    facts = {f["id"]: f["text"] for f in chain["facts"]}
    ch = [f"- {facts.get(c['fact_id'])} was replaced by {facts.get(c['superseded_by'])} (give the latter as current)"
          for c in chain.get("changes", []) if c["seq"] <= hi]
    return CHECKLIST.format(facts="\n".join(f"- {t}" for t in carry + new) or "- (none)",
                            changes=("Changed values:\n" + "\n".join(ch)) if ch else "",
                            distractors="; ".join(d["text"] for d in chain.get("distractors", [])) or "(none)")


def over_budget(row) -> bool:
    return bool(row.get("summary")) and bool(row.get("output_words")) and row.get("words", 0) > TARGET_RATIO * row["output_words"]


def gave_up(row) -> bool:
    return be.failed(row) and row.get("attempts", 0) >= MAX_ATTEMPTS


def make_job(chain, kind, k, previous, variant, a):
    step = sc.build_step(chain, kind, k, previous, a.max_tokens_short, a.max_tokens_long)
    system = SYSTEM.format(budget=int(PROMPT_RATIO * step["output_words"]), limit=step["output_words"])
    if not a.blind:
        system += checklist(chain, kind, k)
    return {"custom_id": sc.step_id(chain["id"], kind, k, variant), "system": system, "user": step["prompt"],
            "max_tokens": TEACHER_MAX_TOKENS[kind], "_step": step}


def make_compress_job(chain, kind, k, previous, variant, a, draft, last=False):
    """Retry of an over-budget row: rewrite the teacher's own draft to the budget (cheap, converges).
    last=True (the MAX_ATTEMPTS-th attempt) may drop up to a tenth of the checklist values."""
    step = sc.build_step(chain, kind, k, previous, a.max_tokens_short, a.max_tokens_long)
    system = COMPRESS_SYSTEM.format(budget=int(PROMPT_RATIO * step["output_words"]), limit=step["output_words"])
    if last:
        new, carry = ss.facts_due(sc.chain_view(chain, kind), k)
        system += COMPRESS_LAST_PASS.format(n_drop=max(1, (len(new) + len(carry)) // 10))
    if not a.blind:
        system += checklist(chain, kind, k)
    return {"custom_id": sc.step_id(chain["id"], kind, k, variant), "system": system,
            "user": f"Draft ({ss.words(draft)} words; rewrite to at most {int(PROMPT_RATIO * step['output_words'])} words):\n\n{draft}",
            "max_tokens": TEACHER_MAX_TOKENS[kind], "effort": COMPRESS_EFFORT, "_step": step}


def job_for(chain, kind, k, previous, variant, a, have):
    """Fresh write, or a compress pass when the previous attempt was a complete but over-budget draft."""
    if have is not None and be.failed(have) and over_budget(have) and have.get("stop_reason") not in ("max_tokens", "length"):
        return make_compress_job(chain, kind, k, previous, variant, a, have["summary"], last=have.get("attempts", 0) + 1 >= MAX_ATTEMPTS)
    return make_job(chain, kind, k, previous, variant, a)


def _pick_base_prev(cid, kind, k, share):
    h = int(hashlib.sha1(f"{cid}-{kind}-{k}".encode()).hexdigest(), 16) % 1000
    return h < int(share * 1000)


def to_row(chain, kind, k, variant, previous_from, step, result, teacher, attempts=1):
    row = {"id": sc.step_id(chain["id"], kind, k, variant), "chain": chain["id"], "category": chain.get("category"),
           "kind": kind, "k": k, "variant": "base_prev" if variant else "clean", "previous_from": previous_from,
           "previous_summary": step["previous_summary"], "output_words": step["output_words"],
           "max_tokens": step["max_tokens"], "teacher": teacher, "attempts": attempts, "target_ratio": TARGET_RATIO}
    text = (result.get("text") or "").strip()
    row["stop_reason"] = result.get("stop_reason")
    if result.get("error") or not text:
        row.update(summary="", words=0, __failed__=f"__FAILED__: {result.get('error') or 'empty response'}")
        return row
    if row["stop_reason"] in ("max_tokens", "length"):
        # a cut summary must not be trained on and must not feed the next step's previous summary
        row.update(summary=text, words=ss.words(text), __failed__=f"__FAILED__: truncated at max_tokens ({ss.words(text)} words)")
        return row
    row.update(summary=text, words=ss.words(text), score=sc.score_step(chain, kind, k, text, step["output_words"]))
    if over_budget(row):
        # a length rule broken at generation time is a failed generation: retried (with dependants) up to MAX_ATTEMPTS
        row["__failed__"] = f"__FAILED__: over budget ({row['words']} words > {TARGET_RATIO} x {row['output_words']})"
    return row


def load_rejected(path):
    """{(chain, kind, k): summary} of good, non-empty base summaries."""
    out = {}
    for r in be.read_jsonl(path) if path else []:
        if not be.failed(r) and (r.get("summary") or "").strip():
            out[(r["chain"], r["kind"], r["k"])] = r["summary"].strip()
    return out


def wanted_steps(chain, rejected, share):
    """[(kind, k, variant)] every row this chain should end up with."""
    out = []
    for kind in sc.KINDS:
        for k in range(sc.n_steps(chain, kind)):
            out.append((kind, k, ""))
            if k >= 1 and (chain["id"], kind, k - 1) in rejected and _pick_base_prev(chain["id"], kind, k, share):
                out.append((kind, k, "b"))
    return out


def previous_for(chain, kind, k, variant, chosen, rejected):
    """(previous_text, previous_from) or (None, None) if the dependency is not ready."""
    if k == 0:
        return "", None
    if variant == "b":
        return rejected[(chain["id"], kind, k - 1)], f"rejected:{sc.step_id(chain['id'], kind, k - 1)}"
    pid = sc.step_id(chain["id"], kind, k - 1)
    prev = chosen.get(pid)
    if prev is None or be.failed(prev):
        return None, None
    return prev["summary"], f"chosen:{pid}"


def report(rows, path):
    good = [r for r in rows if not be.failed(r)]
    print(f"wrote {path}: {len(rows)} rows, {len(good)} good, {len(rows) - len(good)} failed "
          f"({sum(1 for r in rows if over_budget(r) and be.failed(r))} over budget, {sum(1 for r in rows if gave_up(r))} given up)")
    if good:
        ratios = sorted(r["words"] / r["output_words"] for r in good)
        print(f"  good rows limit_ratio median {ratios[len(ratios)//2]:.2f} max {ratios[-1]:.2f} (budget asked {PROMPT_RATIO}, accepted <= {TARGET_RATIO}); "
              f"{sum(1 for r in good if r.get('compressed_from_words'))} reached it through a compress pass")
    for kind in sc.KINDS:
        rs = [r for r in good if r["kind"] == kind]
        if rs:
            agg = ss.aggregate([r["score"] for r in rs])
            agg["base_prev_rows"] = sum(1 for r in rs if r["variant"] == "base_prev")
            print(f"  {kind}: {json.dumps(agg)}")
    if len(good) < len(rows):
        print("  failed rows keep a __failed__ marker; re-run the same command to retry them.")


# ------------------------------------------------------------------ commands
CUT_END = __import__("re").compile(r'[.!?"\u201d)\]]\s*$')


def invalidate_chain(chosen):
    """A step whose previous summary came from a failed (or invalidated) chosen row is stale: it was
    built on text that will be regenerated. Mark it failed so it is redone in dependency order.
    Rows written before stop_reason was recorded (the 2026-09-13 smoke) are failed when they end
    mid-sentence — the signature of the max_tokens cut."""
    for r in chosen.values():
        r.setdefault("attempts", 1)
        if r.get("target_ratio") != TARGET_RATIO:
            # attempts were spent against a different length target (2026-09-14: 0.9 -> 0.8 declared 12 steps given
            # up without one try at the new budget); give the row a fresh compress budget under the current target
            if be.failed(r) and over_budget(r) or (not be.failed(r) and over_budget(r)):
                r["attempts"] = min(r["attempts"], 1)
            r["target_ratio"] = TARGET_RATIO
        if "stop_reason" not in r and not be.failed(r) and r.get("summary") and not CUT_END.search(r["summary"]):
            r["__failed__"] = f"__FAILED__: legacy row cut mid-sentence ({r.get('words')} words)"
        elif not be.failed(r) and over_budget(r):
            r["__failed__"] = f"__FAILED__: over budget ({r['words']} words > {TARGET_RATIO} x {r['output_words']})"
    changed = True
    while changed:
        changed = False
        for r in chosen.values():
            if be.failed(r) or not (r.get("previous_from") or "").startswith("chosen:"):
                continue
            prev = chosen.get(r["previous_from"].split(":", 1)[1])
            if prev is None or be.failed(prev):
                r["__failed__"] = "__FAILED__: stale (previous summary was regenerated)"
                changed = True
    return chosen


def write_out(out, new_rows, chosen):
    """Merge this run's rows over the invalidated view of the file (not the raw disk rows: a row that
    is over budget on disk carries no marker, and llm_backend.merge_rows would let it beat its own
    compressed-but-still-failed retry forever — the wave-17 loop of 2026-09-13). A good row is kept
    over a new failure; an invalidated row is always replaced, even by a failure with attempts+1."""
    merged = dict(chosen)
    for r in new_rows:
        old = merged.get(r["id"])
        if old is not None and not be.failed(old) and be.failed(r):
            continue
        merged[r["id"]] = r
    rows = sorted(merged.values(), key=lambda r: r["id"])
    be.write_jsonl(out, rows)
    return rows


def _setup(a):
    chains = sc.load_chains(a.chains, set(a.only.split(",")) if getattr(a, "only", "") else None)
    rejected = load_rejected(a.rejected)
    out = a.out or os.path.join(os.path.dirname(a.chains), "chosen.jsonl")
    chosen = invalidate_chain({r["id"]: r for r in be.read_jsonl(out)})
    return chains, rejected, out, chosen


def _all_jobs_estimate(a, chains, rejected):
    jobs, tout = [], 0
    for c in chains:
        for kind, k, variant in wanted_steps(c, rejected, a.base_prev_share):
            prev = "x" * (600 if kind == "short" else 3000) * 5 if k else ""     # stand-in for a previous summary
            j = make_job(c, kind, k, prev, variant, a)
            jobs.append(j); tout += int(j["_step"]["output_words"] * TARGET_RATIO * 1.4)
    return jobs, tout


def cmd_estimate(a):
    spec = be.resolve_model(a.model)
    chains, rejected, _, _ = _setup(a)
    jobs, tout = _all_jobs_estimate(a, chains, rejected)
    per = tout // max(1, len(jobs))
    for batch in ((False, True) if spec.provider == "anthropic" else (False,)):
        usd, tin, to = be.estimate_usd(spec, jobs, out_tokens_per_job=per, batch=batch)
        print(f"{spec}  steps={len(jobs)} ({sum(1 for j in jobs if j['custom_id'].endswith('b'))} base-prev)  "
              f"{'batch' if batch else 'sync '}  in≈{tin/1e6:.2f}M out≈{to/1e6:.2f}M  ≈ ${usd:.2f}")
    if not rejected:
        print("  (no --rejected given or no usable base summaries: no base-prev variants)")
    return 0


def cmd_run(a):
    spec = be.resolve_model(a.model)
    chains, rejected, out, chosen = _setup(a)
    key = be.key_for(spec)
    todo = [c for c in chains if any(sc.step_id(c["id"], kd, k, v) not in chosen or be.failed(chosen[sc.step_id(c["id"], kd, k, v)])
                                     for kd, k, v in wanted_steps(c, rejected, a.base_prev_share))]
    jobs, tout = _all_jobs_estimate(a, todo, rejected)
    usd, _, _ = be.estimate_usd(spec, jobs, tout // max(1, len(jobs)))
    print(f"{spec}: {len(todo)} chains with missing steps ({len(jobs)} steps at most); estimate ≈ ${usd:.2f}"
          + (f" (cap ${a.max_usd:.2f})" if a.max_usd is not None else ""))
    if not todo:
        return 0
    import concurrent.futures
    import threading
    lock, spent, stop = threading.Lock(), [0.0], threading.Event()
    new_rows = []

    def work(c):
        local = {}
        rows = []
        for kind, k, variant in wanted_steps(c, rejected, a.base_prev_share):
            sid = sc.step_id(c["id"], kind, k, variant)
            have = chosen.get(sid)
            if have is not None and not be.failed(have):
                local[sid] = have
                continue
            if have is not None and gave_up(have):
                continue                                    # MAX_ATTEMPTS reached: reported, not retried
            prev, pfrom = previous_for(c, kind, k, variant, {**chosen, **local}, rejected)
            if prev is None or stop.is_set():
                continue                                    # previous step failed / cap: retried next run
            job = job_for(c, kind, k, prev, variant, a, have)
            r = be.complete_with_retry(spec, key, job, job.get("effort", a.effort))
            with lock:
                spent[0] += be.usage_usd(spec, r.get("usage") or {})
                if a.max_usd is not None and spent[0] > a.max_usd:
                    stop.set()
            row = to_row(c, kind, k, variant, pfrom, job["_step"], r, a.model, attempts=(have or {}).get("attempts", 0) + 1)
            if "effort" in job:
                row["compressed_from_words"] = ss.words(have["summary"])
            local[sid] = row
            rows.append(row)
        return rows

    # passes: a failed (over-budget) step blocks its chain within a pass; the next pass compresses it and
    # continues the chain, until a pass produces nothing new (all done or given up) or MAX_ATTEMPTS+1 passes
    for pass_no in range(1, MAX_ATTEMPTS + 2):
        pass_rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
            for i, rows in enumerate(ex.map(work, todo), 1):
                pass_rows.extend(rows)
                for r in rows:
                    tag = " (compressed)" if r.get("compressed_from_words") else ""
                    if be.failed(r):
                        print(f"  [pass {pass_no} {i}/{len(todo)}] {r['id']} FAIL {r['__failed__'][:70]}{tag}", flush=True)
                    else:
                        s = r["score"]
                        print(f"  [pass {pass_no} {i}/{len(todo)}] {r['id']:14} {r['words']:4}w/{r['output_words']} new={s.get('fact_coverage_new')} "
                              f"carry={s.get('fact_coverage_carry')} fab={s['fabrication']}"
                              f"{' bullets' if s['bullets'] else ''}{' meta' if s['meta'] else ''}{tag}", flush=True)
        new_rows.extend(pass_rows)
        chosen = invalidate_chain({r["id"]: r for r in write_out(out, new_rows, chosen)})
        if not pass_rows or stop.is_set():
            break
    merged = write_out(out, new_rows, chosen)
    report(merged, out)
    print(f"usage-based spend ≈ ${spent[0]:.3f}" + (" (cost cap reached; rerun to resume)" if stop.is_set() else ""))
    return 0


def _wave(a, chains, rejected, chosen):
    """Jobs for every wanted step whose dependency is satisfied and that is not yet good."""
    jobs, metas = [], []
    for c in chains:
        for kind, k, variant in wanted_steps(c, rejected, a.base_prev_share):
            sid = sc.step_id(c["id"], kind, k, variant)
            have = chosen.get(sid)
            if have is not None and (not be.failed(have) or gave_up(have)):
                continue
            prev, pfrom = previous_for(c, kind, k, variant, chosen, rejected)
            if prev is None:
                continue
            j = job_for(c, kind, k, prev, variant, a, have)
            jobs.append({k_: v for k_, v in j.items() if k_ != "_step"})
            metas.append({"id": sid, "chain": c["id"], "kind": kind, "k": k, "variant": variant, "previous_from": pfrom,
                          "attempts": (have or {}).get("attempts", 0) + 1, "compressed_from_words": ss.words(have["summary"]) if "effort" in j else None,
                          "previous_summary": j["_step"]["previous_summary"], "output_words": j["_step"]["output_words"],
                          "max_tokens": j["_step"]["max_tokens"]})
    return jobs, metas


def _remaining(chains, rejected, chosen, share):
    """(steps still to do, steps given up after MAX_ATTEMPTS)."""
    todo = gu = 0
    for c in chains:
        for kd, k, v in wanted_steps(c, rejected, share):
            have = chosen.get(sc.step_id(c["id"], kd, k, v))
            if have is not None and gave_up(have):
                gu += 1
            elif have is None or be.failed(have):
                todo += 1
    return todo, gu


def cmd_submit(a):
    spec = be.resolve_model(a.model)
    chains, rejected, out, chosen = _setup(a)
    jobs, metas = _wave(a, chains, rejected, chosen)
    left, gu = _remaining(chains, rejected, chosen, a.base_prev_share)
    if gu:
        print(f"{gu} steps given up after {MAX_ATTEMPTS} over-budget attempts (their later steps are not generated)")
    if not jobs:
        print(f"nothing submittable: {left} steps remain" + (" (their previous steps failed — fix/retry those)" if left else " — all done"))
        return 0
    usd, _, _ = be.estimate_usd(spec, jobs, int(sum(m["output_words"] for m in metas) / len(metas) * TARGET_RATIO * 1.4), batch=True)
    # the whole job, not just this wave: every remaining step at this wave's average, plus the compress passes the
    # smoke needed (~0.6 per k>=1 short step at ~half the price). 2026-09-14: the smoke's first wave read $0.53 and
    # the remaining waves cost $3.16 more — the per-wave number alone misled.
    per_step = usd / max(1, len(jobs))
    whole = left * per_step * 1.3
    print(f"{spec} batch wave: {len(jobs)} steps now ({sum(1 for j in jobs if 'effort' in j)} compress passes) ≈ ${usd:.2f}; "
          f"{left - len(jobs)} more steps wait for later waves — WHOLE JOB ≈ ${whole:.2f} over all waves incl. retries (reference only)")
    b = be.batch_submit(spec, jobs, effort=a.effort)
    path = be.write_manifest(KIND, {"batch_id": b["id"], "model": str(spec), "model_alias": a.model, "n": len(jobs), "out": out,
                                    "chains": os.path.abspath(a.chains), "rejected": os.path.abspath(a.rejected) if a.rejected else None,
                                    "base_prev_share": a.base_prev_share, "blind": a.blind, "effort": a.effort,
                                    "max_tokens_short": a.max_tokens_short, "max_tokens_long": a.max_tokens_long,
                                    "est_usd": round(usd, 3), "metas": metas})
    print(f"submitted batch {b['id']} ({len(jobs)} requests) -> manifest {path}")
    print("next: python3 gen_summary_chosen.py fetch --waves   (fetch, then submit+fetch the remaining waves)")
    return 0


def cmd_status(a):
    rc = 0
    for p in be.list_manifests(KIND):
        m = json.load(open(p))
        try:
            b = be.batch_status(m["batch_id"])
            st = b.get("processing_status")
            print(f"{p}: {m['model']} n={m['n']} status={st} {b.get('request_counts', {})}")
            rc = rc or (0 if st == "ended" else 1)
        except Exception as e:  # noqa: BLE001
            print(f"{p}: status check failed ({type(e).__name__}: {e})"); rc = 1
    return rc


SPENT = [0.0]     # running total across the waves of one `fetch --waves` invocation


def _fetch_one(path, poll, no_wait):
    m = json.load(open(path))
    print(f"manifest {path}: batch {m['batch_id']} {m['model']} n={m['n']} -> {m['out']}")
    if no_wait:
        st = be.batch_status(m["batch_id"]).get("processing_status")
        if st != "ended":
            print(f"still {st}; run fetch again later."); return None
    else:
        be.batch_wait(m["batch_id"], poll)
    chains = {c["id"]: c for c in sc.load_chains(m["chains"])}
    results = be.batch_results(m["batch_id"])
    spec = be.resolve_model(m.get("model_alias") or m["model"].split(":", 1)[1])
    rows = []
    for meta in m["metas"]:
        c = chains[meta["chain"]]
        step = {"previous_summary": meta["previous_summary"], "output_words": meta["output_words"], "max_tokens": meta["max_tokens"]}
        row = to_row(c, meta["kind"], meta["k"], meta["variant"], meta["previous_from"], step,
                     results.get(meta["id"], {"error": "no result for this id"}), m.get("model_alias", m["model"]),
                     attempts=meta.get("attempts", 1))
        if meta.get("compressed_from_words"):
            row["compressed_from_words"] = meta["compressed_from_words"]
        rows.append(row)
    spent = sum(be.usage_usd(spec, r.get("usage") or {}, batch=True) for r in results.values())
    SPENT[0] += spent
    merged = write_out(m["out"], rows, invalidate_chain({r["id"]: r for r in be.read_jsonl(m["out"])}))
    report(merged, m["out"])
    print(f"usage-based spend for this batch ≈ ${spent:.3f}; running total this fetch ≈ ${SPENT[0]:.3f}")
    return m


def cmd_fetch(a):
    path = be.newest_manifest(KIND, a.manifest)
    m = _fetch_one(path, a.poll_interval, a.no_wait)
    if m is None:
        return 1
    if not a.waves:
        return 0
    # loop: submit the next wave with the manifest's settings, wait, fetch — until nothing is left
    ns = argparse.Namespace(chains=m["chains"], rejected=m.get("rejected"), out=m["out"], model=m.get("model_alias", m["model"]),
                            effort=m.get("effort", "medium"), base_prev_share=m.get("base_prev_share", 0.3), blind=m.get("blind", False),
                            max_tokens_short=m.get("max_tokens_short", sp.MAX_TOKENS_SHORT_DEFAULT),
                            max_tokens_long=m.get("max_tokens_long", sp.MAX_TOKENS_LONG_DEFAULT))
    for wave in range(1, 20):
        before = len(be.list_manifests(KIND))
        cmd_submit(ns)
        if len(be.list_manifests(KIND)) == before:
            break
        print(f"-- wave {wave + 1}")
        if _fetch_one(be.newest_manifest(KIND), a.poll_interval, False) is None:
            return 1
    print(f"all waves done; usage-based spend across them ≈ ${SPENT[0]:.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--chains", required=True)
        p.add_argument("--rejected", default=None, help="stage-2 rows; enables the base-previous variant")
        p.add_argument("--out", default=None, help="default: chosen.jsonl next to --chains")
        p.add_argument("--model", default="opus", help=f"alias ({', '.join(sorted(be.MODELS))}), anthropic:<id> or openrouter:<vendor/model>")
        p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max", ""])
        p.add_argument("--base-prev-share", type=float, default=0.3, help="share of k>=1 steps that also get a base-previous variant")
        p.add_argument("--blind", action="store_true", help="no ledger checklist in the system prompt")
        p.add_argument("--max-tokens-short", type=int, default=sp.MAX_TOKENS_SHORT_DEFAULT, help="Honcho SUMMARY_MAX_TOKENS_SHORT")
        p.add_argument("--max-tokens-long", type=int, default=sp.MAX_TOKENS_LONG_DEFAULT, help="Honcho SUMMARY_MAX_TOKENS_LONG")
        p.add_argument("--only", default="", help="comma-separated chain ids (pilot a few chains before a full batch)")

    p = sub.add_parser("estimate"); common(p); p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("run", help="concurrent chains, sequential steps (Anthropic sync or OpenRouter)"); common(p)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-usd", type=float, default=None)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("submit", help="Anthropic Message Batch: one wave (the next ready step of every chain)"); common(p); p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("fetch"); p.add_argument("--manifest"); p.add_argument("--no-wait", action="store_true")
    p.add_argument("--poll-interval", type=int, default=60)
    p.add_argument("--waves", action="store_true", help="after fetching, keep submitting and fetching waves until every step is done")
    p.set_defaults(fn=cmd_fetch)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
