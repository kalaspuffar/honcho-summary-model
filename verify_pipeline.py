#!/usr/bin/env python3
"""verify_pipeline.py — $0 self-test. Must print GO before a commit.

1. every .py parses            2. one prompt builder / one scorer
3. parity: summary_prompt renders Honcho's prompt shape
4. summary_scoring unit checks 5. train_lora data prep on a summary-shaped row (fake tokenizer)
6. llm_backend estimate + mock OpenRouter round trip (skip with --quick)
7. summary_chain step logic (parity of chunking / output_words / ids)
8. $0 end-to-end: mock OpenRouter + mock Anthropic -> gen_sessions -> gen_summary_rejected ->
   gen_summary_chosen (run and submit/fetch) -> build_summary_dataset -> eval_summary (skip with --quick)
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
QUICK = "--quick" in sys.argv
FAILS = []


def ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  -- {detail}"))
    if not cond:
        FAILS.append(name)


print("== 1. every .py parses ==")
for f in sorted(os.listdir(HERE)):
    if f.endswith(".py"):
        try:
            compile(open(f).read(), f, "exec"); ok(f, True)
        except SyntaxError as e:
            ok(f, False, str(e))

print("== 2. single prompt builder / single scorer ==")
defs = {f: open(f).read() for f in os.listdir(HERE) if f.endswith(".py")}
for rx in ("BULLET", "META", "THINK_LEAK", "NARRATION", "ECHO"):
    where = [f for f, s in defs.items() if re.search(rf"^{rx}\s*=", s, re.M)]
    ok(f"{rx} defined only in summary_scoring.py", where == ["summary_scoring.py"], str(where))
where = [f for f, s in defs.items() if "You are a system that summarizes" in s and f != "verify_pipeline.py"]
ok("Honcho prompt text only in summary_prompt.py", where == ["summary_prompt.py"], str(where))
ok("summary_prompt.py names its Honcho commit", re.search(r"commit [0-9a-f]{40}", defs["summary_prompt.py"]) is not None)

print("== 3. parity ==")
import summary_prompt as sp  # noqa: E402
msgs = [{"peer_name": "Anna", "content": "I moved the deadline to April 22."},
        {"peer_name": "assistant", "content": "Noted."}]
short = sp.build_messages("short", msgs, None, 300)
ok("one user turn, no system, no tools", len(short) == 1 and short[0]["role"] == "user")
c = short[0]["content"]
ok("short prompt shape", c.startswith("You are a system that summarizes") and "Hard limit: 300 words maximum" in c
   and "<conversation>\nAnna: I moved the deadline to April 22.\nassistant: Noted.\n</conversation>" in c
   and f"<previous_summary>\n{sp.NO_PREVIOUS_SUMMARY}\n</previous_summary>" in c)
lg = sp.build_messages("long", msgs, "Earlier: Anna set April 25.", 3000)[0]["content"]
ok("long prompt shape", lg.startswith("You are a system that creates thorough") and "<previous_summary>\nEarlier: Anna set April 25.\n</previous_summary>" in lg)
ok("output_words rules", sp.output_words_short(600) == 450 and sp.output_words_short(5000) == 750 and sp.output_words_long() == 3000)

print("== 4. scoring ==")
import summary_scoring as ss  # noqa: E402
chain = {"id": "c1", "facts": [{"id": "f1", "text": "April 25", "first_seq": 3, "kind": "date"},
                                {"id": "f2", "text": "Boulder Gran Fondo", "first_seq": 5, "kind": "event"},
                                {"id": "f3", "text": "April 22", "first_seq": 25, "kind": "date"}],
         "changes": [{"fact_id": "f1", "superseded_by": "f3", "seq": 25}],
         "distractors": [{"text": "Thermo Fisher"}],
         "chunks": [{"k": 0, "seqs": [1, 20]}, {"k": 1, "seqs": [21, 40]}]}
r0 = ss.score_summary(chain, 0, "Anna set the deadline to April 25 and signed up for the Boulder Gran Fondo.", 300)
ok("chunk 0: both new facts covered, nothing carried", r0["fact_coverage_new"] == 1.0 and r0["fact_coverage_carry"] is None and not r0["fabrication"])
r1 = ss.score_summary(chain, 1, "Anna moved the deadline from April 25 to April 22. She still plans the Boulder Gran Fondo.", 300)
ok("chunk 1: new value present, carried fact kept, superseded value excluded from due lists",
   r1["fact_coverage_new"] == 1.0 and r1["fact_coverage_carry"] == 1.0 and r1["latest_state"] == 1.0 and r1["n_carry"] == 1)
r1b = ss.score_summary(chain, 1, "Anna's deadline is April 25.", 300)
ok("chunk 1: stale value only -> latest_state 0, carry dropped", r1b["latest_state"] == 0.0 and r1b["fact_coverage_carry"] == 0.0)
ok("fabrication: distractor asserted", ss.score_summary(chain, 0, "Anna uses a Thermo Fisher centrifuge.", 300)["fabrication"])
ok("over limit", ss.score_summary(chain, 0, "word " * 301, 300)["over_limit"] and not ss.score_summary(chain, 0, "word " * 300, 300)["over_limit"])
ok("bullets flagged", ss.score_summary(chain, 0, "- April 25\n- Gran Fondo", 300)["bullets"])
ok("meta flagged", ss.score_summary(chain, 0, "Here is a summary of the conversation: Anna set April 25.", 300)["meta"])
ok("think leak flagged", ss.score_summary(chain, 0, "Thinking Process:\n1. read\n</think>\nAnna set April 25.", 300)["think_leak"])
ok("narration flagged and zeroes coverage", ss.score_summary(chain, 0, "I'll summarize the conversation now.", 300)["narration"]
   and ss.score_summary(chain, 0, "I'll summarize the conversation now.", 300)["fact_coverage_new"] == 0.0)
ok("empty", ss.score_summary(chain, 0, "", 300)["empty"])
ok("echo flagged", ss.score_summary(chain, 0, "There is no previous summary -- the messages are the beginning.\n\n<conversation>\nAnna: hi", 300)["echo"]
   and not r0["echo"])
pc = {"peers": [{"name": "Fenna"}, {"name": "Kestrel"}], "facts": [], "changes": [], "distractors": [{"text": "Fenna's friend Vera"}, {"text": "the city of Ashvale"}, {"text": "Batch nine"}],
      "chunks": [{"k": 0, "seqs": [1, 20]}]}
ok("matcher: peer name + possessive fragment are not evidence", not ss.score_summary(pc, 0, "Fenna tells Kestrel the emptiness feels odd.", 300)["fabrication"])
ok("matcher: stopword + one anchor is not evidence", not ss.has_fact("They left the city at dawn.", "the city of Ashvale"))
ok("matcher: distant co-occurrence is not evidence", not ss.has_fact("Beatriz names the batch Thistle. " + "word " * 30 + "Nine bottles were left.", "Batch nine"))
ok("matcher: adjacent paraphrase counts", ss.has_fact("They agreed that Viktor will run the demo on Friday.", "let Viktor run the demo")
   and ss.has_fact("Vera, a friend of Fenna, joins.", "Fenna's friend Vera", ignore={"Fenna"}))
ok("matcher: the number is mandatory", ss.has_fact("she bought 4.2 kg of pilsner malt", "4.2 kilos of pilsner malt") and not ss.has_fact("she bought pilsner malt", "4.2 kilos of pilsner malt"))
ok("matcher: thousands separators and curly quotes normalise", ss.has_fact("gross takings were £14380", "£14,380") and ss.has_fact("Fenna’s friend Vera came", "Fenna's friend Vera"))
nm = {"peers": [{"name": "Priya"}], "facts": [{"id": "f1", "text": "Nine slats per side", "first_seq": 2, "kind": "number"}], "changes": [],
      "distractors": [{"text": "seven slats per side"}, {"text": "Copenhagen office pilot"}], "chunks": [{"k": 0, "seqs": [1, 20]}]}
rn = ss.score_summary(nm, 0, "Priya plans 9 slats per side; the pilot runs from the Lisbon office.", 300)
ok("matcher: number words normalise (nine == 9) and a near-miss distractor is not a fabrication", rn["fact_coverage_new"] == 1.0 and not rn["fabrication"])
ok("matcher: the near-miss IS a fabrication when actually asserted", ss.score_summary(nm, 0, "Priya plans seven slats per side.", 300)["fabrication"]
   and ss.score_summary(nm, 0, "the Copenhagen office runs the pilot", 300)["fabrication"])
fp = {"peers": [], "facts": [], "changes": [], "distractors": [{"text": "fifteen prints"}], "chunks": [{"k": 0, "seqs": [1, 20]}]}
ok("matcher: strict window — '12 prints ... 15 inches' is not 'fifteen prints', 'fifteen prints on the wall' is",
   not ss.score_summary(fp, 0, "mounting 12 potential prints (reduced to 8 at 10 by 15 inches) on boards", 300)["fabrication"]
   and ss.score_summary(fp, 0, "she hung fifteen prints on the wall", 300)["fabrication"])
ok("matcher: a bare name fact matches by substring", ss.has_fact("Nils arrived late", "Nils") and not ss.has_fact("Nora arrived late", "Nils"))
agg = ss.aggregate([r0, r1, r1b])
ok("aggregate", agg["n"] == 3 and agg["over_limit_rows"] == 0 and agg["median_fact_coverage_new"] == 1.0)

print("== 5. train prep ==")
import train_lora as td  # noqa: E402


class FakeTok:
    pad_token_id = 0
    chat_template = "... {%- if enable_thinking is defined and enable_thinking is false %} ..."

    def _turns(self, msgs, tools):
        return " ".join(f"<|{m['role']}|> {m.get('content') or ''} <|end|>" for m in msgs)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, tools=None, enable_thinking=None, **kw):
        if add_generation_prompt:
            return self._turns(msgs, tools) + " <|assistant|>" + (" <think> \n\n </think> \n\n" if enable_thinking is False else " <think> \n")
        return self._turns(msgs[:-1], tools) + " <|assistant|> <think> \n\n </think> \n\n " + (msgs[-1].get("content") or "") + " <|end|>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [abs(hash(w)) % 30000 + 1 for w in text.split()]}

    def decode(self, ids):
        return f"<{len(ids)} tokens>"


row = {"messages": [{"role": "user", "content": short[0]["content"]}, {"role": "assistant", "content": "Anna moved the deadline to April 22."}]}
samples, dropped, kinds = td.prepare_sft([row], FakeTok(), 100000)
ok("summary row -> one answer turn, no tool turns", kinds == {"answer": 1, "tool": 0} and dropped == 0)
served = FakeTok()(FakeTok().apply_chat_template(row["messages"][:-1], add_generation_prompt=True))["input_ids"]
want = FakeTok()("\n</think>\n\n Anna moved the deadline to April 22. <|end|>")["input_ids"]
s0 = samples[0]
ok("served prompt masked, completion closes the think block", s0["labels"][:len(served)] == [-100] * len(served) and s0["input_ids"][len(served):] == want)
ok("over-long rows dropped, never truncated", td.prepare_sft([row], FakeTok(), 10)[1] == 1)
ok("no hardcoded split", "samples[:7]" not in defs["train_lora.py"])
ok("resolve_base rejects the Ollama tag", (lambda: (td.resolve_base("/data/smoke/qwen35-9b-text") == "/data/smoke/qwen35-9b-text"))())

if not QUICK:
    print("== 6. backend ==")
    import llm_backend as be  # noqa: E402
    spec = be.resolve_model("opus")
    usd, tin, tout = be.estimate_usd(spec, [{"system": "", "user": "x" * 4000}], 1000)
    ok("estimate_usd positive", usd > 0 and tin > 0 and tout == 1000)
    port = 18400 + int(time.time()) % 100
    srv = subprocess.Popen([sys.executable, "mock_or_server.py", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=0.5); break
            except Exception:
                time.sleep(0.1)
        os.environ["OPENROUTER_BASE"] = f"http://127.0.0.1:{port}/v1"
        os.environ["OPENROUTER_API_KEY"] = "mock"
        body = json.dumps({"model": "x", "messages": short, "max_tokens": 50}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        ok("mock OpenRouter answers the summary prompt shape", bool(data.get("choices")))
    finally:
        srv.terminate()

print("== 7. summary_chain ==")
import summary_chain as sc  # noqa: E402
def _chain(n):
    return sc.with_chunks({"id": "cX", "messages": [{"seq": i, "peer": "A" if i % 2 else "B", "text": f"m{i} " * 30} for i in range(1, n + 1)],
                           "facts": [], "changes": [], "distractors": [], "peers": [{"name": "A"}, {"name": "B"}]})
ok("60 msgs -> 3 short + 1 long", sc.steps(_chain(60)) == [("short", 0), ("short", 1), ("short", 2), ("long", 0)])
ok("80 msgs -> 4 short + 1 long", sc.n_steps(_chain(80), "short") == 4 and sc.n_steps(_chain(80), "long") == 1)
ok("120 msgs -> 6 short + 2 long, long 1 covers 61-120", sc.n_steps(_chain(120), "long") == 2 and sc.chain_view(_chain(120), "long")["chunks"][1]["seqs"] == [61, 120])
st0 = sc.build_step(_chain(60), "short", 0, "")
st1 = sc.build_step(_chain(60), "short", 1, "prev " * 400)
ok("short chunk 0 limit from message tokens only", st0["output_words"] == sp.output_words_short(sum(be_te(m["text"]) for m in _chain(60)["messages"][:20])) if (be_te := __import__("llm_backend").tokens_estimate) else False)
ok("previous summary tokens raise the short limit toward the cap", st1["output_words"] > st0["output_words"] and st1["output_words"] <= 750)
ok("short step sends max_tokens 1000, long 4000 and 3000 words", st0["max_tokens"] == 1000 and sc.build_step(_chain(60), "long", 0, "")["max_tokens"] == 4000
   and sc.build_step(_chain(60), "long", 0, "")["output_words"] == 3000)
ok("operator caps flow through", sc.build_step(_chain(60), "long", 0, "", max_tokens_long=2667)["output_words"] == 2000)
ok("step prompt is the parity prompt", st1["prompt"] == sp.build_messages("short", sc.step_messages(_chain(60), "short", 1), "prev " * 400, st1["output_words"])[0]["content"])
ok("no previous -> NO_PREVIOUS_SUMMARY sentence", sp.NO_PREVIOUS_SUMMARY in st0["prompt"] and st0["previous_summary"] == "")
ok("step ids round-trip", sc.parse_step_id("c00003-s2b") == ("c00003", "short", 2, "b") and sc.parse_step_id("c00003-l1") == ("c00003", "long", 1, "")
   and sc.step_id("c00003", "long", 1) == "c00003-l1")
walked = sc.walk(_chain(60), "short", lambda step: {"summary": "", "reasoning": "answer in think"}, answer_from_reasoning=False)
ok("walk: empty content chains an empty previous and flags answered_in_thinking", len(walked) == 3 and all(r["answered_in_thinking"] for r in walked)
   and sp.NO_PREVIOUS_SUMMARY in sc.build_step(_chain(60), "short", 1, walked[0]["summary"])["prompt"])
walked2 = sc.walk(_chain(60), "short", lambda step: {"summary": "", "reasoning": "answer in think"}, answer_from_reasoning=True)
ok("walk --answer-from-reasoning scores and chains the reasoning text", walked2[1]["previous_summary"] == "answer in think" and walked2[1]["words"] == 3)
import mock_or_server as mo  # noqa: E402
import gen_sessions as gs  # noqa: E402
meta = gs.plan(1, 0, 7)[0]
mc = mo.mock_chain(80, 3, three=True)
row, why = gs.validate(mc, dict(meta, category="dense-facts", shape="three-people"))
ok("gen_sessions.validate accepts the mock chain and cuts 80 -> 4 short chunks + 1 long", row is not None and len(row["chunks"]) == 4 and len(row["chunks_long"]) == 1, why)
bad = dict(mc, facts=mc["facts"] + [{"id": "zz", "text": "never said", "first_seq": 3, "kind": "x"}], distractors=[{"text": "Lake Vesna"}, {"text": "Lake Orrin"}])
row2, _ = gs.validate(bad, dict(meta, category="dense-facts", shape="three-people"))
ok("validate drops non-verbatim facts and distractors that appear in the text", row2 is not None and not any(f["text"] == "never said" for f in row2["facts"])
   and [d["text"] for d in row2["distractors"]] == ["Lake Orrin"])
ok("validate rejects < 60 messages", gs.validate(dict(mc, messages=mc["messages"][:59]), meta)[0] is None)
ok("category mix covers all six in 20 rows", {m["category"] for m in gs.plan(20, 0, 7)} == set(gs.CATEGORIES))
import gen_summary_chosen as gc  # noqa: E402
import llm_backend as be  # noqa: E402,F811
legacy = {"c1-s0": {"id": "c1-s0", "summary": "Complete sentence.", "words": 2, "previous_from": None},
          "c1-s1": {"id": "c1-s1", "summary": "Cut mid-sentence and the new chunk is", "words": 7, "previous_from": "chosen:c1-s0"},
          "c1-s2": {"id": "c1-s2", "summary": "Fine on its own.", "words": 4, "previous_from": "chosen:c1-s1"},
          "c1-s2b": {"id": "c1-s2b", "summary": "Fine too.", "words": 2, "previous_from": "rejected:c1-s1"},
          "c2-s0": {"id": "c2-s0", "summary": "Start.", "words": 1, "previous_from": None, "stop_reason": "end_turn"},
          "c2-s1": {"id": "c2-s1", "summary": 'She said "done."', "words": 3, "previous_from": "chosen:c2-s0", "stop_reason": "end_turn"}}
inv = gc.invalidate_chain(dict(legacy))
ok("chosen: legacy cut row fails, its dependants go stale, base-prev and complete rows survive",
   be.failed(inv["c1-s1"]) and "cut" in inv["c1-s1"]["__failed__"] and be.failed(inv["c1-s2"]) and "stale" in inv["c1-s2"]["__failed__"]
   and not be.failed(inv["c1-s2b"]) and not be.failed(inv["c1-s0"]) and not be.failed(inv["c2-s1"]))
ob = {"c3-s0": {"id": "c3-s0", "summary": "Long enough. " * 100, "words": 200, "output_words": 200, "previous_from": None, "stop_reason": "end_turn"},
      "c3-s1": {"id": "c3-s1", "summary": "Fine.", "words": 1, "output_words": 200, "previous_from": "chosen:c3-s0", "stop_reason": "end_turn"},
      "c3-s2": {"id": "c3-s2", "summary": "Fine.", "words": 1, "output_words": 200, "previous_from": "chosen:c3-s1", "stop_reason": "end_turn", "attempts": 3,
                "__failed__": "__FAILED__: over budget"}}
inv2 = gc.invalidate_chain(dict(ob))
ok("chosen: over-budget row fails and its dependant goes stale; attempts default to 1",
   be.failed(inv2["c3-s0"]) and "over budget" in inv2["c3-s0"]["__failed__"] and be.failed(inv2["c3-s1"]) and inv2["c3-s1"]["attempts"] == 1)
ok("chosen: a row at MAX_ATTEMPTS is given up, a first failure is not", gc.gave_up(inv2["c3-s2"]) and not gc.gave_up(inv2["c3-s0"]))
ok("chosen: --only restricts the chains", [c["id"] for c in sc.load_chains(chains if False else "data/chains.jsonl", {"c00000"})] == ["c00000"] if os.path.exists("data/chains.jsonl") else True)
ok("scorer: 1/15 is not the number 15", not ss.has_fact("whether 1/15 sec is too slow for her prints", "fifteen prints", strict=True)
   and ss.has_fact("she chose 1/15 sec", "1/15 sec"))
_a = argparse.Namespace(max_tokens_short=1000, max_tokens_long=4000, blind=False) if (argparse := __import__("argparse")) else None
_ch = _chain(60); _ch["facts"] = [{"id": "f1", "text": "m3", "first_seq": 3, "kind": "x"}]
_over = {"id": "cX-s1", "summary": "word " * 900, "words": 900, "output_words": 750, "stop_reason": "end_turn", "__failed__": "__FAILED__: over budget"}
_cj = gc.job_for(_ch, "short", 1, "prev text", "", _a, _over)
ok("chosen: an over-budget draft is retried as a low-effort COMPRESS pass carrying the draft and checklist",
   "COMPRESS:" in _cj["system"] and _cj.get("effort") == "low" and "word word" in _cj["user"] and f"at most {int(gc.PROMPT_RATIO * _cj['_step']['output_words'])} words" in _cj["system"] and "- m3" in _cj["system"])
ok("chosen: a truncated or missing row gets a fresh write", "effort" not in gc.job_for(_ch, "short", 1, "prev", "", _a, None)
   and "effort" not in gc.job_for(_ch, "short", 1, "prev", "", _a, dict(_over, stop_reason="max_tokens")))
ok("chosen: teacher budget is its own, not Honcho's max_tokens", gc.TEACHER_MAX_TOKENS["short"] >= 4000 and gc.TEACHER_MAX_TOKENS["long"] >= 8000)
ok("no real-deployment seeds: no name list in gen_sessions", not re.search(r"^NAMES\s*=", defs["gen_sessions.py"], re.M))

if not QUICK:
    print("== 8. end-to-end on mocks ($0) ==")
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="summary-verify-")
    p_or, p_an = 18600 + int(time.time()) % 100, 18700 + int(time.time()) % 100
    srv_or = subprocess.Popen([sys.executable, "mock_or_server.py", str(p_or)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    srv_an = subprocess.Popen([sys.executable, "mock_anth_batch.py", str(p_an)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env = dict(os.environ, OPENROUTER_BASE=f"http://127.0.0.1:{p_or}/v1", OPENROUTER_API_KEY="mock",
               ANTHROPIC_API_BASE=f"http://127.0.0.1:{p_an}", ANTHROPIC_API_KEY="mock", DIALECTIC_RESULTS_DIR=os.path.join(tmp, "results"))

    def sh(*args, expect=0):
        r = subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True, timeout=600)
        if expect is not None and r.returncode != expect:
            print("      $", " ".join(args)); print("      " + (r.stdout + r.stderr)[-1500:].replace("\n", "\n      "))
        return r
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{p_or}/v1/models", timeout=0.5); break
            except Exception:
                time.sleep(0.1)
        chains = os.path.join(tmp, "chains.jsonl")
        r = sh("gen_sessions.py", "run", "--n", "4", "--model", "deepseek", "--out", chains, "--concurrency", "2")
        rows = [json.loads(l) for l in open(chains)] if os.path.exists(chains) else []
        ok("gen_sessions run: 4 good chains", r.returncode == 0 and len(rows) == 4 and not any(r_.get("__failed__") for r_ in rows), r.stdout[-300:])
        r = sh("gen_sessions.py", "run", "--n", "4", "--model", "deepseek", "--out", chains)
        ok("gen_sessions run is resume-safe (nothing to do)", "0 chains to generate" in r.stdout, r.stdout[-200:])
        r = sh("gen_sessions.py", "estimate", "--n", "30", "--model", "opus")
        ok("gen_sessions estimate prints sync and batch", "batch" in r.stdout and "sync" in r.stdout and "$" in r.stdout)
        rej = os.path.join(tmp, "rejected.jsonl")
        r = sh("gen_summary_rejected.py", "--chains", chains, "--out", rej, "--base", f"http://127.0.0.1:{p_or}/v1", "--model", "mock", "--concurrency", "2")
        rrows = [json.loads(l) for l in open(rej)] if os.path.exists(rej) else []
        n_steps = sum(len(sc.steps(sc.with_chunks(c))) for c in rows)
        ok(f"gen_summary_rejected: one scored row per step ({n_steps})", r.returncode == 0 and len(rrows) == n_steps and all("score" in x for x in rrows), r.stdout[-300:])
        ok("rejected rows chain their own previous summary", any(x["k"] >= 1 and x["previous_summary"] for x in rrows))
        ok("mock summaries cover the ledger", all((x["score"].get("fact_coverage_new") or 1.0) >= 0.99 for x in rrows))
        r = sh("gen_summary_rejected.py", "--chains", chains, "--out", rej, "--base", f"http://127.0.0.1:{p_or}/v1", "--model", "mock")
        ok("gen_summary_rejected is resume-safe", "0 chain-kinds" in r.stdout, r.stdout[-200:])
        rej_t = os.path.join(tmp, "rejected_thinker.jsonl")
        r = sh("gen_summary_rejected.py", "--chains", chains, "--out", rej_t, "--base", f"http://127.0.0.1:{p_or}/v1", "--model", "mock-thinker", "--only", rows[0]["id"], "--kind", "short")
        trows = [json.loads(l) for l in open(rej_t)]
        ok("answered-inside-thinking is recorded (empty content, reasoning_chars > 0)", all(x["answered_in_thinking"] and x["words"] == 0 and x["reasoning_chars"] > 0 for x in trows), str(trows[:1])[:200])
        cho = os.path.join(tmp, "chosen.jsonl")
        r = sh("gen_summary_chosen.py", "estimate", "--chains", chains, "--model", "opus", "--rejected", rej)
        ok("gen_summary_chosen estimate", r.returncode == 0 and "base-prev" in r.stdout and "$" in r.stdout, r.stdout[-300:])
        r = sh("gen_summary_chosen.py", "run", "--chains", chains, "--model", "deepseek", "--out", cho, "--rejected", rej, "--base-prev-share", "0.5", "--concurrency", "2")
        crows = [json.loads(l) for l in open(cho)] if os.path.exists(cho) else []
        clean = [x for x in crows if x["variant"] == "clean"]
        bp = [x for x in crows if x["variant"] == "base_prev"]
        ok(f"gen_summary_chosen run: every step clean ({len(clean)}/{n_steps}) plus base-prev variants ({len(bp)})",
           r.returncode == 0 and len(clean) == n_steps and len(bp) > 0 and not any(x.get("__failed__") for x in crows), r.stdout[-400:])
        ok("base-prev variant's previous summary is the rejected k-1 output",
           all(x["previous_summary"] == next(y["summary"] for y in rrows if y["id"] == x["previous_from"].split(":")[1]) for x in bp))
        ok("clean variant's previous summary is the chosen k-1 output",
           all(x["previous_summary"] == next(y["summary"] for y in clean if y["id"] == x["previous_from"].split(":")[1]) for x in clean if x["k"] >= 1))
        cho2 = os.path.join(tmp, "chosen_batch.jsonl")
        r = sh("gen_summary_chosen.py", "submit", "--chains", chains, "--model", "opus", "--out", cho2, "--rejected", rej)
        ok("chosen submit: first wave = k=0 steps + ready base-prev", r.returncode == 0 and "wave" in r.stdout and "manifest" in r.stdout, r.stdout[-300:])
        r = sh("gen_summary_chosen.py", "fetch", "--waves", "--poll-interval", "0")
        crows2 = [json.loads(l) for l in open(cho2)] if os.path.exists(cho2) else []
        ok("chosen fetch --waves completes every step through the batch mock", r.returncode == 0 and len([x for x in crows2 if x["variant"] == "clean"]) == n_steps
           and "all done" in r.stdout, r.stdout[-400:])
        r = sh("gen_summary_chosen.py", "status")
        ok("chosen status lists ended batches", r.returncode == 0 and "ended" in r.stdout, r.stdout[-200:])
        ds = os.path.join(tmp, "dataset")
        r = sh("build_summary_dataset.py", "--chains", chains, "--chosen", cho, "--rejected", rej, "--out", ds, "--eval-frac", "0.25")
        tr = [json.loads(l) for l in open(ds + "_train.sft.jsonl")]; ev = [json.loads(l) for l in open(ds + "_eval.sft.jsonl")]
        dp = [json.loads(l) for l in open(ds + "_train.dpo.jsonl")] + [json.loads(l) for l in open(ds + "_eval.dpo.jsonl")]
        ok("build_summary_dataset: SFT rows written, eval chains disjoint", r.returncode == 0 and tr and ev and not ({x["chain"] for x in tr} & {x["chain"] for x in ev}), r.stdout[-400:])
        ok("SFT row = exact Honcho prompt + chosen, no system turn", all(len(x["messages"]) == 2 and x["messages"][0]["role"] == "user"
           and "<previous_summary>" in x["messages"][0]["content"] and x["messages"][1]["role"] == "assistant" for x in tr))
        ok("DPO pairs exist and share the prompt with the rejected side (k=0 or base-prev)", dp and all(x["k"] == 0 or x["id"].endswith("b") for x in dp), str(len(dp)))
        ok("kept rows are under 0.9x the limit", all(len(x["messages"][1]["content"].split()) <= 0.9 * int(re.search(r"Hard limit: (\d+)", x["messages"][0]["content"]).group(1)) for x in tr + ev))
        ev_out = os.path.join(tmp, "eval_mock.jsonl")
        r = sh("eval_summary.py", "--chains", chains, "--ids-from", ds + "_eval.sft.jsonl", "--model", "mock", "--base", f"http://127.0.0.1:{p_or}/v1", "--out", ev_out)
        summ = json.load(open(os.path.join(tmp, "eval_mock.summary.json"))) if os.path.exists(os.path.join(tmp, "eval_mock.summary.json")) else {}
        ok("eval_summary run writes per-kind aggregates over the eval chains only", r.returncode == 0 and summ.get("short", {}).get("n") == sum(sc.n_steps(sc.with_chunks(c), "short") for c in rows if c["id"] in {x["chain"] for x in ev})
           and summ.get("long") is not None, r.stdout[-400:] + r.stderr[-400:])
        ev_b = os.path.join(tmp, "eval_bullety.jsonl")
        r = sh("eval_summary.py", "--chains", chains, "--ids-from", ds + "_eval.sft.jsonl", "--model", "mock-bullety", "--base", f"http://127.0.0.1:{p_or}/v1", "--out", ev_b, "--kind", "short")
        r = sh("eval_summary.py", "compare", ev_out, ev_b)
        ok("eval_summary compare shows the bullet rows of the bullety model", r.returncode == 0 and "bullet_rows" in r.stdout and json.load(open(os.path.join(tmp, "eval_bullety.summary.json")))["short"]["bullet_rows"] > 0, r.stdout[-400:])
        r = sh("eval_summary.py", "rescore", ev_out, "--chains", chains)
        ok("eval_summary rescore", r.returncode == 0 and json.load(open(os.path.join(tmp, "eval_mock.summary.json"))).get("rescored") is True, r.stdout[-200:])
        # the trainer's data prep accepts the built rows
        samples, dropped, kinds = td.prepare_sft(tr, FakeTok(), 100000)
        ok("train_lora.prepare_sft on the built SFT rows", dropped == 0 and kinds == {"answer": len(tr), "tool": 0})
        r = sh("train_lora.py", "--stage", "check", "--model", "/nonexistent", "--data", ds + "_train.sft.jsonl", expect=None)
        ok("train_lora --stage check runs on the dataset shape (fails only on the missing tokenizer)", "not implemented" not in (r.stdout + r.stderr))
    finally:
        srv_or.terminate(); srv_an.terminate()
        shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILS:
    print(f"NO-GO — {len(FAILS)} failed: {FAILS}"); sys.exit(1)
print("GO — all checks passed.")
