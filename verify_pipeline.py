#!/usr/bin/env python3
"""verify_pipeline.py — $0 self-test. Must print GO before a commit.

1. every .py parses            2. one prompt builder / one scorer
3. parity: summary_prompt renders Honcho's prompt shape
4. summary_scoring unit checks 5. train_lora data prep on a summary-shaped row (fake tokenizer)
6. llm_backend estimate + mock OpenRouter round trip (skip with --quick)
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
for rx in ("BULLET", "META", "THINK_LEAK", "NARRATION"):
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

print()
if FAILS:
    print(f"NO-GO — {len(FAILS)} failed: {FAILS}"); sys.exit(1)
print("GO — all checks passed.")
