#!/usr/bin/env python3
"""summary_chain.py — the one place that turns a session chain into Honcho summary *steps*.

A chain row (gen_sessions.py, PLAN §2.1) has 60–120 messages. Honcho summarises every
MESSAGES_PER_SHORT_SUMMARY (20) messages with a short summary whose previous summary is the stored
previous SHORT summary, and every MESSAGES_PER_LONG_SUMMARY (60) messages with a long summary whose
previous is the previous LONG summary (summarizer.py at the commit named in summary_prompt.py:
`get_summary(db, ws, session, summary_type)` — the two chains are independent). A step is therefore
(kind, k): short k covers messages [20k+1, 20k+20], long j covers [60j+1, 60j+60].

output_words parity: short = int(min(input_tokens, MAX_TOKENS_SHORT) * 0.75) where input_tokens =
tokens(messages) + tokens(previous summary) (Honcho's per-message token_count plus the stored
summary's token_count; we approximate both with llm_backend.tokens_estimate); long =
int(MAX_TOKENS_LONG * 0.75). max_tokens sent = MAX_TOKENS_SHORT / MAX_TOKENS_LONG. Both caps are
operator settings (SUMMARY_MAX_TOKENS_SHORT/LONG) — every script takes --max-tokens-short/--max-tokens-long.

Everything here is derived data; the prompt text itself comes only from summary_prompt.py and the
scores only from summary_scoring.py. Stdlib only.
"""
import json
import time
import urllib.request

import llm_backend as be
import summary_prompt as sp
import summary_scoring as ss

CHUNK = sp.MESSAGES_PER_SHORT_SUMMARY        # 20
LONG_EVERY = sp.MESSAGES_PER_LONG_SUMMARY    # 60
KINDS = ("short", "long")


# ------------------------------------------------------------------ chunking
def chunks_for(n_messages: int, size: int = CHUNK):
    """[{k, seqs:[lo, hi]}] for the complete blocks of `size` messages (1-based, inclusive)."""
    return [{"k": k, "seqs": [k * size + 1, (k + 1) * size]} for k in range(n_messages // size)]


def with_chunks(chain: dict) -> dict:
    """Return the chain with `chunks` (short) and `chunks_long` filled from its message count."""
    n = len(chain["messages"])
    return {**chain, "chunks": chunks_for(n, CHUNK), "chunks_long": chunks_for(n, LONG_EVERY)}


def chain_view(chain: dict, kind: str) -> dict:
    """The chain as summary_scoring.py wants it for `kind`: `chunks` = the blocks of that kind."""
    if kind == "long":
        return {**chain, "chunks": chain.get("chunks_long") or chunks_for(len(chain["messages"]), LONG_EVERY)}
    return {**chain, "chunks": chain.get("chunks") or chunks_for(len(chain["messages"]), CHUNK)}


def n_steps(chain: dict, kind: str) -> int:
    return len(chain_view(chain, kind)["chunks"])


def steps(chain: dict):
    """Every (kind, k) of a chain in the order Honcho reaches them (short before long at the same
    message; Honcho actually gathers both concurrently, the order does not matter for scoring)."""
    out = []
    for k, ch in enumerate(chain_view(chain, "short")["chunks"]):
        out.append(("short", k))
        hi = ch["seqs"][1]
        if hi % LONG_EVERY == 0:
            out.append(("long", hi // LONG_EVERY - 1))
    return out


def step_id(chain_id: str, kind: str, k: int, variant: str = "") -> str:
    return f"{chain_id}-{'s' if kind == 'short' else 'l'}{k}{variant}"


def parse_step_id(sid: str):
    """'c00003-s2b' -> ('c00003', 'short', 2, 'b')."""
    head, tail = sid.rsplit("-", 1)
    kind = "short" if tail[0] == "s" else "long"
    num = ""
    for ch in tail[1:]:
        if ch.isdigit():
            num += ch
        else:
            break
    return head, kind, int(num), tail[1 + len(num):]


def step_messages(chain: dict, kind: str, k: int):
    """The messages Honcho formats for this step, as {"peer_name", "content"} dicts."""
    lo, hi = chain_view(chain, kind)["chunks"][k]["seqs"]
    return [{"peer_name": m["peer"], "content": m["text"]} for m in chain["messages"] if lo <= m["seq"] <= hi]


def output_words_for(kind: str, msgs, previous, max_tokens_short=sp.MAX_TOKENS_SHORT_DEFAULT,
                     max_tokens_long=sp.MAX_TOKENS_LONG_DEFAULT) -> int:
    if kind == "long":
        return sp.output_words_long(max_tokens_long)
    input_tokens = sum(be.tokens_estimate(m["content"]) for m in msgs) + (be.tokens_estimate(previous) if previous else 0)
    return sp.output_words_short(input_tokens, max_tokens_short)


def build_step(chain: dict, kind: str, k: int, previous, max_tokens_short=sp.MAX_TOKENS_SHORT_DEFAULT,
               max_tokens_long=sp.MAX_TOKENS_LONG_DEFAULT) -> dict:
    """Exactly what Honcho sends for this step: {"messages": [one user turn], "prompt", "output_words",
    "max_tokens", "previous_summary"}. `previous` empty/None -> the no-previous-summary sentence."""
    msgs = step_messages(chain, kind, k)
    prev = previous if (previous or "").strip() else None      # whitespace-only = nothing stored; text passes through untouched
    ow = output_words_for(kind, msgs, prev, max_tokens_short, max_tokens_long)
    payload = sp.build_messages(kind, msgs, prev, ow)
    return {"messages": payload, "prompt": payload[0]["content"], "output_words": ow,
            "max_tokens": max_tokens_long if kind == "long" else max_tokens_short,
            "previous_summary": prev or ""}


def score_step(chain: dict, kind: str, k: int, summary: str, output_words: int) -> dict:
    return ss.score_summary(chain_view(chain, kind), k, summary, output_words)


# ------------------------------------------------------------------ student call (OpenAI-compatible /v1)
OPENROUTER_HEADERS = {"HTTP-Referer": "https://github.com/kalaspuffar/honcho-summary-model",
                      "X-Title": "honcho-summary-model"}


def chat_v1(base: str, body: dict, api_key=None, timeout=900):
    """POST /chat/completions on Ollama's /v1 or OpenRouter. Returns
    {"content", "reasoning", "finish_reason", "usage"}. Ollama and OpenRouter put the model's
    <think> text in a side field; qwen3.5:9b on Ollama can put its whole answer there and return
    EMPTY content (honcho-dialectic-model TRAIN.md §7) — Honcho would then store an empty summary."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
        headers.update(OPENROUTER_HEADERS)
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    if not data.get("choices"):
        raise RuntimeError(f"no choices: {str(data.get('error') or data)[:300]}")
    ch = data["choices"][0]
    m = ch.get("message") or {}
    content = m.get("content")
    return {"content": content if isinstance(content, str) else "",
            "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "finish_reason": ch.get("finish_reason"), "usage": data.get("usage") or {}}


def summarise_step(base: str, model: str, step: dict, api_key=None, temperature=None, extra_body=None, attempts=2):
    """Run the student on one step the way Honcho does (one user message, max_tokens as Honcho sends
    it, temperature only if given — Honcho leaves it to the Modelfile). Never raises: errors land in
    "error" and the summary is ""."""
    body = {"model": model, "messages": step["messages"], "max_tokens": step["max_tokens"]}
    if temperature is not None:
        body["temperature"] = temperature
    body.update(extra_body or {})
    err = ""
    for i in range(attempts):
        t0 = time.time()
        try:
            r = chat_v1(base, body, api_key=api_key)
            return {"summary": (r["content"] or "").strip(), "reasoning": r["reasoning"], "finish_reason": r["finish_reason"],
                    "usage": r["usage"], "latency_s": round(time.time() - t0, 1), "error": ""}
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            if i + 1 < attempts:
                time.sleep(2)
    return {"summary": "", "reasoning": "", "finish_reason": None, "usage": {}, "latency_s": 0.0, "error": err}


def walk(chain: dict, kind: str, generate, answer_from_reasoning=False, max_tokens_short=sp.MAX_TOKENS_SHORT_DEFAULT,
         max_tokens_long=sp.MAX_TOKENS_LONG_DEFAULT, stop_on_error=True):
    """The honest setting: run every step of `kind` in order, previous summary = the model's OWN
    output for the previous step. `generate(step)` -> dict with at least "summary" (+ optional
    reasoning/finish_reason/usage/latency_s/error). Returns one row per step, scored.

    answer_from_reasoning: when content is empty but reasoning text came back, use (and chain) the
    reasoning text instead — the baseline column of the dialectic eval; the row still records
    answered_in_thinking=True. Off by default (what Honcho would actually store)."""
    rows, previous = [], ""
    for k in range(n_steps(chain, kind)):
        step = build_step(chain, kind, k, previous, max_tokens_short, max_tokens_long)
        r = generate(step)
        summary = (r.get("summary") or "").strip()
        reasoning = r.get("reasoning") or ""
        in_thinking = (not summary) and bool(reasoning.strip())
        if in_thinking and answer_from_reasoning:
            summary = reasoning.strip()
        row = {"id": step_id(chain["id"], kind, k), "chain": chain["id"], "category": chain.get("category"),
               "kind": kind, "k": k, "previous_summary": step["previous_summary"],
               "output_words": step["output_words"], "max_tokens": step["max_tokens"],
               "summary": summary, "words": ss.words(summary),
               "finish_reason": r.get("finish_reason"), "answered_in_thinking": in_thinking,
               "reasoning_chars": len(reasoning), "latency_s": r.get("latency_s", 0.0),
               "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
               "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
               "error": r.get("error", "")}
        if row["error"]:
            row["__failed__"] = f"__FAILED__: {row['error']}"
            rows.append(row)
            if stop_on_error:
                break
            previous = ""
            continue
        row["score"] = score_step(chain, kind, k, summary, step["output_words"])
        rows.append(row)
        previous = summary
    return rows


def load_chains(path, ids=None):
    chains = [with_chunks(c) for c in be.read_jsonl(path) if not be.failed(c)]
    if ids is not None:
        chains = [c for c in chains if c["id"] in ids]
    if not chains:
        raise SystemExit(f"no good chains in {path}")
    seen = [c["id"] for c in chains]
    if len(seen) != len(set(seen)):
        raise SystemExit(f"duplicate chain ids in {path}")
    return chains


def peer_names(chain: dict, humans_only=False):
    """Every peer name in the chain. humans_only drops peers with role "assistant" — an assistant's
    product name recurs across sessions in a real deployment too, so it must not glue chains together
    in the persona split."""
    names = {p["name"] for p in chain.get("peers", [])} | {m["peer"] for m in chain.get("messages", [])}
    if humans_only:
        names -= {p["name"] for p in chain.get("peers", []) if p.get("role") == "assistant"}
    return sorted(names)
