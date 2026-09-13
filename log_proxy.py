#!/usr/bin/env python3
"""log_proxy.py — transparent logging proxy in front of an OpenAI-compatible endpoint (Ollama /v1).

Records every request body and the response to a JSONL file and forwards the call unchanged, so you can
see EXACTLY what Honcho sends the summary model (prompt text, max_tokens, temperature, extra fields) and
what came back — the 2026-09-13 live check stored one summary that could not have come from the
messages the harness thought Honcho had (TRAIN.md). Stdlib only; never writes into data/.

  python3 log_proxy.py --listen 0.0.0.0:11435 --upstream http://node7.ea.org:11434 --log results/proxy-honcho.jsonl
  # Honcho: SUMMARY_MODEL_CONFIG__OVERRIDES__BASE_URL=http://<this host>:11435/v1
  python3 log_proxy.py diff results/proxy-honcho.jsonl --chains data/chains.jsonl --chain c00022 --step s1
      # compare the logged prompt of a step with what summary_chain.build_step would have sent

Log rows: {ts, method, path, status, latency_s, request: <json body>, response: <json body or text>}.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

STATE = {"upstream": "", "log": ""}


def _append(row):
    os.makedirs(os.path.dirname(os.path.abspath(STATE["log"])) or ".", exist_ok=True)
    with open(STATE["log"], "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _forward(self, body):
        url = STATE["upstream"].rstrip("/") + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding", "connection")}
        req = urllib.request.Request(url, data=body if body else None, headers=headers, method=self.command)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                data, status, ctype = r.read(), r.status, r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            data, status, ctype = e.read(), e.code, e.headers.get("Content-Type", "application/json")
        except Exception as e:  # noqa: BLE001
            data, status, ctype = json.dumps({"error": f"proxy: {type(e).__name__}: {e}"}).encode(), 502, "application/json"
        dt = round(time.time() - t0, 1)
        try:
            req_j = json.loads(body) if body else None
        except Exception:  # noqa: BLE001
            req_j = body.decode(errors="replace")[:5000]
        try:
            resp_j = json.loads(data)
        except Exception:  # noqa: BLE001
            resp_j = data.decode(errors="replace")[:5000]
        _append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "method": self.command, "path": self.path, "status": status,
                 "latency_s": dt, "request": req_j, "response": resp_j})
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        sys.stderr.write(f"{self.command} {self.path} -> {status} {dt}s\n")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self._forward(self.rfile.read(n) if n else b"")

    def do_GET(self):
        self._forward(b"")


def serve(a):
    host, port = a.listen.rsplit(":", 1)
    STATE.update(upstream=a.upstream, log=a.log)
    srv = ThreadingHTTPServer((host, int(port)), H)
    sys.stderr.write(f"log_proxy: {a.listen} -> {a.upstream}, logging to {a.log}\n")
    srv.serve_forever()


def diff(a):
    """Find the logged request whose <conversation> block is closest to a step's messages and report
    what differs from summary_chain.build_step: message set, previous summary, output_words, extra fields."""
    import summary_chain as sc
    rows = [json.loads(l) for l in open(a.log) if l.strip()]
    chain = sc.load_chains(a.chains, {a.chain})[0]
    kind = "short" if a.step.startswith("s") else "long"
    k = int(a.step[1:])
    want_msgs = sc.step_messages(chain, kind, k)
    want_lines = [f"{m['peer_name']}: {m['content']}" for m in want_msgs]
    best, best_hits = None, -1
    for r in rows:
        req = r.get("request") or {}
        if not isinstance(req, dict) or "messages" not in req:
            continue
        text = "\n".join(str(m.get("content") or "") for m in req["messages"])
        hits = sum(1 for ln in want_lines if ln in text)
        if hits > best_hits:
            best, best_hits = r, hits
    if best is None:
        print("no chat request in the log"); return 1
    req = best["request"]
    text = "\n".join(str(m.get("content") or "") for m in req["messages"])
    conv = text.split("<conversation>\n", 1)[1].split("\n</conversation>", 1)[0] if "<conversation>" in text else ""
    prev = text.split("<previous_summary>\n", 1)[1].split("\n</previous_summary>", 1)[0] if "<previous_summary>" in text else ""
    got_lines = conv.split("\n") if conv else []
    print(f"best match {best['ts']} status {best['status']} {best['latency_s']}s: {best_hits}/{len(want_lines)} of step {kind} {k}'s messages present")
    print(f"  request fields: {sorted(kk for kk in req if kk != 'messages')}  roles: {[m.get('role') for m in req['messages']]}")
    print(f"  max_tokens={req.get('max_tokens')} temperature={req.get('temperature')} model={req.get('model')}")
    import re
    lim = re.search(r"Hard limit: (\d+) words", text)
    print(f"  Hard limit in prompt: {lim.group(1) if lim else '?'} words; previous summary {len(prev.split())} words"
          + (" (NO previous)" if "There is no previous summary" in prev else ""))
    print(f"  conversation lines sent: {len(got_lines)} (expected {len(want_lines)})")
    missing = [ln[:80] for ln in want_lines if ln not in text]
    extra = [ln[:80] for ln in got_lines if ln not in want_lines]
    if missing:
        print(f"  MISSING from the sent conversation ({len(missing)}): {missing[:5]}")
    if extra:
        print(f"  EXTRA lines not in this step ({len(extra)}): {extra[:5]}")
    resp = best.get("response") or {}
    if isinstance(resp, dict) and resp.get("choices"):
        m = resp["choices"][0].get("message") or {}
        print(f"  response: finish={resp['choices'][0].get('finish_reason')} content {len((m.get('content') or '').split())}w reasoning {len(m.get('reasoning') or '')} chars")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("serve"); p.set_defaults(fn=serve)
    p.add_argument("--listen", default="0.0.0.0:11435")
    p.add_argument("--upstream", default="http://node7.ea.org:11434")
    p.add_argument("--log", default="results/proxy.jsonl")
    p = sub.add_parser("diff"); p.set_defaults(fn=diff)
    p.add_argument("log"); p.add_argument("--chains", default="data/chains.jsonl")
    p.add_argument("--chain", required=True); p.add_argument("--step", required=True, help="s1, l0, ...")
    argv = sys.argv[1:]
    if argv and argv[0] not in ("serve", "diff", "-h", "--help"):
        argv = ["serve"] + argv
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
