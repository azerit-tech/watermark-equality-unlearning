"""Local fake of the competition API, for tests only.

Behaviour mirrors what is known about the real server:
  GET  /models            -> {"object":"list","data":[{"id","object","owned_by","budget":{"used","limit"}}]}
  POST /chat/completions  -> OpenAI-style chat.completion, X-API-Key required,
                             system messages / max_tokens>256 rejected (400, not counted),
                             budget exhausted -> 429 without Retry-After,
                             more than `max_concurrency` in flight -> 429 with Retry-After.
Scripted faults per endpoint (consumed in order, one per request):
  "busy"     429 + Retry-After: 0          (not counted)
  "502"/"503"/"504"                        (not counted by this fake)
  "drop"     counts the response, then closes the socket without answering
  "exhaust"  sets used=limit (teammate spent everything) then 429 w/o Retry-After
  "slow:S"   sleeps S seconds then answers normally
  "teammate:K" before answering, adds K to used (someone else spending meanwhile)
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFIX = "/33-model-equality/v1"


class FakeState:
    def __init__(self, key: str, ep_limit: int = 100):
        self.key = key
        self.lock = threading.Lock()
        self.models = {f"ep-{i:02d}": {"used": 0, "limit": ep_limit} for i in range(1, 34)}
        self.models.update({f"ref-{i:02d}": {"used": 0, "limit": 1_000_000} for i in range(1, 10)})
        self.faults: dict[str, list[str]] = {}
        self.delay = 0.0
        self.inflight = 0
        self.max_inflight = 0
        self.max_concurrency = 4
        self.posts = 0
        self.requests: list[dict] = []
        self.seen_headers: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence
        pass

    @property
    def st(self) -> FakeState:
        return self.server.state  # type: ignore[attr-defined]

    def _send(self, code: int, obj, headers: dict | None = None):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _auth(self) -> bool:
        k = self.headers.get("X-API-Key")
        if k is None:
            self._send(401, {"detail": "Missing X-API-Key"})
            return False
        if k != self.st.key:
            self._send(401, {"detail": "Invalid API key"})
            return False
        return True

    def do_GET(self):
        if self.path != PREFIX + "/models":
            return self._send(404, {"detail": "Not Found"})
        if not self._auth():
            return
        with self.st.lock:
            data = [{"id": k, "object": "model", "owned_by": "hackathon",
                     "budget": dict(v)} for k, v in self.st.models.items()]
        self._send(200, {"object": "list", "data": data})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if self.path != PREFIX + "/chat/completions":
            return self._send(404, {"detail": "Not Found"})
        st = self.st
        with st.lock:
            st.posts += 1
            st.seen_headers.append(dict(self.headers))
            st.inflight += 1
            st.max_inflight = max(st.max_inflight, st.inflight)
            over = st.inflight > st.max_concurrency
        try:
            if over:
                return self._send(429, {"detail": "too many concurrent requests"},
                                  {"Retry-After": "0"})
            if not self._auth():
                return
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return self._send(400, {"detail": "bad json"})
            st.requests.append(body)
            ep = body.get("model")
            msgs = body.get("messages") or []
            mt = body.get("max_tokens")
            if ep not in st.models:
                return self._send(404, {"detail": f"unknown model {ep}"})
            if any(m.get("role") == "system" for m in msgs):
                return self._send(400, {"detail": "system messages are not allowed"})
            if mt is not None and (not isinstance(mt, int) or mt > 256 or mt < 1):
                return self._send(400, {"detail": "max_tokens must be <= 256"})
            if any("INVALID" in (m.get("content") or "") for m in msgs):
                return self._send(400, {"detail": "invalid prompt"})
            with st.lock:
                fl = st.faults.get(ep) or []
                fault = fl.pop(0) if fl else None
            if fault == "busy":
                return self._send(429, {"detail": "busy"}, {"Retry-After": "0"})
            if fault in ("502", "503", "504"):
                return self._send(int(fault), {"detail": "upstream error"})
            if fault == "exhaust":
                with st.lock:
                    st.models[ep]["used"] = st.models[ep]["limit"]
            if fault and fault.startswith("teammate:"):
                with st.lock:
                    st.models[ep]["used"] += int(fault.split(":")[1])
            if fault and fault.startswith("slow:"):
                time.sleep(float(fault.split(":")[1]))
            with st.lock:
                b = st.models[ep]
                if b["used"] >= b["limit"]:
                    exhausted = True
                else:
                    exhausted = False
                    b["used"] += 1
                    n = b["used"]
            if exhausted:
                return self._send(429, {"detail": "budget exhausted for this endpoint"})
            if st.delay:
                time.sleep(st.delay)
            if fault == "drop":
                self.close_connection = True
                try:
                    import socket
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return
            content = f"answer from {ep} #{n} to: {msgs[-1]['content'][:30]}"
            self._send(200, {
                "id": f"chatcmpl-{ep}-{n}", "object": "chat.completion",
                # same shape as the real server (checked 2026-11-21): no "usage",
                # created=0, finish_reason "stop" even when cut at max_tokens
                "created": 0, "model": ep,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "budget": {"used": n, "limit": st.models[ep]["limit"]}})
        finally:
            with st.lock:
                st.inflight -= 1


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # clients killed mid-request (crash tests) -> broken pipes; not interesting
        import sys
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


class FakeServer:
    def __init__(self, key: str = "test-key-SECRET-123", ep_limit: int = 100):
        self.state = FakeState(key, ep_limit)
        self.httpd = _QuietServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.httpd.state = self.state  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}{PREFIX}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
