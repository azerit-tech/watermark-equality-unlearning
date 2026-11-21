"""Append-only JSONL ledger.

Every HTTP attempt to /chat/completions is written (and fsync'ed) BEFORE the
request is sent ("attempt" record), and its outcome is written (and fsync'ed)
as soon as it is known ("response" or "error" record). Consequences:

* a response received from the server is on disk before anything else happens;
* a crash between send and receive leaves a dangling "attempt", which is
  counted as spent (conservative budget accounting);
* resuming a plan skips every item whose key already has a "response".

Records are never modified or deleted.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def item_key(endpoint: str, prompt_id: str, rep: int, max_tokens: int) -> str:
    return f"{endpoint}|{prompt_id}|{rep}|{max_tokens}"


class LedgerBusy(RuntimeError):
    pass


class PromptConflict(ValueError):
    pass


@dataclass
class LedgerState:
    done_keys: set[str] = field(default_factory=set)
    # attempts that may have been charged by the server, per endpoint
    local_used: Counter = field(default_factory=Counter)
    responses: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)
    dangling: Counter = field(default_factory=Counter)  # attempt with no outcome
    prompt_texts: dict[str, str] = field(default_factory=dict)
    corrupt_lines: int = 0
    n_records: int = 0


class Ledger:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._fh = None
        self._run_lock_fh = None

    # ------------------------------------------------------------------ read
    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "rb") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    yield {"type": "_corrupt"}
                    continue
                if isinstance(rec, dict):
                    yield rec

    def state(self) -> LedgerState:
        st = LedgerState()
        attempts: dict[str, str] = {}   # attempt_id -> endpoint
        outcome: dict[str, dict] = {}   # attempt_id -> outcome record
        for rec in self.records():
            st.n_records += 1
            t = rec.get("type")
            if t == "_corrupt":
                st.corrupt_lines += 1
                continue
            pid, ptxt = rec.get("prompt_id"), rec.get("prompt")
            if pid is not None and ptxt is not None:
                st.prompt_texts.setdefault(pid, ptxt)
            if t == "attempt":
                attempts[rec["attempt_id"]] = rec["endpoint"]
            elif t in ("response", "error"):
                outcome[rec.get("attempt_id")] = rec
                if t == "response":
                    st.done_keys.add(rec["key"])
                    st.responses[rec["endpoint"]] += 1
                else:
                    st.errors[rec["endpoint"]] += 1
        for aid, ep in attempts.items():
            o = outcome.get(aid)
            if o is None:
                st.dangling[ep] += 1
                st.local_used[ep] += 1
            elif o["type"] == "response" or o.get("charged") != "no":
                st.local_used[ep] += 1
        return st

    def responses(self) -> list[dict[str, Any]]:
        return [r for r in self.records() if r.get("type") == "response"]

    # ----------------------------------------------------------------- write
    def _open(self):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists()
            self._fh = open(self.path, "ab")
            if new:
                dfd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            else:
                # repair a torn last line left by a crash mid-write
                size = self.path.stat().st_size
                if size:
                    with open(self.path, "rb") as r:
                        r.seek(size - 1)
                        if r.read(1) != b"\n":
                            self._fh.write(b"\n")
                            self._fh.flush()
                            os.fsync(self._fh.fileno())
        return self._fh

    def append(self, rec: dict[str, Any]) -> None:
        line = (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with self._lock:
            fh = self._open()
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    # -------------------------------------------------------------- run lock
    def acquire_run_lock(self) -> None:
        """Only one executing run per ledger at a time (across processes)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(str(self.path) + ".lock", "a")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            raise LedgerBusy(f"another run is already using {self.path}")
        self._run_lock_fh = fh

    def release_run_lock(self) -> None:
        if self._run_lock_fh is not None:
            fcntl.flock(self._run_lock_fh.fileno(), fcntl.LOCK_UN)
            self._run_lock_fh.close()
            self._run_lock_fh = None

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
        self.release_run_lock()


def load_responses(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Flat view of all successful responses, for analysis code."""
    out = []
    for r in Ledger(path).responses():
        out.append({k: r.get(k) for k in (
            "endpoint", "prompt_id", "prompt", "rep", "max_tokens", "text",
            "finish_reason", "ts", "attempt_id")} | {"usage": (r.get("response") or {}).get("usage")
                                                     if isinstance(r.get("response"), dict) else None})
    return out


def summarize(path: str | os.PathLike) -> dict[str, Any]:
    st = Ledger(path).state()
    per_prompt: dict[str, Counter] = defaultdict(Counter)
    for r in Ledger(path).responses():
        per_prompt[r["endpoint"]][r["prompt_id"]] += 1
    return {"state": st, "per_prompt": per_prompt}
