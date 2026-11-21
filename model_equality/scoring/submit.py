"""Guarded submission of a CSV to the Model Equality scoring server.

    python3.12 -m model_equality.scoring.submit SUBMISSION.csv            # dry run (default)
    python3.12 -m model_equality.scoring.submit SUBMISSION.csv --confirm-submit

Steps: validate the CSV (ep-01..ep-33) -> check the 30-minute guard against
the local append-only log -> (only with --confirm-submit) POST multipart to
the server, wait up to 600 s, record the answer.

API key: env MODEL_EQUALITY_API_KEY (priority) else API_KEY_BOARD; the
``.env`` at the repository root is loaded without overriding the
environment.  The key and the .env content are never printed.

Journal (append-only JSON lines): model_equality/scoring/submissions/log.jsonl
  * an "attempt" entry is written BEFORE the request (UTC timestamp, sha256,
    path of the archived copy of the CSV) -- the guard counts attempts, so a
    request that timed out still blocks the next 30 minutes;
  * a "result" entry after (HTTP status, score, previous_best, improved, body).
The guard only knows submissions made through this script on this machine;
teammates must coordinate the team-wide 30-minute limit themselves.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .csvio import validate

# Request shape copied from the organisers' submission.py (copy in data/submission.py):
#   requests.post(f"{SERVER_URL}/submit/{TASK_ID}", headers={"X-API-Key": KEY},
#                 files={"file": (file.name, f)}, timeout=(600, 600))
SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
URL = f"{SERVER_URL}/submit/{TASK_ID}"
API_HEADER = "X-API-Key"
FILE_FIELD = "file"
KEY_VARS = ("MODEL_EQUALITY_API_KEY", "API_KEY_BOARD")  # first non-empty wins
MIN_INTERVAL = dt.timedelta(minutes=30)
TIMEOUT = (600, 600)              # as submission.py: the server waits up to 500 s for the score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_LOG_DIR = HERE / "submissions"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal KEY=VALUE parser; never overrides existing variables; prints nothing."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        os.environ.setdefault(k, v)


def get_api_key() -> tuple[str, str]:
    """(variable name, key).  Raises RuntimeError without revealing anything."""
    load_dotenv()
    for var in KEY_VARS:
        v = os.environ.get(var, "").strip()
        if v:
            return var, v
    raise RuntimeError(f"no API key: set one of {', '.join(KEY_VARS)} (env or repo .env)")


def read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def last_attempt(log_path: Path) -> dt.datetime | None:
    ts = [dt.datetime.fromisoformat(e["ts"]) for e in read_log(log_path) if e.get("kind") == "attempt"]
    return max(ts) if ts else None


def guard(log_path: Path, now: dt.datetime | None = None) -> tuple[bool, str]:
    """(allowed, message) according to the 30-minute rule."""
    now = now or now_utc()
    last = last_attempt(log_path)
    if last is None:
        return True, "no previous submission in the local log"
    wait = last + MIN_INTERVAL - now
    if wait > dt.timedelta(0):
        return False, (f"last submission at {last.isoformat()} ; next allowed at "
                       f"{(last + MIN_INTERVAL).isoformat()} (in {int(wait.total_seconds() // 60)} min "
                       f"{int(wait.total_seconds() % 60)} s)")
    return True, f"last submission at {last.isoformat()} (more than 30 min ago)"


def append_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


@contextlib.contextmanager
def locked(log_dir: Path):
    """Exclusive lock so two local runs cannot both pass the guard."""
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / ".lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another submission is in progress (lock held)")
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "<redacted>") if secret else text


def submit(csv_path: str, confirm: bool, log_dir: Path = DEFAULT_LOG_DIR, poster=None,
           now: dt.datetime | None = None, out=print) -> int:
    """Return code: 0 ok/dry-run ok, 1 invalid CSV, 2 guard, 3 no key, 4 HTTP/network error.

    ``poster(url, headers, files, timeout)`` defaults to requests.post (injectable for tests).
    """
    log_path = log_dir / "log.jsonl"
    rep = validate(csv_path)
    out(rep.format())
    if not rep.ok:
        out("REFUSED: invalid CSV, nothing sent.")
        return 1
    data = Path(csv_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    out(f"sha256 {sha}")
    for e in read_log(log_path):
        if e.get("kind") == "attempt" and e.get("sha256") == sha:
            out(f"NOTE: this exact file was already submitted at {e['ts']}")
            break

    try:
        with locked(log_dir):
            ok, msg = guard(log_path, now)
            out(("guard OK: " if ok else "REFUSED (30-minute rule): ") + msg)
            if not ok:
                return 2
            try:
                var, key = get_api_key()
            except RuntimeError as e:
                out(f"REFUSED: {e}")
                return 3
            out(f"API key found in ${var} (not shown)")
            if not confirm:
                out("DRY RUN: nothing sent. Re-run with --confirm-submit to submit.")
                return 0

            ts = now or now_utc()
            stamp = ts.strftime("%Y%m%dT%H%M%SZ")
            copy = log_dir / f"{stamp}_{sha[:12]}.csv"
            shutil.copyfile(csv_path, copy)
            os.chmod(copy, 0o444)
            append_log(log_path, {"kind": "attempt", "ts": ts.isoformat(), "sha256": sha,
                                  "source": str(Path(csv_path).resolve()), "copy": copy.name,
                                  "url": URL})
            if poster is None:
                import requests
                poster = requests.post
            out(f"POST {URL} (may take up to ~500 s) ...")
            t0 = time.monotonic()
            entry = {"kind": "result", "sha256": sha, "attempt_ts": ts.isoformat()}
            try:
                r = poster(URL, headers={API_HEADER: key},
                           files={FILE_FIELD: (Path(csv_path).name, data)}, timeout=TIMEOUT)
            except Exception as e:  # network error / timeout: attempt stays logged
                entry.update(ts=now_utc().isoformat(), elapsed_s=round(time.monotonic() - t0, 1),
                             error=_redact(f"{type(e).__name__}: {e}", key))
                append_log(log_path, entry)
                out(f"ERROR: {entry['error']}")
                return 4
            body = _redact(r.text, key)
            entry.update(ts=now_utc().isoformat(), elapsed_s=round(time.monotonic() - t0, 1),
                         http_status=r.status_code, body=body[:4000])
            # Response handling as in submission.py: JSON body; HTTP 200 and
            # status == "success" -> score (public split), previous_best, improved, message;
            # otherwise "detail" (rejected) or status/message.
            try:
                js = r.json()
            except ValueError:
                js = None
            if isinstance(js, dict):
                for k in ("status", "score", "previous_best", "improved", "message", "detail"):
                    if k in js:
                        entry[k] = js[k]
            append_log(log_path, entry)
            if js is None:
                out(f"ERROR: HTTP {r.status_code}, non-JSON body: {body[:300]}")
                return 4
            if r.status_code != 200:
                out(f"REJECTED (HTTP {r.status_code}): {_redact(str(js.get('detail', js)), key)}")
                return 4
            if js.get("status") != "success":
                out(f"{js.get('status')}: {_redact(str(js.get('message')), key)}")
                return 4
            out(f"Score (public split): {js.get('score')}")
            if "previous_best" in js:
                out(f"Previous best: {js['previous_best']}  (improved: {js.get('improved')})")
            if js.get("message"):
                out(_redact(str(js["message"]), key))
            return 0
    except RuntimeError as e:
        out(f"REFUSED: {e}")
        return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--confirm-submit", action="store_true",
                    help="actually send (uses one of the team's 30-minute slots)")
    ap.add_argument("--status", action="store_true", help="show the guard state and the log, send nothing")
    a = ap.parse_args(argv)
    if a.status:
        print(guard(DEFAULT_LOG_DIR / "log.jsonl")[1])
        for e in read_log(DEFAULT_LOG_DIR / "log.jsonl"):
            print(json.dumps({k: v for k, v in e.items() if k != "body"}))
        return 0
    if not a.csv:
        ap.error("csv path required (or use --status)")
    return submit(a.csv, a.confirm_submit)


if __name__ == "__main__":
    sys.exit(main())
