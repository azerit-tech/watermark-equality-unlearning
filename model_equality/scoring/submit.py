"""Guarded submission of a CSV to the Model Equality scoring server.

    python3.12 -m model_equality.scoring.submit SUBMISSION.csv            # dry run (default)
    python3.12 -m model_equality.scoring.submit SUBMISSION.csv --confirm-submit

Steps, all before anything is sent:
  1. validate the CSV (ep-01..ep-33);                          refusal -> exit 1
  2. refuse a degenerate partition (one family for all, or every endpoint its
     own model) unless --allow-degenerate;                     refusal -> exit 5
  3. show the CSV age; refuse a CSV older than the query ledger
     model_equality/api/ledger/ledger.jsonl unless --allow-stale;  -> exit 5
  4. refuse a file already submitted (same sha256) unless --force;  -> exit 5
  5. 32-minute guard on the local log (server rule: 30 min; margin because we
     do not know whether it counts from reception or from scoring);  -> exit 2
  6. find the API key;                                          refusal -> exit 3
  7. only with --confirm-submit: POST multipart, wait up to 600 s.  error -> exit 4

API key: env MODEL_EQUALITY_API_KEY (priority, warned about) else API_KEY_BOARD;
the ``.env`` at the repository root is loaded without overriding the
environment.  The key and the .env content are never printed.

Journal (append-only JSON lines): model_equality/scoring/submissions/log.jsonl
  * an "attempt" entry is written BEFORE the request: UTC timestamp, sha256,
    archived copy of the CSV, provenance (ledger line count + sha256, git HEAD,
    uncommitted-changes flag, copies of summary.txt / labels.csv found next to
    the CSV).  The guard counts attempts, so a timed-out request still blocks;
  * a "result" entry after (HTTP status, status, score, previous_best, improved,
    message/detail, body).
The guard only knows submissions made through this script on this machine;
teammates must coordinate the team-wide limit themselves.
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
import subprocess
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
MIN_INTERVAL = dt.timedelta(minutes=32)   # server rule is 30 min; 2 min margin
TIMEOUT = (600, 600)              # as submission.py: the server waits up to 500 s for the score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_LOG_DIR = HERE / "submissions"
DEFAULT_LEDGER = REPO_ROOT / "model_equality" / "api" / "ledger" / "ledger.jsonl"
SIDE_FILES = ("summary.txt", "labels.csv")   # copied next to the archived CSV if present


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


def get_api_key_info() -> tuple[str, str, str]:
    """(variable name, key, origin) with origin "shell environment" or ".env".
    Raises RuntimeError without revealing anything."""
    before = {k: os.environ.get(k) for k in KEY_VARS}
    load_dotenv()
    for var in KEY_VARS:
        v = os.environ.get(var, "").strip()
        if v:
            return var, v, ("shell environment" if before[var] else ".env")
    raise RuntimeError(f"no API key: set one of {', '.join(KEY_VARS)} (env or repo .env)")


def get_api_key() -> tuple[str, str]:
    """(variable name, key)."""
    var, key, _ = get_api_key_info()
    return var, key


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
    """(allowed, message) according to the 32-minute guard."""
    now = now or now_utc()
    last = last_attempt(log_path)
    if last is None:
        return True, "no previous submission in the local log"
    wait = last + MIN_INTERVAL - now
    if wait > dt.timedelta(0):
        return False, (f"last submission at {last.isoformat()} ; next allowed (32-minute guard) at "
                       f"{(last + MIN_INTERVAL).isoformat()} (in {int(wait.total_seconds() // 60)} min "
                       f"{int(wait.total_seconds() % 60)} s)")
    return True, f"last submission at {last.isoformat()} (more than 32 min ago)"


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


def degenerate(rows: dict) -> list[str]:
    """Reasons why a validated partition is degenerate (empty list if fine)."""
    out = []
    if len({f for f, _ in rows.values()}) == 1:
        out.append("all endpoints share ONE family label (MCC_family = 0)")
    if len(set(rows.values())) == len(rows):
        out.append("every endpoint has its OWN model (MCC_model = 0)")
    return out


def ledger_info(ledger: Path) -> dict | None:
    """Line count, sha256 and mtime of the query ledger, or None if absent."""
    try:
        data = ledger.read_bytes()
        mtime = ledger.stat().st_mtime
    except OSError:
        return None
    return {"path": str(ledger), "lines": sum(1 for ln in data.splitlines() if ln.strip()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "mtime": dt.datetime.fromtimestamp(mtime, dt.timezone.utc).isoformat(), "_mtime": mtime}


def git_info(repo: Path = REPO_ROOT) -> dict:
    """HEAD commit and an uncommitted-changes flag (read-only git commands)."""
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_head": head or None, "git_dirty": None if status is None else bool(status)}


def _age(seconds: float) -> str:
    seconds = int(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    return f"{sign}{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def submit(csv_path: str, confirm: bool, log_dir: Path = DEFAULT_LOG_DIR, poster=None,
           now: dt.datetime | None = None, out=print, allow_degenerate: bool = False,
           allow_stale: bool = False, force: bool = False, ledger_path: Path = DEFAULT_LEDGER,
           repo: Path = REPO_ROOT) -> int:
    """Return code: 0 ok/dry-run ok, 1 invalid CSV, 2 guard/lock, 3 no key,
    4 HTTP/network error, 5 refused by a safety check (degenerate, stale, resend).

    ``poster(url, headers, files, timeout)`` defaults to requests.post (injectable for tests).
    """
    log_path = log_dir / "log.jsonl"
    rep = validate(csv_path)
    out(rep.format())
    if not rep.ok:
        out("REFUSED: invalid CSV, nothing sent.")
        return 1

    reasons = degenerate(rep.rows)
    if reasons:
        if not allow_degenerate:
            out("REFUSED: degenerate partition: " + "; ".join(reasons)
                + ". Use --allow-degenerate if this is intended (e.g. a families-only submission).")
            return 5
        out("WARNING (allowed by --allow-degenerate): " + "; ".join(reasons))

    csv_mtime = Path(csv_path).stat().st_mtime
    out(f"CSV last modified {dt.datetime.fromtimestamp(csv_mtime, dt.timezone.utc).isoformat()} "
        f"(age {_age(time.time() - csv_mtime)})")
    led = ledger_info(Path(ledger_path))
    if led is None:
        out(f"WARNING: query ledger not found at {ledger_path}; staleness not checked")
    else:
        out(f"ledger {led['path']}: {led['lines']} lines, last modified {led['mtime']} "
            f"(age {_age(time.time() - led['_mtime'])})")
        if csv_mtime < led["_mtime"]:
            msg = (f"CSV is older than the ledger by {_age(led['_mtime'] - csv_mtime)}: "
                   f"answers collected since were not used")
            if not allow_stale:
                out(f"REFUSED: {msg}. Regenerate the CSV or use --allow-stale.")
                return 5
            out(f"WARNING (allowed by --allow-stale): {msg}")

    data = Path(csv_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    out(f"sha256 {sha}")
    prev = [e for e in read_log(log_path) if e.get("kind") == "attempt" and e.get("sha256") == sha]
    if prev:
        if not force:
            out(f"REFUSED: this exact file was already submitted at {prev[-1]['ts']}. Use --force to resend.")
            return 5
        out(f"WARNING (allowed by --force): this exact file was already submitted at {prev[-1]['ts']}")

    try:
        with locked(log_dir):
            ok, msg = guard(log_path, now)
            out(("guard OK: " if ok else "REFUSED (32-minute guard): ") + msg)
            if not ok:
                return 2
            try:
                var, key, origin = get_api_key_info()
            except RuntimeError as e:
                out(f"REFUSED: {e}")
                return 3
            out(f"API key taken from ${var} ({origin}); value not shown")
            if var == "MODEL_EQUALITY_API_KEY":
                out("WARNING: $MODEL_EQUALITY_API_KEY overrides $API_KEY_BOARD; if it is an old value "
                    "exported in this shell, run `unset MODEL_EQUALITY_API_KEY`.")
            if not confirm:
                out("DRY RUN: nothing sent. Re-run with --confirm-submit to submit.")
                return 0

            ts = now or now_utc()
            stamp = ts.strftime("%Y%m%dT%H%M%SZ")
            base = f"{stamp}_{sha[:12]}"
            copy = log_dir / f"{base}.csv"
            shutil.copyfile(csv_path, copy)
            os.chmod(copy, 0o444)
            side = {}
            for name in SIDE_FILES:
                src = Path(csv_path).resolve().parent / name
                if src.is_file():
                    dst = log_dir / f"{base}_{name}"
                    shutil.copyfile(src, dst)
                    os.chmod(dst, 0o444)
                    side[name] = dst.name
            attempt = {"kind": "attempt", "ts": ts.isoformat(), "sha256": sha,
                       "source": str(Path(csv_path).resolve()), "copy": copy.name,
                       "side_files": side, "url": URL,
                       "csv_mtime": dt.datetime.fromtimestamp(csv_mtime, dt.timezone.utc).isoformat(),
                       "ledger_lines": led and led["lines"], "ledger_sha256": led and led["sha256"],
                       "ledger_mtime": led and led["mtime"],
                       "flags": {"allow_degenerate": allow_degenerate, "allow_stale": allow_stale,
                                 "force": force}}
            attempt.update(git_info(repo))
            append_log(log_path, attempt)
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
            # Response handling as in submission.py: success only if HTTP 200 AND
            # status == "success"; previous_best / improved are optional.
            try:
                js = r.json()
            except ValueError:
                js = None
            if isinstance(js, dict):
                for k in ("status", "score", "previous_best", "improved", "message", "detail"):
                    if k in js:
                        entry[k] = js[k]
            append_log(log_path, entry)
            if not isinstance(js, dict):
                out(f"ERROR: HTTP {r.status_code}, non-JSON body: {body[:300]}")
                return 4
            if r.status_code != 200:
                out(f"REJECTED (HTTP {r.status_code}): {_redact(str(js.get('detail', js)), key)}")
                return 4
            if js.get("status") != "success":
                out(f"FAILED: status={js.get('status')!r} message={_redact(str(js.get('message')), key)}")
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
    ap.add_argument("--allow-degenerate", action="store_true",
                    help="allow one family for all endpoints or one model per endpoint (families-only submission)")
    ap.add_argument("--allow-stale", action="store_true",
                    help="allow a CSV older than model_equality/api/ledger/ledger.jsonl")
    ap.add_argument("--force", action="store_true", help="allow resending a file already submitted (same sha256)")
    ap.add_argument("--status", action="store_true", help="show the guard state and the log, send nothing")
    a = ap.parse_args(argv)
    if a.status:
        print(guard(DEFAULT_LOG_DIR / "log.jsonl")[1])
        for e in read_log(DEFAULT_LOG_DIR / "log.jsonl"):
            print(json.dumps({k: v for k, v in e.items() if k != "body"}))
        return 0
    if not a.csv:
        ap.error("csv path required (or use --status)")
    return submit(a.csv, a.confirm_submit, allow_degenerate=a.allow_degenerate,
                  allow_stale=a.allow_stale, force=a.force)


if __name__ == "__main__":
    sys.exit(main())
