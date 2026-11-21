"""Submit a checkpoint to the leaderboard server and log the score.

  python submit.py runs/first/submission.pt

The API key is read from the HACKATHON_API_KEY environment variable, or from a `.env` file next to this
script containing the line  HACKATHON_API_KEY=...  (`.env` is git-ignored: never commit the key).
Run this from a JURECA login node: compute nodes have no internet access.

Same request as the organisers' submission.py, plus a local format check before uploading and a
`submissions.jsonl` log, because the leaderboard only shows the team's best score.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from common import check_submission

TASK_ID = "31-unlearning"
SERVER_URL = "http://34.147.229.55"
HERE = Path(__file__).resolve().parent


def api_key() -> str:
    key = os.environ.get("HACKATHON_API_KEY", "")
    env = HERE / ".env"
    if not key and env.is_file():
        for line in env.read_text().splitlines():
            if line.strip().startswith("HACKATHON_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path)
    p.add_argument("--note", default="", help="free text saved in the log (e.g. the hyper-parameters)")
    args = p.parse_args()

    key = api_key()
    if not key:
        print("No API key: set HACKATHON_API_KEY or create unlearning/.env", file=sys.stderr)
        return 1
    check_submission(args.file)
    print(f"Format checks passed. Submitting {args.file} to {TASK_ID} (scoring can take a few minutes)...")
    with args.file.open("rb") as f:
        reply = requests.post(f"{SERVER_URL}/submit/{TASK_ID}", headers={"X-API-Key": key},
                              files={"file": (args.file.name, f)}, timeout=(10, 600))
    try:
        body = reply.json()
    except ValueError:
        body = {"raw": reply.text[:300]}
    with (HERE / "submissions.jsonl").open("a") as log:
        log.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "file": str(args.file),
                              "note": args.note, "http": reply.status_code, "reply": body}) + "\n")
    if reply.status_code != 200 or body.get("status") != "success":
        print(f"Rejected (HTTP {reply.status_code}): {body}", file=sys.stderr)
        return 1
    print(f"Score (public split): {body.get('score')}")
    if "previous_best" in body:
        print(f"Previous best: {body['previous_best']}  (improved: {body.get('improved')})")
    print(body.get("message", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
