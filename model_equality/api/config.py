"""Constants and API-key loading for the Model Equality Testing client.

The key is never printed, logged, or written to the ledger.
Lookup order (first non-empty wins):
  1. env MODEL_EQUALITY_API_KEY   (explicit override)
  2. env API_KEY_BOARD
  3. same two names, in that order, from the .env file at the repository root
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_BASE_URL = "http://34.147.229.55/33-model-equality/v1"
API_KEY_HEADER = "X-API-Key"  # confirmed: server answers 401 {"detail":"Missing X-API-Key"}

KEY_ENV_NAMES = ("MODEL_EQUALITY_API_KEY", "API_KEY_BOARD")

# Competition limits
SERVER_EP_LIMIT = 100          # responses per scored endpoint, whole competition
MAX_GENERATED_TOKENS = 256
MAX_INPUT_TOKENS = 1024
MAX_TEAM_CONCURRENCY = 4

# Client defaults (deliberately prudent)
DEFAULT_EP_CAP = 50            # hard cap per ep- endpoint, counting team-wide usage
DEFAULT_CONCURRENCY = 4

SCORED_RE = re.compile(r"^ep-(0[1-9]|[12][0-9]|3[0-3])$")
REF_RE = re.compile(r"^ref-0[1-9]$")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = REPO_ROOT / "model_equality" / "api" / "ledger" / "ledger.jsonl"


class MissingAPIKey(RuntimeError):
    pass


def is_scored(endpoint: str) -> bool:
    """True for budgeted endpoints. Anything that is not a known ref- id is
    treated as scored (fail safe)."""
    return not REF_RE.match(endpoint)


def _parse_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        out[k] = v
    return out


def load_api_key(dotenv_path: Path | None = None) -> str:
    for name in KEY_ENV_NAMES:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    dotenv_path = dotenv_path or (REPO_ROOT / ".env")
    env = _parse_dotenv(dotenv_path)
    for name in KEY_ENV_NAMES:
        v = env.get(name, "").strip()
        if v:
            return v
    raise MissingAPIKey(
        "No API key found: set MODEL_EQUALITY_API_KEY or API_KEY_BOARD in the "
        f"environment or in {dotenv_path}."
    )


def base_url() -> str:
    return os.environ.get("MODEL_EQUALITY_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
