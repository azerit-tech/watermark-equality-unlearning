"""Download tokenizer/config files (never weights) for every candidate into hf/<name>/.

Usage: .venv/bin/python fetch.py [name ...]
Writes hf/_sources.json : name -> {repo, files, errors}.
"""
from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from candidates import CANDIDATES, FILES

ROOT = Path(__file__).resolve().parent / "hf"
CACHE = Path(__file__).resolve().parent / ".hfcache"


def fetch_one(name: str) -> tuple[str, dict]:
    errs = []
    for repo in CANDIDATES[name]:
        got = []
        dest = ROOT / name
        try:
            for fn in FILES:
                if fn == "tokenizer.model" and "tokenizer.json" in got:
                    continue
                try:
                    p = hf_hub_download(repo, fn, cache_dir=CACHE)
                except EntryNotFoundError:
                    continue
                except Exception as e:  # 404 on file is fine; gated / missing repo is not
                    msg = str(e)
                    if "404" in msg and "Entry" in msg or "does not exist" in msg and fn != "tokenizer.json":
                        continue
                    if fn in ("tokenizer.json", "tokenizer_config.json"):
                        raise
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dest / fn)
                got.append(fn)
            if "tokenizer.json" in got or "tokenizer.model" in got:
                return name, {"repo": repo, "files": got, "errors": errs}
            errs.append(f"{repo}: no tokenizer file")
        except Exception as e:
            errs.append(f"{repo}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    return name, {"repo": None, "files": [], "errors": errs}


def main(argv):
    names = argv or list(CANDIDATES)
    ROOT.mkdir(exist_ok=True)
    src_path = ROOT / "_sources.json"
    sources = json.loads(src_path.read_text()) if src_path.exists() else {}
    with ThreadPoolExecutor(6) as ex:
        for name, info in ex.map(fetch_one, names):
            sources[name] = info
            print(f"{name:32s} {info['repo'] or 'FAILED'}  {info['files'] if info['repo'] else info['errors']}")
    src_path.write_text(json.dumps(sources, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
