"""Probe bank and ledger loading.

* ``load_probes``: probes.jsonl -> {probe_id: Probe}.
* ``load_dataset``: ledger.jsonl -> Dataset (endpoint -> probe_id -> [Response]),
  read through ``model_equality.api.ledger.load_responses`` (read-only).
  A ledger record is attached to a probe by ``prompt_id`` (the client uses the
  probe ``id``), or, failing that, by exact prompt text.  Records of prompts that
  are not probes (e.g. ``sig_fact``) are kept apart in ``Dataset.other``.

Analysis unit ("unit"): the table on which a statistic is computed.  Its id is
``wm_block`` when set, else ``group`` (KT tool blocks), else the probe id.  K
probes of one block (KD01..KD08) form one unit whose contexts are the prompts;
every stream / form probe is its own unit; every free probe is its own unit.
"""
from __future__ import annotations

import fnmatch
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBES = ROOT / "probes" / "probes.jsonl"
EXTRA_PROBES = ROOT / "analysis" / "probes_extra.jsonl"
DEFAULT_LEDGER = ROOT / "api" / "ledger" / "ledger.jsonl"
DEFAULT_TRUTH = ROOT / "scoring" / "data" / "practice_answers.csv"

DISCRETE_PARSERS = ("stream", "open_list", "form", "form_open", "first_word", "first_match", "tree")
TREE_SUFFIX = "@tree"
LONG_CATEGORIES = ("SL_stream_diluted", "F_form")


@dataclass
class Probe:
    id: str
    category: str
    group: str
    parser: dict
    text: str = ""
    wm_block: str | None = None
    max_tokens: int = 256
    priority: int = 2
    phases: list = field(default_factory=list)
    long_flag: bool | None = None     # set on virtual probes (tree views) to inherit the base probe's class

    @property
    def ptype(self) -> str:
        return self.parser.get("type", "free")

    @property
    def unit(self) -> str:
        return self.wm_block or self.group or self.id

    @property
    def discrete(self) -> bool:
        return self.ptype in DISCRETE_PARSERS

    @property
    def long(self) -> bool:
        """Long-context probe (system-prompt dilution, METHOD 5.4 / 9.1)."""
        if self.long_flag is not None:
            return self.long_flag
        return self.category in LONG_CATEGORIES or "_long" in self.id


@dataclass
class Response:
    text: str
    finish_reason: str | None = None
    usage: dict | None = None
    rep: int | None = None
    max_tokens: int | None = None


@dataclass
class Dataset:
    probes: dict                      # probe_id -> Probe
    responses: dict                   # endpoint -> probe_id -> [Response]
    other: dict = field(default_factory=dict)   # endpoint -> prompt_id -> [Response] (non-probe prompts)

    @property
    def endpoints(self) -> list:
        return sorted(self.responses)

    def n(self, ep: str, pid: str) -> int:
        return len(self.responses.get(ep, {}).get(pid, []))

    def subset(self, endpoints: Iterable[str] | None = None, probes: Iterable[str] | None = None) -> "Dataset":
        eps = set(self.responses) if endpoints is None else set(endpoints)
        pids = None if probes is None else set(probes)
        resp = {e: {p: list(v) for p, v in d.items() if pids is None or p in pids}
                for e, d in self.responses.items() if e in eps}
        return Dataset(self.probes, resp, {e: d for e, d in self.other.items() if e in eps})


def load_probes(path: str | os.PathLike = DEFAULT_PROBES) -> dict:
    """Probe bank.  When ``path`` is the default bank, the supplementary probes of
    analysis/probes_extra.jsonl (written by build_extra_probes.py) are merged in (the bank wins on a clash)."""
    out = {}
    paths = [Path(path)]
    if Path(path).resolve() == DEFAULT_PROBES.resolve() and EXTRA_PROBES.exists():
        paths.append(EXTRA_PROBES)
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["id"] in out:
                    continue
                out[d["id"]] = Probe(id=d["id"], category=d.get("category", ""), group=d.get("group") or d["id"],
                                     parser=d.get("parser") or {"type": "free"}, text=d.get("text", ""),
                                     wm_block=d.get("wm_block"), max_tokens=d.get("max_tokens", 256),
                                     priority=d.get("priority", 2), phases=d.get("phases", []))
    return out


def units_of(probes: dict, probe_ids: Iterable[str] | None = None) -> dict:
    """unit_id -> sorted list of probe ids."""
    u = defaultdict(list)
    for pid in (probes if probe_ids is None else probe_ids):
        if pid in probes:
            u[probes[pid].unit].append(pid)
    return {k: sorted(v) for k, v in sorted(u.items())}


def expand_endpoints(spec: str | Iterable[str] | None, available: Iterable[str]) -> list:
    """'ref-*', 'ref-01,ref-02', 'ep-01..ep-33' or a list -> sorted endpoints present."""
    avail = sorted(set(available))
    if spec is None:
        return avail
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    out = []
    for it in items:
        it = it.strip()
        if ".." in it:
            a, b = it.split("..")
            pre = a.rstrip("0123456789")
            lo, hi = int(a[len(pre):]), int(b[len(pre):] if b.startswith(pre) else b)
            width = len(a) - len(pre)
            out += [f"{pre}{i:0{width}d}" for i in range(lo, hi + 1) if f"{pre}{i:0{width}d}" in avail]
        else:
            out += [e for e in avail if fnmatch.fnmatch(e, it)]
    return sorted(set(out))


def load_dataset(ledger: str | os.PathLike = DEFAULT_LEDGER, probes: dict | None = None,
                 endpoints: str | Iterable[str] | None = None) -> Dataset:
    from model_equality.api.ledger import load_responses
    probes = load_probes() if probes is None else probes
    by_text = {p.text: pid for pid, p in probes.items() if p.text}
    resp: dict = defaultdict(lambda: defaultdict(list))
    other: dict = defaultdict(lambda: defaultdict(list))
    seen = set()
    for r in load_responses(ledger):
        key = (r.get("endpoint"), r.get("prompt_id"), r.get("rep"), r.get("max_tokens"), r.get("attempt_id"))
        if key in seen:
            continue
        seen.add(key)
        ep, pid = r.get("endpoint"), r.get("prompt_id")
        if ep is None or r.get("text") is None:
            continue
        x = Response(text=r["text"], finish_reason=r.get("finish_reason"), usage=r.get("usage"),
                     rep=r.get("rep"), max_tokens=r.get("max_tokens"))
        if pid not in probes:
            pid2 = by_text.get(r.get("prompt") or "")
            if pid2 is None:
                other[ep][pid].append(x)
                continue
            pid = pid2
        resp[ep][pid].append(x)
    ds = Dataset(probes, {e: dict(d) for e, d in resp.items()}, {e: dict(d) for e, d in other.items()})
    if endpoints is not None:
        eps = expand_endpoints(endpoints, set(ds.responses) | set(ds.other))
        ds = ds.subset(eps)
    return ds


def add_tree_views(ds: Dataset, K: int = 12) -> list:
    """Add a virtual probe ``<probe>@tree`` for every probe that has responses (same Response objects).

    Its parser (extract.parse_tree) turns the first K word-level tokens of a response into events
    (exact prefix -> next token): repeated *exact* contexts, hence true next-token laws, for any probe type
    (free text included).  Category ``T_tree``; one unit per view; ``long`` inherited from the base probe.
    A view and its base probe are the same responses: never count both in a budget.
    """
    added = []
    for pid in sorted({p for d in ds.responses.values() for p in d}):
        if pid.endswith(TREE_SUFFIX) or pid not in ds.probes:
            continue
        base = ds.probes[pid]
        vid = pid + TREE_SUFFIX
        ds.probes[vid] = Probe(id=vid, category="T_tree", group=vid, parser={"type": "tree", "K": int(K)},
                               text=base.text, wm_block=None, max_tokens=base.max_tokens, priority=base.priority,
                               phases=[], long_flag=base.long)
        for d in ds.responses.values():
            if pid in d:
                d[vid] = d[pid]
        added.append(vid)
    return added


def read_truth(path: str | os.PathLike = DEFAULT_TRUTH) -> dict:
    """endpoint -> (family, model, kind) (practice_answers.csv format)."""
    from model_equality.scoring.csvio import read_rows
    return read_rows(str(path))


def lure_kind(kind: str) -> str:
    """Map a practice 'kind' string to NONE / TEMP / WM / SP / DET / VAR."""
    k = (kind or "").lower()
    if "watermark" in k:
        return "WM"
    if "system" in k or "persona" in k:
        return "SP"
    if "greedy" in k or "beam" in k or "determin" in k:
        return "DET"
    if "temperature" in k or "sampling" in k or "decod" in k:
        return "TEMP"
    if k.strip() in ("plain", "base", ""):
        return "NONE"
    return "VAR"
