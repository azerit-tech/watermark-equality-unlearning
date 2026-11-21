#!/usr/bin/env python3
"""Lure evidence per endpoint, family by family, without any model (read-only on the ledger).

    python3.12 -m model_equality.analysis.lures_ep                      # ep-, families from analysis/families_ep.csv
    python3.12 -m model_equality.analysis.lures_ep --prefix ref-        # the bench, one family
    python3.12 -m model_equality.analysis.lures_ep --out FILE.csv

Three signals, each compared INSIDE the family (the family median is the reference, never an absolute value):

* ``rare`` / ``rare_z``: share of an endpoint's word tokens that no other endpoint of the family uses on the
  same prompt ("private words"), on equal numbers of responses per prompt; robust z (median / MAD of the others).
  Bench (69 free responses per endpoint): temperature 1.0 = 0.384 (z +8.4), LoRA 1.5 = 0.282 (z +3.8), the seven
  others 0.171 to 0.208 (|z| < 1).  A high value means "higher temperature OR a strong weight change": it does
  not tell which, the local replay (T_mle) does.
* ``rep_share`` / ``rep_excess``: answers to the same prompt that are identical or share a 100-character prefix
  (labels.repeat_stats).  Bench: watermark 0.149 against 0.000 to 0.078.
* ``id_name``: share of identity answers (E01 / D01) that give the family's usual name (Qwen, Llama, SmolLM ...);
  ``leak``: an excerpt when a leak answer (D03) returns something else than NOTHING or the prompt itself.
  A persona or a system prompt replaces the name and often leaks.

The thresholds printed in the last column are hand-set on the single bench family: suggestions to review, not
measurements.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .data import DEFAULT_LEDGER, load_dataset, load_probes
from .labels import repeat_stats

HERE = Path(__file__).parent
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# probes whose answers are not free prose (digits, forms) or are written in another language than the prompt set
NOT_PROSE = ("E02_pi_digits", "E03_story_zh", "D03_leak_repeat", "D11_token_boundary", "D13_year", "D14_sum")
NAMES = ("qwen", "llama", "smollm", "mistral", "gemma", "phi", "olmo")
RARE_Z, REP_EXCESS = 6.0, 0.08


def private_word_share(ds, eps: list, seed: int = 0, min_n: int = 1) -> dict:
    """endpoint -> (mean share of private words, number of responses used)."""
    rng = random.Random(seed)
    per = defaultdict(list)
    pids = sorted({p for e in eps for p in ds.responses.get(e, {})})
    for pid in pids:
        pr = ds.probes[pid]
        if pr.ptype != "free" or pid in NOT_PROSE or pid.endswith("@tree"):
            continue
        by = {e: defaultdict(list) for e in eps}
        for e in eps:
            for r in ds.responses.get(e, {}).get(pid, []):
                if r.text:
                    by[e][r.max_tokens].append(r.text)
        for mt in sorted({m for e in eps for m in by[e]}, key=lambda x: (x is None, x)):
            n = min(len(by[e].get(mt, [])) for e in eps)
            if n < min_n:
                continue
            toks = {e: [[w.lower() for w in WORD.findall(t)] for t in rng.sample(by[e][mt], n)] for e in eps}
            df = Counter()
            for e in eps:
                for w in {w for t in toks[e] for w in t}:
                    df[w] += 1
            for e in eps:
                for t in toks[e]:
                    if len(t) >= 8:
                        per[e].append(sum(df[w] == 1 for w in t) / len(t))
    return {e: ((statistics.mean(per[e]) if per[e] else float("nan")), len(per[e])) for e in eps}


def robust_z(vals: dict) -> dict:
    out = {}
    for e, v in vals.items():
        others = [x for k, x in vals.items() if k != e and x == x]
        if len(others) < 3 or v != v:
            out[e] = float("nan")
            continue
        med = statistics.median(others)
        mad = statistics.median(abs(x - med) for x in others) * 1.4826 or 1e-9
        out[e] = (v - med) / mad
    return out


def identity(ds, e: str) -> tuple:
    """(share of identity answers naming a known model family, number of answers, most frequent other name)."""
    texts = [r.text for p in ("E01_identity_long", "D01_identity") for r in ds.responses.get(e, {}).get(p, []) if r.text]
    if not texts:
        return float("nan"), 0, ""
    named = sum(any(n in t.lower() for n in NAMES) for t in texts)
    other = Counter()
    for t in texts:
        if not any(n in t.lower() for n in NAMES):
            m = re.search(r"(?:I am|I'm|name is|Name:\**)\s+\**([A-Z][\w-]+(?: [A-Z][\w-]+)?)", t)
            other[m.group(1) if m else "(no name)"] += 1
    return named / len(texts), len(texts), (other.most_common(1)[0][0] if other else "")


def leak(ds, e: str) -> str:
    for r in ds.responses.get(e, {}).get("D03_leak_repeat", []):
        t = re.sub(r"</?(begin|end)>", " ", r.text or "", flags=re.I).strip()
        if len(t) > 40 and "NOTHING" not in t[:20] and not t.lower().startswith("repeat all the text"):
            return re.sub(r"\s+", " ", t)[:90]
    return ""


def table(ds, families: dict) -> list:
    rows = []
    for fam in sorted(families):
        eps = sorted(families[fam])
        rare = private_word_share(ds, eps)
        rz = robust_z({e: rare[e][0] for e in eps})
        rep = repeat_stats(ds, eps)
        for e in eps:
            idn, nid, other = identity(ds, e)
            lk = leak(ds, e)
            hints = []
            if rz[e] == rz[e] and rz[e] >= RARE_Z:
                hints.append("TEMP-or-strong-variant")
            if rep[e]["rep_flag"] or (rep[e]["rep_pairs"] >= 20 and rep[e]["rep_excess"] >= REP_EXCESS):
                hints.append("WM-or-low-T")
            if nid >= 2 and idn <= 0.34:
                hints.append("SP")
            if lk:
                hints.append("leak")
            rows.append({"family": fam, "endpoint": e, "rare": round(rare[e][0], 3), "rare_n": rare[e][1],
                         "rare_z": round(rz[e], 1), "rep_pairs": rep[e]["rep_pairs"],
                         "rep_share": round(rep[e]["rep_share"], 3), "rep_excess": round(rep[e]["rep_excess"], 3),
                         "id_name": round(idn, 2), "id_n": nid, "id_other": other, "leak": lk,
                         "hints": " ".join(hints)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefix", default="ep-")
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    ap.add_argument("--families", default=str(HERE / "families_ep.csv"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ds = load_dataset(a.ledger, load_probes(), a.prefix + "*")
    if a.prefix == "ep-":
        fam = {r[0]: r[1] for r in csv.reader(open(a.families, encoding="utf-8-sig")) if r and r[0] != "endpoint"}
    else:
        fam = {e: a.prefix.rstrip("-") for e in ds.endpoints}
    families = defaultdict(list)
    for e in ds.endpoints:
        families[fam.get(e, "?")].append(e)
    rows = table(ds, families)
    cols = list(rows[0]) if rows else []
    print(" ".join(f"{c:>10s}" if c not in ("leak", "hints", "id_other") else c for c in cols))
    for r in rows:
        print(" ".join(f"{str(r[c]):>10s}" if c not in ("leak", "hints", "id_other") else str(r[c]) + " |" for c in cols))
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
