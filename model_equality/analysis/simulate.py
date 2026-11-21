"""Budget simulation on ref- (METHOD 6, R3).

Replays pairs -> labels -> decision on ``draws`` sub-budgets of the stock, each
endpoint receiving ``battery[probe]`` responses per probe (default: the budget,
100, split evenly over the probes of the selected units).  Draws are disjoint
when the stock allows it (slice d of a fixed shuffle), otherwise random
subsamples (flagged ``overlap``).  Each partition is scored with
``model_equality.scoring.metrics.score`` against the truth (one family).

pi_fp8 = share of draws where z* of the (plain, quantized) pair exceeds 4.
"""
from __future__ import annotations

import math

import numpy as np

from model_equality.scoring.metrics import score

from .data import lure_kind
from .decision import decide
from .labels import PairIndex
from .pairs import compute_pairs


def default_battery(units: dict, probes: dict, budget: int = 100) -> dict:
    pids = [p for u in units.values() for p in u if probes[p].discrete]
    if not pids:
        return {}
    k = max(1, budget // len(pids))
    return {p: k for p in pids}


def parse_battery(spec: str) -> dict:
    out = {}
    for it in spec.split(","):
        if it.strip():
            p, n = it.split(":")
            out[p.strip()] = int(n)
    return out


def draw_budgets(parsed: dict, battery: dict, draws: int, rng) -> tuple:
    """List of index subsets (draw -> endpoint -> probe -> [indices]) and an overlap flag."""
    order = {}
    overlap = False
    for e, d in parsed.items():
        for p, k in battery.items():
            lst = d.get(p, [])
            order[(e, p)] = rng.permutation(len(lst))
            if len(lst) < k * draws:
                overlap = True
    subs = []
    for di in range(draws):
        sub = {}
        for e, d in parsed.items():
            sub[e] = {}
            for p, k in battery.items():
                lst = d.get(p, [])
                if not lst:
                    continue
                idx = order[(e, p)]
                if len(lst) >= k * draws:
                    sel = idx[di * k:(di + 1) * k]
                else:
                    sel = rng.choice(len(lst), size=min(k, len(lst)), replace=False)
                sub[e][p] = sorted(int(i) for i in sel)
        subs.append(sub)
    return subs, overlap


def run_simulation(ds, parsed: dict, truth: dict, units: dict, battery: dict, draws: int = 10, B: int = 200,
                   seed: int = 0, jobs: int = 1, n_boot: int = 20, pi_fp8=None, min_cell=None) -> tuple:
    eps = [e for e in ds.endpoints if e in truth]
    rng = np.random.default_rng(seed)
    parsed = {e: parsed[e] for e in eps if e in parsed}
    subs, overlap = draw_budgets(parsed, battery, draws, rng)
    used_units = {u: [p for p in ps if p in battery] for u, ps in units.items()}
    used_units = {u: ps for u, ps in used_units.items() if ps}
    kind = {e: lure_kind(truth[e][2]) for e in eps}
    plain = [e for e in eps if kind[e] == "NONE"]
    quant = [e for e in eps if "quant" in truth[e][2].lower()]
    rows = []
    kw = {} if min_cell is None else {"min_cell": min_cell}
    for di, sidx in enumerate(subs):
        sub = {e: {p: [parsed[e][p][i] for i in ix] for p, ix in d.items()} for e, d in sidx.items()}
        prows, _ = compute_pairs(sub, ds.probes, eps, used_units, B=B, seed=seed * 7919 + di, jobs=jobs,
                                 n_boot=n_boot, per_unit=False, **kw)
        dsub = ds.subset(eps)   # the decision reads ds (DET, leaks): restrict it to the draw
        dsub.responses = {e: {p: [ds.responses[e][p][i] for i in ix] for p, ix in d.items()}
                          for e, d in sidx.items()}
        dsub.other = {}
        res = decide(eps, prows, dsub, sub, pi_fp8=pi_fp8)
        tf = {e: truth[e][0] for e in eps}
        tm = {e: truth[e][1] for e in eps}
        sc = score(res["family"], res["model"], tf, tm, "true")
        fam = next(iter(res["families"].values()))
        pidx = PairIndex(prows)
        row = {"draw": di, "mcc_model": sc["mcc_model"], "mcc_family": sc["mcc_family"], "score": sc["score"],
               "base": fam["base"], "groups": " | ".join(" ".join(g) for g in fam["groups"]),
               "labels": " ".join(f"{e[-2:]}:{fam['label'][e]}" for e in eps),
               "wrong_pairs": " ".join(f"{a[-2:]}-{b[-2:]}{k}" for a, b, k in
                                       sc["model_confusion"]["true"]["false_pairs"])}
        for a in plain[:1]:
            for q in quant[:1]:
                r = pidx.get(a, q)
                row["z_temp_plain_quant"] = r.get("z_temp") if r else float("nan")
        rows.append(row)
    m = np.array([r["mcc_model"] for r in rows])
    zq = np.array([r.get("z_temp_plain_quant", float("nan")) for r in rows], float)
    summary = {"draws": len(rows), "overlap": overlap, "battery_per_endpoint": sum(battery.values()),
               "mcc_model_mean": float(m.mean()), "mcc_model_sd": float(m.std(ddof=1)) if len(m) > 1 else 0.0,
               "mcc_model_min": float(m.min()), "mcc_model_max": float(m.max()),
               "share_below_0.5": float(np.mean(m < 0.5)),
               "pi_fp8": float(np.mean(zq > 4)) if np.isfinite(zq).any() else float("nan"),
               "z_plain_quant_median": float(np.nanmedian(zq)) if np.isfinite(zq).any() else float("nan")}
    return rows, summary
