"""All pairwise statistics for a set of endpoints (METHOD 5.2, 5.4 signatures, 8.3).

``compute_pairs(parsed, probes, endpoints, units, B, seed, jobs)`` returns
(pair_rows, unit_rows): one row per pair with aggregated z (all units, short
units, long units), kappa, watermark signature and a bootstrap interval of
ln kappa; one row per (pair, unit) with the per-unit z.
"""
from __future__ import annotations

import itertools
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .stats import MIN_CELL, PairTables, t_temp_inv, _build_discrete

WM_DELTA_MIN = 1.0
WM_RESID_MAX = 0.20
WM_MIN_OPTIONS = 8
WM_MIN_OPT_BLOCK = 4
WM_STABILITY = 0.30


# --------------------------------------------------------------- WM signature
def _two_means(x: np.ndarray) -> tuple:
    """1-D 2-means by exhaustive split of the sorted values: (delta, within_ss, total_ss)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    tot = ((x - x.mean()) ** 2).sum()
    best = (0.0, tot)
    for k in range(1, n):
        a, b = x[:k], x[k:]
        w = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if w < best[1]:
            best = (b.mean() - a.mean(), w)
    return best[0], best[1], tot


def wm_signature(inv_result) -> dict:
    """Two-valued per-option offsets b_Y - kappa b_X (METHOD 5.4), from a T_inv fit.

    Per wm_block with >= 4 options (each seen in >= 2 kept contexts): centre the
    offsets, 1-D 2-means -> delta_b, residual share r_b.  A block "splits" if
    delta_b >= 1 and r_b < 0.20.  Signature: the splitting blocks hold >= 8
    options and >= half of the informative options, and their delta_b agree
    within +-30 %.  (Choice: blocks where the watermark has pushed the red
    options below the common-support threshold cannot split and are not
    counted against the signature unless they hold most of the options.)
    """
    r = inv_result
    out = {"wm_sig": False, "wm_delta": float("nan"), "wm_resid": float("nan"), "wm_options": 0,
           "wm_stable": False, "wm_blocks": 0}
    if r is None or r.bX is None:
        return out
    kap = float(r.kappa[0])
    d = r.bY - kap * r.bX
    seen = r.opt_seen >= 2
    info_opts = 0
    split = []   # (delta_b, within, total, n_b)
    for blk in np.unique(r.block):
        sel = (r.block == blk) & seen
        nb = int(sel.sum())
        if nb < WM_MIN_OPT_BLOCK:
            continue
        info_opts += nb
        x = d[sel] - d[sel].mean()
        dl, w, t = _two_means(x)
        if t > 0 and dl >= WM_DELTA_MIN and w / t < WM_RESID_MAX:
            split.append((dl, w, t, nb))
    out["wm_options"] = info_opts
    if not split:
        return out
    n_split = sum(s_[3] for s_ in split)
    delta = float(np.average([s_[0] for s_ in split], weights=[s_[3] for s_ in split]))
    resid = sum(s_[1] for s_ in split) / sum(s_[2] for s_ in split)
    stable = all(abs(s_[0] - delta) <= WM_STABILITY * delta for s_ in split)
    out.update(wm_delta=delta, wm_resid=float(resid), wm_stable=bool(stable), wm_blocks=len(split))
    out["wm_sig"] = bool(n_split >= WM_MIN_OPTIONS and n_split >= 0.5 * info_opts and stable)
    return out


# --------------------------------------------------------------- kappa bootstrap
def kappa_interval(parsed, X, Y, probes, units, n_boot: int, rng, min_cell=MIN_CELL, level=0.95) -> tuple:
    """Percentile interval of ln kappa_hat (T_temp), resampling responses within probes."""
    if n_boot <= 0:
        return float("nan"), float("nan")
    ks = []
    for _ in range(n_boot):
        dus = []
        for uid, pids in units.items():
            if not probes[pids[0]].discrete:
                continue
            ex, ey = {}, {}
            for p in pids:
                lx = parsed.get(X, {}).get(p, [])
                ly = parsed.get(Y, {}).get(p, [])
                ex[p] = [lx[i].events for i in rng.integers(0, len(lx), len(lx))] if lx else []
                ey[p] = [ly[i].events for i in rng.integers(0, len(ly), len(ly))] if ly else []
            if not any(ex.values()) or not any(ey.values()):
                continue
            u = _build_discrete(uid, False, pids, ex, ey, 0, rng)
            if u is not None:
                dus.append(u)
        if not dus:
            return float("nan"), float("nan")
        r = t_temp_inv(dus, "temp", min_cell=min_cell)
        if r.n_cells:
            ks.append(math.log(r.kappa[0]))
    if len(ks) < 5:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(ks, a)), float(np.quantile(ks, 1 - a))


# --------------------------------------------------------------- one pair
_G: dict = {}


def _init(parsed, probes, units, cfg):
    _G.update(parsed=parsed, probes=probes, units=units, cfg=cfg)


def _clean(d: dict, prefix: str = "") -> dict:
    return {prefix + k: v for k, v in d.items() if not k.startswith("_")}


def pair_task(args) -> tuple:
    X, Y, seed = args
    parsed, probes, units, cfg = _G["parsed"], _G["probes"], _G["units"], _G["cfg"]
    rng = np.random.default_rng(seed)
    pt = PairTables.build(parsed, X, Y, probes, units, cfg["B"], rng, free=cfg.get("free", True))
    mc = cfg.get("min_cell", MIN_CELL)
    allu = pt.stat(min_cell=mc)
    row = {"a": X, "b": Y}
    row.update(_clean(allu))
    short = pt.stat(pt.units(long=False), which=("temp", "inv"), min_cell=mc)
    long = pt.stat(pt.units(long=True), which=("temp", "inv"), min_cell=mc)
    for k in ("z_temp", "z_inv", "kappa_temp", "cells_temp", "log10_splits", "low_power"):
        row[f"{k}_short"] = short.get(k, float("nan"))
        row[f"{k}_long"] = long.get(k, float("nan"))
    row.update(wm_signature(allu.get("_inv")))
    lo, hi = kappa_interval(parsed, X, Y, probes, units, cfg.get("n_boot", 0), rng, mc)
    row["lnk_lo"], row["lnk_hi"] = lo, hi
    urows = []
    if cfg.get("per_unit", True):
        for uid in pt.units():
            s = pt.stat([uid], min_cell=mc)
            urows.append({"a": X, "b": Y, "unit": uid, "long": uid in pt.units(long=True)} | _clean(s))
    return row, urows


def compute_pairs(parsed: dict, probes: dict, endpoints: list, units: dict, B: int = 500, seed: int = 0,
                  jobs: int = 1, n_boot: int = 30, min_cell: int = MIN_CELL, per_unit: bool = True,
                  pairs: list | None = None, free: bool = True) -> tuple:
    cfg = {"B": B, "n_boot": n_boot, "min_cell": min_cell, "per_unit": per_unit, "free": free}
    pairs = list(itertools.combinations(sorted(endpoints), 2)) if pairs is None else pairs
    tasks = [(a, b, seed * 100003 + i) for i, (a, b) in enumerate(pairs)]
    rows, urows = [], []
    if jobs <= 1 or len(tasks) <= 1:
        _init(parsed, probes, units, cfg)
        res = map(pair_task, tasks)
    else:
        try:   # fork: no re-import of __main__, no pickling of the data
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = None
        ex = ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(parsed, probes, units, cfg),
                                 mp_context=ctx)
        res = ex.map(pair_task, tasks, chunksize=max(1, len(tasks) // (4 * jobs)))
    for r, u in res:
        rows.append(r)
        urows.extend(u)
    if jobs > 1 and len(tasks) > 1:
        ex.shutdown()
    return rows, urows


def default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))
