"""Probe selection on ref- (METHOD 6, R2 criterion).

For every unit and every draw (subsample of ``n_sub`` responses per endpoint
and probe), per-unit z of the pairs that matter:
  * the "same model" pairs (truth), with the planned statistic: z_inv when one
    endpoint is the watermark lure, z_mmd for free probes, else z_temp;
  * the hard pairs (default (01,02) fp8, (01,09) LoRA 0.3, (09,04));
  * control: two random halves of the same endpoint (z must be centred on 0).
A unit is kept if
  1. conformity: parsable rate >= 85 % on every endpoint (70 % for persona,
     60 % for streams),
  2. safety: z <= 2.5 in >= 90 % of the draws for every same pair; a unit that
     fails only on system-prompt / persona pairs is kept with ``safe_except=SP``
     (SP endpoints are judged on long probes, METHOD 9.1),
  3. sensitivity: z > 3 in >= 50 % of the draws for at least one hard pair,
  4. control: |mean z| <= 1 on the split halves (when available).
Only these pairs are computed (13 + controls instead of 36).

Low volume: when an endpoint has fewer than ``n_sub`` responses to a probe,
all of them are used and the draws differ only by the permutations; the table
reports the responses actually used and ``low_power``.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .data import lure_kind
from .extract import conformity_rows
from .pairs import compute_pairs
from .stats import subsample_parsed

SAFE_Z, SAFE_FRAC = 2.5, 0.90
SENS_Z, SENS_FRAC = 3.0, 0.50
CONF_MIN, CONF_MIN_PERSONA = 0.85, 0.70
CONF_MIN_STREAM = 0.60   # METHOD rev. 17:45: streams usable at 60 % when valid items are many
DEFAULT_HARD = (("ref-01", "ref-02"), ("ref-01", "ref-09"), ("ref-09", "ref-04"))


def planned_z(row: dict, kind_a: str, kind_b: str, free: bool) -> float:
    if free:
        return float(row.get("z_mmd", float("nan")))
    if "WM" in (kind_a, kind_b):
        return float(row.get("z_inv", float("nan")))
    return float(row.get("z_temp", float("nan")))


def _split_halves(parsed: dict, eps: list, rng) -> tuple:
    out, pairs = {}, []
    for e in eps:
        A, B = {}, {}
        for pid, lst in parsed.get(e, {}).items():
            if len(lst) < 4:
                continue
            idx = rng.permutation(len(lst))
            h = len(lst) // 2
            A[pid] = [lst[i] for i in idx[:h]]
            B[pid] = [lst[i] for i in idx[h:2 * h]]
        if A:
            out[f"{e}#A"], out[f"{e}#B"] = A, B
            pairs.append((f"{e}#A", f"{e}#B"))
    return out, pairs


def run_selection(ds, parsed: dict, truth: dict, units: dict, n_sub: int = 12, draws: int = 20, B: int = 200,
                  seed: int = 0, jobs: int = 1, hard=DEFAULT_HARD, min_cell=None, control: bool = True) -> tuple:
    eps = [e for e in ds.endpoints if e in truth]
    kind = {e: lure_kind(truth[e][2]) for e in eps}
    same = [(a, b) for a, b in itertools.combinations(eps, 2) if truth[a][:2] == truth[b][:2]]
    hard = [(a, b) for a, b in hard if a in eps and b in eps]
    rng = np.random.default_rng(seed)
    kw = {} if min_cell is None else {"min_cell": min_cell}
    draw_rows = []
    for d in range(draws):
        sub = subsample_parsed(parsed, n_sub, rng, endpoints=set(eps))
        pairs = same + hard
        if control:
            halves, cpairs = _split_halves(sub, eps, rng)
            sub = sub | halves
            pairs = pairs + cpairs
        _, urows = compute_pairs(sub, ds.probes, list(sub), units, B=B, seed=seed * 1000 + d, jobs=jobs,
                                 n_boot=0, pairs=pairs, **kw)
        for r in urows:
            a, b = r["a"], r["b"]
            role = "control" if "#" in a else ("same" if (a, b) in same else "hard")
            free = r.get("n_units", 0) == 0
            ka, kb = kind.get(a.split("#")[0], "NONE"), kind.get(b.split("#")[0], "NONE")
            draw_rows.append({"draw": d, "unit": r["unit"], "a": a, "b": b, "role": role, "free": free,
                              "kind_a": ka, "kind_b": kb, "z_planned": planned_z(r, ka, kb, free),
                              "z_raw": r.get("z_raw"), "z_temp": r.get("z_temp"), "z_inv": r.get("z_inv"),
                              "z_mmd": r.get("z_mmd"), "cells_temp": r.get("cells_temp"),
                              "n_resp_a": r.get("n_resp_X"), "n_resp_b": r.get("n_resp_Y"),
                              "low_power": r.get("low_power")})
    ranked = summarize_selection(ds, parsed, truth, units, draw_rows, same, hard, kind)
    return ranked, draw_rows


def summarize_selection(ds, parsed, truth, units, draw_rows, same, hard, kind) -> list:
    conf = conformity_rows(ds, parsed)
    by_unit: dict = {}
    for r in draw_rows:
        by_unit.setdefault(r["unit"], []).append(r)
    out = []
    for uid, pids in units.items():
        rows = by_unit.get(uid, [])
        # conformity per endpoint over the unit's probes
        cr = [c for c in conf if c["unit"] == uid and c["endpoint"] in truth]
        per_ep = {}
        for c in cr:
            n, k = per_ep.get(c["endpoint"], (0, 0))
            per_ep[c["endpoint"]] = (n + c["n_responses"], k + c["n_parsable"])
        rates = {e: k / n for e, (n, k) in per_ep.items() if n}
        is_stream = ds.probes[pids[0]].ptype in ("stream", "open_list")
        thr = lambda e: min(CONF_MIN_STREAM if is_stream else CONF_MIN,  # noqa: E731
                            CONF_MIN_PERSONA if "persona" in truth[e][2].lower() else 1.0)
        conform_ok = bool(rates) and all(v >= thr(e) for e, v in rates.items())
        res = {"unit": uid, "probes": " ".join(pids), "category": ds.probes[pids[0]].category,
               "long": all(ds.probes[p].long for p in pids),
               "conf_min": min(rates.values()) if rates else float("nan"),
               "conf_worst": min(rates, key=rates.get) if rates else "", "conform_ok": conform_ok}
        # safety
        fails, safe_fracs, fail_pairs = set(), [], []
        for a, b in same:
            z = np.array([r["z_planned"] for r in rows if r["a"] == a and r["b"] == b], float)
            if len(z) == 0:
                continue
            frac = float(np.mean(np.where(np.isfinite(z), z <= SAFE_Z, True)))
            safe_fracs.append(frac)
            if frac < SAFE_FRAC:
                ks = {kind[a], kind[b]}
                # a pair with an SP endpoint is blamed on SP (it is judged on long probes, 9.1)
                fails |= {"SP"} if "SP" in ks else (ks - {"NONE"} or {"NONE"})
                fail_pairs.append(f"{a[-2:]}-{b[-2:]}")
        res["safe_min_frac"] = min(safe_fracs) if safe_fracs else float("nan")
        res["safe_fail_lures"] = ",".join(sorted(fails))
        res["safe_fail_pairs"] = " ".join(fail_pairs)
        res["safe_ok"] = bool(safe_fracs) and not fails
        res["safe_except"] = "SP" if fails and fails <= {"SP"} else ""
        # sensitivity
        best, best_pair, med = 0.0, "", []
        for a, b in hard:
            z = np.array([r["z_planned"] for r in rows if r["a"] == a and r["b"] == b], float)
            if len(z) == 0:
                continue
            frac = float(np.mean(np.where(np.isfinite(z), z > SENS_Z, False)))
            res[f"sens_{a[-2:]}_{b[-2:]}"] = frac
            res[f"zmed_{a[-2:]}_{b[-2:]}"] = float(np.nanmedian(z)) if np.isfinite(z).any() else float("nan")
            if np.isfinite(z).any():
                med.append(float(np.nanmedian(z)))
            if frac > best:
                best, best_pair = frac, f"{a}/{b}"
        res["sens_max"] = best
        res["sens_pair"] = best_pair
        res["sens_ok"] = best >= SENS_FRAC
        res["zmed_hard"] = max(med) if med else float("nan")
        cz = np.array([r["z_planned"] for r in rows if r["role"] == "control"], float)
        res["control_mean_z"] = float(np.nanmean(cz)) if np.isfinite(cz).any() else float("nan")
        res["control_ok"] = not (np.isfinite(res["control_mean_z"]) and abs(res["control_mean_z"]) > 1.0)
        nr = [r["n_resp_a"] for r in rows if r["role"] != "control" and r["n_resp_a"]]
        res["resp_per_ep"] = float(np.median(nr)) if nr else 0.0
        lp = [bool(r["low_power"]) for r in rows if r["role"] != "control"]
        res["low_power_share"] = float(np.mean(lp)) if lp else 1.0
        res["kept"] = bool(conform_ok and (res["safe_ok"] or res["safe_except"] == "SP") and res["sens_ok"]
                           and res["control_ok"])
        out.append(res)
    out.sort(key=lambda r: (not r["kept"], not (r["conform_ok"] and (r["safe_ok"] or r["safe_except"])),
                            -r["sens_max"], -(r["zmed_hard"] if math.isfinite(r["zmed_hard"]) else -99)))
    return out
