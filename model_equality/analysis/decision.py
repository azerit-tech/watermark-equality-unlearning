"""From pair statistics to a partition (METHOD 9).

Per family:
1. provisional reference = non-DET endpoint with the smallest sum of z_temp to
   the others; labels relative to it (labels.label_endpoint);
2. base candidate (9.2) = NONE endpoint with the smallest sum of z* to the
   lure-labelled endpoints; labels recomputed relative to it (iterated, <= 3x);
3. z* per pair (9.1), weight table + H-struct priors (9.3), pi_fp8 rule;
4. partition maximising the sum of within-group weights (exact by subset DP up
   to 12 endpoints, else average-linkage seed + single-endpoint moves);
5. gray pairs (9.4) with the reserve priority 1..4.

A pair without statistic (no common data) gets weight 0; an endpoint without
any usable pair statistic (and not DET) is forced alone and reported in
``no_data`` (never merged through a weight-0 tie).
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np

from .labels import (PairIndex, Z_DIFF, Z_SAME, det_flags, identity_drift, label_endpoint, modal_agreement,
                     repeat_stats, _f)

LURES = ("DET", "WM", "SP", "TEMP")
NO_DATA_WEIGHT = -1e6
WEIGHT_TABLE = ((2.0, 2.0), (3.0, 0.5), (4.0, -0.5), (6.0, -2.0), (math.inf, -4.0))


def weight_of_z(z: float) -> float:
    if not math.isfinite(z):
        return 0.0
    for hi, w in WEIGHT_TABLE:
        if z <= hi:
            return w
    return -4.0


def z_star(row: dict | None, la: str, lb: str) -> tuple:
    """(z*, which statistic) for a pair of sampled endpoints (METHOD 9.1)."""
    if row is None:
        return float("nan"), "none"
    if la == "DET" or lb == "DET":
        return float("nan"), "det"
    if la == "SP" or lb == "SP":
        z = _f(row.get("z_temp_long"))
        if math.isfinite(z):
            if z > Z_SAME and bool(row.get("wm_sig")):
                zi = _f(row.get("z_inv_long"))
                if math.isfinite(zi):
                    return zi, "inv_long"
            return z, "temp_long"
        return _f(row.get("z_temp")), "temp(no long)"
    z = _f(row.get("z_temp"))
    if math.isfinite(z) and z > Z_SAME and bool(row.get("wm_sig")):
        return _f(row.get("z_inv")), "inv"
    return z, "temp"


# ------------------------------------------------------------------ optimiser
def best_partition(W: np.ndarray) -> tuple:
    """Partition of range(n) maximising the sum of W[i,j] over pairs in a group."""
    n = W.shape[0]
    if n == 0:
        return [], 0.0
    if n <= 12:
        return _dp_partition(W)
    return _heuristic_partition(W)


def _dp_partition(W):
    n = W.shape[0]
    full = (1 << n) - 1
    gain = np.zeros(1 << n)
    for m in range(1, 1 << n):
        low = (m & -m).bit_length() - 1
        rest = m & ~(1 << low)
        s = 0.0
        r = rest
        while r:
            b = (r & -r).bit_length() - 1
            s += W[low, b]
            r &= r - 1
        gain[m] = gain[rest] + s

    @lru_cache(maxsize=None)
    def f(mask):
        if mask == 0:
            return 0.0, ()
        low = mask & -mask
        rest = mask & ~low
        best = (-math.inf, ())
        sub = rest
        while True:
            g = sub | low
            v, parts = f(mask & ~g)
            v += gain[g]
            # tie-break: fewer groups (merging on exact ties)
            if v > best[0] + 1e-12 or (abs(v - best[0]) <= 1e-12 and len(parts) + 1 < len(best[1])):
                best = (v, (g,) + parts)
            if sub == 0:
                break
            sub = (sub - 1) & rest
        return best

    v, parts = f(full)
    groups = [[i for i in range(n) if g >> i & 1] for g in parts]
    return groups, float(v)


def _score(W, lab):
    s = 0.0
    n = len(lab)
    for i in range(n):
        for j in range(i + 1, n):
            if lab[i] == lab[j]:
                s += W[i, j]
    return s


def _heuristic_partition(W):
    n = W.shape[0]
    groups = [[i] for i in range(n)]
    while True:   # agglomerate while the best merge (average weight) is positive
        best = None
        for a, b in itertools.combinations(range(len(groups)), 2):
            gain = sum(W[i, j] for i in groups[a] for j in groups[b])
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, a, b)
        if best is None:
            break
        _, a, b = best
        groups[a] += groups[b]
        del groups[b]
    lab = np.zeros(n, int)
    for k, g in enumerate(groups):
        lab[g] = k
    improved = True
    while improved:
        improved = False
        for i in range(n):
            cur = _score(W, lab)
            for k in list(set(lab)) + [lab.max() + 1]:
                if k == lab[i]:
                    continue
                old = lab[i]
                lab[i] = k
                if _score(W, lab) > cur + 1e-9:
                    improved = True
                    cur = _score(W, lab)
                else:
                    lab[i] = old
    groups = [list(np.flatnonzero(lab == k)) for k in sorted(set(lab))]
    return groups, _score(W, lab)


# ------------------------------------------------------------------ family decision
def decide_family(eps: list, pidx: PairIndex, ds, parsed: dict, det: dict | None = None,
                  pi_fp8: float | None = None, prior: float = 1.0, forced: dict | None = None,
                  no_data: str = "alone") -> dict:
    """``forced``: endpoint -> label (DET / WM / SP / TEMP / NONE) imposed by hand, e.g. a persona read in the
    leak answers; it replaces the computed label (the computed criteria stay in ``labels`` for review)."""
    forced = {e: l for e, l in (forced or {}).items() if e in eps}
    eps = sorted(eps)
    det = det_flags(ds) if det is None else det
    sampled = [e for e in eps if not det.get(e, {}).get("det")]
    ident = identity_drift(ds, eps)
    rep = repeat_stats(ds, eps)

    def sum_z(e, others, key="z_temp"):
        s = 0.0
        for o in others:
            if o == e:
                continue
            r = pidx.get(e, o)
            z = _f(r.get(key)) if r else float("nan")
            s += z if math.isfinite(z) else 0.0
        return s

    ref = min(sampled, key=lambda e: sum_z(e, sampled)) if sampled else None
    labels = {}
    for _ in range(3):
        labels = {e: label_endpoint(e, ref, pidx, ds, det, ident=ident, family=eps, rep=rep) for e in eps}
        for e, l in forced.items():
            labels[e]["criteria"].append(f"forced:{l}")
            labels[e]["label"] = l
        lab = {e: labels[e]["label"] for e in eps}
        lures = [e for e in eps if lab[e] in LURES]
        nones = [e for e in sampled if lab[e] == "NONE"]
        if not lures or not nones:
            base = None
            break

        def zs_to_lures(e):
            s = 0.0
            for l in lures:
                if l == e or lab[l] == "DET":
                    continue
                z, _ = z_star(pidx.get(e, l), lab[e], lab[l])
                s += z if math.isfinite(z) else 0.0
            return s
        base = min(nones, key=zs_to_lures)
        if base == ref:
            break
        ref = base
    lab = {e: labels[e]["label"] for e in eps}
    if base is not None and base not in forced:
        lab[base] = "NONE"

    # modal agreement for DET endpoints
    agree = {}
    for g in eps:
        if lab[g] != "DET":
            continue
        for x in sampled:
            agree[(g, x)] = modal_agreement(parsed, ds.probes, g, x)

    n = len(eps)
    W = np.zeros((n, n))
    prow = []
    for i, j in itertools.combinations(range(n), 2):
        a, b = eps[i], eps[j]
        la, lb = lab[a], lab[b]
        row = pidx.get(a, b)
        z, which = z_star(row, la, lb)
        w = weight_of_z(z)
        notes = []
        is_lure = lambda e: lab[e] in LURES  # noqa: E731
        is_base_or_lure = lambda e: e == base or is_lure(e)  # noqa: E731
        if la == "DET" or lb == "DET":
            g, x = (a, b) if la == "DET" else (b, a)
            if lab[x] == "DET":
                same = _det_identical(ds, g, x)
                w = prior + (2.0 if same else 0.0)
                notes.append(f"DET-DET identical={same}")
            else:
                to_base = is_base_or_lure(x)
                w = prior if to_base else -2.0
                if base is not None and x != base and not to_base:
                    A_x, _ = agree.get((g, x), (float("nan"), 0))
                    A_b, _ = agree.get((g, base), (float("nan"), 0))
                    if math.isfinite(A_x) and math.isfinite(A_b) and A_x - A_b >= 0.15:
                        w = prior
                        notes.append(f"modal agreement favours variant ({A_x:.2f} vs base {A_b:.2f})")
                elif base is not None and x == base:
                    best_var = max((agree.get((g, v), (float('nan'), 0))[0] - agree.get((g, base), (float('nan'), 0))[0]
                                    for v in sampled if v != base and lab[v] == "NONE"), default=float("nan"))
                    if math.isfinite(best_var) and best_var >= 0.15:
                        w = -2.0
                        notes.append("modal agreement favours a variant over the base")
        else:
            if la == "NONE" and lb == "NONE" and pi_fp8 is not None and pi_fp8 < 0.5 and math.isfinite(z) and z <= Z_SAME:
                w = 0.5
                notes.append("pi_fp8<0.5: NONE-NONE weak evidence")
            if (is_lure(a) and is_base_or_lure(b)) or (is_lure(b) and is_base_or_lure(a)):
                w += prior
                notes.append("+prior lure-base")
            elif la == "NONE" and lb == "NONE":
                w -= prior
                notes.append("-prior NONE-NONE")
        W[i, j] = W[j, i] = w
        prow.append({"a": a, "b": b, "label_a": la, "label_b": lb, "z_star": z, "stat": which, "weight": w,
                     "notes": "; ".join(notes), "low_power": bool(row.get("low_power")) if row else True})
    # endpoints without any usable pair statistic stay alone (audit point 3): a weight-0 tie
    # must never attach them to a group
    no_data_mode, no_data = no_data, []
    for i, e in enumerate(eps):
        if lab[e] == "DET":
            continue
        has = any(math.isfinite(p["z_star"]) for p in prow if e in (p["a"], p["b"]))
        if not has:
            no_data.append(e)
            # "merge" (explicit choice, first "families only" submission): in doubt, merge (METHOD 1) --
            # an endpoint without statistics joins the group of its family instead of staying alone
            W[i, :] = W[:, i] = prior if no_data_mode == "merge" else NO_DATA_WEIGHT
            W[i, i] = 0.0
    if no_data_mode == "lures":
        # endpoints labelled as lures (by hand or by a criterion) are the base model by definition: they are
        # grouped even without statistics; unlabelled endpoints without statistics stay alone
        for i, a in enumerate(eps):
            for j, b in enumerate(eps):
                if i < j and (a in no_data or b in no_data) and lab[a] in LURES and lab[b] in LURES:
                    W[i, j] = W[j, i] = prior
    for p in prow:
        if p["a"] in no_data or p["b"] in no_data:
            both_lures = no_data_mode == "lures" and lab[p["a"]] in LURES and lab[p["b"]] in LURES
            p["weight"] = prior if (no_data_mode == "merge" or both_lures) else NO_DATA_WEIGHT
            p["notes"] = (p["notes"] + "; " if p["notes"] else "") + (
                "NO_DATA endpoint merged (doubt)" if no_data_mode == "merge" else
                "NO_DATA, two lures grouped" if both_lures else "NO_DATA endpoint kept alone")
    groups, total = best_partition(W)
    model_of = {}
    for k, g in enumerate(groups):
        for i in g:
            model_of[eps[i]] = k
    gray = gray_pairs(prow, lab, base, model_of, pi_fp8)
    return {"endpoints": eps, "labels": labels, "label": lab, "base": base, "ref": ref, "W": W, "no_data": no_data,
            "groups": [[eps[i] for i in g] for g in groups], "objective": total, "pairs": prow,
            "model_of": model_of, "gray": gray}


def _det_identical(ds, g, x) -> bool:
    common = set(ds.responses.get(g, {})) & set(ds.responses.get(x, {}))
    if not common:
        return False
    return all(ds.responses[g][p][0].text == ds.responses[x][p][0].text for p in common)


def gray_pairs(prow: list, lab: dict, base, model_of: dict, pi_fp8=None) -> list:
    out = []
    for r in prow:
        a, b, z = r["a"], r["b"], r["z_star"]
        gray = math.isfinite(z) and Z_SAME < z <= Z_DIFF
        nn_close = lab[a] == "NONE" and lab[b] == "NONE" and math.isfinite(z) and z <= Z_SAME
        if pi_fp8 is not None and 0.5 <= pi_fp8 < 0.8 and nn_close:
            gray = True
        if not gray and not nn_close:
            continue
        if base in (a, b) and (lab[a] == "NONE" and lab[b] == "NONE") and gray:
            prio = 1
        elif base in (a, b) and (lab[a] in LURES or lab[b] in LURES):
            prio = 2
        elif nn_close:
            prio = 3
        elif lab[a] == "NONE" and lab[b] == "NONE":
            prio = 4
        else:
            prio = 2 if (lab[a] in LURES or lab[b] in LURES) else 4
        out.append(r | {"priority": prio, "same_group": model_of[a] == model_of[b],
                        "action": "never spend reserve" if prio == 4 else "reserve: +10 responses each on the 2 best probes"})
    return sorted(out, key=lambda r: (r["priority"], -r["z_star"] if math.isfinite(r["z_star"]) else 0))


def decide(endpoints: list, pair_rows: list, ds, parsed: dict, families: dict | None = None,
           pi_fp8: float | None = None, prior: float = 1.0, forced_labels: dict | None = None,
           no_data: str = "alone", labels_only: bool = False) -> dict:
    """Full decision. families: endpoint -> family label (None: one family).

    ``labels_only`` (ep- campaign, after the first scored submission): the partition is read from the labels
    alone -- in every family, one group = the endpoints labelled as a lure (DET / WM / SP / TEMP) or ``BASE``
    (white-box verdict BASE_OR_LURE), every other endpoint alone.  Pair statistics are not used: on the bench
    and on the twin the black-box z scores do not separate the variants at this budget, and inside a family most
    pairs are "different", so an endpoint we know nothing about is better left alone."""
    pidx = PairIndex(pair_rows)
    det = det_flags(ds)
    fams = families or {e: "F" for e in endpoints}
    by_fam: dict = {}
    for e in sorted(endpoints):
        by_fam.setdefault(fams.get(e, "F"), []).append(e)
    family, model, results = {}, {}, {}
    for fi, (f, eps) in enumerate(sorted(by_fam.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
        res = decide_family(eps, pidx, ds, parsed, det, pi_fp8, prior, forced=forced_labels, no_data=no_data)
        if labels_only:
            grp = [e for e in eps if res["label"][e] in LURES or res["label"][e] == "BASE"]
            res["groups"] = ([grp] if grp else []) + [[e] for e in eps if e not in grp]
            res["model_of"] = {e: k for k, g in enumerate(res["groups"]) for e in g}
            res["gray"] = []
        results[f] = res
        flab = f"F{fi + 1:02d}"
        for k, g in enumerate(sorted(res["groups"], key=lambda g: (-len(g), g))):
            for e in g:
                family[e] = flab
                model[e] = f"{flab}M{k + 1:02d}"
    return {"family": family, "model": model, "families": results}
