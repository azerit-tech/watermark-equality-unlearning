"""White-box (replay) method on the twin: mescore outputs -> per-endpoint tests on 100-response draws.

  $ME_PY -m twin.wb_eval --root RUN_ROOT --fam Q [--base base --alts fp8t,int8c,sysreplay] [--ndiv 5]

For every endpoint and every disjoint draw of ~100 responses (same composition as the black-box
budget: the plan divided by ndiv, restricted to the scored reps):
  z_exc07   excess surprise (surprise - entropy, rows as units) under BASE at T=0.7 (the known T)
  T_mle     best temperature on the grid under BASE;  z_excT  excess surprise at T_mle
  rank1     share of tokens that are the argmax (greedy detector)
  llr_<alt> paired per-row LLR z of hypothesis ALT vs BASE at T=0.7 (positive = data prefer ALT)
Flags (default thresholds): 'not-base' if z_excT > 4 ; '<alt>' if llr_<alt> > 4.
Also the number of responses needed for z = 4, extrapolated from all scored rows (16 R / z^2).
Tokens flagged as alternative segmentation (flag bit 2) are excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

from mescore import store

from .spec import PLAN


def zrows(d):
    d = np.asarray(d, float)
    if len(d) < 3 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))


class Scored:
    def __init__(self, path):
        arr, meta = store.load(path)
        self.meta = meta
        self.temps = meta["temps"]
        off = arr["offsets"]
        self.R = len(off) - 1
        rows = np.repeat(np.arange(self.R), np.diff(off))
        keep = ((arr["flags"] & 2) == 0).astype(float)
        self.tok = arr["tok"]
        self.off = off
        lo = arr["logit_obs"].astype(np.float64)
        self.ll = np.zeros((len(self.temps), self.R))     # row log-lik at each T
        self.exc = np.zeros((len(self.temps), self.R))    # row excess surprise at each T
        for m, T in enumerate(self.temps):
            lp = lo / T - arr["lse"][:, m]
            self.ll[m] = np.bincount(rows, lp * keep, minlength=self.R)
            self.exc[m] = np.bincount(rows, (-lp - arr["ent_T"][:, m]) * keep, minlength=self.R)
        self.ntok = np.bincount(rows, keep, minlength=self.R)
        self.r1 = np.bincount(rows, (arr["rank"] == 1) * keep, minlength=self.R)
        self.pid = [r["prompt_id"] for r in meta["rows"]]
        self.rep = [int(r["key"].split("|")[2]) for r in meta["rows"]]
        self.ids = [r["id"] for r in meta["rows"]]

    def m(self, T):
        return min(range(len(self.temps)), key=lambda i: abs(self.temps[i] - T))


def draws_of(sc: Scored, ndiv: int, n_draws: int):
    k = {p: max(1, n // ndiv) for p, n in PLAN.items()}
    out = []
    for d in range(n_draws):
        idx = [i for i in range(sc.R) if d * k[sc.pid[i]] <= sc.rep[i] < (d + 1) * k[sc.pid[i]]]
        out.append(np.array(idx, int))
    return out


def endpoint_stats(B: Scored, alts: dict, idx, T0=0.7):
    m0 = B.m(T0)
    llsum = B.ll[:, idx].sum(1)
    mT = int(np.argmax(llsum))
    s = {"rows": len(idx), "tokens": int(B.ntok[idx].sum()), "rank1": float(B.r1[idx].sum() / B.ntok[idx].sum()),
         "z_exc07": zrows(B.exc[m0, idx]), "exc07_per_tok": float(B.exc[m0, idx].sum() / B.ntok[idx].sum()),
         "T_mle": B.temps[mT], "z_excT": zrows(B.exc[mT, idx])}
    for name, A in alts.items():
        s[f"llr_{name}"] = zrows(A.ll[m0, idx] - B.ll[m0, idx])
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fam", default="Q")
    ap.add_argument("--base", default="base")
    ap.add_argument("--alts", default="fp8t,int8c,sysreplay")
    ap.add_argument("--ndiv", type=int, default=5)
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--zthr", type=float, default=4.0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    sdir = os.path.join(a.root, "score", a.fam)
    truth = {r["endpoint"]: r for r in csv.DictReader(open(os.path.join(a.root, f"truth_{a.fam}.csv")))}
    eps = sorted(truth)
    alts = [x for x in a.alts.split(",") if x]
    tag = a.tag or a.base
    out_rows = []
    lines = [f"base hypothesis = {a.base}; alternatives = {alts}; draws of ~100 responses (plan/{a.ndiv}); "
             f"threshold z > {a.zthr}", ""]
    hdr = (f"{'endpoint':7s} {'kind':48s} {'z_exc@0.7':>16s} {'T_mle':>9s} {'z_exc@Tmle':>16s} {'rank1':>6s} " +
           " ".join(f"{'llr_' + x:>16s}" for x in alts) + "   resp_for_z4(excT," + ",".join(alts) + ")")
    lines.append(hdr)
    for ep in eps:
        p = os.path.join(sdir, a.base, ep + ".npz")
        if not store.is_done(p):
            continue
        B = Scored(p)
        A = {}
        for x in alts:
            px = os.path.join(sdir, x, ep + ".npz")
            if store.is_done(px):
                S = Scored(px)
                if np.array_equal(S.tok, B.tok) and np.array_equal(S.off, B.off):
                    A[x] = S
        D = draws_of(B, a.ndiv, a.n_draws)
        st = [endpoint_stats(B, A, idx) for idx in D]
        full = endpoint_stats(B, A, np.arange(B.R))
        need = []
        for key in ["z_excT"] + [f"llr_{x}" for x in alts]:
            z = full.get(key, float("nan"))
            need.append(str(int(math.ceil(16 * B.R / z ** 2))) if z == z and z > 0 else "-")

        def fmt(key, nd=1):
            v = [s.get(key, float("nan")) for s in st]
            return " ".join(f"{x:5.{nd}f}" for x in v)
        kind = truth[ep]["kind"][:48]
        lines.append(f"{ep:7s} {kind:48s} {fmt('z_exc07'):>16s} {fmt('T_mle', 2):>9s} {fmt('z_excT'):>16s} "
                     f"{np.mean([s['rank1'] for s in st]):6.3f} " +
                     " ".join(f"{fmt('llr_' + x):>16s}" for x in alts) + "   " + ",".join(need))
        for d, s in enumerate(st):
            out_rows.append(dict(endpoint=ep, kind=truth[ep]["kind"], model=truth[ep]["model"], draw=d, base=a.base,
                                 **s, flag_notbase=s["z_excT"] > a.zthr,
                                 **{f"flag_{x}": s.get(f"llr_{x}", float("nan")) > a.zthr for x in alts}))
        out_rows.append(dict(endpoint=ep, kind=truth[ep]["kind"], model=truth[ep]["model"], draw="all",
                             base=a.base, **full))
    od = os.path.join(a.root, "wb_eval")
    os.makedirs(od, exist_ok=True)
    keys = sorted({k for r in out_rows for k in r}, key=lambda k: (k not in ("endpoint", "kind", "model", "draw"), k))
    with open(os.path.join(od, f"{a.fam}_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(out_rows)
    txt = "\n".join(lines) + "\n"
    open(os.path.join(od, f"{a.fam}_{tag}.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
