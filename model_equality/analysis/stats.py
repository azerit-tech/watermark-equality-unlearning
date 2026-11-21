"""Pairwise statistics T_raw, T_temp, T_inv and the free-text MMD (METHOD 5.2),
calibrated by permuting whole responses between the two endpoints, within each
probe (B permutations; row 0 of every array is the observed split).

Main entry points
-----------------
``PairTables.build(parsed, X, Y, probes, units, B, rng)``
    count tables N_X, N_Y of shape (B+1, n_cells) for every discrete unit and
    MMD values (B+1,) for every free unit, all permutations drawn once.
``PairTables.stat(unit_ids)``
    T_raw / T_temp / T_inv (+ kappa) on the union of the given units (one kappa
    for the pair, METHOD 5.2 step 4), with z and permutation p-values.
``pair_report(...)``
    per-unit and aggregated z for one pair (dict rows).

Definitions (METHOD 5.2, with the choices made where it is silent):
* T_raw: for every context whose pooled total reaches 10, G statistic of the
  2 x m table; an option is merged into an "other" column when its expected
  count on the smaller side, M[c,i] * min(nX, nY) / (nX + nY), is below 3
  (nX, nY = number of responses of the probe; computed on the pooled counts,
  which are permutation-invariant, so the statistic is a fixed function).
* l[c,i] = log((N[c,i] + .5) / (N[c,.] + .5 m_c)), v = 1 / (N[c,i] + .5);
  m_c = number of distinct options seen in context c (pooled).
* common support: N_X >= 5 and N_Y >= 5 (``min_cell``); contexts with fewer
  than 2 kept options are dropped.
* T_temp: centre l over the kept options of each context (same projection for X
  and Y), kappa in [1/4, 4] minimising
  Q = sum (Th_Y - k Th_X)^2 / (k^2 v_X + v_Y); T_temp = Q(kappa_hat).
* T_inv: same, but residuals of the additive fit l ~ a_c + b_i inside each
  wm_block (weighted backfitting, weights 1 / (v_X + v_Y), identical operator
  for X and Y so that the null relation Th_Y = k Th_X is preserved).
* free probes: MET-style MMD^2 (unbiased) with a character Hamming kernel on the
  first 200 characters padded with a filler; summed over probes.
* z = (T_obs - mean(T_perm)) / sd(T_perm); p = (1 + #{T_perm >= T_obs}) / (B + 1).
* z_temp / z_inv are NaN when fewer than ``MIN_CELLS_Z`` cells survive the common-support filter in the
  observed split (too few cells: the statistic is erratic, seen on real forms at 12 repetitions).

Power with few responses: with n responses per endpoint and probe, only
C(2n, n) distinct splits exist (20 for n = 3): z and p are then coarse.
``stat`` reports ``log10_splits`` (log10 of the number of distinct splits,
product over probes) and ``low_power`` = True when fewer than 100 splits exist
(smallest attainable p > 0.01).  Aggregating several probes multiplies the
number of splits, which is the main remedy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np
import scipy.sparse as sp

MIN_CELL = 3  # METHOD default 5; 3 chosen for low repetition counts (tested on synthetic data)
MIN_CELLS_Z = 10  # real data (ref-, forms at 12 repetitions): with fewer kept cells, z_temp / z_inv are erratic
                  # (F05, pair ref-01/ref-09: 8 cells, z_inv = 8.1 for z_temp = 1.1); below this floor z is NaN
MIN_CTX_RAW = 10
MIN_EXPECTED = 3.0
KAPPA_LO, KAPPA_HI = 0.25, 4.0
MMD_LEN = 200
PAD = 0  # filler code for the MMD kernel
_KGRID = np.exp(np.linspace(np.log(KAPPA_LO), np.log(KAPPA_HI), 29))


# --------------------------------------------------------------------- helpers
def _group_matrix(ids: np.ndarray, n_groups: int) -> sp.csr_matrix:
    """C x G indicator matrix (cell -> group)."""
    C = len(ids)
    return sp.csr_matrix((np.ones(C), (np.arange(C), ids)), shape=(C, n_groups))


def _gsum(A: np.ndarray, S: sp.csr_matrix) -> np.ndarray:
    """(R x C) dense @ (C x G) indicator -> (R x G)."""
    return np.asarray((S.T @ A.T).T)


def _zscore(T: np.ndarray) -> tuple:
    obs, perm = T[0], T[1:]
    if len(perm) == 0 or not np.isfinite(obs):
        return float("nan"), float("nan")
    sd = perm.std(ddof=1)
    p = (1 + np.sum(perm >= obs - 1e-12)) / (len(perm) + 1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return (0.0 if abs(obs - perm.mean()) < 1e-9 else float("nan")), float(p)
    return float((obs - perm.mean()) / sd), float(p)


def permutation_assignments(nx: int, ny: int, B: int, rng) -> np.ndarray:
    """(B+1, nx+ny) boolean, True = assigned to X; row 0 is the observed split."""
    n = nx + ny
    A = np.zeros((B + 1, n), dtype=bool)
    A[0, :nx] = True
    if B:
        keys = rng.random((B, n))
        idx = np.argsort(keys, axis=1)[:, :nx]
        np.put_along_axis(A[1:], idx, True, axis=1)
    return A


# ------------------------------------------------------------- unit structures
@dataclass
class DiscreteUnit:
    uid: str
    long: bool
    ctx: np.ndarray        # cell -> context index (local)
    opt: np.ndarray        # cell -> option index within the block (local)
    ctx_names: list
    opt_names: list
    NX: np.ndarray         # (B+1, C)
    NY: np.ndarray
    ratio: np.ndarray      # cell -> min(nX,nY)/(nX+nY) of its probe (for raw merging)
    n_resp: tuple          # (nX, nY) total responses in the unit
    log10_splits: float = 0.0


@dataclass
class FreeUnit:
    uid: str
    long: bool
    T: np.ndarray          # (B+1,) MMD^2
    log10_splits: float = 0.0


def _build_discrete(uid, long, probe_ids, ex, ey, B, rng) -> DiscreteUnit | None:
    """ex / ey: probe_id -> [Counter events] of X / Y."""
    cells: dict = {}
    for pid in probe_ids:
        for evs in (ex.get(pid, []), ey.get(pid, [])):
            for ev in evs:
                for k in ev:
                    if k not in cells:
                        cells[k] = len(cells)
    if not cells:
        return None
    keys = sorted(cells, key=lambda k: (k[0], k[1]))
    cells = {k: i for i, k in enumerate(keys)}
    C = len(keys)
    ctx_names = sorted({k[0] for k in keys})
    opt_names = sorted({k[1] for k in keys})
    cidx = {c: i for i, c in enumerate(ctx_names)}
    oidx = {o: i for i, o in enumerate(opt_names)}
    ctx = np.array([cidx[k[0]] for k in keys])
    opt = np.array([oidx[k[1]] for k in keys])
    NX = np.zeros((B + 1, C))
    tot = np.zeros(C)
    ratio = np.zeros(C)
    nX_tot = nY_tot = 0
    lsplit = 0.0
    for pid in probe_ids:
        lx, ly = ex.get(pid, []), ey.get(pid, [])
        nx, ny = len(lx), len(ly)
        if nx + ny == 0:
            continue
        nX_tot += nx
        nY_tot += ny
        lsplit += math.log10(math.comb(nx + ny, nx))
        R = np.zeros((nx + ny, C))
        for r, ev in enumerate(list(lx) + list(ly)):
            for k, v in ev.items():
                R[r, cells[k]] += v
        tot += R.sum(0)
        used = R.sum(0) > 0
        ratio[used] = min(nx, ny) / (nx + ny)
        if nx == 0:
            continue
        A = permutation_assignments(nx, ny, B, rng)
        NX += A.astype(float) @ R
    NY = tot[None, :] - NX
    return DiscreteUnit(uid, long, ctx, opt, ctx_names, opt_names, NX, NY, ratio, (nX_tot, nY_tot), lsplit)


def _encode(text: str, L: int = MMD_LEN) -> np.ndarray:
    a = np.full(L, PAD, dtype=np.int32)
    t = [ord(c) for c in (text or "")[:L]]
    a[:len(t)] = t
    return a


def mmd_values(tx: list, ty: list, B: int, rng) -> np.ndarray:
    """Unbiased MMD^2 with a normalised Hamming kernel, for the observed split and B permutations."""
    nx, ny = len(tx), len(ty)
    if nx < 2 or ny < 2:
        return np.full(B + 1, np.nan)
    E = np.stack([_encode(t) for t in list(tx) + list(ty)])
    G = (E[:, None, :] == E[None, :, :]).mean(-1)
    np.fill_diagonal(G, 0.0)
    A = permutation_assignments(nx, ny, B, rng).astype(float)
    Bm = 1.0 - A
    sxx = np.einsum("bi,ij,bj->b", A, G, A)
    syy = np.einsum("bi,ij,bj->b", Bm, G, Bm)
    sxy = np.einsum("bi,ij,bj->b", A, G, Bm)
    return sxx / (nx * (nx - 1)) + syy / (ny * (ny - 1)) - 2 * sxy / (nx * ny)


# ------------------------------------------------------------- statistics
def t_raw(NX, NY, ctx, ratio):
    """Sum of G statistics over contexts with pooled total >= 10 (row-wise over permutations)."""
    M = NX[0] + NY[0]
    nctx = ctx.max() + 1
    Mc = np.bincount(ctx, weights=M, minlength=nctx)
    keep_ctx = Mc >= MIN_CTX_RAW
    small = M * ratio < MIN_EXPECTED
    # raw column id: (ctx, option) or (ctx, other)
    col_key = np.where(small, -1, np.arange(len(ctx)))
    pairs = {}
    col = np.empty(len(ctx), dtype=int)
    for i, (c, k) in enumerate(zip(ctx, col_key)):
        col[i] = pairs.setdefault((c, k), len(pairs))
    ncol = len(pairs)
    col_ctx = np.empty(ncol, dtype=int)
    for (c, _), j in pairs.items():
        col_ctx[j] = c
    ncols_per_ctx = np.bincount(col_ctx, minlength=nctx)
    good_ctx = keep_ctx & (ncols_per_ctx >= 2)
    good_col = good_ctx[col_ctx]
    if not good_col.any():
        return np.zeros(NX.shape[0])
    S = _group_matrix(col, ncol)
    OX = _gsum(NX, S)[:, good_col]
    OY = _gsum(NY, S)[:, good_col]
    cc = col_ctx[good_col]
    _, cc = np.unique(cc, return_inverse=True)
    Sc = _group_matrix(cc, cc.max() + 1)
    tX = _gsum(OX, Sc)
    tY = _gsum(OY, Sc)
    Mj = OX + OY
    Mtot = tX + tY
    EX = tX[:, cc] * Mj / Mtot[:, cc]
    EY = tY[:, cc] * Mj / Mtot[:, cc]
    with np.errstate(divide="ignore", invalid="ignore"):
        gx = np.where(OX > 0, OX * np.log(OX / EX), 0.0)
        gy = np.where(OY > 0, OY * np.log(OY / EY), 0.0)
    return 2.0 * (gx + gy).sum(1)


@dataclass
class TempInv:
    T: np.ndarray
    kappa: np.ndarray
    n_cells: int            # kept cells in the observed split
    thX: np.ndarray = None  # observed centred log-probs (cells), for diagnostics
    thY: np.ndarray = None
    mask0: np.ndarray = None
    bX: np.ndarray = None   # option effects of the additive fit (T_inv only), observed
    bY: np.ndarray = None


def _fit_kappa(thX, thY, vX, vY, mask):
    """Grid then local refinement of kappa per row; returns (Qmin, kappa)."""
    def Q(k):
        k = np.asarray(k)
        if k.ndim == 0:
            num = (thY - k * thX) ** 2
            den = k * k * vX + vY
        else:
            num = (thY - k[:, None] * thX) ** 2
            den = (k * k)[:, None] * vX + vY
        return np.where(mask, num / den, 0.0).sum(1)
    R = thX.shape[0]
    vals = np.stack([Q(k) for k in _KGRID], 1)   # (R, K)
    j = vals.argmin(1)
    lo = np.log(_KGRID[np.maximum(j - 1, 0)])
    hi = np.log(_KGRID[np.minimum(j + 1, len(_KGRID) - 1)])
    best_q = vals[np.arange(R), j]
    best_k = _KGRID[j]
    for t in np.linspace(0, 1, 9):
        k = np.exp(lo + t * (hi - lo))
        q = Q(k)
        better = q < best_q
        best_q = np.where(better, q, best_q)
        best_k = np.where(better, k, best_k)
    return best_q, best_k


def _logp(N, ctx, nctx, m_c):
    Sc = _group_matrix(ctx, nctx)
    Nc = _gsum(N, Sc)[:, ctx]
    return np.log((N + 0.5) / (Nc + 0.5 * m_c[ctx])), 1.0 / (N + 0.5)


def t_temp_inv(units: list, mode: str = "temp", min_cell: int = MIN_CELL, backfit_iter: int = 25) -> TempInv:
    """T_temp (mode='temp') or T_inv (mode='inv') over the union of DiscreteUnits."""
    if not units:
        return TempInv(np.zeros(1), np.ones(1), 0)
    # concatenate cells with globally unique context / option-in-block ids
    ctx_l, opt_l, NX_l, NY_l = [], [], [], []
    c_off = o_off = 0
    for u in units:
        ctx_l.append(u.ctx + c_off)
        opt_l.append(u.opt + o_off)
        NX_l.append(u.NX)
        NY_l.append(u.NY)
        c_off += len(u.ctx_names)
        o_off += len(u.opt_names)
    ctx = np.concatenate(ctx_l)
    opt = np.concatenate(opt_l)
    NX = np.concatenate(NX_l, 1)
    NY = np.concatenate(NY_l, 1)
    nctx, nopt = c_off, o_off
    m_c = np.bincount(ctx, minlength=nctx).astype(float)
    lX, vX = _logp(NX, ctx, nctx, m_c)
    lY, vY = _logp(NY, ctx, nctx, m_c)
    mask = (NX >= min_cell) & (NY >= min_cell)
    Sc = _group_matrix(ctx, nctx)
    kc = _gsum(mask.astype(float), Sc)
    mask &= (kc[:, ctx] >= 2)
    mf = mask.astype(float)
    bX = bY = None
    if mode == "temp":
        kc = np.maximum(_gsum(mf, Sc), 1)
        thX = lX - (_gsum(lX * mf, Sc) / kc)[:, ctx]
        thY = lY - (_gsum(lY * mf, Sc) / kc)[:, ctx]
    else:
        So = _group_matrix(opt, nopt)
        w = mf / (vX + vY)
        wc = np.maximum(_gsum(w, Sc), 1e-300)
        wo = np.maximum(_gsum(w, So), 1e-300)

        def backfit(l):
            b = np.zeros((l.shape[0], nopt))
            for _ in range(backfit_iter):
                a = _gsum(w * (l - b[:, opt]), Sc) / wc
                b = _gsum(w * (l - a[:, ctx]), So) / wo
            a = _gsum(w * (l - b[:, opt]), Sc) / wc
            return l - a[:, ctx] - b[:, opt], b
        thX, bXa = backfit(lX)
        thY, bYa = backfit(lY)
        bX, bY = bXa[0], bYa[0]
        # an option seen in a single kept context carries no interaction
        nob = _gsum(mf, So)
        mask &= nob[:, opt] >= 2
    T, kap = _fit_kappa(thX, thY, vX, vY, mask)
    out = TempInv(T, kap, int(mask[0].sum()), thX[0], thY[0], mask[0])
    out.vX0, out.vY0 = vX[0], vY[0]
    if mode == "inv":
        out.bX, out.bY = bX, bY
        out.opt = opt
        out.block = np.concatenate([np.full(len(u.opt_names), i) for i, u in enumerate(units)])
        out.opt_seen = _gsum(mask[:1].astype(float), _group_matrix(opt, nopt))[0]
    return out


# ------------------------------------------------------------- pair tables
@dataclass
class PairTables:
    X: str
    Y: str
    B: int
    discrete: dict = field(default_factory=dict)   # uid -> DiscreteUnit
    free: dict = field(default_factory=dict)       # uid -> FreeUnit

    @classmethod
    def build(cls, parsed: dict, X: str, Y: str, probes: dict, units: dict, B: int, rng,
              free: bool = True) -> "PairTables":
        """parsed: endpoint -> probe_id -> [Parsed]; units: uid -> [probe ids]."""
        pt = cls(X, Y, B)
        px, py = parsed.get(X, {}), parsed.get(Y, {})
        for uid, pids in units.items():
            pr = probes[pids[0]]
            long = all(probes[p].long for p in pids)
            if pr.discrete:
                ex = {p: [q.events for q in px.get(p, [])] for p in pids}
                ey = {p: [q.events for q in py.get(p, [])] for p in pids}
                if not any(ex.values()) or not any(ey.values()):
                    continue
                u = _build_discrete(uid, long, pids, ex, ey, B, rng)
                if u is not None:
                    pt.discrete[uid] = u
            elif free:
                T = np.zeros(B + 1)
                ok = False
                ls = 0.0
                for p in pids:
                    tx = [q.items[0] for q in px.get(p, []) if q.items]
                    ty = [q.items[0] for q in py.get(p, []) if q.items]
                    v = mmd_values(tx, ty, B, rng)
                    if np.isfinite(v).all():
                        T += v
                        ok = True
                        ls += math.log10(math.comb(len(tx) + len(ty), len(tx)))
                if ok:
                    pt.free[uid] = FreeUnit(uid, long, T, ls)
        return pt

    def stat(self, uids=None, which=("raw", "temp", "inv", "mmd"), min_cell: int = MIN_CELL) -> dict:
        """Aggregated statistics over the units ``uids`` (default: all)."""
        uids = list(self.discrete) + list(self.free) if uids is None else list(uids)
        du = [self.discrete[u] for u in uids if u in self.discrete]
        fu = [self.free[u] for u in uids if u in self.free]
        out = {"n_units": len(du), "n_free": len(fu)}
        ld = sum(u.log10_splits for u in du)
        out["log10_splits"] = ld if du else sum(u.log10_splits for u in fu)
        out["low_power"] = bool(out["log10_splits"] < 2.0)
        out["n_resp_X"] = sum(u.n_resp[0] for u in du)
        out["n_resp_Y"] = sum(u.n_resp[1] for u in du)
        if "raw" in which:
            if du:
                T = sum(t_raw(u.NX, u.NY, u.ctx, u.ratio) for u in du)
                out["T_raw"] = float(T[0])
                out["z_raw"], out["p_raw"] = _zscore(T)
            else:
                out["T_raw"] = out["z_raw"] = out["p_raw"] = float("nan")
        for mode in ("temp", "inv"):
            if mode not in which:
                continue
            if du:
                r = t_temp_inv(du, mode, min_cell=min_cell)
                out[f"T_{mode}"] = float(r.T[0])
                out[f"z_{mode}"], out[f"p_{mode}"] = (_zscore(r.T) if r.n_cells >= MIN_CELLS_Z
                                                      else (float("nan"), float("nan")))
                out[f"kappa_{mode}"] = float(r.kappa[0])
                out[f"cells_{mode}"] = r.n_cells
                if r.n_cells:   # effect size (METHOD R1 item 4): RMS residual, raw and noise-corrected
                    m = r.mask0
                    k = float(r.kappa[0])
                    res2 = (r.thY[m] - k * r.thX[m]) ** 2
                    noise = k * k * r.vX0[m] + r.vY0[m]
                    out[f"eps_{mode}"] = float(np.sqrt(res2.mean()))
                    out[f"eps_{mode}_corr"] = float(np.sqrt(max(0.0, res2.mean() - noise.mean())))
                if mode == "inv":
                    out["_inv"] = r
            else:
                for k in ("T", "z", "p", "kappa"):
                    out[f"{k}_{mode}"] = float("nan")
                out[f"cells_{mode}"] = 0
        if "mmd" in which:
            if fu:
                T = sum(u.T for u in fu)
                out["T_mmd"] = float(T[0])
                out["z_mmd"], out["p_mmd"] = _zscore(T)
            else:
                out["T_mmd"] = out["z_mmd"] = out["p_mmd"] = float("nan")
        return out

    def units(self, long: bool | None = None) -> list:
        allu = list(self.discrete.values()) + list(self.free.values())
        return [u.uid for u in allu if long is None or u.long == long]


def subsample_parsed(parsed: dict, n: int | dict, rng, endpoints=None) -> dict:
    """Random subsample (without replacement) of n responses per (endpoint, probe).
    ``n`` may be a dict probe_id -> n (probes absent from it are dropped)."""
    out = {}
    for e, d in parsed.items():
        if endpoints is not None and e not in endpoints:
            continue
        out[e] = {}
        for pid, lst in d.items():
            k = n.get(pid, 0) if isinstance(n, dict) else n
            if k <= 0:
                continue
            if len(lst) <= k:
                out[e][pid] = list(lst)
            else:
                idx = rng.choice(len(lst), size=k, replace=False)
                out[e][pid] = [lst[i] for i in sorted(idx)]
    return out
