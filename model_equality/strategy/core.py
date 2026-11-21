"""Belief model and proposal engine of the submission strategy.

Setting (checked on the first real submission: 3 right families, one model per
family -> public score exactly 0.1):

    public score = 0.1 * MCC_family + 0.9 * MCC_model,
    MCC_model over the PUBLIC pairs inside one family, 0 if the prediction is constant.

Families are taken as known (``cf`` = family term, 0.1, re-measured by any
submission whose model prediction is constant inside the families).  Everything
below is about the within-family pairs (``nw`` of them).

Unknowns: the true model partition T and the public pair set S (each pair is
public with probability ``q``).  A submission is a 0/1 vector ``s`` over the
within-family pairs (1 = predicted same model).

* ``Decode``: T-free arithmetic.  The pairs are grouped in classes sharing the
  same prediction history; a hypothesis gives, per class, the number of public
  truly-same pairs (p) and public truly-different pairs (d).  Every public
  score is an exact function of these integers, so each new score prunes the
  hypotheses (usually down to one).
* ``Belief``: Markov chains over T.  Prior: pairwise log-odds from p_model
  (full strength inside a block of the candidate partition, divided by the
  block size across blocks, so that "one endpoint against one block" is one
  decision) plus a per-endpoint "hidden variant" switch (an undetectable LoRA
  or quantisation inside a block, probability >= ``pi_hidden``).  Likelihood:
  product over classes of Binomial(p; #truly-same pairs of the class under T, q)
  * Binomial(d; #truly-different, q), summed over the decode hypotheses.
  Rounded scores: when no observed score shows more than 6 decimals, the scores
  are taken as rounded at the largest number of decimals seen (tolerance = half
  a unit: "approx" mode).  When no hypothesis fits (other formula, wrong
  families): "sign mode" (only the sign of each score difference).
* ``Belief.propose``: samples worlds (T, S) that reproduce every observed score,
  evaluates every single move from the incumbent (best public score so far) and
  greedy combinations, and returns the partition maximising
      V = E[ 1(public score beats the incumbent) * (private gain) ].
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

W_FAMILY, W_MODEL = 0.1, 0.9
PENALTY = 6.0                       # log-penalty per impossible unit (soft constraint)


def n_decimals(x: float) -> int:
    """Decimals in the shortest representation of a score (17 with an exponent)."""
    r = repr(float(x))
    if "e" in r or "E" in r:
        return 17
    return len(r.split(".")[1]) if "." in r else 0
MODES = ("exact", "approx", "sign")


def mcc(tp, fp, fn, tn):
    """Vectorised MCC clipped at 0; 0 when the denominator is 0 (as scoring.metrics)."""
    tp, fp, fn, tn = (np.asarray(x, dtype=float) for x in (tp, fp, fn, tn))
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    num = tp * tn - fp * fn
    r = np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1.0)), 0.0)
    return np.maximum(r, 0.0)


def canon(L):
    """Canonical labels: every endpoint gets the smallest index of its cluster."""
    L = np.asarray(L)
    L2 = L[None, :] if L.ndim == 1 else L
    out = (L2[:, :, None] == L2[:, None, :]).argmax(axis=2)
    return out[0] if L.ndim == 1 else out


def binom_table(nmax: int, q: float) -> np.ndarray:
    """LB[n, k] = log Binomial(k; n, q); soft penalty when k > n."""
    n = np.arange(nmax + 1)[:, None].astype(float)
    k = np.arange(nmax + 1)[None, :].astype(float)
    lb = (gammaln(n + 1) - gammaln(k + 1) - gammaln(np.maximum(n - k, 0) + 1)
          + k * math.log(q) + np.maximum(n - k, 0) * math.log1p(-q))
    return np.where(k <= n, lb, -PENALTY * (k - n) - 1.0)


def sign_table(q: float, nmax: int = 45, floor: float = 1e-3) -> np.ndarray:
    """log P(sign of the score change | #good flips, #bad flips); index 0 = down, 1 = equal, 2 = up."""
    k = np.arange(nmax + 1)
    pm = np.exp(binom_table(nmax, q))
    pm = np.where(k[None, :] <= k[:, None], pm, 0.0)
    out = np.zeros((nmax + 1, nmax + 1, 3))
    for g in range(nmax + 1):
        for u in range(nmax + 1):
            joint = np.outer(pm[g, :g + 1], pm[u, :u + 1])        # [public good, public bad]
            up = np.tril(joint, -1).sum()
            down = np.triu(joint, 1).sum()
            tie = np.trace(joint) - joint[0, 0]
            out[g, u] = (down + tie / 2, joint[0, 0], up + tie / 2)
    return np.log(np.maximum(out, floor))


# --------------------------------------------------------------------------- problem
class Problem:
    """Endpoints, families (fixed), candidate partition and prior built from p_model."""

    def __init__(self, endpoints, family: dict, model: dict, p_model, q: float = 0.3,
                 pi_hidden: float = 0.12, p_lo: float = 0.03, p_hi: float = 0.98, anchored: bool = False):
        """``anchored``: for an endpoint that the candidate leaves alone, joining a cluster that holds
        a candidate block (>= 2 endpoints) is judged on its pairs with that block only; its pairs with
        the other lone endpoints of the cluster are ignored (two lone endpoints may both belong to the
        base group even if nothing links them to each other)."""
        self.eps = list(endpoints)
        n = self.n = len(self.eps)
        self.q, self.pi_hidden, self.p_lo, self.p_hi = q, pi_hidden, p_lo, p_hi
        self.fam_names = sorted({family[e] for e in self.eps})
        self.fam = np.array([self.fam_names.index(family[e]) for e in self.eps])
        self.n_fam = len(self.fam_names)
        ids: dict = {}
        self.cand = canon(np.array([ids.setdefault((family[e], model[e]), len(ids)) for e in self.eps]))
        ia, ib = np.triu_indices(n, 1)
        keep = self.fam[ia] == self.fam[ib]
        self.ia, self.ib = ia[keep], ib[keep]
        self.nw = len(self.ia)
        self.fam_idx = [np.flatnonzero(self.fam == self.fam[i]) for i in range(n)]
        self.fam_pos = [int(np.flatnonzero(self.fam_idx[i] == i)[0]) for i in range(n)]

        P = np.clip((np.asarray(p_model, float) + np.asarray(p_model, float).T) / 2, p_lo, p_hi)
        self.P = P
        size = np.array([(self.cand == self.cand[i]).sum() for i in range(n)])
        W = np.zeros((n, n))
        for i, j in zip(self.ia, self.ib):
            lg = math.log(P[i, j] / (1 - P[i, j]))
            if self.cand[i] == self.cand[j]:
                if size[i] == 2:        # a block of two: the pair itself carries the hidden-variant floor
                    lg = min(lg, math.log((1 - pi_hidden) / pi_hidden))
            else:
                lg /= max(size[i], size[j])
            W[i, j] = W[j, i] = lg
        self.Wmat = W
        self.anchor = (size >= 2) if anchored else None
        eta = np.zeros(n)
        for i in range(n):
            mates = np.flatnonzero((self.cand == self.cand[i]) & (np.arange(n) != i))
            if len(mates) >= 2:
                eta[i] = min(0.5, max(pi_hidden, 1.0 - P[i, mates].mean()))
        self.eta = eta

    def pairs_same(self, L):
        L = np.asarray(L)
        return L[..., self.ia] == L[..., self.ib]

    def labels_from(self, family: dict, model: dict):
        """Canonical labels of another partition, or None if its families differ from the problem's."""
        fam = [family[e] for e in self.eps]
        fid: dict = {}
        f = np.array([fid.setdefault(x, len(fid)) for x in fam])
        if not np.array_equal(canon(f), canon(self.fam)):
            return None
        ids: dict = {}
        return canon(np.array([ids.setdefault((family[e], model[e]), len(ids)) for e in self.eps]))

    def names(self, lab) -> tuple:
        """(family, model) dicts for a label vector; models named <family>_mNN by first endpoint."""
        lab = canon(lab)
        family, model = {}, {}
        count: dict = {}
        name: dict = {}
        for i, e in enumerate(self.eps):
            f = self.fam_names[self.fam[i]]
            if lab[i] not in name:
                count[f] = count.get(f, 0) + 1
                name[lab[i]] = f"{f}_m{count[f]:02d}"
            family[e], model[e] = f, name[lab[i]]
        return family, model

    def neighbours(self, lab) -> list:
        """Partitions one elementary move away: eject an endpoint, move it to another cluster, merge two clusters."""
        lab = canon(lab)
        n, seen, out = self.n, set(), []

        def add(new):
            new = canon(new)
            key = new.tobytes()
            if key not in seen:
                seen.add(key)
                out.append(new)

        for f in range(self.n_fam):
            idx = np.flatnonzero(self.fam == f)
            clusters = sorted(set(lab[idx].tolist()))
            size = {c: int((lab[idx] == c).sum()) for c in clusters}
            for i in idx:
                if size[lab[i]] > 1:
                    new = lab.copy()
                    new[i] = n + i
                    add(new)
                for c in clusters:
                    if c != lab[i]:
                        new = lab.copy()
                        new[i] = c
                        add(new)
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    if size[clusters[a]] > 1 and size[clusters[b]] > 1:
                        new = lab.copy()
                        new[lab == clusters[b]] = clusters[a]
                        add(new)
        return out

    def diff(self, a, b) -> list:
        """Readable difference a -> b: ("split", endpoints, the core they leave) or ("join", parts, None)."""
        a, b = canon(a), canon(b)
        out = []
        for c in sorted(set(b.tolist())):
            K = np.flatnonzero(b == c)
            src = sorted(set(a[K].tolist()))
            whole = np.flatnonzero(a == src[0])
            if len(src) == 1 and len(whole) == len(K):
                continue                                            # unchanged cluster
            if len(src) == 1:
                pieces = sorted(set(b[whole].tolist()), key=lambda x: (-int((b[whole] == x).sum()), x))
                if pieces[0] == c:
                    continue                                        # K is the core that stays
                core = [self.eps[i] for i in whole if b[i] == pieces[0]]
                out.append(("split", [self.eps[i] for i in K], core))
            else:
                out.append(("join", [[self.eps[i] for i in K if a[i] == s_] for s_ in src], None))
        return out


# --------------------------------------------------------------------------- decode
class TooComplex(Exception):
    pass


class Decode:
    """Public-count hypotheses consistent with every observed score (no reference to the truth)."""

    def __init__(self, nw: int, q: float, cf: float = W_FAMILY, tol: float = 1e-9,
                 max_hyp: int = 20000, limit: int = 2_000_000):
        self.nw, self.q, self.cf, self.tol = nw, q, cf, tol
        self.max_hyp, self.limit = max_hyp, limit
        self.cls = np.zeros(nw, dtype=np.intp)
        self.sizes = np.array([nw])
        self.bits: list = []            # per submission: class-level prediction (0/1)
        self.hp = self.hd = None        # (H x m) public truly-same / truly-different counts per class
        self.n_sub = 0

    def add(self, s, score: float) -> bool:
        s = np.asarray(s, bool)
        try:
            ok = self._first(s, score) if self.n_sub == 0 else self._next(s, score)
        except TooComplex:
            ok = False
        if ok:
            self.n_sub += 1
        return ok

    def _first(self, s, score):
        A = int(s.sum())
        B = self.nw - A
        if A == 0 or B == 0:
            return False
        qh = min(0.95, 1.8 * self.q)

        def grid(n):
            cap = int(min(n, math.ceil(n * qh + 4 * math.sqrt(n * qh * (1 - qh)) + 1)))
            g = np.arange(cap + 1)
            X, Y = np.meshgrid(g, g, indexing="ij")
            m = X + Y <= cap
            return X[m], Y[m]

        tp, fp = grid(A)
        fn, tn = grid(B)
        rows = []
        step = max(1, 1_500_000 // len(fn))
        for a in range(0, len(tp), step):
            sc = self.cf + W_MODEL * mcc(tp[a:a + step, None], fp[a:a + step, None], fn[None, :], tn[None, :])
            i, j = np.nonzero(np.abs(sc - score) <= self.tol)
            if len(i):
                rows.append(np.stack([fn[j], tn[j], tp[a + i], fp[a + i]], axis=1))
        if not rows:
            return False
        R = np.concatenate(rows)
        self.cls = s.astype(np.intp)                     # class 0 = predicted different, 1 = predicted same
        self.sizes = np.array([B, A])
        self.hp, self.hd = R[:, [0, 2]], R[:, [1, 3]]
        self.bits = [np.array([0, 1])]
        self._prune()
        return True

    def _split(self, s):
        key = self.cls * 2 + s.astype(np.intp)
        uniq, inv = np.unique(key, return_inverse=True)
        inv = inv.reshape(-1)
        sizes = np.bincount(inv, minlength=len(uniq))
        parent, bit = uniq // 2, uniq % 2
        m_old = len(self.sizes)
        c1 = -np.ones(m_old, dtype=np.intp)
        c0 = -np.ones(m_old, dtype=np.intp)
        c1[parent[bit == 1]] = np.flatnonzero(bit == 1)
        c0[parent[bit == 0]] = np.flatnonzero(bit == 0)
        return inv, sizes, parent, bit, c1, c0

    def _options(self, h, split, n1, n0):
        opts, shape = [], []
        for sg in split:
            p, d, a1, a0 = int(self.hp[h, sg]), int(self.hd[h, sg]), int(n1[sg]), int(n0[sg])
            pr = np.arange(max(0, p - a0), min(p, a1) + 1)
            dr = np.arange(max(0, d - a0), min(d, a1) + 1)
            PP, DD = np.meshgrid(pr, dr, indexing="ij")
            ok = (PP + DD <= a1) & (p - PP + d - DD <= a0)
            opts.append((PP[ok], DD[ok]))
            shape.append(int(ok.sum()))
        return opts, shape

    def cost(self, s) -> int:
        """Number of count splits to enumerate for the first hypothesis if ``s`` were submitted."""
        if self.hp is None:
            return 1
        _, sizes, _, _, c1, c0 = self._split(np.asarray(s, bool))
        both = (c1 >= 0) & (c0 >= 0)
        n1 = np.where(c1 >= 0, sizes[np.maximum(c1, 0)], 0)
        n0 = np.where(c0 >= 0, sizes[np.maximum(c0, 0)], 0)
        _, shape = self._options(0, np.flatnonzero(both), n1, n0)
        return int(np.prod(shape, dtype=float)) if shape else 1

    def _next(self, s, score):
        inv, sizes, parent, bit, c1, c0 = self._split(s)
        both = (c1 >= 0) & (c0 >= 0)
        split = np.flatnonzero(both)
        unsplit = np.flatnonzero(~both)
        child_unsplit = np.where(c1[unsplit] >= 0, c1[unsplit], c0[unsplit])
        only1 = (c1 >= 0) & ~both
        n1 = np.where(c1 >= 0, sizes[np.maximum(c1, 0)], 0)
        n0 = np.where(c0 >= 0, sizes[np.maximum(c0, 0)], 0)
        hp, hd = self.hp, self.hd
        Ptot, Ntot = hp.sum(1), hd.sum(1)
        tpfix, fpfix = hp[:, only1].sum(1), hd[:, only1].sum(1)
        m_new = len(sizes)
        newp, newd = [], []
        for h in range(len(hp)):
            opts, shape = self._options(h, split, n1, n0)
            total = int(np.prod(shape, dtype=float)) if shape else 1
            if total == 0:
                continue
            if total > self.limit:
                raise TooComplex
            tp = np.full(1, tpfix[h])
            fp = np.full(1, fpfix[h])
            for pp, dd in opts:
                tp = (tp[:, None] + pp[None, :]).ravel()
                fp = (fp[:, None] + dd[None, :]).ravel()
            sc = self.cf + W_MODEL * mcc(tp, fp, Ptot[h] - tp, Ntot[h] - fp)
            hit = np.flatnonzero(np.abs(sc - score) <= self.tol)
            if not len(hit):
                continue
            rp = np.zeros((len(hit), m_new), dtype=hp.dtype)
            rd = np.zeros((len(hit), m_new), dtype=hd.dtype)
            rp[:, child_unsplit] = hp[h, unsplit]
            rd[:, child_unsplit] = hd[h, unsplit]
            if shape:
                ix = np.unravel_index(hit, shape)
                for k, sg in enumerate(split):
                    pp, dd = opts[k]
                    rp[:, c1[sg]] = pp[ix[k]]
                    rp[:, c0[sg]] = hp[h, sg] - pp[ix[k]]
                    rd[:, c1[sg]] = dd[ix[k]]
                    rd[:, c0[sg]] = hd[h, sg] - dd[ix[k]]
            newp.append(rp)
            newd.append(rd)
        if not newp:
            return False
        self.hp, self.hd = np.concatenate(newp), np.concatenate(newd)
        self.cls, self.sizes = inv, sizes
        self.bits = [b[parent] for b in self.bits] + [bit]
        self._prune()
        return True

    def _prune(self):
        if len(self.hp) <= self.max_hyp:
            return
        lb = binom_table(int(self.sizes.max()), self.q)
        w = lb[self.sizes[None, :], self.hp + self.hd].sum(1)
        keep = np.argsort(-w)[:self.max_hyp]
        self.hp, self.hd = self.hp[keep], self.hd[keep]

    def counts(self, k: int):
        """Public (TP, FP, FN, TN) of submission k for every hypothesis: (H x 4)."""
        b = self.bits[k].astype(bool)
        tp, fp = self.hp[:, b].sum(1), self.hd[:, b].sum(1)
        return np.stack([tp, fp, self.hp.sum(1) - tp, self.hd.sum(1) - fp], axis=1)


# --------------------------------------------------------------------------- belief
class Belief:
    def __init__(self, pb: Problem, n_chains: int = 256, seed: int = 0, k_top: int = 24,
                 prior_sweeps: int = 12, tol: float = 1e-9, use_swap: bool = True, rescue: int = 3,
                 k_top_approx: int = 300, max_hyp_approx: int = 4000):
        self.pb = pb
        self.use_swap, self.rescue = use_swap, rescue
        self.k_top_approx, self.max_hyp_approx = k_top_approx, max_hyp_approx
        self.C = n_chains
        self.rng = np.random.default_rng(seed)
        self.k_top = k_top
        self.L = np.tile(pb.cand, (n_chains, 1))
        self.h = np.zeros((n_chains, pb.n), dtype=bool)
        self.LB = binom_table(pb.nw, pb.q)
        self.subs: list = []          # dicts: lab, s, score, kind ('model' | 'probe'), dup
        self.cf = W_FAMILY
        self.tol = tol
        self.tol_used = tol
        self.mode = "exact"
        self.dec: Decode | None = None
        self._hp = self._hd = None
        self._tests: list = []
        self._sign_tab = None
        for _ in range(prior_sweeps):
            self.sweep(beta=0.0)

    # ---- observations
    def observe(self, lab, score: float) -> dict:
        pb = self.pb
        lab = canon(np.asarray(lab))
        s = pb.pairs_same(lab)
        kind = "probe" if (s.all() or not s.any()) else "model"
        dup = any(np.array_equal(lab, x["lab"]) for x in self.subs)
        sub = dict(lab=lab, s=s, score=float(score), kind=kind, dup=dup)
        self.subs.append(sub)
        if kind == "probe":
            if abs(score - self.cf) > 1e-12:
                self.cf = float(score)
                self._rebuild()
        elif not dup:
            if self.mode == "sign":
                self._build_sign()
            elif self.dec is not None and self._tol_now() == self.dec.tol and self.dec.add(s, float(score)):
                self._sync()
            else:
                self._rebuild()
        return sub

    def _items(self):
        return [x for x in self.subs if x["kind"] == "model" and not x["dup"]]

    def _tol_now(self) -> float:
        """Exact unless every score so far shows few decimals (then: rounded at that many decimals)."""
        d = max((n_decimals(x["score"]) for x in self._items()), default=17)
        return self.tol if d > 6 else max(self.tol, 0.51 * 10.0 ** -d)

    def _rebuild(self):
        items = self._items()
        if not items:
            self.dec, self.mode, self._hp, self._hd = None, "exact", None, None
            return
        tol = self._tol_now()
        exact = tol <= self.tol
        dec = Decode(self.pb.nw, self.pb.q, self.cf, tol=tol, max_hyp=20000 if exact else self.max_hyp_approx)
        if all(dec.add(x["s"], x["score"]) for x in items):
            self.dec, self.tol_used = dec, tol
            self.mode = "exact" if exact else "approx"
            self._sync()
            return
        self.dec, self._hp, self._hd = None, None, None
        self.mode = "sign"
        self._build_sign()

    def _k(self) -> int:
        return self.k_top if self.mode == "exact" else self.k_top_approx

    def _sync(self):
        d = self.dec
        self._onehot = (d.cls[:, None] == np.arange(len(d.sizes))[None, :]).astype(np.float32)
        self._sizes = d.sizes.astype(np.intp)
        self._hp, self._hd = d.hp[:self._k()].astype(np.intp), d.hd[:self._k()].astype(np.intp)

    def _build_sign(self):
        if self._sign_tab is None:
            self._sign_tab = sign_table(self.pb.q)
        items = self._items()
        self._tests = []
        for k in range(1, len(items)):
            r = min(range(k), key=lambda j: int((items[j]["s"] != items[k]["s"]).sum()))
            sk, sr = items[k]["s"], items[r]["s"]
            dlt = items[k]["score"] - items[r]["score"]
            sg = 1 if abs(dlt) <= 1e-9 else (2 if dlt > 0 else 0)
            self._tests.append((np.flatnonzero(sr & ~sk), np.flatnonzero(~sr & sk), sg))

    # ---- likelihood
    def _truth_pairs(self, L, h):
        Lt = np.where(h, self.pb.n + np.arange(self.pb.n)[None, :], L)
        return Lt[:, self.pb.ia] == Lt[:, self.pb.ib]

    def _class_counts(self, t):
        S = np.rint(t.astype(np.float32) @ self._onehot).astype(np.intp)
        return S, self._sizes[None, :] - S

    def _loglik_h(self, t, hp, hd):
        S, D = self._class_counts(t)
        return (self.LB[S[:, None, :], hp[None, :, :]] + self.LB[D[:, None, :], hd[None, :, :]]).sum(2)

    def _loglik(self, t):
        if self.mode in ("exact", "approx") and self._hp is not None:
            return logsumexp(self._loglik_h(t, self._hp, self._hd), axis=1)
        if self.mode == "sign" and self._tests:
            ll = np.zeros(t.shape[0])
            M = self._sign_tab.shape[0] - 1
            for f1, f2, sg in self._tests:
                good = (~t[:, f1]).sum(1) + t[:, f2].sum(1)
                bad = t[:, f1].sum(1) + (~t[:, f2]).sum(1)
                ll += self._sign_tab[np.minimum(good, M), np.minimum(bad, M), sg]
            return ll
        return np.zeros(t.shape[0])

    def _select_top(self):
        """Keep in the chains' likelihood the hypotheses with the largest marginal weight."""
        if self.dec is None or self.dec.hp is None:
            return
        hp, hd = self.dec.hp.astype(np.intp), self.dec.hd.astype(np.intp)
        if len(hp) <= self._k():
            self._hp, self._hd = hp, hd
            return
        t = self._truth_pairs(self.L[:64], self.h[:64])
        w = np.concatenate([logsumexp(self._loglik_h(t, hp[a:a + 512], hd[a:a + 512]), axis=0)
                            for a in range(0, len(hp), 512)])
        keep = np.argsort(-w)[:self._k()]
        self._hp, self._hd = hp[keep], hd[keep]

    # ---- MCMC
    def sweep(self, beta: float = 1.0):
        pb, C, n, rng = self.pb, self.C, self.pb.n, self.rng
        ar = np.arange(C)
        cur = self._loglik(self._truth_pairs(self.L, self.h)) if beta > 0 else None
        for i in rng.permutation(n):
            idx, pos = pb.fam_idx[i], pb.fam_pos[i]
            if len(idx) < 2:
                continue
            Lf = self.L[:, idx]
            same = Lf[:, :, None] == Lf[:, None, :]
            same[:, :, pos] = False
            w = pb.Wmat[i, idx]
            if pb.anchor is not None and not pb.anchor[i]:
                anc = same & pb.anchor[idx][None, None, :]
                gains = anc @ w + np.where(anc.any(2), 0.0, (same & ~anc) @ w)
            else:
                gains = same @ w
            logit = gains - np.log(np.maximum(same.sum(2), 1))
            logit[:, pos] = 0.0                                    # option "alone"
            choice = (logit + rng.gumbel(size=logit.shape)).argmax(1)
            pL = self.L.copy()
            pL[:, i] = np.where(choice == pos, n + i, Lf[ar, choice])
            ph, dprior = self.h, 0.0
            if pb.eta[i] > 0:
                flip = rng.random(C) < 0.5
                ph = self.h.copy()
                ph[:, i] ^= flip
                lo = math.log(pb.eta[i] / (1 - pb.eta[i]))
                dprior = np.where(flip, np.where(ph[:, i], lo, -lo), 0.0)
            lu = np.log(rng.random(C))
            if beta > 0:
                new = self._loglik(self._truth_pairs(pL, ph))
                acc = lu < beta * (new - cur) + dprior
                cur = np.where(acc, new, cur)
            else:
                acc = lu < dprior
            self.L = canon(np.where(acc[:, None], pL, self.L))
            if ph is not self.h:
                self.h = np.where(acc[:, None], ph, self.h)

    def swap_moves(self, beta: float = 1.0, n_moves: int | None = None):
        """Exchange the 'hidden variant' switch between two endpoints of one candidate block
        (which member of the block is the odd one?)."""
        pb, C, rng = self.pb, self.C, self.rng
        cands = np.flatnonzero(pb.eta > 0)
        if len(cands) < 2 or not self.use_swap:
            return
        cur = self._loglik(self._truth_pairs(self.L, self.h))
        lo = np.log(pb.eta[cands] / (1 - pb.eta[cands]))
        lo = dict(zip(cands.tolist(), lo.tolist()))
        for _ in range(n_moves if n_moves is not None else len(cands)):
            i = int(rng.choice(cands))
            mates = cands[(pb.cand[cands] == pb.cand[i]) & (cands != i)]
            if not len(mates):
                continue
            j = int(rng.choice(mates))
            differ = self.h[:, i] != self.h[:, j]
            if not differ.any():
                continue
            ph = self.h.copy()
            ph[:, i], ph[:, j] = self.h[:, j], self.h[:, i]
            dprior = np.where(ph[:, i], lo[i] - lo[j], lo[j] - lo[i])
            new = self._loglik(self._truth_pairs(self.L, ph))
            acc = differ & (np.log(rng.random(C)) < beta * (new - cur) + dprior)
            cur = np.where(acc, new, cur)
            self.h = np.where(acc[:, None], ph, self.h)

    def feasible(self) -> float:
        """Share of the chains whose truth can produce the decoded public counts (1.0 in sign mode)."""
        if self.mode == "sign" or self._hp is None:
            return 1.0
        S, D = self._class_counts(self._truth_pairs(self.L, self.h))
        ok = ((self._hp[None, :, :] <= S[:, None, :]) & (self._hd[None, :, :] <= D[:, None, :])).all(2).any(1)
        return float(ok.mean())

    def refresh(self, sweeps: int = 8, rescue: int | None = None):
        """Move the chains to the posterior of the current observations.  When most chains cannot
        reproduce the decoded counts (stuck in a wrong mode), run tempered rounds (rescue)."""
        if self.mode != "sign":
            self._select_top()
        for k in range(sweeps):
            self.sweep(1.0)
            self.swap_moves(1.0)
            if k == sweeps // 2 - 1 and self.mode != "sign":
                self._select_top()
        for _ in range(self.rescue if rescue is None else rescue):
            if self.feasible() >= 0.5:
                break
            for beta in np.linspace(0.1, 1.0, 2 * sweeps):
                self.sweep(float(beta))
                self.swap_moves(float(beta))
            self._select_top()
            for _ in range(max(2, sweeps // 2)):
                self.sweep(1.0)
                self.swap_moves(1.0)

    # ---- worlds
    def sample_worlds(self, n_rep: int = 8):
        """Samples (truth-same, public) over the within-family pairs: (Wn x nw) bool each."""
        pb, rng = self.pb, self.rng
        t = np.repeat(self._truth_pairs(self.L, self.h), n_rep, axis=0)
        Wn, nw = t.shape
        if self.mode != "sign" and self._hp is not None:
            lh = self._loglik_h(t, self._hp, self._hd)
            hc = (lh + rng.gumbel(size=lh.shape)).argmax(1)
            S, D = self._class_counts(t)
            cs, cd = np.minimum(self._hp[hc], S), np.minimum(self._hd[hc], D)
            cnt = np.stack([cd, cs], axis=2).reshape(Wn, -1)          # cell = 2 * class + truth
            cell = 2 * self.dec.cls[None, :] + t
            order = np.argsort(cell + rng.random((Wn, nw)), axis=1, kind="stable")
            csort = np.take_along_axis(cell, order, axis=1)
            idx = np.arange(nw)[None, :]
            first = np.concatenate([np.ones((Wn, 1), bool), csort[:, 1:] != csort[:, :-1]], axis=1)
            start = np.maximum.accumulate(np.where(first, idx, 0), axis=1)
            pub_sorted = (idx - start) < np.take_along_axis(cnt, csort, axis=1)
            pub = np.empty_like(pub_sorted)
            np.put_along_axis(pub, order, pub_sorted, axis=1)
        else:
            pub = rng.random((Wn, nw)) < pb.q
        return t, pub

    @staticmethod
    def _ctx(t, pub):
        f = np.float32
        return dict(A=(pub & t).astype(f), B=(pub & ~t).astype(f), C=(~pub & t).astype(f), D=(~pub & ~t).astype(f))

    def evaluate(self, cands, ctx):
        """Model-MCC of candidate partitions in every world: (public, private, all pairs), each (Wn x K)."""
        Sc = np.ascontiguousarray(self.pb.pairs_same(np.atleast_2d(cands)).astype(np.float32).T)
        out = []
        tps, fps, Ps, Ns = [], [], [], []
        for same, diff in ((ctx["A"], ctx["B"]), (ctx["C"], ctx["D"])):
            tp, fp = same @ Sc, diff @ Sc
            P, N = same.sum(1)[:, None], diff.sum(1)[:, None]
            out.append(mcc(tp, fp, P - tp, N - fp))
            tps.append(tp), fps.append(fp), Ps.append(P), Ns.append(N)
        tp, fp, P, N = sum(tps), sum(fps), sum(Ps), sum(Ns)
        out.append(mcc(tp, fp, P - tp, N - fp))
        return out

    # ---- decisions
    def scored(self):
        return [x for x in self.subs if x["kind"] == "model"]

    def incumbent(self):
        """Submission kept by the server: best public score, the earliest on ties."""
        best = None
        for x in self.scored():
            if best is None or x["score"] > best["score"] + 1e-12:
                best = x
        return best

    def propose(self, n_rep: int = 8, v_min: float = 1e-4, depth: int = 3, top: int = 6,
                max_cost: int = 200_000, risk: float = 2.0, safe_first: bool = False):
        """Next partition to submit.  Returns a dict (``lab`` is None when nothing is worth a slot).

        A single move from the incumbent is chosen by V = E[ 1(public beats the incumbent) * private gain ].
        Further moves are bundled with it only if they also raise the loss-averse value
        Vr = E[ 1(adopted) * u(gain) ], u(g) = g if g >= 0 else (1 + risk) * g: a change that the public
        split cannot confirm rides along only when the belief is strong (> (1 + risk) / (2 + risk))."""
        pb = self.pb
        inc = self.incumbent()
        if inc is None:
            return dict(lab=pb.cand.copy(), first=True, V=None, alternatives=[])
        t, pub = self.sample_worlds(n_rep)
        ctx = self._ctx(t, pub)
        p0, r0, f0 = (x[:, 0] for x in self.evaluate(inc["lab"][None], ctx))
        seen = {x["lab"].tobytes() for x in self.subs}

        def stats(lab, pubs, privs, fulls):
            adopt = pubs > p0 + 1e-9
            gain = W_MODEL * (privs - r0)
            na = max(int(adopt.sum()), 1)
            return dict(lab=lab, V=float((adopt * gain).mean()),
                        Vr=float((adopt * np.where(gain < 0, (1 + risk) * gain, gain)).mean()),
                        p_adopt=float(adopt.mean()),
                        p_tie=float((np.abs(pubs - p0) <= 1e-9).mean()),
                        gain_if_adopted=float((adopt * gain).sum() / na),
                        p_harm_if_adopted=float((adopt & (gain < -1e-9)).sum() / na),
                        p_right=float((fulls > f0 + 1e-9).mean()),
                        blind_gain=float(gain.mean()))

        best, base, alternatives = None, inc["lab"], []
        for d in range(depth):
            cands = [c for c in pb.neighbours(base) if c.tobytes() not in seen]
            cands = [c for c in cands if 0 < pb.pairs_same(c).sum() < pb.nw]
            if not cands:
                break
            arr = np.array(cands)
            pubs, privs, fulls = self.evaluate(arr, ctx)
            adopt = pubs > p0[:, None] + 1e-9
            gain = W_MODEL * (privs - r0[:, None])
            V = (adopt * gain).mean(0)
            Vr = (adopt * np.where(gain < 0, (1 + risk) * gain, gain)).mean(0)
            if d == 0:
                order = np.argsort(-V)
                alternatives = [stats(arr[k], pubs[:, k], privs[:, k], fulls[:, k]) for k in order[:top]]
                if safe_first and Vr.max() > v_min:        # tests whose adoption is safe go first
                    order = np.argsort(-Vr)
            else:                                   # bundle only what improves both values
                ok = (V >= best["V"]) & (Vr > best["Vr"] + 0.05 * abs(best["Vr"]) + 1e-6)
                order = [k for k in np.argsort(-Vr) if ok[k]]
            pick = None
            for k in order[:8]:
                if self.dec is None or self.dec.cost(pb.pairs_same(arr[k])) <= max_cost:
                    pick = k
                    break
            if pick is None or (d == 0 and V[pick] <= 0):
                break
            best = stats(arr[pick], pubs[:, pick], privs[:, pick], fulls[:, pick])
            best["depth"] = d + 1
            base = arr[pick]
        if best is None or best["V"] < v_min:
            return dict(lab=None, first=False, V=best["V"] if best else 0.0, alternatives=alternatives,
                        best=best)
        best.update(first=False, alternatives=alternatives)
        return best

    def assess(self, labs, n_rep: int = 8, risk: float = 2.0) -> list:
        """What given partitions would bring if submitted now (same quantities as ``propose``)."""
        inc = self.incumbent()
        t, pub = self.sample_worlds(n_rep)
        ctx = self._ctx(t, pub)
        arr = np.array([canon(np.asarray(x)) for x in labs])
        pubs, privs, fulls = self.evaluate(arr, ctx)
        if inc is None:
            p0 = r0 = f0 = np.zeros(t.shape[0])
        else:
            p0, r0, f0 = (x[:, 0] for x in self.evaluate(inc["lab"][None], ctx))
        out = []
        for k in range(len(arr)):
            adopt = pubs[:, k] > p0 + 1e-9
            gain = W_MODEL * (privs[:, k] - r0)
            na = max(int(adopt.sum()), 1)
            out.append(dict(lab=arr[k], V=float((adopt * gain).mean()),
                            Vr=float((adopt * np.where(gain < 0, (1 + risk) * gain, gain)).mean()),
                            p_adopt=float(adopt.mean()), p_tie=float((np.abs(pubs[:, k] - p0) <= 1e-9).mean()),
                            gain_if_adopted=float((adopt * gain).sum() / na),
                            p_harm_if_adopted=float((adopt & (gain < -1e-9)).sum() / na),
                            p_right=float((fulls[:, k] > f0 + 1e-9).mean()), blind_gain=float(gain.mean()),
                            exp_public=float(self.cf + W_MODEL * pubs[:, k].mean()),
                            pub_lo=float(self.cf + W_MODEL * np.quantile(pubs[:, k], 0.1)),
                            pub_hi=float(self.cf + W_MODEL * np.quantile(pubs[:, k], 0.9)),
                            exp_private=float(self.cf + W_MODEL * privs[:, k].mean())))
        return out

    def summary(self, n_rep: int = 8) -> dict:
        """Expected private model-MCC of every scored submission and probability to beat the first one."""
        t, pub = self.sample_worlds(n_rep)
        ctx = self._ctx(t, pub)
        subs = self.scored()
        if not subs:
            return dict(rows=[])
        arr = np.array([x["lab"] for x in subs])
        _, privs, _ = self.evaluate(arr, ctx)
        rows = []
        for k, x in enumerate(subs):
            sc = self.cf + W_MODEL * privs[:, k]
            rows.append(dict(score=x["score"], priv_mean=float(sc.mean()), priv_lo=float(np.quantile(sc, 0.1)),
                             priv_hi=float(np.quantile(sc, 0.9)),
                             p_worse_than_first=float((privs[:, k] < privs[:, 0] - 1e-9).mean())))
        return dict(rows=rows)

    def marginals(self) -> np.ndarray:
        """Posterior probability 'same model' for every within-family pair."""
        return self._truth_pairs(self.L, self.h).mean(0)

    def decoded(self, k: int | None = None):
        """Distinct public (TP, FP, FN, TN) decodes of model submission k (default: the last one)."""
        if self.dec is None or self.dec.hp is None:
            return None
        k = self.dec.n_sub - 1 if k is None else k
        c = self.dec.counts(k)
        return np.unique(c, axis=0)
