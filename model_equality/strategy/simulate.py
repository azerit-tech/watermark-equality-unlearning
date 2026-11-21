"""Simulation of submission strategies on synthetic truths (no API call).

World: 33 endpoints in families (default 3 x 11, as observed on the real task).
Each family has a base model with lures (system prompt, persona, watermark,
temperature: SAME model) and real variants (fp8, int8, LoRA 0.3 / 0.7 / 1.5,
sometimes a lure copy of a variant).  Evidence (p_model per pair) mimics what
our methods see: LoRA >= 0.7 clearly different, fp8 borderline, LoRA 0.3 and
half of the int8 indistinguishable from the base, some lures looking different.

Server: public split = a random 30 % of the 528 pairs; the score is exact
(or rounded with --decimals); the submission kept for the final ranking is the
best public one (the earliest on ties) and is judged on the other pairs.

Strategies: ``naive`` (the candidate partition only), ``greedy`` (hill
climbing on single moves ranked by p_model, keeps what improves the public
score), ``bayes`` (model_equality.strategy.core).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .core import W_FAMILY, W_MODEL, Belief, Problem, canon, mcc

EPS = [f"ep-{i:02d}" for i in range(1, 34)]


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def make_world(rng, sizes=(11, 11, 11), p_fa: float = 0.12, p_int8_hidden: float = 0.5) -> dict:
    """Random truth and evidence.  Returns fam, truth, kind, P (n x n p_model)."""
    n = int(sum(sizes))
    perm = rng.permutation(n)
    fam = np.empty(n, int)
    truth = np.empty(n, int)
    look = np.empty(n, int)             # model the endpoint is indistinguishable from
    amb = np.full(n, np.nan)            # borderline evidence against its base group (endpoint-level log-odds)
    kind = [""] * n
    ref = np.empty(n, int)
    nxt, start = 0, 0
    for f, m in enumerate(sizes):
        idx = perm[start:start + m]
        start += m
        fam[idx] = f
        n_same = min(int(rng.choice([4, 5, 6, 7], p=[0.2, 0.3, 0.3, 0.2])), m - 2)
        base = nxt
        nxt += 1
        for k, i in enumerate(idx[:n_same]):
            truth[i], look[i] = base, base
            kind[i] = "base" if k == 0 else str(rng.choice(["sysprompt", "persona", "watermark", "temp"]))
            if k > 0 and rng.random() < p_fa:
                amb[i] = rng.normal(-0.3, 1.0)          # lure that looks different
        core = ["fp8", "lora0.3", "lora0.7", "lora1.5"]
        rng.shuffle(core)
        placed = []
        for j, i in enumerate(idx[n_same:]):
            kd = core[j] if j < len(core) else str(rng.choice(["int8", "lora_other", "twin"], p=[0.4, 0.3, 0.3]))
            if kd == "twin" and placed:
                src = placed[int(rng.integers(len(placed)))]
                truth[i], look[i], kind[i] = truth[src], look[src], "twin:" + kind[src]
                if not np.isnan(amb[src]):
                    amb[i] = amb[src] + rng.normal(0, 0.3)
                continue
            if kd == "twin":
                kd = "lora_other"
            truth[i], kind[i] = nxt, kd
            nxt += 1
            if kd == "lora0.3" or (kd == "int8" and rng.random() < p_int8_hidden):
                look[i] = base                           # undetectable
            else:
                look[i] = truth[i]
                if kd in ("fp8", "int8"):
                    amb[i] = rng.normal(-1.0, 1.3)       # borderline against the base group
            placed.append(i)
        ref[idx] = base                                  # the group a borderline endpoint is compared with
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if fam[i] != fam[j]:
                continue
            ai, aj = not np.isnan(amb[i]), not np.isnan(amb[j])
            if not ai and not aj:
                lg = rng.normal(2.2, 0.6) if look[i] == look[j] else rng.normal(-4.0, 1.0)
            elif ai and aj:
                lg = (rng.normal((amb[i] + amb[j]) / 2 + 1.0, 0.7) if truth[i] == truth[j]
                      else rng.normal(min(amb[i], amb[j]) - 1.0, 0.8))
            else:
                a, c = (i, j) if ai else (j, i)          # a borderline, c clear
                lg = rng.normal(amb[a], 0.5) if look[c] == ref[a] else rng.normal(-4.0, 1.0)
            P[i, j] = P[j, i] = _sig(lg)
    return dict(fam=fam, truth=canon(truth), kind=kind, P=P, n=n)


def make_world_split(rng, sizes=(11, 11, 11), p_wrong: float = 0.15, hint: float = 1.3,
                     p_twin: float = 0.15) -> dict:
    """The situation met on the real task: per family the candidate groups 2 to 4 lures we recognised
    (with probability ``p_wrong`` one of them is in fact another model) and leaves every other endpoint
    alone, although the base group really holds 5 to 8 endpoints.  Evidence is weak: one log-odds per lone
    endpoint against the group (true members higher by ``hint`` on average), nothing between lone endpoints."""
    n = int(sum(sizes))
    perm = rng.permutation(n)
    fam, truth, cand = np.empty(n, int), np.empty(n, int), np.empty(n, int)
    kind = [""] * n
    P = np.zeros((n, n))
    nxt, start = 0, 0
    for f, m in enumerate(sizes):
        idx = perm[start:start + m]
        start += m
        fam[idx] = f
        b = min(int(rng.choice([5, 6, 7, 8], p=[0.3, 0.3, 0.25, 0.15])), m - 2)
        base = nxt
        nxt += 1
        truth[idx[:b]] = base
        for i in idx[:b]:
            kind[i] = "base-like"
        var = idx[b:]
        for k, i in enumerate(var):
            if k == len(var) - 1 and len(var) >= 2 and rng.random() < p_twin:
                truth[i], kind[i] = truth[var[0]], "twin"
            else:
                truth[i], kind[i] = nxt, "variant"
                nxt += 1
        g = int(rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2]))
        group = list(idx[:g])
        if rng.random() < p_wrong:
            group[-1] = var[0]                           # a variant wrongly taken for a lure
        cand[idx] = n + idx                              # everyone alone ...
        cand[group] = n + group[0]                       # ... except the recognised lures
        conf = {i: (rng.uniform(0.8, 0.95) if truth[i] == base else rng.uniform(0.65, 0.9)) for i in group}
        for i in idx:
            for j in idx:
                if i >= j:
                    continue
                if i in conf and j in conf:
                    v = (conf[i] + conf[j]) / 2
                elif i in conf or j in conf:
                    continue
                else:
                    v = 0.05
                P[i, j] = P[j, i] = v
        for i in idx:
            if i in conf:
                continue
            v = _sig(rng.normal(-0.85 + (hint if truth[i] == base else 0.0), 1.0))
            for j in group:
                P[i, j] = P[j, i] = v
    return dict(fam=fam, truth=canon(truth), kind=kind, P=P, n=n, cand=canon(cand))


def naive_candidate(world) -> np.ndarray:
    """Our 'best estimate' without the leaderboard: average linkage at 0.5 inside each family."""
    from model_equality.scoring.partition import partition
    n = world["n"]
    eps = EPS[:n]
    M = world["P"].copy()
    np.fill_diagonal(M, 1.0)
    _, mod = partition(eps, M, 0.5, families={e: f"F{world['fam'][i]}" for i, e in enumerate(eps)})
    ids: dict = {}
    return canon(np.array([ids.setdefault(mod[e], len(ids)) for e in eps]))


class Server:
    """Simulated scoring server: fixed public split, exact score."""

    def __init__(self, world, rng, q: float = 0.3, decimals: int | None = None, iid: bool = False):
        n = world["n"]
        self.ia, self.ib = np.triu_indices(n, 1)
        npairs = len(self.ia)
        if iid:
            self.pub = rng.random(npairs) < q
        else:
            self.pub = np.zeros(npairs, bool)
            self.pub[rng.choice(npairs, size=int(round(q * npairs)), replace=False)] = True
        self.fam, self.truth, self.decimals = world["fam"], world["truth"], decimals
        self.fsame = self.fam[self.ia] == self.fam[self.ib]
        self.tsame = self.truth[self.ia] == self.truth[self.ib]

    def _score(self, lab, mask) -> float:
        psame = (lab[self.ia] == lab[self.ib]) & self.fsame
        m = mask & self.fsame
        t, p = self.tsame[m], psame[m]
        mm = mcc((t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum())
        f = self.fsame[mask]                             # predicted families = true families
        mf = 1.0 if 0 < f.sum() < len(f) else 0.0
        return float(W_FAMILY * mf + W_MODEL * mm)

    def public(self, lab) -> float:
        s = self._score(lab, self.pub)
        return round(s, self.decimals) if self.decimals is not None else s

    def private(self, lab) -> float:
        return self._score(lab, ~self.pub)

    def full(self, lab) -> float:
        return self._score(lab, np.ones_like(self.pub))


def problem_of(world, cand, q=0.3, **kw) -> Problem:
    eps = EPS[:world["n"]]
    return Problem(eps, {e: f"F{world['fam'][i]}" for i, e in enumerate(eps)},
                   {e: f"m{cand[i]}" for i, e in enumerate(eps)}, world["P"], q=q, **kw)


def _kept(history):
    """Index of the submission the server keeps: best public score, earliest on ties."""
    best = 0
    for k, (_, s) in enumerate(history):
        if s > history[best][1] + 1e-12:
            best = k
    return best


def run_bayes(world, server, cand, slots: int, q: float = 0.3, chains: int = 256, sweeps: int = 8,
              seed: int = 0, propose_kw: dict | None = None, belief_kw: dict | None = None, **kw) -> dict:
    pb = problem_of(world, cand, q=q, **kw)
    bel = Belief(pb, n_chains=chains, seed=seed, **(belief_kw or {}))
    history, modes = [], []
    lab = pb.cand
    for _ in range(slots):
        sc = server.public(lab)
        history.append((lab, sc))
        bel.observe(lab, sc)
        modes.append(bel.mode)
        if len(history) == slots:
            break
        bel.refresh(sweeps)
        prop = bel.propose(**(propose_kw or {}))
        if prop["lab"] is None:
            break
        lab = prop["lab"]
    return dict(history=history, mode=modes[-1], belief=bel)


def run_greedy(world, server, cand, slots: int, q: float = 0.3, floor: float = 0.05) -> dict:
    """Hill climbing without a model: try the single move that p_model finds most plausible, keep it if
    the public score strictly improves."""
    pb = problem_of(world, cand, q=q)
    history = [(pb.cand, server.public(pb.cand))]
    tried = {pb.cand.tobytes()}
    P = np.clip(world["P"], 0.02, 0.98)[pb.ia, pb.ib]
    while len(history) < slots:
        inc_lab, _ = history[_kept(history)]
        s0 = pb.pairs_same(inc_lab)
        best, best_v = None, floor
        for c in pb.neighbours(inc_lab):
            if c.tobytes() in tried:
                continue
            s1 = pb.pairs_same(c)
            if not 0 < s1.sum() < pb.nw:
                continue
            flip = s0 != s1
            v = float(np.where(s1[flip], P[flip], 1 - P[flip]).mean())
            if v > best_v:
                best, best_v = c, v
        if best is None:
            break
        tried.add(best.tobytes())
        history.append((best, server.public(best)))
    return dict(history=history)


def one_world(seed: int, slots: int = 20, q_true: float = 0.3, q_model: float = 0.3, chains: int = 256,
              sweeps: int = 8, decimals: int | None = None, sizes=(11, 11, 11), greedy: bool = True,
              world_kw: dict | None = None, model_kw: dict | None = None, propose_kw: dict | None = None,
              belief_kw: dict | None = None, scenario: str = "merge") -> dict:
    """scenario "merge": our best estimate merges the undetectable variants (make_world);
    scenario "split": recognised lures grouped, everything else alone (make_world_split)."""
    rng = np.random.default_rng(seed)
    if scenario == "split":
        world = make_world_split(rng, sizes=sizes, **(world_kw or {}))
        cand = world["cand"]
        model_kw = dict(anchored=True, **(model_kw or {}))
    else:
        world = make_world(rng, sizes=sizes, **(world_kw or {}))
        cand = naive_candidate(world)
    server = Server(world, rng, q=q_true, decimals=decimals)
    out = dict(seed=seed, naive_pub=server.public(cand), naive_priv=server.private(cand),
               naive_full=server.full(cand))
    t0 = time.time()
    runs = {"bayes": run_bayes(world, server, cand, slots, q=q_model, chains=chains, sweeps=sweeps, seed=seed,
                               propose_kw=propose_kw, belief_kw=belief_kw, **(model_kw or {}))}
    out["bayes_time"] = time.time() - t0
    out["bayes_mode"] = runs["bayes"]["mode"]
    if greedy:
        runs["greedy"] = run_greedy(world, server, cand, slots, q=q_model)
    for name, r in runs.items():
        hist = r["history"]
        k = _kept(hist)
        out[f"{name}_n"] = len(hist)
        out[f"{name}_pub"] = hist[k][1]
        out[f"{name}_priv"] = server.private(hist[k][0])
        out[f"{name}_full"] = server.full(hist[k][0])
        traj, harm = [], 0
        for j in range(1, slots + 1):
            kk = _kept(hist[:j])
            traj.append(server.private(hist[kk][0]))
            if j > 1 and traj[-1] < traj[-2] - 1e-12:
                harm += 1
        out[f"{name}_traj"] = traj
        out[f"{name}_harmful_adoptions"] = harm
        out[f"{name}_adoptions"] = sum(1 for j in range(1, len(hist)) if _kept(hist[:j + 1]) == j)
    return out


def _job(args):
    seed, kw = args
    return one_world(seed, **kw)


def run_many(n_worlds: int = 100, jobs: int = 4, seed0: int = 1000, **kw) -> list:
    args = [(seed0 + i, kw) for i in range(n_worlds)]
    if jobs <= 1:
        return [_job(a) for a in args]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(jobs) as pool:
        return pool.map(_job, args, chunksize=1)


def report(res: list, slots: int) -> str:
    def col(k):
        return np.array([r[k] for r in res], float)

    n = len(res)
    lines = [f"mondes simulés : {n} ; créneaux : {slots}"]
    nv = col("naive_priv")
    lines.append(f"{'stratégie':10s} {'privé moyen':>11s} {'gain moyen':>10s} {'± e.t.':>7s} {'médiane':>8s} "
                 f"{'P(gain>0)':>9s} {'P(perte)':>8s} {'perte moy.':>10s} {'pire':>7s} {'public-privé':>12s} {'soumis':>6s}")
    lines.append(f"{'naive':10s} {nv.mean():11.4f} {'':>10s} {'':>7s} {'':>8s} {'':>9s} {'':>8s} {'':>10s} {'':>7s} "
                 f"{(col('naive_pub') - nv).mean():12.4f} {1:6.1f}")
    for name in ("greedy", "bayes"):
        if f"{name}_priv" not in res[0]:
            continue
        pv = col(f"{name}_priv")
        g = pv - nv
        loss = g < -1e-9
        lines.append(f"{name:10s} {pv.mean():11.4f} {g.mean():+10.4f} {g.std(ddof=1) / math.sqrt(n):7.4f} "
                     f"{np.median(g):+8.4f} {(g > 1e-9).mean():9.2f} {loss.mean():8.2f} "
                     f"{(g[loss].mean() if loss.any() else 0.0):+10.4f} {g.min():+7.3f} "
                     f"{(col(f'{name}_pub') - pv).mean():12.4f} {col(f'{name}_n').mean():6.1f}")
    for name in ("greedy", "bayes"):
        if f"{name}_traj" not in res[0]:
            continue
        tr = np.array([r[f"{name}_traj"] for r in res])
        marks = [k for k in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30) if k <= slots]
        lines.append(f"{name} : gain privé moyen après k créneaux : " +
                     "  ".join(f"k={k}:{(tr[:, k - 1] - nv).mean():+.4f}" for k in marks))
        ad, hm = col(f"{name}_adoptions").sum(), col(f"{name}_harmful_adoptions").sum()
        lines.append(f"{name} : adoptions (le public s'améliore) : {int(ad)} ; dont nuisibles au privé : {int(hm)} "
                     f"({hm / max(ad, 1):.1%})")
    if "bayes_mode" in res[0]:
        modes = [r["bayes_mode"] for r in res]
        lines.append("bayes : mode de décodage final : " + ", ".join(f"{m}={modes.count(m)}" for m in sorted(set(modes))))
        lines.append(f"bayes : temps moyen par monde : {col('bayes_time').mean():.1f} s")
    return "\n".join(lines)
