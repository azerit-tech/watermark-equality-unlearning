"""Honest selection of FEW-parameter recipes on the twin: fit on half A of F against 2 golds, score on half B against
the held-out 3rd gold (2 random class-stratified splits x 2 directions x 3 golds = 12 out-of-sample scores).

  python recipe_cv.py same|view [n_splits]
mode same: one target per image, also on the altered copies (what unlearn.py trains)
mode view: altered copies get recipe_train.py's per-view target (view-loss rank: teacher view logits, ViewRanker)
Fixed everywhere (choices made a priori / by earlier twin work): ranking = global rank of the original's clean CE,
class tilt beta 0.25, other classes softmax(logits) tau 1 + 1e-3 floor. Free parameters per family:
  ln2   mean, std               (lognormal losses by rank)
  two3  err, mr, mw             (wrong share; lognormal mean of right losses, sd 0.3*... fixed; excess of wrong ones)
  cal2  a, b                    (T=2 logistic p_y = sigmoid(a - b probit(u)))
each with rest = orig (original's softmax) or cfmix (0.5/0.5 log-mix with the cf agent's onestep LOO prediction).
"""
import sys
import time

import torch
import torch.nn.functional as F

from recipe_fit import assemble, difficulty, flip_of, lognorm_q, load_twin, score
from recipe_train import ViewRanker, view_targets

torch.set_num_threads(4)
MODE = sys.argv[1] if len(sys.argv) > 1 else "same"
NSPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 2
tw = load_twin()
y, z = tw["y"], tw["O"]["F"]
n, C = z.shape
SEEDS = (1, 2, 3)  # seed 0 is the ViewRanker's reference pass
u = difficulty(z, y, tw["ccount"], "ce", False, 0.25)
order = u.argsort()


def rest(kind):
    lq = F.log_softmax(z.double().scatter(1, y.view(-1, 1), float("-inf")), 1)
    if kind == "cfmix":
        cf = torch.load("runs/cf/twin_pb10q/cf_all_F.pt", weights_only=False)["onestep"]
        p1 = cf["probs"].double().clamp_min(1e-12)
        lo = F.log_softmax(p1.log().scatter(1, y.view(-1, 1), float("-inf")), 1)
        lq = F.log_softmax((0.5 * lq + 0.5 * lo).scatter(1, y.view(-1, 1), float("-inf")), 1)
    q = (1 - 1e-3) * lq.exp() + 1e-3 / (C - 1)
    return q.scatter(1, y.view(-1, 1), 0).clamp_min(1e-300).log().scatter(1, y.view(-1, 1), float("-inf"))


LQ = {k: rest(k) for k in ("orig", "cfmix")}
FLIP = {k: flip_of(v) for k, v in LQ.items()}


def target(fam, P, rk):
    lq = LQ[rk]
    L = torch.empty(n, dtype=torch.float64)
    if fam == "ln2":
        L[order] = lognorm_q(n, P["mean"], P["std"]).clamp_max(18.0)
    elif fam == "two3":
        nw = round(P["err"] * n)
        right, wrong = order[:n - nw], order[n - nw:]
        L[right] = torch.minimum(lognorm_q(len(right), P["mr"], 2 * P["mr"]), FLIP[rk][right] - 0.05)
        if nw:
            L[wrong] = FLIP[rk][wrong] + 0.05 + lognorm_q(nw, P["mw"], P["mw"])
    elif fam == "cal2":
        p2y = torch.sigmoid(P["a"] - P["b"] * torch.special.ndtri(u.clamp(1e-6, 1 - 1e-6)))
        # T=2 p_y -> T=1 loss given the rest shape at T=2: p1_y = p2y^2 / (p2y^2 + (1-p2y)^2 * sum q2^2)
        q2 = F.softmax(lq / 2, 1)
        s2 = (q2 ** 2).sum(1)
        p1y = p2y ** 2 / (p2y ** 2 + (1 - p2y) ** 2 * s2)
        L = -p1y.clamp_min(1e-8).log()
    return assemble(L, lq, y), L.float()


RANKER = None


def ev(fam, P, rk, idx, gold_idx):
    global RANKER
    t, L = target(fam, P, rk)
    ta = None
    if MODE == "view":
        if RANKER is None or RANKER[0] is not L:
            RANKER = (L, ViewRanker(tw["O"]["F_alter_seeds"][0], y, L, 0.25))
        ta = [None] + [view_targets(tw["O"]["F_alter_seeds"][s], y, L, 1.0, 1e-3, "rank", RANKER[1]) for s in SEEDS]
    return score(tw, t, ta, gold_idx, idx, seeds=SEEDS)


def lin(a, b, k):
    return [round(a + (b - a) * i / (k - 1), 4) for i in range(k)]


FAMS = {"ln2": ({"mean": 0.45, "std": 1.0}, {"mean": lin(0.25, 0.8, 12), "std": lin(0.5, 2.0, 16)}),
        "two3": ({"err": 0.16, "mr": 0.15, "mw": 1.0}, {"err": lin(0.06, 0.26, 11), "mr": [0.03, 0.06, 0.1, 0.15, 0.2, 0.3],
                                                        "mw": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]}),
        "cal2": ({"a": -0.5, "b": 1.0}, {"a": lin(-2, 2, 17), "b": lin(0.25, 2.5, 10)})}


def fit(fam, rk, idx, gold_idx):
    P0, grid = FAMS[fam]
    P, best = dict(P0), ev(fam, P0, rk, idx, gold_idx)["forget"]
    for _ in range(4):
        changed = False
        for k, vals in grid.items():
            for v in vals:
                if P[k] == v:
                    continue
                Q = dict(P, **{k: v})
                f = ev(fam, Q, rk, idx, gold_idx)["forget"]
                if f > best + 1e-6:
                    P, best, changed = Q, f, True
        if not changed:
            break
    return P, best


g = torch.Generator().manual_seed(1)
splits = []
for r in range(NSPLIT):
    a = torch.zeros(n, dtype=torch.bool)
    for c in y.unique():
        s = (y == c).nonzero().squeeze(1)
        s = s[torch.randperm(len(s), generator=g)]
        a[s[: (len(s) + r % 2) // 2]] = True
    A, B = a.nonzero().squeeze(1), (~a).nonzero().squeeze(1)
    splits += [(A, B), (B, A)]

t0 = time.time()
print(f"mode {MODE}: {len(splits)} half-splits x 3 held-out golds; scores = forget on the OTHER half vs the HELD-OUT gold")
print(f"{'recipe':<12} {'in-sample':>9} | {'CV mean':>7} {'sd':>6} | {'CV ms':>6} {'CV pred':>7} | params fitted on all F/3 golds")
res = {}
for fam in FAMS:
    for rk in ("orig", "cfmix"):
        Pall, fall = fit(fam, rk, None, (0, 1, 2))
        cv = []
        for A, B in splits:
            for k in range(3):
                pair = tuple(j for j in range(3) if j != k)
                P, _ = fit(fam, rk, A, pair)
                cv.append(ev(fam, P, rk, B, (k,)))
        f = torch.tensor([c["forget"] for c in cv])
        ms = torch.tensor([c["ms"] for c in cv]).mean()
        pr = torch.tensor([c["pred"] for c in cv]).mean()
        res[f"{fam}/{rk}"] = {"insample": fall, "P": Pall, "cv": f.tolist(), "cv_ms": ms.item(), "cv_pred": pr.item()}
        print(f"{fam + '/' + rk:<12} {fall:9.4f} | {f.mean():7.4f} {f.std():6.4f} | {ms:6.3f} {pr:7.3f} | {Pall}  ({time.time() - t0:.0f}s)", flush=True)
# reference: the a-priori G2 at the twin-gold level, no fit at all, scored the same way (other half vs held-out gold)
for name, (fam, P) in {"G2 prior": ("ln2", {"mean": 0.53, "std": 1.17}), "G2 low": ("ln2", {"mean": 0.4, "std": 0.9})}.items():
    cv = [ev(fam, P, "orig", B, (k,)) for A, B in splits for k in range(3)]
    f = torch.tensor([c["forget"] for c in cv])
    print(f"{name:<12} {'-':>9} | {f.mean():7.4f} {f.std():6.4f} | {torch.tensor([c['ms'] for c in cv]).mean():6.3f} "
          f"{torch.tensor([c['pred'] for c in cv]).mean():7.3f} | {P} (no fit)")
    res[name] = {"cv": f.tolist(), "P": P}
# ceiling under the same protocol: each gold vs the held-out gold, other half
cv = []
for A, B in splits:
    for k in range(3):
        for j in range(3):
            if j != k:
                cv.append(score(tw, tw["G"][j]["F"], [None] + [tw["G"][j]["F_alter_seeds"][s] for s in SEEDS], (k,), B, seeds=SEEDS)["forget"])
print(f"{'gold ceiling':<12} {'-':>9} | {torch.tensor(cv).mean():7.4f} {torch.tensor(cv).std():6.4f} | (another gold's own outputs)")
torch.save(res, f"runs/recipe/cv_{MODE}.pt")
print(f"done {time.time() - t0:.0f}s")
