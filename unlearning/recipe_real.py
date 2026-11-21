"""Apply the CV-selected 'two3' recipe (recipe_cv.py) to the REAL model (runs/diag/diag.pt) and to the twin.

two3: images ranked by the global rank of the original's clean CE + 0.25 * z(log class forget-count); the hardest `err`
share is made wrong: L = flip + 0.05 + lognormal(mean mw, sd mw) excess (flip = loss at which p_y equals the runner-up);
the others get L = lognormal(mean mr, sd 2 mr) by rank, capped at flip - 0.05; other classes softmax(clean logits) + 1e-3
floor (tau 1). Three free parameters, fitted on the twin with CV over halves x held-out gold.

  python recipe_real.py      -> runs/recipe/real_T3*.pt, runs/recipe/twin_T3_probs.pt, prints realised stats
"""
import torch
import torch.nn.functional as F

from recipe_fit import assemble, describe, difficulty, flip_of, lognorm_q, load_twin, score, fmt

torch.set_num_threads(4)


def two3(z, y, ccount, err, mr, mw, beta=0.25, floor=1e-3):
    n, C = z.shape
    u = difficulty(z, y, ccount, "ce", False, beta)
    order = u.argsort()
    lq = F.log_softmax(z.double().scatter(1, y.view(-1, 1), float("-inf")), 1)
    q = ((1 - floor) * lq.exp() + floor / (C - 1)).scatter(1, y.view(-1, 1), 0)
    lq = q.clamp_min(1e-300).log().scatter(1, y.view(-1, 1), float("-inf"))
    flip = flip_of(lq)
    L = torch.empty(n, dtype=torch.float64)
    nw = round(err * n)
    right, wrong = order[:n - nw], order[n - nw:]
    L[right] = torch.minimum(lognorm_q(len(right), mr, 2 * mr), flip[right] - 0.05)
    L[wrong] = (flip[wrong] + 0.05 + lognorm_q(nw, mw, mw)).clamp_max(18.0)
    return assemble(L, lq, y), flip


def two3_sw(z, y, ccount, err, mr, mw, sw, beta=0.25, floor=1e-3):
    """two3 with a separate sd of the wrong images' excess loss (for calibrating a heavy real-like tail)."""
    n, C = z.shape
    u = difficulty(z, y, ccount, "ce", False, beta)
    order = u.argsort()
    lq = F.log_softmax(z.double().scatter(1, y.view(-1, 1), float("-inf")), 1)
    q = ((1 - floor) * lq.exp() + floor / (C - 1)).scatter(1, y.view(-1, 1), 0)
    lq = q.clamp_min(1e-300).log().scatter(1, y.view(-1, 1), float("-inf"))
    flip = flip_of(lq)
    L = torch.empty(n, dtype=torch.float64)
    nw = round(err * n)
    right, wrong = order[:n - nw], order[n - nw:]
    L[right] = torch.minimum(lognorm_q(len(right), mr, 2 * mr), flip[right] - 0.05)
    L[wrong] = (flip[wrong] + 0.05 + lognorm_q(nw, mw, sw)).clamp_max(18.0)
    return assemble(L, lq, y), flip


def calibrate(z, y, cc, err, mr, want_m, want_s):
    """Grid + refine over (mw, sw) so the realised (1e-8 clamped) CE mean/std hit the request."""
    best = None
    for mw in [0.5 * 1.15 ** i for i in range(25)]:
        for sw in [0.5 * 1.15 ** i for i in range(25)]:
            _, st = stats(two3_sw(z, y, cc, err, mr, mw, sw)[0], y)
            d = abs(st["ce"] - want_m) + abs(st["std"] - want_s)
            if best is None or d < best[0]:
                best = (d, mw, sw, st)
    return best


def stats(t, y):
    p = t.double().exp()
    p = p / p.sum(1, keepdim=True)
    return p, describe(p.clamp_min(1e-8).log().float(), y)


LEVELS = {"T3": (0.20, 0.15, 1.0),      # twin CV fit, verbatim (twin realised CE 0.53 std 0.93 acc 0.80)
          "T3lo": (0.16, 0.12, 1.0),    # err low
          "T3hi": (0.20, 0.15, 1.8)}    # heavier wrong tail: towards the bridge-rule std
tw = load_twin()
for name, (e, mr, mw) in LEVELS.items():
    t, _ = two3(tw["O"]["F"], tw["y"], tw["ccount"], e, mr, mw)
    print(fmt(f"twin {name}", score(tw, t)))
    if name == "T3":
        p, _ = stats(t, tw["y"])
        torch.save({"probs": p.float(), "info": f"twin two3 {LEVELS[name]}", "pseudo_labels": tw["y"]}, "runs/recipe/twin_T3_probs.pt")

HEAVY = {"T3x": (0.16, 0.15, 0.65, 1.85), "T3xx": (0.16, 0.15, 0.80, 2.20)}  # err, mr, wanted realised mean, std
d = torch.load("runs/diag/diag.pt", map_location="cpu", weights_only=False)
z, y = d["clean_logits"].float(), d["pseudo_labels"].long()
cc = torch.bincount(y, minlength=100).float()
for name, (e, mr, mw) in LEVELS.items():
    t, flip = two3(z, y, cc, e, mr, mw)
    p, st = stats(t, y)
    info = (f"real_{name}: recipe 'two3' (recipe_real.py) err {e} mr {mr} (sd {2 * mr}) mw {mw} (sd {mw}); ranking = clean CE + "
            f"0.25*z(log class count); others softmax(clean logits) tau 1 + 1e-3 floor. Realised CE {st['ce']:.3f} std {st['std']:.3f} "
            f"acc {st['acc']:.3f} T2-entropy {st['ent2']:.2f}; flip loss median {flip.median():.2f}")
    torch.save({"probs": p.float(), "info": info, "pseudo_labels": y, "params": {"err": e, "mr": mr, "mw": mw}}, f"runs/recipe/real_{name}.pt")
    print(info)

for name, (e, mr, wm, ws) in HEAVY.items():
    dist, mw, sw, _ = calibrate(z, y, cc, e, mr, wm, ws)
    t, flip = two3_sw(z, y, cc, e, mr, mw, sw)
    p, st = stats(t, y)
    Lw = -t.double().log_softmax(1).gather(1, y.view(-1, 1)).squeeze(1)
    info = (f"real_{name}: recipe 'two3' heavy tail (recipe_real.py) err {e} mr {mr} (sd {2 * mr}); wrong excess lognormal(mean {mw:.2f}, "
            f"sd {sw:.2f}) calibrated to realised {wm}/{ws}; ranking = clean CE + 0.25*z(log class count); others softmax(clean logits) "
            f"tau 1 + 1e-3 floor. Realised CE {st['ce']:.3f} std {st['std']:.3f} acc {st['acc']:.3f} T2-entropy {st['ent2']:.2f}; "
            f"wrong-image CE median {Lw[t.argmax(1) != y].median():.2f}")
    torch.save({"probs": p.float(), "info": info, "pseudo_labels": y, "params": {"err": e, "mr": mr, "mw": mw, "sw": sw}},
               f"runs/recipe/real_{name}.pt")
    print(info, flush=True)
    tt, _ = two3_sw(tw["O"]["F"], tw["y"], tw["ccount"], e, mr, mw, sw)
    print(fmt(f"  same params on the twin (gold 0.53/0.96)", score(tw, tt)))
