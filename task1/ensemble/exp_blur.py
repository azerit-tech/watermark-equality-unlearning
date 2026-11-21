"""Essai complémentaire : empilement avec contexte spatial (logits et cartes de signal lissés) et lissage des probas.
    python t1/ensemble/exp_blur.py trustmark frozen_tm,tmdense/best
"""
import sys
import numpy as np
from scipy import ndimage
from explore import (discover, method_names, score, feats, stack_cv, logit, sigmoid, fuse, signal_feats, fullacc,
                     boot_paired, load_regimes, load_train)

method, names = sys.argv[1], sys.argv[2].split(",")
meta, gt = load_train(); reg, treg = load_regimes("train"), load_regimes("test")
vn, tn = method_names(method)
srcs = {s["name"]: (V, T) for s, V, T in discover(method, vn, tn)}
V = [srcs[n][0] for n in names]; T = [srcs[n][1] for n in names]
base = score(V[0], vn, reg, gt)
print(f"{method} base {names[0]} {base.mean():.4f}")
def blur(a, s):
    return np.stack([ndimage.gaussian_filter(x, s) for x in a])
for sig in (1, 2, 4):
    P = blur(V[0], sig); s = score(P, vn, reg, gt); print(f"  lissage {names[0]} sigma={sig}: {s.mean():.4f} gain {s.mean()-base.mean():+.4f} P={boot_paired(s, base)[0]:.2f}")
    if len(V) > 1:
        P = blur(fuse(V, None, "max"), sig); s = score(P, vn, reg, gt); print(f"  lissage max sigma={sig}: {s.mean():.4f} gain {s.mean()-base.mean():+.4f} P={boot_paired(s, base)[0]:.2f}")
r_v = np.array([1.0 if reg[n] else -1.0 for n in vn]); r_t = np.array([1.0 if treg[n] else -1.0 for n in tn])
acc_v, acc_t = fullacc(method, vn, "train"), fullacc(method, tn, "test")
sig_v, sig_t = signal_feats(method, vn, "train"), signal_feats(method, tn, "test")
Y = np.stack([gt[n] for n in vn]).astype(np.float32)
Lv = [logit(v) for v in V]; Lt = [logit(t) for t in T]
for use_sig in ((False, True) if sig_v is not None else (False,)):
    ev = Lv + [blur(l, 3) for l in Lv] + [blur(l, 8) for l in Lv]
    et = Lt + [blur(l, 3) for l in Lt] + [blur(l, 8) for l in Lt]
    sv = [blur(np.clip(x, -5, 5), 4) for x in sig_v] if use_sig else None
    st = [blur(np.clip(x, -5, 5), 4) for x in sig_t] if use_sig else None
    Fv = feats(ev, sv, r_v, acc_v, True); Ft = feats(et, st, r_t, acc_t, True)
    oof, pt, _ = stack_cv(Fv, Y, Ft)
    s = score(oof, vn, reg, gt)
    p, q5 = boot_paired(s, base)
    print(f"  stack contexte sig={int(use_sig)}: {s.mean():.4f} gain {s.mean()-base.mean():+.4f} P={p:.2f} q5={q5:+.3f}")
