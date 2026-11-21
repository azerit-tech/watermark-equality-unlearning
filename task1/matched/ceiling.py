"""Ceiling of the matched filter: oracle filter = TRUE residual (Cw - C) on a half-watermarked synthetic image;
several whitenings and windows. Pixel AUC (right half watermarked). Run from TASK: python t1/matched/ceiling.py --method trustmark"""
import argparse, os, sys
from pathlib import Path
import cv2, numpy as np, torch
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/matched"))
from corr1 import lcorr, to128, auc  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=10)
a = ap.parse_args()
d = TASK / "t1/out/synth/v0" / a.method / "full" / "s0_0000"
C_all, Cw_all = np.load(d / "C.npy", mmap_mode="r"), np.load(d / "Cw.npy", mmap_mode="r")
G = np.zeros((128, 128), bool); G[:, 64:] = True; R = {}
for i in range(a.n):
    C, Cw = np.asarray(C_all[i]), np.asarray(Cw_all[i]); rt = Cw.astype(np.float32) - C.astype(np.float32)
    x = C.copy(); x[:, 256:] = Cw[:, 256:]; xf = x.astype(np.float32)
    W = {"hp07": xf - cv2.GaussianBlur(xf, (0, 0), 0.7), "hp15": xf - cv2.GaussianBlur(xf, (0, 0), 1.5),
         "hp4": xf - cv2.GaussianBlur(xf, (0, 0), 4), "med3": xf - cv2.medianBlur(x, 3).astype(np.float32),
         "med5": xf - cv2.medianBlur(x, 5).astype(np.float32)}
    for wn, h in W.items():
        for k in (16, 32, 64):
            R.setdefault(f"{wn}_k{k}", []).append(auc(to128(lcorr(h, rt, k)), G))
            R.setdefault(f"{wn}_k{k}_whitefilt", []).append(auc(to128(lcorr(h, rt - cv2.GaussianBlur(rt, (0, 0), 1.5), k)), G))
for k, v in sorted(R.items(), key=lambda kv: -np.mean(kv[1])): print(a.method, f"{k:22s} {np.mean(v):.3f}")
