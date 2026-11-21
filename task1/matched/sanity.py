"""Upper bound / sanity of the matched filter on synthetic pairs (C clean, Cw = embed(C, m)):
(1) global corr between true residual r = Cw - C and re-computed r^ = Embed(Cw, m) - Cw (and its message part);
(2) pixel AUC of local corr(HP(x), r_true) on a half-watermarked image x (oracle filter = true residual).
Run from TASK: CUDA_VISIBLE_DEVICES= python t1/matched/sanity.py --method trustmark"""
import argparse, os, sys
from pathlib import Path
import cv2, numpy as np, torch
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1/matched"))
from wmloc.watermarks import load_watermark  # noqa
from corr1 import embf, lcorr, to128, auc  # noqa


def gc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=8)
a = ap.parse_args(); torch.set_num_threads(8)
wm = load_watermark(a.method, device="cpu"); rng = np.random.default_rng(0)
d = TASK / "t1/out/synth/v0" / a.method / "full" / "s0_0000"
C_all, Cw_all, msg = np.load(d / "C.npy", mmap_mode="r"), np.load(d / "Cw.npy", mmap_mode="r"), np.load(d / "msg.npy")
G = np.zeros((128, 128), bool); G[:, 64:] = True
R = {k: [] for k in ("g_rm", "g_d", "g_rC", "g_rand", "auc_oracle32", "auc_oracle16", "auc_rm32", "auc_d32", "auc_oracle_raw32")}
for i in range(a.n):
    C, Cw, m = np.asarray(C_all[i]), np.asarray(Cw_all[i]), msg[i].astype(np.uint8)
    rt = Cw.astype(np.float32) - C.astype(np.float32)
    rm = embf(wm, Cw, m); rr = np.mean([embf(wm, Cw, rng.integers(0, 2, len(m)).astype(np.uint8)) for _ in range(3)], 0)
    rC = embf(wm, C, m); rx = embf(wm, C, rng.integers(0, 2, len(m)).astype(np.uint8))
    R["g_rm"].append(gc(rt, rm)); R["g_d"].append(gc(rt, rm - rr)); R["g_rC"].append(gc(rt, rC)); R["g_rand"].append(gc(rt, rx))
    x = C.copy(); x[:, 256:] = Cw[:, 256:]; xf = x.astype(np.float32)
    hp = xf - cv2.GaussianBlur(xf, (0, 0), 1.5)
    R["auc_oracle32"].append(auc(to128(lcorr(hp, rt, 32)), G)); R["auc_oracle16"].append(auc(to128(lcorr(hp, rt, 16)), G))
    R["auc_oracle_raw32"].append(auc(to128(lcorr(xf, rt, 32)), G))
    rmx = embf(wm, x, m); rrx = np.mean([embf(wm, x, rng.integers(0, 2, len(m)).astype(np.uint8)) for _ in range(3)], 0)
    R["auc_rm32"].append(auc(to128(lcorr(hp, rmx, 32)), G)); R["auc_d32"].append(auc(to128(lcorr(hp, rmx - rrx, 32)), G))
print(a.method, " ".join(f"{k} {np.mean(v):+.3f}" for k, v in R.items()), flush=True)
print(a.method, "rms true residual", float(np.sqrt(np.mean(rt ** 2))), "rms rm", float(np.sqrt(np.mean(rm ** 2))))
