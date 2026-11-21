"""RISE maps from raw scores + AUC per edit type (CPU). Run from TASK:
    python t1/rise/maps.py t1/out/rise/ps_nf0_blur1.npz [more.npz ...]
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "t1", "signal")); sys.path.insert(0, os.path.join(os.getcwd(), "t1", "rise"))
from lib import auc  # noqa: E402
from rise import make_masks  # noqa: E402

IDX = json.load(open("t1/common/train_index.json"))
TYPES = ("hole", "transplant", "outpainting")
_M = {}


def masks(N):
    if N not in _M:
        _M[N] = make_masks(N)
    return _M[N]


def map_cov(s, M):
    """RISE / covariance estimator: E[(s - mean s)(M - mean M)] per cell."""
    sc = (s - s.mean()).astype(np.float64)
    return np.tensordot(sc, M - M.mean(0), axes=(0, 0)).astype(np.float32) / len(s)


def map_ridge(s, M, lo=32, lam=1.0):
    X = F.interpolate(torch.from_numpy(M[:, None]), size=(lo, lo), mode="area")[:, 0].reshape(len(M), -1).double().numpy()
    X = X - X.mean(0); y = (s - s.mean()).astype(np.float64)
    A = X.T @ X; A += lam * np.trace(A) / A.shape[0] * np.eye(A.shape[0])
    b = np.linalg.solve(A, X.T @ y).reshape(1, 1, lo, lo)
    return F.interpolate(torch.from_numpy(b), size=(128, 128), mode="bilinear", align_corners=False)[0, 0].numpy().astype(np.float32)


def all_maps(z, n=None, kind="cov"):
    S = z["scores"]; n = n or S.shape[1]; M = masks(S.shape[1])[:n]
    f = map_cov if kind == "cov" else map_ridge
    return np.stack([f(s[:n], M) for s in S])


def auc_line(lab, maps, names, gt):
    a = np.array([auc(m, gt[n]) for m, n in zip(maps, names)])
    t = {k: np.nanmean([v for v, n in zip(a, names) if IDX[n]["edit_type"] == k]) for k in TYPES}
    print(f"  {lab:28s} AUC {np.nanmean(a):.3f} " + " ".join(f"{k[:5]} {t[k]:.3f}" for k in TYPES), flush=True)
    return a


if __name__ == "__main__":
    gz = np.load("release/train/masks_128.npz"); gt = dict(zip(gz["names"].tolist(), gz["masks"]))
    for f in sys.argv[1:]:
        z = np.load(f); names = [str(n) for n in z["names"]]
        print(f, f"{float(z['sec_per_img']):.2f} s/img", {k: sum(IDX[n]['edit_type'] == k for n in names) for k in TYPES})
        for n in (500, z["scores"].shape[1]):
            auc_line(f"cov N={n}", all_maps(z, n), names, gt)
            for lam in (0.1, 1.0):
                auc_line(f"ridge32 lam={lam} N={n}", all_maps(z, n, "ridge") if lam == 1.0 else np.stack([map_ridge(s[:n], masks(z['scores'].shape[1])[:n], lam=lam) for s in z["scores"]]), names, gt)
