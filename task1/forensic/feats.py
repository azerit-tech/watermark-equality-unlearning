"""Étape 1 forensic : features bas niveau par patch 16x16 sur les outpaintings train, AUC intra-image dedans/dehors.
Depuis TASK (CPU) : CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python t1/forensic/feats.py [--split train|test] [--all]
Écrit OUT/feats_<split>.npz : names, F [N,32,32,K], lab [N,32,32] (fraction dans le masque, -1 si inconnu)."""
from __future__ import annotations
import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import median_filter

TASK = Path(os.environ.get("TASK", "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"))
OUT = Path(os.environ.get("FOUT", str(TASK / "t1/out/forensic")))
P = 16
FEATS = ["rmed_var", "rgau_var", "rmed_kurt", "dct_hf", "dct_mf", "grad", "lap_abs",
         "block_max", "block_gph", "sat", "chroma_res", "rmed_abs", "y_std", "rgau_var_s2"]


def bm(x, p=P):  # moyenne par bloc p x p sur image 512x512
    h, w = x.shape
    return x[: h // p * p, : w // p * p].reshape(h // p, p, w // p, p).mean((1, 3))


def dct_maps(y):
    # énergie DCT 8x8 par bloc 8x8 aligné (0,0), haute (u+v>=8) et moyenne fréquence
    blocks = y.reshape(64, 8, 64, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    D = np.stack([cv2.dct(b) for b in blocks]).reshape(64, 64, 8, 8)
    u, v = np.meshgrid(range(8), range(8), indexing="ij")
    hf = np.log1p((D ** 2)[..., (u + v) >= 8].mean(-1))
    mf = np.log1p((D ** 2)[..., ((u + v) >= 3) & ((u + v) < 8)].mean(-1))
    return hf.reshape(32, 2, 32, 2).mean((1, 3)), mf.reshape(32, 2, 32, 2).mean((1, 3))


def one(args):
    split, n = args
    im = cv2.imread(str(TASK / "release" / split / "images" / f"{n}.png"), cv2.IMREAD_COLOR).astype(np.float32)
    y = cv2.cvtColor(im / 255.0, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = y[..., 0] * 255, y[..., 1] * 255, y[..., 2] * 255
    rmed = Y - median_filter(Y, 3)
    rgau = Y - cv2.GaussianBlur(Y, (0, 0), 1.0)
    rgau2 = Y - cv2.GaussianBlur(Y, (0, 0), 2.0)
    f = {}
    f["rmed_var"] = np.log1p(bm(rmed ** 2))
    f["rgau_var"] = np.log1p(bm(rgau ** 2))
    f["rgau_var_s2"] = np.log1p(bm(rgau2 ** 2))
    m2 = bm(rgau ** 2); m4 = bm(rgau ** 4)
    f["rmed_kurt"] = np.log((m4 + 1e-6) / (m2 ** 2 + 1e-6))
    f["rmed_abs"] = np.log1p(bm(np.abs(rmed)))
    f["dct_hf"], f["dct_mf"] = dct_maps(Y)
    gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0); gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1)
    f["grad"] = np.log1p(bm(np.sqrt(gx ** 2 + gy ** 2)))
    f["lap_abs"] = np.log1p(bm(np.abs(cv2.Laplacian(Y, cv2.CV_32F))))
    # grille JPEG : différences absolues aux frontières de bloc, par phase (0..7), fenêtre 32x32 autour du patch
    dh = np.abs(np.diff(Y, axis=1)); dh = np.pad(dh, ((0, 0), (0, 1)))
    dv = np.abs(np.diff(Y, axis=0)); dv = np.pad(dv, ((0, 1), (0, 0)))
    ph = np.zeros((8, 32, 32), np.float32)
    for k in range(8):
        mh = np.zeros_like(dh); mh[:, k::8] = dh[:, k::8]
        mv = np.zeros_like(dv); mv[k::8, :] = dv[k::8, :]
        s = cv2.blur(mh, (32, 32)) * 8 + cv2.blur(mv, (32, 32)) * 8   # moyenne aux frontières de phase k
        ph[k] = s[8::16, 8::16]
    tot = (cv2.blur(dh, (32, 32)) + cv2.blur(dv, (32, 32)))[8::16, 8::16] + 1e-3
    f["block_max"] = np.log(ph.max(0) / tot)
    gph = int(np.argmax(ph.mean((1, 2))))   # phase globale de l'image
    f["block_gph"] = np.log((ph[gph] + 1e-3) / tot)
    hsv = cv2.cvtColor(im / 255.0, cv2.COLOR_BGR2HSV)
    f["sat"] = bm(hsv[..., 1])
    rc = (Cr - cv2.GaussianBlur(Cr, (0, 0), 1.0)) ** 2 + (Cb - cv2.GaussianBlur(Cb, (0, 0), 1.0)) ** 2
    f["chroma_res"] = np.log1p(bm(rc))
    f["y_std"] = np.log1p(np.sqrt(bm(Y ** 2) - bm(Y) ** 2 + 1e-6))
    F = np.stack([f[k] for k in FEATS], -1).astype(np.float32)
    lab = -np.ones((32, 32), np.float32)
    mp = TASK / "release" / split / "masks" / f"{n}.png"
    if mp.exists():
        m = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        lab = bm(m)
    return n, F, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--all", action="store_true", help="toutes les images (sinon outpaintings train)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((TASK / "release" / a.split / "metadata.json").read_text())
    names = [m["image_name"] for m in meta]
    if a.split == "train" and not a.all:
        tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
        names = [n for n in names if tidx[n]["edit_type"] == "outpainting"]
    with Pool(8) as pool:
        res = pool.map(one, [(a.split, n) for n in names], chunksize=4)
    tag = a.split + ("_all" if a.all else "")
    np.savez(OUT / f"feats_{tag}.npz", names=np.array([r[0] for r in res]), F=np.stack([r[1] for r in res]),
             lab=np.stack([r[2] for r in res]), feats=np.array(FEATS))
    print("ok", len(res))


if __name__ == "__main__":
    main()
