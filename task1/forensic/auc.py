"""AUC intra-image (dedans = patch entièrement dans le masque, dehors = entièrement hors) par feature et par méthode,
plus logistique / MLP (torch CPU) appris sur folds 1-4, évalués sur fold 0 (et en CV 5 folds pour info).
Depuis TASK : CUDA_VISIBLE_DEVICES= python t1/forensic/auc.py"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import torch

TASK = Path(os.environ.get("TASK", "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"))
OUT = Path(os.environ.get("FOUT", str(TASK / "t1/out/forensic")))
torch.set_num_threads(8)


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    x = np.concatenate([pos, neg]); r = np.argsort(np.argsort(x, kind="mergesort")) + 1.0
    # rangs moyens en cas d'égalité négligés (features continues)
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def znorm(F):  # standardisation par image (les AUC intra-image y sont invariantes)
    mu = F.reshape(F.shape[0], -1, F.shape[-1]).mean(1)[:, None, None]
    sd = F.reshape(F.shape[0], -1, F.shape[-1]).std(1)[:, None, None] + 1e-6
    return ((F - mu) / sd).astype(np.float32)


def ctx(Z):  # ajoute des moyennes locales 3x3 et 5x5 des features (contexte spatial)
    t = torch.from_numpy(Z).permute(0, 3, 1, 2)
    a3 = torch.nn.functional.avg_pool2d(t, 3, 1, 1, count_include_pad=False)
    a5 = torch.nn.functional.avg_pool2d(t, 5, 1, 2, count_include_pad=False)
    return torch.cat([t, a3, a5], 1).permute(0, 2, 3, 1).numpy()


def fit(X, y, w, hidden=0, epochs=300, seed=0):
    torch.manual_seed(seed)
    X = torch.from_numpy(X).float(); y = torch.from_numpy(y).float(); w = torch.from_numpy(w).float()
    k = X.shape[1]
    net = torch.nn.Linear(k, 1) if hidden == 0 else torch.nn.Sequential(
        torch.nn.Linear(k, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2 if hidden == 0 else 3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        l = (torch.nn.functional.binary_cross_entropy_with_logits(net(X)[:, 0], y, reduction="none") * w).sum() / w.sum()
        l.backward(); opt.step()
    return net


def per_image_auc(S, lab):
    out = []
    for s, l in zip(S, lab):
        out.append(auc(s[l >= 0.999], s[l <= 0.001]))
    return np.array(out)


def main():
    z = np.load(OUT / "feats_train.npz")
    names = [str(n) for n in z["names"]]; F = np.clip(np.nan_to_num(z["F"].astype(np.float64), posinf=0, neginf=0), -1e3, 1e3); lab = z["lab"]; feats = list(z["feats"])
    meta = {m["image_name"]: m for m in json.loads((TASK / "release/train/metadata.json").read_text())}
    meth = np.array([meta[n]["method"] for n in names])
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    fold = np.array([next(i for i, f in enumerate(folds) if n in f) for n in names])
    rep = {"n": len(names), "inside_frac_mean": float((lab >= 0.999).mean()), "features": {}}
    print("n", len(names), "fraction patches dedans", rep["inside_frac_mean"])
    for j, fn in enumerate(feats):
        a = per_image_auc(F[..., j], lab)
        r = {m: float(np.nanmean(a[meth == m])) for m in ("stegastamp", "trustmark", "pixelseal")}
        r["all"] = float(np.nanmean(a)); r["frac_img_auc_gt_0.8_or_lt_0.2"] = float(np.nanmean((a > 0.8) | (a < 0.2)))
        rep["features"][fn] = r
        print(f"{fn:14s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))
    Z = ctx(znorm(F))
    res = {}
    for name, hid in (("logit", 0), ("mlp", 32)):
        for scheme in ("fold0", "cv5"):
            S = np.zeros(lab.shape, np.float32)
            test_folds = [0] if scheme == "fold0" else range(5)
            for k in test_folds:
                tr = fold != k
                Xtr = Z[tr].reshape(-1, Z.shape[-1]); ltr = lab[tr].reshape(-1)
                keep = (ltr >= 0.999) | (ltr <= 0.001)
                ytr = (ltr[keep] > 0.5).astype(np.float32)
                # pondération : chaque image et chaque classe pèse pareil (approximé globalement)
                wv = np.where(ytr > 0, 0.5 / ytr.mean(), 0.5 / (1 - ytr.mean())).astype(np.float32)
                net = fit(Xtr[keep], ytr, wv, hidden=hid)
                with torch.no_grad():
                    te = fold == k
                    S[te] = net(torch.from_numpy(Z[te].reshape(-1, Z.shape[-1])).float())[:, 0].numpy().reshape(-1, 32, 32)
            sel = fold == 0 if scheme == "fold0" else np.ones(len(names), bool)
            a = per_image_auc(S[sel], lab[sel])
            r = {m: float(np.nanmean(a[meth[sel] == m])) for m in ("stegastamp", "trustmark", "pixelseal")}
            r["all"] = float(np.nanmean(a)); r["n"] = int(sel.sum())
            res[f"{name}_{scheme}"] = r
            print(f"{name}_{scheme:6s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
            if scheme == "cv5":
                np.savez(OUT / f"oof_{name}.npz", names=np.array(names), S=S, lab=lab, fold=fold, meth=meth)
    rep["classifiers"] = res
    (OUT / "auc_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
