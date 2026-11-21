"""Fold-0 analysis of occlusion maps: AUC per edit type, thresholded official score, combination with sources.
Run from TASK (CPU):  python t1/occl/analyze.py --method pixelseal --raw t1/out/occl/raw_pixelseal_f0.npz --src dense_login_ps
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import rankdata

TASK = os.getcwd()
for p in ("t1/signal", "t1/train", "t1/synth", "t1/common", "."):
    sys.path.insert(0, os.path.join(TASK, p))
from lib import reproject, auc, otsu  # noqa: E402
from regime_guard import guard, load_regimes  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402

IDX = json.load(open("t1/common/train_index.json"))
TYPES = ("hole", "transplant", "outpainting")


def build_maps(zl):
    """list of raw npz (same names order; later files override) -> dict variant -> (n,128,128)"""
    out = {}
    for z in zl:
        out.update(single_maps(z))
    add_multi(out)
    return out


def single_maps(z):
    out = {}
    base = z["base"][:, None, None]
    for k in z.files:
        if k.startswith("occ_") or k.startswith("inv_"):
            md, s = k.split("_"); s = int(s); pos = z[f"pos_{s}"].tolist()
            v = base - z[k] if md == "occ" else z[k]
            out[k] = np.stack([reproject(x, pos, s) for x in v])
    return out


def add_multi(out):
    def zs(m):
        return (m - m.mean((1, 2), keepdims=True)) / (m.std((1, 2), keepdims=True) + 1e-9)
    occ = [k for k in out if k.startswith("occ")]; inv = [k for k in out if k.startswith("inv")]
    if len(occ) > 1:
        out["occ_multi"] = np.mean([zs(out[k]) for k in occ], 0)
    if len(inv) > 1:
        out["inv_multi"] = np.mean([zs(out[k]) for k in inv], 0)
    if occ and inv:
        out["occ+inv"] = np.mean([zs(out[k]) for k in occ + inv], 0)


def pct(m):
    return (rankdata(m.ravel()) / m.size).reshape(m.shape)


def score_by_type(preds, names, gt):
    r = {t: [] for t in TYPES}
    for n, p in zip(names, preds):
        r[IDX[n]["edit_type"]].append(miou_image(p, gt[n]))
    allv = [v for t in TYPES for v in r[t]]
    return np.mean(allv), {t: (np.mean(r[t]) if r[t] else np.nan) for t in TYPES}


def fmt(tot, d):
    return f"{tot:.4f} | " + " ".join(f"{t[:5]} {d[t]:.3f}" for t in TYPES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--src", default="")
    a = ap.parse_args()
    zl = [np.load(f) for f in a.raw.split(",")]; z = zl[0]; names = [str(n) for n in z["names"]]
    gz = np.load("release/train/masks_128.npz"); gt = dict(zip(gz["names"].tolist(), gz["masks"]))
    reg = load_regimes("train")
    maps = build_maps(zl)
    print('nan check', {k: int(np.isnan(v).any((1, 2)).sum()) for k, v in maps.items()})
    cnt = {t: sum(IDX[n]["edit_type"] == t for n in names) for t in TYPES}
    print(f"{a.method}: n={len(names)} {cnt} | {sum(float(q['sec_per_img']) for q in zl):.2f} s/img")
    print("== AUC per variant (mean over images, by edit type) and thresholded score (Otsu + regime guard)")
    for k, M in maps.items():
        A = {t: np.nanmean([auc(m, gt[n]) for m, n in zip(M, names) if IDX[n]["edit_type"] == t]) for t in TYPES}
        Aall = np.nanmean([auc(m, gt[n]) for m, n in zip(M, names)])
        preds = []
        for m, n in zip(M, names):
            q = pct(m)
            preds.append(guard(np.where(m > otsu(m), 0.5 + 0.5 * q, 0.5 * q), reg[n])[0])
        tot, d = score_by_type(preds, names, gt)
        print(f"  {k:10s} AUC {Aall:.3f} (hole {A['hole']:.3f} trans {A['transplant']:.3f} outp {A['outpainting']:.3f}) | score {fmt(tot, d)}")
    if not a.src:
        return
    src = [s for s in json.load(open("t1/mgr/sources.json"))[a.method] if s["name"] == a.src][0]
    P = {}
    for f in src["val"]:
        sz = np.load(f); P.update(dict(zip([str(n) for n in sz["names"]], sz["probs"].astype(np.float32))))
    Ps = np.stack([P[n] for n in names])
    print(f"== combination with source {a.src}")
    A = np.nanmean([auc(p, gt[n]) for p, n in zip(Ps, names)])
    base_pred = [guard(p, reg[n])[0] for p, n in zip(Ps, names)]
    tot, d = score_by_type(base_pred, names, gt)
    print(f"  source alone: AUC {A:.3f} | score {fmt(tot, d)}")
    for k in [k for k in maps if k in ("occ_32", "occ_64", "occ_96", "occ_multi", "inv_multi", "occ+inv")]:
        M = maps[k]
        Q = np.stack([pct(m) for m in M])
        for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            row = []
            # (a) weighted mean of probabilities with the per-image percentile of the occlusion map
            pa = (1 - w) * Ps + w * Q
            # (b) rerank: order by mixed percentile, keep the source's coverage (after guard)
            pb = []
            for p, q, n, bm in zip(Ps, Q, names, base_pred):
                r = pct((1 - w) * pct(p) + w * q); c = bm.mean()
                pb.append((r > 1 - c).astype(np.uint8))
            # (c) product-like: logit sum with percentile logit
            lq = np.log(np.clip(Q, 1e-3, 1 - 1e-3) / np.clip(1 - Q, 1e-3, 1))
            lp = np.log(np.clip(Ps, 1e-4, 1 - 1e-4) / np.clip(1 - Ps, 1e-4, 1))
            pc = 1 / (1 + np.exp(-((1 - w) * lp + w * lq)))
            for nm, pp in (("mean", pa), ("rerank", pb), ("logit", pc)):
                if nm == "rerank":
                    preds = pp
                else:
                    preds = [guard(p, reg[n])[0] for p, n in zip(pp, names)]
                t2, d2 = score_by_type(preds, names, gt)
                row.append(f"{nm} {t2:.4f} ({d2['hole']:.3f}/{d2['transplant']:.3f}/{d2['outpainting']:.3f})")
            print(f"  {k:10s} w={w:.1f}  " + "  ".join(row))


if __name__ == "__main__":
    main()
