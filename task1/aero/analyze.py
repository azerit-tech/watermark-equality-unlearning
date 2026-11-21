import json, sys, numpy as np
from scipy.stats import rankdata
from scipy import ndimage as ndi

TASK = "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"
idx = json.load(open(f"{TASK}/t1/common/train_index.json"))
z = np.load(f"{TASK}/release/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
r = np.load(sys.argv[1]); names = r["names"].tolist()


def auc(s, m):
    m = m.astype(bool).ravel(); s = s.ravel()
    if m.all() or not m.any(): return np.nan
    rk = rankdata(s); n1 = m.sum(); n0 = (~m).sum()
    return (rk[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def variants(M):
    l1, l2, l3, g = M
    c = np.median(g)
    out = {"lp1": l1, "lp2": l2, "lp3": l3, "lp2/sqrtg": l2 / np.sqrt(g + c), "lp2/g": l2 / (g + c),
           "lp2_s": ndi.gaussian_filter(l2 / np.sqrt(g + c), 2), "grad": g}
    return out


def box_iou(a, b):
    y0, y1, x0, x1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y1 - y0) * max(0, x1 - x0)
    ar = lambda t: (t[1] - t[0]) * (t[3] - t[2])
    return inter / (ar(a) + ar(b) - inter)


for key in [k[5:] for k in r.files if k.startswith("maps_")]:
    maps = r[f"maps_{key}"]; mse = r[f"mse_{key}"]
    print(f"== VAE {key}: PSNR moyen {np.mean(10 * np.log10(4 / mse)):.1f} dB")
    tab = {}
    ious = []
    for i, n in enumerate(names):
        et, m = idx[n]["edit_type"], idx[n]["method"]
        for v, s in variants(maps[i]).items():
            tab.setdefault(v, {}).setdefault((et, m), []).append(auc(s, gt[n]))
        if et == "outpainting":
            s = variants(maps[i])["lp2_s"]; b = s > np.quantile(s, 0.7)
            lab, k = ndi.label(b)
            if k:
                big = 1 + np.argmax(ndi.sum(b, lab, range(1, k + 1))); ys, xs = np.where(lab == big)
                pb = [ys.min(), ys.max() + 1, xs.min(), xs.max() + 1]
                gy, gx = np.where(gt[n]); gb = [gy.min(), gy.max() + 1, gx.min(), gx.max() + 1]
                ious.append((m, box_iou(pb, gb), gt[n].mean()))
    cols = [(et, m) for et in ["outpainting", "transplant"] for m in ["pixelseal", "trustmark", "stegastamp"]]
    print(f"{'variant':10s} " + " ".join(f"{e[:4]}-{m[:2]}" for e, m in cols) + "  outp  transp  all")
    for v, d in tab.items():
        row = [np.nanmean(d.get(c, [np.nan])) for c in cols]
        o = np.nanmean(sum((d[c] for c in cols[:3]), [])); t = np.nanmean(sum((d[c] for c in cols[3:]), []))
        print(f"{v:10s} " + " ".join(f"{x:7.3f}" for x in row) + f"  {o:.3f} {t:.3f} {np.nanmean([o, t]):.3f}")
    print("outpainting bbox(top30% lp2_s, plus grande composante) IoU vs rect vrai:",
          " ".join(f"{m[:2]}:{iou:.2f}(cov{c:.2f})" for m, iou, c in ious))
