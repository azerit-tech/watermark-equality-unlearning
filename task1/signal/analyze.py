"""CPU analysis of the GPU caches: regime separation, native maps, sliding sweep, full sliding run, map export.
Usage (from TASK): python t1/signal/analyze.py [--tag _val] [--export]"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from lib import OUT, METHODS, masks128, positions, reproject, auc, best_global_threshold, score_otsu, score, miou_image

SIZES = [64, 96, 128, 192, 256]


def up128(maps):
    t = torch.from_numpy(np.asarray(maps, np.float32))[:, None]
    return F.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)[:, 0].numpy()


def load(path):
    return dict(np.load(path, allow_pickle=True)) if path.exists() else None


def regime(tag):
    print("\n=== 1. Décodage image entière : séparation des régimes ===")
    M = masks128(); preds = {}
    for meth in METHODS:
        tr = load(OUT / f"fullacc_{meth}_train{tag}.npz"); te = load(OUT / f"fullacc_{meth}_test{tag}.npz")
        if tr is None:
            print(meth, "missing"); continue
        acc, y = tr["acc"], tr["fw"] == 1
        hi, lo = acc[y], acc[~y]
        srt = np.sort(acc); cands = (srt[1:] + srt[:-1]) / 2
        errs = [int(((acc > t) != y).sum()) for t in cands]
        k = int(np.argmin(errs)); thr = cands[k]
        print(f"{meth}: full acc mean {hi.mean():.3f} min {hi.min():.3f} max {hi.max():.3f} | patch mean {lo.mean():.3f} "
              f"min {lo.min():.3f} max {lo.max():.3f} | gap(min_full - max_patch) {hi.min()-lo.max():+.3f} | "
              f"best thr {thr:.3f} -> {errs[k]} errors / {len(acc)} | {float(tr['sec_per_img'])*1000:.1f} ms/img")
        mar = tr["margin"]
        print(f"   margin: full min {mar[y].min():.3f} | patch max {mar[~y].max():.3f}")
        preds.update({str(n): bool(a > thr) for n, a in zip(tr["names"], acc)})
        if te is not None:
            ta = te["acc"]
            h, e = np.histogram(ta, bins=10, range=(0.4, 1.0))
            print(f"   test n={len(ta)}: frac > thr = {(ta > thr).mean():.3f} | hist[0.4..1.0 step .06]: {h.tolist()} "
                  f"| in ambiguous zone [{lo.max():.3f},{hi.min():.3f}]: {int(((ta > min(lo.max(), hi.min())) & (ta < max(lo.max(), hi.min()))).sum())}")
            preds.update({f"test:{n}": bool(a > thr) for n, a in zip(te["names"], ta)})
    names = [n for n in M if n in preds]
    if names:
        z = np.zeros((128, 128), np.uint8)
        sc = score([z + 1 if preds[n] else z for n in names], [M[n] for n in names])
        print(f"Baseline régime (tout 1 si full prédit, sinon tout 0), train n={len(names)}: score {sc:.4f}")
        for meth in METHODS:
            tr = load(OUT / f"fullacc_{meth}_train{tag}.npz")
            if tr is None: continue
            nn = [str(n) for n in tr["names"]]
            print(f"   {meth}: {score([z + 1 if preds[n] else z for n in nn], [M[n] for n in nn]):.4f}")
    return preds


def evaluate(label, maps, names, fw, M, preds, extra=""):
    gts = [M[n] for n in names]
    aucs = np.array([auc(m, g) for m, g in zip(maps, gts)])
    thr, sg = best_global_threshold(maps, gts)
    so = score_otsu(maps, gts)
    # threshold per regime, regime predicted by full-image decode
    pr = np.array([preds.get(n, False) for n in names])
    out = np.zeros(len(names)); sr = {}
    for r in (True, False):
        idx = np.where(pr == r)[0]
        if len(idx) == 0: continue
        t, _ = best_global_threshold([maps[i] for i in idx], [gts[i] for i in idx])
        for i in idx:
            out[i] = miou_image(maps[i] > t, gts[i])
        sr[r] = t
    per = {}
    for r, nm in ((1, "full"), (0, "patch")):
        idx = fw == r
        per[nm] = (np.nanmean(aucs[idx]), np.mean([miou_image(maps[i] > thr, gts[i]) for i in np.where(idx)[0]]),
                   out[idx].mean())
    print(f"{label:34s} AUC {np.nanmean(aucs):.3f} (full {per['full'][0]:.3f} / patch {per['patch'][0]:.3f}) | "
          f"score thrGlob {sg:.4f} (full {per['full'][1]:.3f} / patch {per['patch'][1]:.3f}) | Otsu {so:.4f} | "
          f"thr/régime prédit {out.mean():.4f} (full {per['full'][2]:.3f} / patch {per['patch'][2]:.3f}) {extra}")
    return np.nanmean(aucs), sg, out.mean()


def native(tag, preds):
    print("\n=== 2. Cartes natives (features per_position_logits) ===")
    M = masks128(); best = {}
    for meth in ("pixelseal", "trustmark"):
        d = load(OUT / f"native_{meth}_train{tag}.npz")
        if d is None:
            print(meth, "missing"); continue
        names = [str(n) for n in d["names"]]; fw = d["fw"]
        print(f"{meth}: native map {d['signed'].shape[1:]} -> 128 bilinear | {float(d['sec_per_img'])*1000:.1f} ms/img")
        for key in [k for k in ("signed", "frac", "det") if k in d]:
            maps = up128(d[key])
            gts = [M[n] for n in names]
            # alignment check: correlation with mask for identity / flips / transpose
            def corr(tf):
                return np.nanmean([np.corrcoef(m.ravel(), tf(g).ravel().astype(float))[0, 1] for m, g in zip(maps, gts)])
            c = {k: corr(f) for k, f in (("id", lambda g: g), ("flipud", lambda g: g[::-1]), ("fliplr", lambda g: g[:, ::-1]),
                                          ("T", lambda g: g.T), ("rot180", lambda g: g[::-1, ::-1]))}
            print("   align corr " + key + ": " + " ".join(f"{k} {v:+.3f}" for k, v in c.items()))
            r = evaluate(f"  native {meth}.{key}", maps, names, fw, M, preds)
            best[(meth, f"native.{key}")] = r
    return best


def sweep(tag, preds):
    print("\n=== 3. Fenêtre glissante (sous-ensemble train équilibré) ===")
    M = masks128(); best = {}
    for meth in METHODS:
        d = load(OUT / f"sweep_{meth}{tag}.npz")
        if d is None:
            print(meth, "missing"); continue
        names = [str(n) for n in d["names"]]; fw = d["fw"]
        print(f"{meth}: n={len(names)} (full {int((fw==1).sum())} / patch {int((fw==0).sum())})")
        multi = []
        sizes = [s for s in SIZES if f"pos_{s}" in d]
        for s in sizes:
            pos4 = d[f"pos_{s}"].tolist(); A = d[f"win_acc_{s}"]; G = d[f"win_mar_{s}"]
            idx2 = [i for i, p in enumerate(pos4) if p % (s // 2) == 0 or p == 512 - s]
            pos2 = [pos4[i] for i in idx2]
            # local vs global: accuracy of windows entirely inside / outside the watermarked area
            ins, outs = {1: [], 0: []}, {1: [], 0: []}
            for k, n in enumerate(names):
                g = M[n]
                for i, y in enumerate(pos4):
                    for j, x in enumerate(pos4):
                        cov = g[y // 4:(y + s) // 4, x // 4:(x + s) // 4].mean()
                        if cov >= 0.99: ins[int(fw[k])].append(A[k, i, j])
                        elif cov <= 0.01: outs[int(fw[k])].append(A[k, i, j])
            f = lambda v: f"{np.mean(v):.3f} (n={len(v)})" if v else "n/a"
            print(f"  s={s:3d}: crop entièrement watermarké acc: full {f(ins[1])} patch {f(ins[0])} | "
                  f"crop entièrement propre acc: full {f(outs[1])} patch {f(outs[0])} | {float(d[f'sec_{s}']):.2f} s/img @s/4")
            for st, sel, pos in ((s // 4, slice(None), pos4), (s // 2, idx2, pos2)):
                for nm, W in (("acc", A), ("mar", G)):
                    maps = [reproject(w[np.ix_(sel, sel)] if st == s // 2 else w, pos, s) for w in W]
                    if st == s // 2 and nm == "acc": multi.append(maps)
                    best[(meth, f"slide.{nm}.{s}/{st}")] = evaluate(f"  {meth} {nm} s={s} stride={st}", maps, names, fw, M, preds)
        mm = [np.mean([multi[i][k] for i in range(len(sizes))], 0) for k in range(len(names))]
        best[(meth, "slide.acc.multi/s2")] = evaluate(f"  {meth} acc multi-scale s/2", mm, names, fw, M, preds)
        for s in (64, 128):
            if f"occ_gray_{s}" not in d: continue
            pos = d[f"occpos_{s}"].tolist()
            for kind in ("gray", "blur"):
                maps = [reproject(w, pos, s) for w in d[f"occ_{kind}_{s}"]]
                best[(meth, f"occl.{kind}.{s}")] = evaluate(f"  {meth} occlusion {kind} s={s}", maps, names, fw, M, preds)
    return best


def slide_full(tag, preds, export):
    print("\n=== 3b. Fenêtre glissante, train complet (config choisie sur le sous-ensemble) ===")
    M = masks128()
    for meth in METHODS:
        d = load(OUT / f"slide_{meth}_train{tag}.npz")
        if d is None:
            print(meth, "missing"); continue
        names = [str(n) for n in d["names"]]; fw = d["fw"]
        sw = load(OUT / f"sweep_{meth}{tag}.npz"); sub = set(str(n) for n in sw["names"]) if sw else set()
        print(f"{meth}: cfg {d['cfg'].tolist()} n={len(names)} {float(d['sec_per_img']):.2f} s/img")
        for key in ("acc", "margin"):
            maps = d[key].astype(np.float32)
            evaluate(f"  {meth} slide.{key} (all)", list(maps), names, fw, M, preds)
            keep = [i for i, n in enumerate(names) if n not in sub]
            if sub and keep:
                evaluate(f"  {meth} slide.{key} (hors sweep)", [maps[i] for i in keep], [names[i] for i in keep], fw[keep], M, preds)


def export(tag):
    """Save continuous 128x128 float16 maps (all available channels) per method and split."""
    print("\n=== 5. Export des cartes ===")
    for meth in METHODS:
        for split in ("train", "test"):
            ch = {}
            names = None
            nat = load(OUT / f"native_{meth}_{split}{tag}.npz")
            if nat is not None:
                names = nat["names"]
                for k in ("signed", "frac", "det"):
                    if k in nat: ch[f"native_{k}"] = up128(nat[k]).astype(np.float16)
            sl = load(OUT / f"slide_{meth}_{split}{tag}.npz")
            if sl is not None:
                if names is not None: assert (sl["names"] == names).all()
                names = sl["names"]
                ch["slide_acc"] = sl["acc"]; ch["slide_margin"] = sl["margin"]
            fa = load(OUT / f"fullacc_{meth}_{split}{tag}.npz")
            if names is None or not ch: continue
            if fa is not None:
                assert (fa["names"] == names).all()
                ch["full_acc"] = fa["acc"].astype(np.float32); ch["full_margin"] = fa["margin"].astype(np.float32)
            p = OUT / f"maps_{meth}_{split}{tag}.npz"
            np.savez_compressed(p, names=names, **ch)
            print(p, {k: v.shape for k, v in ch.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--export", action="store_true")
    ap.add_argument("--only", default="regime,native,sweep,slide")
    a = ap.parse_args(); only = a.only.split(",")
    preds = regime(a.tag)
    if "native" in only: native(a.tag, preds)
    if "sweep" in only: sweep(a.tag, preds)
    if "slide" in only: slide_full(a.tag, preds, a.export)
    if a.export: export(a.tag)
