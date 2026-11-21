"""Audit de sub11_v6.pt (lecture seule). Depuis TASK, CPU :  python t1/audit6/audit.py"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage

TASK = Path(os.environ.get("TASK", "."))
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK, TASK / "t1/mgr"):
    sys.path.insert(0, str(p))
from regime_guard import load_regimes  # noqa
from wmloc.utils import miou_image  # noqa

SUB = TASK / "t1/out/mgr/submissions"
M = ("stegastamp", "trustmark", "pixelseal")

def P(*a): print(*a, flush=True)

# ---------------------------------------------------------------- (1) format
P("=== (1) FORMAT")
raw = torch.load(SUB / "sub11_v6.pt", weights_only=True)
P("type", type(raw).__name__, "cles", list(raw.keys()))
names = raw["image_names"]; masks = raw["masks"]
P("names type", type(names).__name__, len(names), type(names[0]).__name__, "masks", masks.dtype, tuple(masks.shape),
  "uniques", torch.unique(masks).tolist(), "taille Mo", os.path.getsize(SUB / "sub11_v6.pt") / 1e6)
tm = json.loads((TASK / "release/test/metadata.json").read_text())
tnames = [m["image_name"] for m in tm]; tmeth = {m["image_name"]: m["method"] for m in tm}
imgs = sorted(p.stem for p in (TASK / "release/test/images").glob("*.png"))
P("ordre == metadata.json:", names == tnames, "| ensemble == images/*.png:", set(names) == set(imgs), len(imgs),
  "| doublons:", len(names) - len(set(names)), "| suffixe .png:", sum(n.endswith(".png") for n in names))
P("methodes:", {m: sum(tmeth[n] == m for n in names) for m in M})
# alignement : chaque masque = sortie de la source de sa methode ? (recalcul a partir des probs test)
z = {m: np.load(TASK / f"t1/out/stegatta2/{m}/test_probs128.npz") for m in M}
for m in M:
    zn = [str(x) for x in z[m]["names"]]
    P(f"  probs test {m}: n={len(zn)} toutes de la methode: {all(tmeth.get(n) == m for n in zn)}")
treg = load_regimes("test")
from build_best import predict, DEFAULT  # noqa
mism = 0
for m in M:
    d = {str(n): i for i, n in enumerate(z[m]["names"])}
    for i, n in enumerate(names):
        if tmeth[n] != m: continue
        rec = predict(z[m]["probs"][d[n]].astype(np.float32), treg[n], dict(DEFAULT), None, n)
        mism += int(not np.array_equal(rec.astype(bool), masks[i].numpy()))
P("masques recalcules depuis probs test (meme nom) differents:", mism)

# ---------------------------------------------------------------- (2) couverture train vs test
P("\n=== (2) COUVERTURE (fraction de cellules a 1)")
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
z128 = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(z128["names"].tolist(), z128["masks"]))
trmeta = {m["image_name"]: m for m in json.loads((TASK / "release/train/metadata.json").read_text())}
reg = load_regimes("train")

def rect_fill(mino):
    if not mino.any(): return np.nan
    ys, xs = np.where(mino)
    return mino.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))

def stats(c):
    c = np.asarray(c); return f"n={len(c):3d} moy={c.mean():.3f} med={np.median(c):.3f} min={c.min():.3f} max={c.max():.3f}"

P("-- train vrai, methode x edit_type (et regime decode)")
for m in M:
    for t in ("hole", "transplant", "outpainting"):
        ns = [n for n, v in idx.items() if v["method"] == m and v["edit_type"] == t]
        c = [gt[n].mean() for n in ns]
        fw = np.mean([trmeta[n]["full_watermark"] for n in ns]); rg = np.mean([reg[n] for n in ns])
        rf = [rect_fill(gt[n] == (0 if gt[n].mean() >= .5 else 1)) for n in ns]
        P(f"  {m:10s} {t:11s} {stats(c)} full_wm={fw:.2f} reg_full_decode={rg:.2f} rectfill_min_med={np.nanmedian(rf):.2f}"
          f" rect=1:{np.mean(np.array(rf) > .999):.2f}")
    for r in (True, False):
        ns = [n for n, v in idx.items() if v["method"] == m and reg[n] == r]
        P(f"  {m:10s} regime_decode={'full' if r else 'part'}  {stats([gt[n].mean() for n in ns])}"
          f"  types={ {t: sum(idx[n]['edit_type']==t for n in ns) for t in ('hole','transplant','outpainting')} }")

P("-- test predit, methode x regime decode")
cov = masks.float().mean(dim=(1, 2)).numpy()
ci = {n: i for i, n in enumerate(names)}
for m in M:
    for r in (True, False):
        ns = [n for n in names if tmeth[n] == m and treg[n] == r]
        c = cov[[ci[n] for n in ns]]
        rf = np.array([rect_fill(masks[ci[n]].numpy() == (0 if r else 1)) for n in ns])
        ncomp = np.array([ndimage.label(masks[ci[n]].numpy() == (0 if r else 1))[1] for n in ns])
        P(f"  {m:10s} {'full' if r else 'part'} {stats(c)} contredit={int(((c >= .5) != r).sum())}"
          f" vide/plein={int(((c == 0) | (c == 1)).sum())} cov<.5(full)|>.5(part)={int(((c<.5) if r else (c>.5)).sum())}"
          f" rect=1:{np.mean(rf > .999):.2f} rectfill_med={np.nanmedian(rf):.2f} ncomp_med={np.median(ncomp):.0f}"
          f" cov<0.7(full)|cov>0.4(part)={int(((c<.7) if r else (c>.4)).sum())}")

# ---------------------------------------------------------------- (3) listes
P("\n=== (3) masques vides/pleins, contradictions, minoritaire minuscule (<=1%)")
bad = []
for n in names:
    c = cov[ci[n]]
    tag = []
    if c in (0., 1.): tag.append("vide/plein")
    if (c >= .5) != treg[n]: tag.append("contredit_regime")
    mino = min(c, 1 - c)
    if 0 < mino <= 0.0101: tag.append(f"mino<=1%")
    if tag: bad.append((n, tmeth[n], "full" if treg[n] else "part", round(float(c), 4), tag))
for b in bad: P("  ", b)
P("  total", len(bad))
fa = {m: np.load(TASK / f"t1/out/signal/fullacc_{m}_test.npz") for m in M}

# ---------------------------------------------------------------- (4) comparaison versions
P("\n=== (4) v6 vs anciennes")
for old in ("sub10_v5", "sub09_v4", "sub08_v3"):
    o = torch.load(SUB / f"{old}.pt", weights_only=True)
    assert o["image_names"] == names, old
    om = o["masks"].numpy().astype(bool); nm = masks.numpy()
    oc = om.mean(axis=(1, 2))
    for m in M:
        ii = np.array([ci[n] for n in names if tmeth[n] == m])
        diff = np.array([not np.array_equal(om[i], nm[i]) for i in ii])
        dc = cov[ii] - oc[ii]
        agree = np.array([miou_image(nm[i], om[i]) for i in ii])
        fl = int(((oc[ii] >= .5) != (cov[ii] >= .5)).sum())
        rr = np.array([treg[names[i]] for i in ii])
        P(f"  {old} {m:10s} changees={diff.sum():3d}/{len(ii)} dcov moy={dc.mean():+.4f}"
          f" (full {dc[rr].mean():+.4f}, part {dc[~rr].mean():+.4f}) |dcov|>0.05: {(np.abs(dc)>.05).sum()}"
          f" mIoU(v6,old) moy={agree.mean():.3f} min={agree.min():.3f} bascule_majorite={fl}")
