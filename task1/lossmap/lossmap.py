"""Carte des points perdus de la recette sub04 sur le fold 0 + bornes hautes de réparations parfaites.
Depuis TASK, CPU, ~1 min :
    python t1/lossmap/lossmap.py            # -> t1/out/lossmap/{per_image.tsv, report.txt, panel.png, cases.tsv}
"""
import json, sys, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "train", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from regime_guard import guard, load_regimes  # noqa: E402
from train import binarize  # noqa: E402
from evaluate import load_train  # noqa: E402
from wmloc.utils import miou_image, downsample_mask  # noqa: E402

OUT = TASK / "t1" / "out" / "lossmap"; OUT.mkdir(parents=True, exist_ok=True)
SRC = {"stegastamp": "t1/out/stega/best/fold0_probs128.npz",
       "trustmark": "t1/out/refine/trustmark_fold0_probs128.npz",
       "pixelseal": "t1/out/refine/pixelseal_fold0_probs128.npz"}
BOXF = {m: TASK / f"t1/out/boxsearch/{m}_evaltrain_boxes.json" for m in SRC}
BOXF["pixelseal"] = TASK / "t1/out/boxsearch/pixelseal_evaltrain_grid1_boxes.json"
METHODS = ("stegastamp", "trustmark", "pixelseal"); TYPES = ("hole", "transplant", "outpainting")
N = 102

idx = json.loads((TASK / "t1/common/train_index.json").read_text())
names = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0]
meta, gt = load_train()
reg = load_regimes("train")
probs = {}
for m, f in SRC.items():
    z = np.load(TASK / f)
    for n, p in zip(z["names"], z["probs"]):
        if str(n) in names and idx[str(n)]["method"] == m:
            probs[str(n)] = p.astype(np.float32)
assert len(probs) == N, len(probs)
RAW = {"trustmark": "t1/out/dense/login/tm/fold0_probs128.npz", "pixelseal": "t1/out/dense/login/ps/fold0_probs128.npz"}
raw = {n: probs[n] for n in probs if idx[n]["method"] == "stegastamp"}   # probabilités continues (avant GrabCut)
for m, f in RAW.items():
    z = np.load(TASK / f)
    for n, p in zip(z["names"], z["probs"]):
        if str(n) in names and idx[str(n)]["method"] == m:
            raw[str(n)] = p.astype(np.float32)
assert len(raw) == N
boxes = {}
for m, f in BOXF.items():
    if f.exists():
        for n, v in json.loads(f.read_text())["boxes"].items():
            boxes[n] = (v["rect"], v["conf"])

S = lambda m, n: miou_image(m, gt[n])  # noqa: E731
base = {n: guard(probs[n], reg[n])[0].astype(np.uint8) for n in names}
s0 = {n: S(base[n], n) for n in names}


def mino(g):  # minority class value of the truth
    return int(g.mean() < 0.5)


def maj_of(m):
    return int(m.mean() >= 0.5)


def rectmask(r):
    y0, y1, x0, x1 = [int(v) for v in r]
    b = np.zeros((512, 512), np.uint8); b[max(y0, 0):min(y1, 512), max(x0, 0):min(x1, 512)] = 1
    return downsample_mask(b)


# ------------------------------------------------------------------ réparations (chacune : nom -> masque)
TH = np.round(np.arange(0.02, 0.99, 0.02), 2)
st = np.ones((3, 3), bool)


def r_regime(n):  # régime parfait (garde-fou avec le vrai régime)
    return guard(probs[n], bool(idx[n]["full_watermark"]))[0]


def r_thr(n, full=None):  # meilleur seuil par image (oracle), garde-fou (régime décodé ou donné)
    full = reg[n] if full is None else full
    best, bm = S(base[n], n) if full == reg[n] else -1, None
    for t in TH:
        m = binarize(raw[n], t)
        if maj_of(m) != full:
            continue
        s = S(m, n)
        if s > best:
            best, bm = s, m
    g = base[n] if full == reg[n] else guard(probs[n], full)[0]
    return g if bm is None or S(g, n) >= best else bm


def r_topk(n):  # bon classement, taille parfaite : top-k cellules avec k = vraie taille de la zone watermarkée
    k = int(gt[n].sum()); m = np.zeros(128 * 128, np.uint8)
    m[np.argsort(-raw[n].reshape(-1), kind="stable")[:k]] = 1
    return m.reshape(128, 128)


def r_band(n, d, m0=None):  # contour parfait dans une bande de d cellules autour du bord prédit
    m = base[n] if m0 is None else m0
    edge = ndimage.binary_dilation(m, st, d) ^ ndimage.binary_erosion(m, st, d, border_value=1)
    return np.where(edge, gt[n], m).astype(np.uint8)


def r_gtpm1(n):  # vérité dilatée/érodée de 1 cellule (moyenne des deux) : plafond « forme juste à 1 cellule près »
    g = gt[n]; mi = mino(g); r = (g == mi)
    d = np.where(ndimage.binary_dilation(r, st), mi, 1 - mi); e = np.where(ndimage.binary_erosion(r, st), mi, 1 - mi)
    return d, e


def r_comp(n, m0=None):  # détection parfaite des composantes : on ne garde que les composantes minoritaires prédites qui touchent la vraie
    m = base[n] if m0 is None else m0; mj = maj_of(m); mi = 1 - mj
    lab, k = ndimage.label(m == mi)
    keep = np.zeros_like(m, bool)
    for i in range(1, k + 1):
        c = lab == i
        if (gt[n][c] == mi).mean() > 0.5:
            keep |= c
    if not keep.any():
        return m
    return np.where(keep, mi, mj).astype(np.uint8)


def r_rect(n, m0=None):  # outpainting : vrai rectangle
    m = base[n] if m0 is None else m0
    return gt[n].copy() if idx[n]["is_rect"] else m


def r_boxsearch(n, cthr, m0=None):  # outpainting : rectangle de boxsearch réel (si conf >= cthr), sur TOUTE image
    m = base[n] if m0 is None else m0
    if n in boxes and boxes[n][1] >= cthr and not reg[n]:
        return rectmask(boxes[n][0])
    return m


def r_bbox(n, m0=None):  # outpainting : rectangle englobant de la prédiction (pas d'oracle sauf « c'est un rectangle »)
    m = base[n] if m0 is None else m0
    if not idx[n]["is_rect"]:
        return m
    mi = 1 - maj_of(m); ys, xs = np.where(m == mi)
    if len(ys) == 0:
        return m
    lab, k = ndimage.label(m == mi); sz = np.bincount(lab.ravel())[1:]; c = lab == (1 + sz.argmax())
    ys, xs = np.where(c); o = np.full_like(m, 1 - mi); o[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = mi
    return o


rep = {}
rep["régime parfait"] = {n: S(r_regime(n), n) for n in names}
rep["seuil parfait/image"] = {n: S(r_thr(n), n) for n in names}
def feas_thr():
    out = {}
    for m in METHODS:
        for r in (True, False):
            ns = [n for n in names if idx[n]["method"] == m and reg[n] == r]
            best = max([None] + list(TH), key=lambda t: np.mean([S(base[n] if t is None else guard_t(n, t), n) for n in ns]))
            for n in ns:
                out[n] = S(base[n] if best is None else guard_t(n, best), n)
            L0.append(f"seuil réaliste {m} full={r}: {best}")
    return out


def guard_t(n, t):
    m = binarize(raw[n], t)
    return m if maj_of(m) == reg[n] else base[n]


L0 = []
rep["seuil réaliste méthode×régime (in-sample)"] = feas_thr()
rep["régime+seuil parfaits"] = {n: S(r_thr(n, bool(idx[n]["full_watermark"])), n) for n in names}
rep["taille parfaite (top-k vrai k)"] = {n: S(r_topk(n), n) for n in names}
for d in (1, 2, 4):
    rep[f"contour parfait bande {d} cell."] = {n: S(r_band(n, d), n) for n in names}
rep["vérité ±1 cellule (plafond)"] = {n: np.mean([S(x, n) for x in r_gtpm1(n)]) for n in names}
rep["composantes parfaites (FP retirés)"] = {n: S(r_comp(n), n) for n in names}
rep["outpainting vrai rectangle"] = {n: S(r_rect(n), n) for n in names}
rep["outpainting bbox de la prédiction"] = {n: S(r_bbox(n), n) for n in names}
for c in (0.9, 0.95, 1.0):
    rep[f"boxsearch réel conf>={c}"] = {n: S(r_boxsearch(n, c), n) for n in names}
# meilleure méthode = vérité pour TM/PS jusqu'au niveau stega
rep["TM+PS au niveau StegaStamp (0.763)"] = {n: (0.7629 if idx[n]["method"] != "stegastamp" else s0[n]) for n in names}

# combinaisons
def combo(n, regime=True, thr=True, rect=True, band=0, comp=False):
    full = bool(idx[n]["full_watermark"]) if regime else reg[n]
    m = r_thr(n, full) if thr else guard(probs[n], full)[0]
    if comp: m = r_comp(n, m)
    if band: m = r_band(n, band, m)
    if rect: m = r_rect(n, m)
    return S(m, n)

combos = {
    "régime + rectangle": dict(regime=True, thr=False, rect=True),
    "rectangle + seuil": dict(regime=False, thr=True, rect=True),
    "régime + seuil + rectangle": dict(regime=True, thr=True, rect=True),
    "rectangle + composantes": dict(regime=False, thr=False, rect=True, comp=True),
    "rectangle + contour 1": dict(regime=False, thr=False, rect=True, band=1),
    "rectangle + contour 2": dict(regime=False, thr=False, rect=True, band=2),
    "régime+seuil+rect+composantes": dict(regime=True, thr=True, rect=True, comp=True),
    "régime+seuil+rect+comp+contour 1": dict(regime=True, thr=True, rect=True, comp=True, band=1),
    "régime+seuil+rect+comp+contour 2": dict(regime=True, thr=True, rect=True, comp=True, band=2),
}
for k, kw in combos.items():
    rep["COMBO " + k] = {n: combo(n, **kw) for n in names}

# ------------------------------------------------------------------ attributs par image
frac = {n: float(min(gt[n].mean(), 1 - gt[n].mean())) for n in names}
qs = np.quantile([frac[n] for n in names], [0.25, 0.5, 0.75])
qof = lambda f: "Q" + str(1 + int(np.searchsorted(qs, f, side="right")))  # noqa: E731
attr = {n: dict(method=idx[n]["method"], type=idx[n]["edit_type"], q=qof(frac[n]),
                regerr=reg[n] != bool(idx[n]["full_watermark"])) for n in names}

L = list(L0)
L.append(f"score sub04 fold0 : {np.mean([s0[n] for n in names]):.4f}   (manque total {1-np.mean([s0[n] for n in names]):.4f})")
for m in METHODS:
    v = [s0[n] for n in names if attr[n]["method"] == m]; L.append(f"  {m:11s} {np.mean(v):.4f} n={len(v)}")
L.append(f"quartiles fraction minoritaire (vérité) : {np.round(qs, 4).tolist()}")
L.append(f"erreurs de régime : {sum(a['regerr'] for a in attr.values())} images "
         f"{[ (n, attr[n]['method'], attr[n]['type'], round(s0[n],3)) for n in names if attr[n]['regerr']]}")


def table(keys, title):
    L.append(f"\n== Points perdus par {title} (somme (1-score)/102 ; total {1-np.mean(list(s0.values())):.4f})")
    g = defaultdict(list)
    for n in names:
        g[tuple(attr[n][k] for k in keys)].append(n)
    rows = sorted(g.items(), key=lambda kv: -sum(1 - s0[n] for n in kv[1]))
    for key, ns in rows:
        lost = sum(1 - s0[n] for n in ns) / N
        L.append(f"  {' × '.join(map(str, key)):40s} n={len(ns):3d}  score={np.mean([s0[n] for n in ns]):.3f}  "
                 f"perdu={lost:.4f}  ({100*lost/(1-np.mean(list(s0.values()))):4.1f} %)")


table(["method"], "méthode"); table(["type"], "type"); table(["q"], "quartile taille minoritaire")
table(["regerr"], "erreur de régime"); table(["method", "type"], "méthode × type")
table(["type", "q"], "type × quartile"); table(["method", "type", "q"], "méthode × type × quartile")

L.append("\n== Réparations parfaites : score total (gain) et gain par méthode × type (points sur le total /102)")
cells = [(m, t) for m in METHODS for t in TYPES]
L.append(f"  {'réparation':42s} {'total':>6s} {'gain':>6s}  " + " ".join(f"{m[:2]}/{t[:4]:>4s}" for m, t in cells))
for k, v in sorted(rep.items(), key=lambda kv: -np.mean(list(kv[1].values()))):
    tot = np.mean([v[n] for n in names]); gain = tot - np.mean([s0[n] for n in names])
    per = [sum(v[n] - s0[n] for n in names if attr[n]["method"] == m and attr[n]["type"] == t) / N for m, t in cells]
    L.append(f"  {k:42s} {tot:6.4f} {gain:+6.4f}  " + " ".join(f"{x:+7.4f}" for x in per))
L.append("\n== Plafond intra-classe : score de chaque case si la réparation est appliquée (moyenne dans la case)")
L.append(f"  {'case':28s} base  " + "  ".join(k[:14] for k in ["régime parfait", "seuil parfait/image", "taille parfaite (top-k vrai k)", "contour parfait bande 2 cell.", "composantes parfaites (FP retirés)", "vérité ±1 cellule (plafond)"]))
for m, t in cells:
    ns = [n for n in names if attr[n]["method"] == m and attr[n]["type"] == t]
    vals = [np.mean([rep[k][n] for n in ns]) for k in ["régime parfait", "seuil parfait/image", "taille parfaite (top-k vrai k)", "contour parfait bande 2 cell.", "composantes parfaites (FP retirés)", "vérité ±1 cellule (plafond)"]]
    L.append(f"  {m+'/'+t:28s} {np.mean([s0[n] for n in ns]):.3f} " + "  ".join(f"{x:14.3f}" for x in vals))

L.append("\n== boxsearch sur les images partielles du fold 0 (conf, IoU rectangle/vérité, score base -> box)")
for n in names:
    if not reg[n] and n in boxes:
        bm = rectmask(boxes[n][0]); g = gt[n]; iou = ((bm == 1) & (g == 1)).sum() / max(((bm == 1) | (g == 1)).sum(), 1)
        if boxes[n][1] >= 0.8 or idx[n]["is_rect"]:
            L.append(f"  {n} {idx[n]['method']:10s} {idx[n]['edit_type']:11s} conf={boxes[n][1]:.3f} IoU={iou:.3f} {s0[n]:.3f} -> {S(bm, n):.3f}")
L.append("\n== fiabilité boxsearch sur tout le train (fichiers *_train_boxes / evaltrain) : par seuil de conf, n déclenchés, part outpainting, IoU médiane")
allb = {}
for m in METHODS:
    for f in sorted((TASK / "t1/out/boxsearch").glob(f"{m}_*train*_boxes.json")):
        for k, v in json.loads(f.read_text())["boxes"].items():
            allb[k] = (v["rect"], v["conf"])
for m in METHODS:
    for c in (0.8, 0.9, 0.95):
        ks = [k for k, v in allb.items() if idx[k]["method"] == m and v[1] >= c]
        ious = []
        for k in ks:
            bm = rectmask(allb[k][0]); g = gt[k]; ious.append(((bm == 1) & (g == 1)).sum() / max(((bm == 1) | (g == 1)).sum(), 1))
        nr = sum(idx[k]["is_rect"] for k in ks); ntot = sum(1 for k in allb if idx[k]["method"] == m)
        nrect = sum(1 for k in allb if idx[k]["method"] == m and idx[k]["is_rect"])
        L.append(f"  {m:10s} conf>={c}: {len(ks)}/{ntot} déclenchés, {nr} outpainting (sur {nrect}), IoU méd {np.median(ious) if ious else float('nan'):.3f}, "
                 f"score méd box {np.median([S(rectmask(allb[k][0]), k) for k in ks]) if ks else float('nan'):.3f}")

# diagnostic des cas < 0.5
bad = sorted([n for n in names if s0[n] < 0.5], key=lambda n: s0[n])
L.append(f"\n== {len(bad)} images < 0.5")
cases = []
for n in bad:
    g, m = gt[n], base[n]; mi = mino(g); p = probs[n]
    tr, pr = (g == mi), (m == mi)
    inter = (tr & pr).sum(); iou = inter / max((tr | pr).sum(), 1)
    lab, k = ndimage.label(pr)
    fpc = sum(1 for i in range(1, k + 1) if (tr[lab == i]).mean() <= 0.5)
    auc_in = p[g == 1].mean() - p[g == 0].mean()
    d = dict(name=n, method=attr[n]["method"], type=attr[n]["type"], score=round(s0[n], 3), regerr=attr[n]["regerr"],
             mino_frac=round(frac[n], 4), pred_mino_frac=round(pr.mean(), 4), iou_mino=round(iou, 3), recall=round(inter / max(tr.sum(), 1), 3),
             prec=round(inter / max(pr.sum(), 1), 3), n_pred_comp=k, n_fp_comp=fpc, p_in_minus_out=round(float(auc_in), 3),
             best_thr_score=round(rep["seuil parfait/image"][n], 3), topk_score=round(rep["taille parfaite (top-k vrai k)"][n], 3),
             box=(boxes[n][1] if n in boxes else None))
    cases.append(d); L.append("  " + json.dumps(d))
(OUT / "cases.json").write_text(json.dumps(cases, indent=1))
with open(OUT / "per_image.tsv", "w") as f:
    f.write("name\tmethod\ttype\tq\tminofrac\tregerr\tscore\t" + "\t".join(rep) + "\n")
    for n in names:
        a = attr[n]
        f.write(f"{n}\t{a['method']}\t{a['type']}\t{a['q']}\t{frac[n]:.4f}\t{int(a['regerr'])}\t{s0[n]:.4f}\t" + "\t".join(f"{rep[k][n]:.4f}" for k in rep) + "\n")
(OUT / "report.txt").write_text("\n".join(L)); print("\n".join(L))

# ------------------------------------------------------------------ planche (16 pires)
T = 176
sel = bad[:16]
cols = 4; rows = (len(sel) + cols - 1) // cols
W = cols * (3 * T + 12); H = rows * (T + 30)
canvas = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
for i, n in enumerate(sel):
    x0 = (i % cols) * (3 * T + 12); y0 = (i // cols) * (T + 30)
    img = Image.open(TASK / "release/train/images" / f"{n}.png").convert("RGB").resize((T, T))
    def mk(mm, col):
        a = np.array(img).astype(np.float32) * 0.45
        big = np.kron(mm, np.ones((T // 128 + 1, T // 128 + 1)))[:T, :T] if False else np.array(Image.fromarray(mm * 255).resize((T, T), Image.NEAREST)) > 0
        a[big] = a[big] * 0.4 + np.array(col) * 0.6
        return Image.fromarray(a.astype(np.uint8))
    canvas.paste(img, (x0, y0 + 26)); canvas.paste(mk(gt[n], (0, 200, 0)), (x0 + T, y0 + 26)); canvas.paste(mk(base[n], (230, 0, 0)), (x0 + 2 * T, y0 + 26))
    a = attr[n]
    dr.text((x0 + 2, y0 + 2), f"#{i+1} {n} {a['method'][:5]}/{a['type'][:5]} s={s0[n]:.3f}" + (" REGIME" if a["regerr"] else ""), fill=(0, 0, 0))
    dr.text((x0 + 2, y0 + 13), "image | verite (vert=1) | prediction (rouge=1)", fill=(90, 90, 90))
canvas.save(OUT / "panel.png"); print("panel", len(sel))

# ------------------------------------------------------------------ combinaisons réalistes (boxsearch réel) + objectifs 0.70 / 0.75
def real(n, band=0, thr=False, tmps_level=None):
    if tmps_level is not None and idx[n]["method"] != "stegastamp" and idx[n]["edit_type"] != "outpainting":
        sc = tmps_level[idx[n]["edit_type"]]
        b = r_boxsearch(n, 0.9)
        return sc
    m = r_thr(n) if thr else base[n]
    if band: m = r_band(n, band, m)
    m = r_boxsearch(n, 0.9, m)
    return S(m, n)

stl = {t: np.mean([s0[n] for n in names if idx[n]["method"] == "stegastamp" and idx[n]["edit_type"] == t]) for t in TYPES}
R = {"boxsearch>=0.9": dict(), "boxsearch + contour 1": dict(band=1), "boxsearch + contour 2": dict(band=2),
     "boxsearch + seuil oracle": dict(thr=True), "boxsearch + seuil oracle + contour 1": dict(thr=True, band=1),
     "boxsearch + TM/PS hole&transplant au niveau stega": dict(tmps_level=stl)}
L2 = ["\n== Combinaisons réalistes (boxsearch réel conf>=0.9 sur images de régime partiel)"]
for k, kw in R.items():
    v = [real(n, **kw) for n in names]
    L2.append(f"  {k:52s} {np.mean(v):.4f}  (gain {np.mean(v)-np.mean(list(s0.values())):+.4f})")
sb = {n: real(n) for n in names}
st_ = np.mean([sb[n] for n in names if idx[n]["method"] == "stegastamp"])
for T in (0.70, 0.75):
    for lab, base_st in (("sans box", np.mean([s0[n] for n in names if idx[n]['method'] == 'stegastamp'])), ("avec box", st_)):
        x = (102 * T - 34 * base_st) / 68
        L2.append(f"  cible {T}: TM et PS doivent atteindre en moyenne {x:.3f} ({lab}, stega={base_st:.3f}) ; actuel TM {np.mean([sb[n] for n in names if idx[n]['method']=='trustmark']):.3f} PS {np.mean([sb[n] for n in names if idx[n]['method']=='pixelseal']):.3f} (avec box)")
(OUT / "report.txt").write_text("\n".join(L + L2)); print("\n".join(L2))
