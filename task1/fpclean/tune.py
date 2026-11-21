"""Réglage hors-fold du post-traitement fp.clean (agent fpclean). Depuis TASK, CPU :
  python t1/fpclean/tune.py --src NAME [--e2e] [--workers 8]
Sources (cartes hors-fold) : voir SRC. Pour chaque image : régime = guard2 decide ; score de chaque configuration
  - sans --e2e : masque nettoyé seul (pas de GrabCut / rectangle)
  - avec --e2e : chaîne guard2 complète (nettoyage -> GrabCut TM/PS -> rectangle décodé)
Validation croisée par fold (5 folds réels ; sous-folds aléatoires si la source ne couvre qu'un fold) :
config choisie sur les autres folds, appliquée au fold tenu ; bootstrap apparié (10 000) du gain sur le total.
Écrit t1/out/fpclean/tune_<src>[_e2e].json."""
from __future__ import annotations

import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "fpclean"), str(TASK / "t1" / "guard2"), str(TASK / "t1" / "refine"), str(TASK)]
from fp import clean, clean_probs  # noqa: E402
from rule import decide, THR  # noqa: E402
from refine import refine  # noqa: E402
from wmloc.utils import miou_image, downsample_mask  # noqa: E402

OUT = TASK / "t1" / "out" / "fpclean"
GRABCUT = {"trustmark": True, "pixelseal": True, "stegastamp": False}
BOXES = {"trustmark": "t1/out/boxsearch2/trustmark_train_boxes.json", "stegastamp": "t1/out/boxsearch/stegastamp_train_boxes.json",
         "pixelseal": None}
CONF = {"trustmark": 0.79, "stegastamp": 0.9}

CONFIGS = [{"rule": "none"}]
CONFIGS += [{"rule": "cc", "frac": f} for f in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05)]
CONFIGS += [{"rule": "rel", "r": r} for r in (0.6, 0.7, 0.8, 0.9)]
CONFIGS += [{"rule": "open", "rad": r} for r in (1, 2, 3)]
CONFIGS += [{"rule": "open", "rad": 1, "frac": f} for f in (1.0, 0.2)]
CONFIGS += [{"rule": "topk", "t": t} for t in (0.1, 0.2, 0.3)]


def load(path):
    z = np.load(TASK / path)
    return {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}


def mix(parts):
    ds = [(w, load(f)) for w, f in parts]
    names = set.intersection(*[set(d) for _, d in ds])
    return {n: sum(w * d[n] for w, d in ds) for n in names}


def src_data(name):
    if name == "tm_tilesoof":
        return "trustmark", load("t1/out/tilesoof/trustmark/oof_probs128.npz")
    if name == "tm_frozen":
        return "trustmark", load("t1/out/oof/trustmark/oof_probs128.npz")
    if name == "ss_v1":
        d = load("t1/out/train/v1/stegastamp_f0/fold_probs128.npz"); d.update(load("t1/out/train/v1/stegastamp_f1/fold_probs128.npz"))
        return "stegastamp", d
    if name == "tm_chain0":
        return "trustmark", mix([(0.7, "t1/out/tiles2/trustmark/fold0_probs128.npz"), (0.3, "t1/out/tmdense/best/fold0_probs128.npz")])
    if name == "ps_chain0":
        return "pixelseal", load("t1/out/tiles3/pixelseal/fold0_probs128.npz")
    if name in ("tm_devel", "ps_devel"):   # fold 0 = carte de la chaîne, folds 1-4 = modèles devel hors-fold
        m = {"tm": "trustmark", "ps": "pixelseal"}[name[:2]]
        d = src_data(name[:2] + "_chain0")[1]
        for k in (1, 2, 3, 4):
            f = TASK / "t1/out/devel" / m / f"fold{k}" / "pred" / f"fold{k}_probs128.npz"
            if f.exists():
                d.update(load(f.relative_to(TASK)))
            else:
                print("missing", f, flush=True)
        return m, d
    raise ValueError(name)


def rect128(r):
    big = np.zeros((512, 512), np.uint8)
    big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)


def one(args):
    method, n, p, acc, box, gt, e2e = args
    m_guard, full = decide(p, acc, method)
    out = []
    img = None
    if e2e == "post":    # variante : nettoyage APRÈS GrabCut (masque final, masses de probabilité de la carte)
        return n, np.array(post_scores(method, n, p, m_guard, full, box, gt)), full
    for c in CONFIGS:
        if full:
            out.append(miou_image(m_guard, gt)); continue
        mc = clean(p, False, method, c)
        if not e2e:
            out.append(miou_image(mc, gt)); continue
        if box is not None and box.get("rect") and box["conf"] >= CONF[method]:
            r = rect128([int(v) for v in box["rect"]])
            if r.any() and r.mean() < 0.5:
                out.append(miou_image(r, gt)); continue
        if GRABCUT[method]:
            if img is None:
                im = cv2.imread(str(TASK / "release/train/images" / f"{n}.png"), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            p2 = p if c["rule"] == "none" else clean_probs(p, mc)
            m = refine(p2, img, method, False).astype(np.uint8)
        else:
            m = mc if c["rule"] != "none" else m_guard.astype(np.uint8)
        out.append(miou_image(m, gt))
    return n, np.array(out), full


def post_scores(method, n, p, m_guard, full, box, gt):
    if full:
        return [miou_image(m_guard, gt)] * len(CONFIGS)
    if box is not None and box.get("rect") and box["conf"] >= CONF[method]:
        r = rect128([int(v) for v in box["rect"]])
        if r.any() and r.mean() < 0.5:
            return [miou_image(r, gt)] * len(CONFIGS)
    if GRABCUT[method]:
        im = cv2.imread(str(TASK / "release/train/images" / f"{n}.png"), cv2.IMREAD_COLOR)
        m = refine(p, cv2.cvtColor(im, cv2.COLOR_BGR2RGB), method, False).astype(np.uint8)
    else:
        m = m_guard.astype(np.uint8)
    pp = np.where(m == 1, np.maximum(p, 0.5), np.minimum(p, 0.49)).astype(np.float32)
    return [miou_image(m if c["rule"] == "none" else clean(pp, False, method, c), gt) for c in CONFIGS]


def cfg_name(c):
    return ",".join(f"{k}={v}" for k, v in c.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--e2e", action="store_true"); ap.add_argument("--post", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    method, P = src_data(a.src)
    names = sorted(P)
    meta = {m["image_name"]: m for m in json.loads((TASK / "release/train/metadata.json").read_text())}
    assert all(meta[n]["method"] == method for n in names)
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    if isinstance(tidx, list):
        tidx = {e["image_name"]: e for e in tidx}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    zz = np.load(TASK / "t1/out/signal" / f"fullacc_{method}_train.npz"); acc = {str(n): float(x) for n, x in zip(zz["names"], zz["acc"])}
    sp = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    fold = {}
    for k, v in (sp.items() if isinstance(sp, dict) else enumerate(sp)):
        for n in v:
            fold[str(n).replace(".png", "")] = int(k)
    if a.post:
        a.e2e = "post"
    bx = json.loads((TASK / BOXES[method]).read_text())["boxes"] if (BOXES[method] and a.e2e) else {}
    jobs = [(method, n, P[n], acc[n], bx.get(n), gt[n], a.e2e) for n in names]
    with Pool(a.workers) as pool:
        res = pool.map(one, jobs, chunksize=1)
    S = np.stack([r[1] for r in res])                     # (N, C)
    full = np.array([r[2] for r in res]); et = np.array([tidx[n]["edit_type"] for n in names])
    fo = np.array([fold[n] for n in names])
    if len(set(fo)) < 3:                                  # une seule source de fold : 5 sous-folds aléatoires stratifiés par type
        rng = np.random.default_rng(0); fo = np.zeros(len(names), int)
        for t in set(et):
            idx = np.flatnonzero(et == t); rng.shuffle(idx); fo[idx] = np.arange(len(idx)) % 5
    sel = np.zeros(len(names)); chosen = {}
    for k in sorted(set(fo)):
        tr = fo != k
        j = int(np.argmax(S[tr].mean(0) + 1e-9 * (np.arange(S.shape[1]) == 0)))   # égalité -> "none"
        chosen[int(k)] = cfg_name(CONFIGS[j]); sel[~tr] = S[~tr, j]
    base = S[:, 0]; d = sel - base
    rng = np.random.default_rng(0); B = 10000
    bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    jall = int(np.argmax(S.mean(0) + 1e-9 * (np.arange(S.shape[1]) == 0)))
    R = {"src": a.src, "method": method, "e2e": a.e2e, "n": len(names), "n_minority": int((~full).sum()),
         "folds": sorted(int(k) for k in set(fo)),
         "base": {"all": float(base.mean()), **{t: float(base[et == t].mean()) for t in ("hole", "transplant", "outpainting")}},
         "cv": {"all": float(sel.mean()), **{t: float(sel[et == t].mean()) for t in ("hole", "transplant", "outpainting")}},
         "gain_cv": [float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
         "chosen_per_fold": chosen, "best_all_folds": cfg_name(CONFIGS[jall]),
         "per_config_all": {cfg_name(c): float(S[:, i].mean()) for i, c in enumerate(CONFIGS)},
         "per_config_minority_transplant": {cfg_name(c): float(S[et == "transplant", i].mean()) for i, c in enumerate(CONFIGS)}}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.src + ("_post" if a.e2e == "post" else "_e2e" if a.e2e else "")
    (OUT / f"tune_{tag}.json").write_text(json.dumps(R, indent=1))
    np.savez(OUT / f"tune_{tag}_scores.npz", names=np.array(names), S=S, fold=fo, full=full, et=et)
    print(json.dumps({k: R[k] for k in ("src", "e2e", "n", "n_minority", "base", "cv", "gain_cv", "chosen_per_fold", "best_all_folds")}))
    for k, v in sorted(R["per_config_all"].items(), key=lambda x: -x[1])[:8]:
        print(f"  {v:.4f}  {k}")


if __name__ == "__main__":
    main()
