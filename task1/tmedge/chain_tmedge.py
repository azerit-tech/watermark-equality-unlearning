"""[copie agent stegatta2 de chain_v5.py : StegaStamp = carte TTA (t1/out/stegatta, 4 retournements) si régime décodé majoritaire (fullacc >= THR), carte stega/best sinon ; alignement par nom]
[copie agent guard2 : garde-fou remplacé par t1/guard2/rule.decide ; --maps old|new ; --rule guard|zone ; --out]
Chaîne fixée a priori, par méthode : carte -> seuil 0,5 + garde-fou -> GrabCut (régime minoritaire, TM/PS)
-> rectangle décodé (régime minoritaire, conf >= 0,9). Aucun réglage n'est fait ici.
Depuis TASK (CPU, ~2-5 min) :  OMP_NUM_THREADS=1 python t1/chain/chain.py [--workers 8]
Écrit t1/out/chain/<methode>/{fold0,test}_probs128.npz (probs = masque final 0/1 float16) + t1/out/chain/report.json."""
from __future__ import annotations

import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
sys.path[:0] = [str(TASK / "t1" / "guard2"), str(TASK / "t1" / "refine"), str(TASK / "t1" / "train"), str(TASK / "t1" / "synth"),
                str(TASK / "t1" / "common"), str(TASK)]
from refine import refine, guard  # noqa: E402  (guard identique à t1/train/regime_guard.guard)
from regime_guard import load_regimes  # noqa: E402
from rule import decide, THR  # noqa: E402
from wmloc.utils import miou_image, downsample_mask  # noqa: E402

OUT = TASK / "t1" / "out" / "guard2"
RULE = "zone"
T = "t1/out/tiles"
MAPS = {   # (poids, [fichiers moyennés]) ; {split} = fold0 | test
    "trustmark": [(0.7, [f"{T}/trustmark/{{split}}_t256_probs128.npz", f"{T}/trustmark/{{split}}_t320_probs128.npz"]),
                  (0.3, ["t1/out/tmdense/best/{split}_probs128.npz"])],
    "pixelseal": [(1.0, [f"{T}/pixelseal/{{split}}_t256_probs128.npz", f"{T}/pixelseal/{{split}}_t320_probs128.npz"])],
    "stegastamp": [(1.0, ["t1/out/stega/best/{split}_probs128.npz"])],
}
MAPS_NEW = {   # consignes coordinateur : tuiles TM prolongées (tiles2) et PS prolongées (tiles3), mêmes règles de mélange
    "trustmark": [(0.7, ["t1/out/mgr/ens/trustmark/{split}_probs128.npz"]), (0.3, ["t1/out/tmdense/best/{split}_probs128.npz"])],
    "pixelseal": [(1.0, ["t1/out/mgr/ens/pixelseal/{split}_probs128.npz"])],
    "stegastamp": [(1.0, ["t1/out/stega/best/{split}_probs128.npz"])],
}
GRABCUT = {"trustmark": True, "pixelseal": True, "stegastamp": False}
BOXES = {"trustmark": "t1/out/boxsearch2/trustmark_{s}_boxes.json",   # remplace tmofficial (consigne coordinateur)
         "stegastamp": "t1/out/stegaedge/stegastamp_{s}_boxes.json", "pixelseal": "t1/out/psbox/pixelseal_{s}_boxes.json"}
CONF = {"trustmark": 0.79, "stegastamp": 0.9, "pixelseal": 0.90}   # seuils fixés (TM : validation croisée de l'agent boxsearch2)
STEPS = ["map", "guard", "grabcut", "rect"]


def load(path):
    z = np.load(TASK / path)
    return {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}


STTA = "t1/out/stegatta/stegastamp/{split}_probs128.npz"   # TTA 4 vues (moyenne), ordre des noms différent : aligné par nom
REG = {}   # split -> {nom: fullacc}, rempli dans main


def build_map(method, split, names):
    if method == "stegastamp":
        base = load(MAPS[method][0][1][0].format(split=split)); tta = load(STTA.format(split=split))
        assert set(tta) >= set(names)
        return np.stack([tta[n] if REG[split][n] >= THR[method] else base[n] for n in names])
    acc = np.zeros((len(names), 128, 128), np.float32)
    for w, files in MAPS[method]:
        sub = np.zeros_like(acc)
        for f in files:
            d = load(f.format(split=split))
            sub += np.stack([d[n] for n in names])
        acc += w * sub / len(files)
    return acc


def rect128(r):
    big = np.zeros((512, 512), np.uint8)
    big[max(r[0], 0):min(r[1], 512), max(r[2], 0):min(r[3], 512)] = 1
    return downsample_mask(big).astype(np.uint8)


def read_img(split, n):
    d = "train" if split == "fold0" else "test"
    im = cv2.imread(str(TASK / "release" / d / "images" / f"{n}.png"), cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def chain_one(args):
    method, split, n, p, full, box = args
    m_map = (p >= 0.5).astype(np.uint8)
    acc = full   # ici `full` porte la précision fullacc (voir main)
    if RULE == "zone":
        m_guard, full = decide(p, acc, method)
    else:
        full = bool(acc >= THR[method]); m_guard = guard(p, full)[0]
    m_guard = m_guard.astype(np.uint8)
    if GRABCUT[method] and not full:
        m_gc = refine(p, read_img(split, n), method, full).astype(np.uint8)   # CFG de t1/refine (fixé)
    else:
        m_gc = m_guard
    m_rect, used = m_gc, False
    if box is not None and not full and box.get("rect") and box["conf"] >= CONF[method]:
        r = rect128([int(v) for v in box["rect"]])
        if r.any() and r.mean() < 0.5:
            m_rect, used = r, True
    return n, {"map": m_map, "guard": m_guard, "grabcut": m_gc, "rect": m_rect}, used


def boot(d, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    bm = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--maps", default="new", choices=["old", "new"])
    ap.add_argument("--rule", default="zone", choices=["guard", "zone"])
    ap.add_argument("--out", default="t1/out/guard2")
    ap.add_argument("--tmboxes", default="")   # tmedge : rectangles TrustMark raffinés ({s} = train|test)
    ap.add_argument("--sgboxes", default="")
    ap.add_argument("--methods", default="stegastamp,trustmark,pixelseal")
    a = ap.parse_args()
    global OUT, RULE, MAPS
    OUT = TASK / a.out; RULE = a.rule
    if a.maps == "new":
        MAPS = MAPS_NEW
    if a.tmboxes: BOXES["trustmark"] = a.tmboxes
    if a.sgboxes: BOXES["stegastamp"] = a.sgboxes
    meta = {m["image_name"]: m for m in json.loads((TASK / "release/train/metadata.json").read_text())}
    tmeta = {m["image_name"]: m for m in json.loads((TASK / "release/test/metadata.json").read_text())}
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    if isinstance(tidx, list):
        tidx = {e["image_name"]: e for e in tidx}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    reg = {}   # précision de décodage brute (fullacc) ; le régime est décidé dans chain_one
    for sp, tag in (("fold0", "train"), ("test", "test")):
        reg[sp] = {}
        for mm in THR:
            zz = np.load(TASK / "t1" / "out" / "signal" / f"fullacc_{mm}_{tag}.npz")
            reg[sp].update({str(n): float(x) for n, x in zip(zz["names"], zz["acc"])})
    REG.update(reg)
    report = {}
    allper = {}
    with Pool(a.workers) as pool:
        for method in a.methods.split(","):
            report[method] = {}
            for split in ("fold0", "test"):
                ref = load(MAPS[method][0][1][0].format(split=split))
                names = sorted(ref)
                src = meta if split == "fold0" else tmeta
                assert all(src[n]["method"] == method for n in names), (method, split)
                P = build_map(method, split, names)
                bx = None
                if BOXES[method]:
                    s = "train" if split == "fold0" else "test"
                    bx = json.loads((TASK / BOXES[method].format(s=s)).read_text())["boxes"]
                jobs = [(method, split, n, P[i], reg[split][n], (bx or {}).get(n)) for i, n in enumerate(names)]
                res = pool.map(chain_one, jobs, chunksize=2)
                final = {n: r["rect"] for n, r, _ in res}
                nrect = sum(u for _, _, u in res)
                d = OUT / method; d.mkdir(parents=True, exist_ok=True)
                np.savez(d / f"{split}_probs128.npz", names=np.array(names),
                         probs=np.stack([final[n] for n in names]).astype(np.float16))
                R = {"n": len(names), "rect_applied": int(nrect),
                     "n_minority_decoded": int(sum(reg[split][n] < THR[method] for n in names)),
                     "boxes_available_minority": int(sum(1 for n in names if bx and n in bx and reg[split][n] < THR[method])),
                     "guard_changed": int(sum(int((r["guard"] != r["map"]).any()) for _, r, _ in res)), "grabcut_changed": int(sum(int((r["grabcut"] != r["guard"]).any()) for _, r, _ in res))}
                cov = np.array([final[n].mean() for n in names])
                R["coverage_mean"] = float(cov.mean())
                R["majority_share"] = float((cov >= 0.5).mean())
                R["regime_full_decoded"] = float(np.mean([reg[split][n] >= THR[method] for n in names]))
                R["minority_empty"] = int(sum(1 for n in names if final[n].min() == final[n].max()))
                if split == "fold0":
                    per = {s: np.array([miou_image(r[s], gt[n]) for n, r, _ in res]) for s in STEPS}
                    et = np.array([tidx[n]["edit_type"] for n, _, _ in res])
                    allper[method] = (per, et)
                    d.joinpath("per_fold0.json").write_text(json.dumps({n: float(x) for (n, _, _), x in zip(res, per["rect"])}))
                    R["scores"] = {s: {"all": float(per[s].mean()),
                                       **{t: float(per[s][et == t].mean()) for t in ("hole", "transplant", "outpainting")}}
                                   for s in STEPS}
                    R["delta_boot95"] = {f"{s1}->{s2}": boot(per[s2] - per[s1])
                                         for s1, s2 in zip(STEPS[:-1], STEPS[1:])}
                    R["delta_boot95"]["map->rect"] = boot(per["rect"] - per["map"])
                    R["n_by_type"] = {t: int((et == t).sum()) for t in ("hole", "transplant", "outpainting")}
                report[method][split] = R
                print(method, split, json.dumps(R), flush=True)
    # global = moyenne des 3 méthodes (34 images chacune), bootstrap stratifié par méthode
    rng = np.random.default_rng(1); B = 10000
    glob = {}
    for s in STEPS:
        glob[s] = float(np.mean([allper[m][0][s].mean() for m in allper]))
    for s1, s2 in list(zip(STEPS[:-1], STEPS[1:])) + [("map", "rect")]:
        bm = np.zeros(B)
        for m in allper:
            dd = allper[m][0][s2] - allper[m][0][s1]
            bm += dd[rng.integers(0, len(dd), (B, len(dd)))].mean(1)
        bm /= len(allper)
        glob[f"{s1}->{s2}"] = [glob[s2] - glob[s1], float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    bm = np.zeros(B)
    for m in allper:
        v = allper[m][0]["rect"]; bm += v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    bm /= len(allper)
    glob["final_boot95"] = [glob["rect"], float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    report["global"] = glob
    print("global", json.dumps(glob))
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    (OUT / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
