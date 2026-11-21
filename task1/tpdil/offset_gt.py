"""[agent tpdil] Folds 1-4, transplant, branche GrabCut : décalage signé moyen (px, à 512) entre silhouette et masque vrai
= (aire prédite - aire vraie) / périmètre vrai ; pour la carte seuillée (avant GrabCut) et pour GrabCut."""
import json, sys
from pathlib import Path
import cv2, numpy as np
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1" / "tpdil"), str(TASK / "t1" / "guard2"), str(TASK / "t1" / "refine"), str(TASK)]
from dil import grabcut512  # noqa
from refine import CFG, up, guard  # noqa
F = json.load(open(TASK / "t1/out/tpdil/folds14.json"))
folds = json.load(open(TASK / "t1/common/splits_5fold.json"))["folds"]
maps = {}
z = np.load(TASK / "t1/out/oof/trustmark/oof_probs128.npz"); maps.update({str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])})
for k in (1, 2, 3):
    z = np.load(TASK / f"t1/out/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz"); maps.update({str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])})
mdir = TASK / "release/train/masks"
ex = sorted(mdir.iterdir())[0]
for m in F:
    c = CFG[m]["partial"]["obj"]; om, og = [], []
    for r in F[m]:
        if r["branch"] != "grabcut" or r["type"] != "transplant":
            continue
        n = r["name"]
        gt = cv2.imread(str(mdir / f"{n}{ex.suffix}"), cv2.IMREAD_GRAYSCALE)
        gt = (gt > 127).astype(np.uint8)
        im = cv2.cvtColor(cv2.imread(str(TASK / "release/train/images" / f"{n}.png")), cv2.COLOR_BGR2RGB)
        p = maps[n]
        if gt.mean() > 0.5:
            continue
        per = cv2.Canny(gt * 255, 50, 150).astype(bool).sum() + 1e-6
        mm = (up(p, 512) >= 0.5).astype(np.uint8)
        g = grabcut512(im, p, c["size"], c["band"], c["iters"])
        om.append((int(mm.sum()) - int(gt.sum())) / per); og.append((int(g.sum()) - int(gt.sum())) / per)
    om, og = np.array(om), np.array(og)
    print(m, "n", len(og), "decalage carte median %+.2f px moy %+.2f | GrabCut median %+.2f px moy %+.2f (IQR %+.2f..%+.2f)" % (
        np.median(om), om.mean(), np.median(og), og.mean(), *np.percentile(og, [25, 75])), flush=True)
