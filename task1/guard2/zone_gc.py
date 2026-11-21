"""Zone ambiguë, régime = majorité du modèle (d) ou décodé (b) : effet de l'étape GrabCut de la chaîne sur les
images hors-fold concernées. Depuis TASK : python t1/guard2/zone_gc.py"""
import json, sys
from pathlib import Path
import numpy as np, cv2
TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK / "t1/refine"), str(TASK / "t1/train"), str(TASK / "t1/synth"), str(TASK / "t1/common"), str(TASK)]
from refine import refine  # noqa
from train import binarize  # noqa
from wmloc.utils import miou_image  # noqa
z = np.load(TASK / "release/train/masks_128.npz"); GT = dict(zip([str(n) for n in z["names"]], z["masks"]))
bx = json.loads((TASK / "t1/out/boxsearch2/trustmark_train_boxes.json").read_text())["boxes"]
def ld(p):
    zz = np.load(TASK / p); return {str(n): q.astype(np.float32) for n, q in zip(zz["names"], zz["probs"])}
cases = [("trustmark", "t1/out/oof/trustmark/oof_probs128.npz", n) for n in ("2dfa53b9ecf4", "a558bea42351", "6921edde828e")] + \
        [("pixelseal", "t1/train/../out/train/v0/fold0_probs128.npz", "2e5699234a38")]
for m, f, n in cases:
    p = ld(f)[n]; im = cv2.cvtColor(cv2.imread(str(TASK / f"release/train/images/{n}.png")), cv2.COLOR_BGR2RGB)
    gc = refine(p, im, m, False).astype(np.uint8)
    print(m, n, "bin", round(miou_image(binarize(p), GT[n]), 3), "grabcut(min)", round(miou_image(gc, GT[n]), 3), "box", bx.get(n) if m == "trustmark" else None)
