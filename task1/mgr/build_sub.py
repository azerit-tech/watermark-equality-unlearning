"""Soumission composite (NON envoyée par ce script). Depuis TASK, CPU :
    python t1/mgr/build_sub.py --tag v0 --model stegastamp --out t1/out/mgr/submissions/sub01.pt
Méthodes listées dans --model : probabilités du segmenteur (dossier --tag) + garde-fou de régime.
Autres méthodes : constante selon le régime décodé (tout 1 si « full », sinon tout 0).
Évalue la même recette sur le fold 0 réel avant d'écrire le fichier test."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "train", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from regime_guard import guard, load_regimes  # noqa: E402
from evaluate import evaluate, format_result  # noqa: E402


def build(names, method, probs, reg, model_methods):
    out = {}
    for n in names:
        if method[n] in model_methods:
            out[n] = guard(probs[n].astype(np.float32), reg[n])[0].astype(np.uint8)
        else:
            out[n] = (np.ones if reg[n] else np.zeros)((128, 128), np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0"); ap.add_argument("--model", default="stegastamp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mm = set(a.model.split(","))
    d = TASK / "t1" / "out" / "train" / a.tag
    # 1) fold 0 réel
    z = np.load(d / "fold0_probs128.npz"); names = [str(n) for n in z["names"]]
    probs = dict(zip(names, z["probs"]))
    meta = {m["image_name"]: m["method"] for m in json.loads((TASK / "release/train/metadata.json").read_text())}
    pred = build(names, meta, probs, load_regimes("train"), mm)
    print(format_result(evaluate(pred, names), f"fold 0 reel, modele={sorted(mm)}"))
    # 2) test
    tm = json.loads((TASK / "release/test/metadata.json").read_text())
    tnames = [m["image_name"] for m in tm]; tmeth = {m["image_name"]: m["method"] for m in tm}
    assert len(tnames) == len(set(tnames)) == 1020
    z = np.load(d / "test_probs128.npz"); tprobs = dict(zip([str(n) for n in z["names"]], z["probs"]))
    treg = load_regimes("test")
    tp = build(tnames, tmeth, tprobs, treg, mm)
    masks = torch.from_numpy(np.stack([tp[n] for n in tnames]).astype(bool))
    assert masks.dtype == torch.bool and tuple(masks.shape) == (1020, 128, 128)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": tnames, "masks": masks}, a.out)
    size = os.path.getsize(a.out); assert size <= 20 * 1000 * 1000, size
    chk = torch.load(a.out, weights_only=True)
    assert chk["image_names"] == tnames and torch.equal(chk["masks"], masks)
    for m in ("pixelseal", "trustmark", "stegastamp"):
        idx = [i for i, n in enumerate(tnames) if tmeth[n] == m]
        cov = masks[idx].float().mean(dim=(1, 2))
        print(f"test {m}: n={len(idx)} regime full={np.mean([treg[tnames[i]] for i in idx]):.3f} "
              f"couverture moy={cov.mean():.3f} images>0.5={float((cov > 0.5).float().mean()):.3f}")
    print(f"ecrit {a.out}: {size/1e6:.2f} Mo, couverture moyenne {masks.float().mean():.3f} (NON envoye)")


if __name__ == "__main__":
    main()
