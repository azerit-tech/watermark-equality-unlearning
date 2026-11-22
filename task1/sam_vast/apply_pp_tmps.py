"""Post-traitement CPU d'une soumission tâche 1 (candidat FAIBLE, voir rapport) — TrustMark et PixelSeal seulement.
Règle (N=32, A=164 ; classe minoritaire = classe la moins fréquente du masque ; composantes en 4-connexité) :
  1. si le masque est majoritairement à 1 (régime « full ») : supprimer les composantes minoritaires de moins de N cellules
     (étape annulée si elle viderait la classe minoritaire) ;
  2. tout régime : si la classe minoritaire compte moins de A cellules, la dilater (croix 3x3) jusqu'à >= A cellules.
Garde-fous : jamais de masque vide/plein, jamais de bascule de classe majoritaire. StegaStamp inchangé.
Usage : python3 apply_pp_tmps.py --in sub.pt --out sortie.pt [--meta .../release/test/metadata.json] [-N 32] [-A 164]
"""
import argparse, json
import numpy as np, torch
from scipy import ndimage as ndi

CROSS = ndi.generate_binary_structure(2, 1)


def _join(maj, mino, orig):
    out = np.where(mino, 1 - maj, maj).astype(np.uint8)
    if out.min() == out.max() or (out.mean() >= 0.5) != (orig.mean() >= 0.5):
        return orig
    return out


def rule(m, N=32, A=164):
    m = np.asarray(m).astype(np.uint8)
    maj = int(m.mean() >= 0.5)
    if N and maj == 1:
        mino = m == 0
        lab, _ = ndi.label(mino)
        m = _join(maj, (np.bincount(lab.ravel())[lab] >= N) & mino, m)
    mino = m == 1 - maj
    if A and mino.sum() < A:
        o = mino.copy()
        for _ in range(64):
            if o.sum() >= A:
                break
            o = ndi.binary_dilation(o, CROSS)
        m = _join(maj, o, m)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="/Users/jacquesmougeot/Desktop/Azerit_t1_mirror/release/test/metadata.json")
    ap.add_argument("-N", type=int, default=32)
    ap.add_argument("-A", type=int, default=164)
    ap.add_argument("--methods", default="trustmark,pixelseal")
    a = ap.parse_args()
    sub = torch.load(a.inp, weights_only=True)
    meth = {x["image_name"]: x["method"] for x in json.load(open(a.meta))}
    M = sub["masks"].numpy(); new = M.copy(); stat = {}
    for i, n in enumerate(sub["image_names"]):
        if meth[n] in a.methods.split(","):
            new[i] = rule(M[i], a.N, a.A).astype(bool)
            s = stat.setdefault(meth[n], [0, 0]); d = int((new[i] != M[i]).sum()); s[0] += d > 0; s[1] += d
    out = torch.from_numpy(new)
    assert out.dtype == torch.bool and tuple(out.shape) == tuple(sub["masks"].shape)
    cov = out.flatten(1).float().mean(1); old = sub["masks"].flatten(1).float().mean(1)
    assert not ((cov == 0) | (cov == 1)).any() and bool(((cov >= 0.5) == (old >= 0.5)).all())
    torch.save({"image_names": sub["image_names"], "masks": out}, a.out)
    print({k: {"images_changed": v[0], "cells_changed": v[1]} for k, v in stat.items()}, "->", a.out)
