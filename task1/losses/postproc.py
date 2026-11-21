"""Post-traitements CPU simples sur les masques binaires de la chaine v3, regles par CV imbriquee dans le fold 0.

    CUDA_VISIBLE_DEVICES="" python postproc.py        (depuis TASK, venv active ; importe decompose.py voisin)
Chaque post-traitement a UN parametre, regle separement par regime predit (minoritaire = 0 si la majorite predite est 1).
Protocole : 5 sous-folds du fold 0 stratifies methode x type ; parametre choisi sur 4/5, applique au 1/5 restant ;
le score annonce est celui des images tenues a l'ecart (aucune n'a servi a regler son propre parametre).
"""
import numpy as np
from scipy import ndimage as ndi

from decompose import load, miou_image, cause_maps

CROSS = ndi.generate_binary_structure(2, 1)


def minority(pred):
    mv = 0 if pred.mean() >= 0.5 else 1
    return pred == mv, mv


def back(mm, mv):
    return np.where(mm, mv, 1 - mv).astype(np.uint8)


def op_dilate(pred, r):
    mm, mv = minority(pred)
    if r > 0:
        mm = ndi.binary_dilation(mm, CROSS, iterations=r)
    elif r < 0:
        mm = ndi.binary_erosion(mm, CROSS, iterations=-r, border_value=1)
    return back(mm, mv)


def op_minsize(pred, k):
    mm, mv = minority(pred)
    if k > 0 and mm.any():
        lab, n = ndi.label(mm)
        sizes = ndi.sum(mm, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:] = sizes >= k
        if keep.any():                       # ne jamais vider entierement la minorite
            mm = keep[lab]
    return back(mm, mv)


def op_close(pred, r):
    mm, mv = minority(pred)
    if r > 0:
        mm = ndi.binary_closing(np.pad(mm, r), CROSS, iterations=r)[r:-r, r:-r]
        mm = ndi.binary_fill_holes(mm)
    return back(mm, mv)


OPS = {"dilate": (op_dilate, [-2, -1, 0, 1, 2, 3]),
       "minsize": (op_minsize, [0, 4, 16, 32, 64, 128]),
       "close": (op_close, [0, 1, 2, 3, 4])}


def main():
    idx, gt, probs, fold0 = load()
    names = list(fold0)
    preds = {n: (probs[n] >= 0.5).astype(np.uint8) for n in names}
    reg = {n: preds[n].mean() >= 0.5 for n in names}          # regime predit (connu au test)
    rng = np.random.default_rng(0)
    # sous-folds stratifies methode x type
    sub = {}
    groups = {}
    for n in names:
        groups.setdefault((idx[n]["method"], idx[n]["edit_type"]), []).append(n)
    for g in groups.values():
        g = list(rng.permutation(g))
        for i, n in enumerate(g):
            sub[n] = i % 5
    base = {n: miou_image(preds[n], gt[n]) for n in names}

    # split de la cause "manque" (bande 2) : composantes entierement ratees vs interieur
    whole = inner = 0.0
    for n in names:
        full = bool(idx[n]["full_watermark"]); g = gt[n]; p = preds[n]
        mv = 0 if full else 1
        gm, pm = g == mv, p == mv
        lab, k = ndi.label(gm)
        wmask = np.zeros_like(gm)
        for i in range(1, k + 1):
            c = lab == i
            if (pm & c).sum() < 0.3 * c.sum():
                wmask |= c
        cm = cause_maps(p, g, full, 2)["manque"]
        whole += miou_image(np.where(cm & wmask, g, p), g) - base[n]
        inner += miou_image(np.where(cm & ~wmask, g, p), g) - base[n]
    print(f"manque = composantes entierement ratees {100 * whole / len(names):.2f} pts + interieur rate {100 * inner / len(names):.2f} pts")

    meth = lambda n: idx[n]["method"]
    for name, (fn, grid) in OPS.items():
        sc = {(n, v): miou_image(fn(preds[n], v), gt[n]) for n in names for v in grid}
        # score in-sample par valeur et regime (information seulement)
        for r in (True, False):
            nr = [n for n in names if reg[n] == r]
            ins = {v: np.mean([sc[n, v] for n in nr]) for v in grid}
            print(f"[{name}] regime {'full' if r else 'partial'} in-sample : " + " ".join(f"{v}:{s:.4f}" for v, s in ins.items()))
        held = {}
        chosen = []
        for f in range(5):
            tr = [n for n in names if sub[n] != f]; te = [n for n in names if sub[n] == f]
            for r in (True, False):
                trr = [n for n in tr if reg[n] == r]
                best = max(grid, key=lambda v: (np.mean([sc[n, v] for n in trr]), -abs(v)))
                chosen.append((f, "full" if r else "part", best))
                for n in te:
                    if reg[n] == r:
                        held[n] = sc[n, best]
        d = np.array([held[n] - base[n] for n in names])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"[{name}] choix par sous-fold : {chosen}")
        print(f"[{name}] tenu a l'ecart : {np.mean(list(held.values())):.4f} vs base {np.mean(list(base.values())):.4f} "
              f"delta {d.mean():+.4f} IC95 [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
        for m in ("stegastamp", "trustmark", "pixelseal"):
            nm = [n for n in names if meth(n) == m]
            print(f"    {m:<11} {np.mean([held[n] for n in nm]):.4f} vs {np.mean([base[n] for n in nm]):.4f}")
        for t in ("hole", "transplant", "outpainting"):
            nt = [n for n in names if idx[n]["edit_type"] == t]
            print(f"    {t:<11} {np.mean([held[n] for n in nt]):.4f} vs {np.mean([base[n] for n in nt]):.4f}")


if __name__ == "__main__":
    main()
