"""Audit 1 : format des soumissions, orientation, alignement noms <-> masques, cohérence raffinage / garde-fou.
Depuis TASK (CPU) : OMP_NUM_THREADS=8 python t1/audit/a1_format.py"""
import json, sys, itertools
from pathlib import Path
import numpy as np, torch

TASK = Path.cwd()
sys.path[:0] = [str(TASK / "t1/mgr"), str(TASK / "t1/train"), str(TASK / "t1/synth"), str(TASK / "t1/common"), str(TASK)]
import build_best as bb  # noqa
from regime_guard import load_regimes, guard  # noqa
from wmloc.utils import miou_image, downsample_mask  # noqa

SUB = TASK / "t1/out/mgr/submissions"
tm = json.loads((TASK / "release/test/metadata.json").read_text())
tnames = [m["image_name"] for m in tm]; tmeth = {m["image_name"]: m["method"] for m in tm}
meta, gt = bb.load_train()
fold0 = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0]
reg_tr, reg_te = load_regimes("train"), load_regimes("test")
z = np.load(TASK / "release/train/masks_128.npz"); print("masks_128 keys", z.files, z["masks"].shape, z["masks"].dtype)

if 0:
  pass
print("\n######## 1. format des fichiers")
subs = {}
for f in sorted(SUB.glob("*.pt")):
    size = f.stat().st_size
    d = torch.load(f, weights_only=True)
    names, masks = d["image_names"], d["masks"]
    ok = dict(keys=sorted(d.keys()) == ["image_names", "masks"], names_list=isinstance(names, list) and all(isinstance(n, str) for n in names),
              n=len(names) == 1020, unique=len(set(names)) == 1020, same_set=set(names) == set(tnames), same_order=names == tnames,
              no_ext=not any(n.endswith(".png") for n in names), shape=tuple(masks.shape) == (1020, 128, 128),
              dtype=str(masks.dtype), values=sorted(torch.unique(masks.to(torch.uint8)).tolist()), size_MB=round(size / 1e6, 2))
    print(f.name, ok)
    subs[f.stem] = {n: masks[i].numpy().astype(np.uint8) for i, n in enumerate(names)}

print("\n######## 1b. orientation : vrai masque -> chaîne build_best (défaut thr=0.5 + garde-fou) -> masks_128 ?")
names = sorted(gt)
for reg_name, reg in (("régime vrai", {n: bool(meta[n]["full_watermark"]) for n in names}), ("régime décodé", reg_tr)):
    np.savez(TASK / "t1/out/audit/gt_as_probs.npz", names=np.array(names), probs=np.stack([gt[n] for n in names]).astype(np.float16))
    res = {}
    for method in bb.METHODS:
        probs = bb.load_val(["t1/out/audit/gt_as_probs.npz"], method, {n: m["method"] for n, m in meta.items()})
        pred = {n: bb.predict(p, reg[n], dict(bb.DEFAULT), None, n) for n, p in probs.items()}
        exact = sum(np.array_equal(pred[n], gt[n]) for n in pred)
        bad = [n for n in pred if not np.array_equal(pred[n], gt[n])]
        res[method] = (exact, len(pred), [(n, round(miou_image(pred[n], gt[n]), 3), reg[n], bool(meta[n]["full_watermark"])) for n in bad[:4]])
    print(reg_name, res)
# écriture torch.save -> torch.load -> réindexation par nom (comme main())
tp = {n: gt[n] for n in names}
masks = torch.from_numpy(np.stack([tp[n] for n in names]).astype(bool))
torch.save({"image_names": names, "masks": masks}, TASK / "t1/out/audit/rt.pt")
d = torch.load(TASK / "t1/out/audit/rt.pt", weights_only=True)
print("aller-retour torch.save/load exact :", all(np.array_equal(d["masks"][i].numpy().astype(np.uint8), gt[n]) for i, n in enumerate(d["image_names"])))
# masque pleine résolution officiel -> downsample_mask == masks_128 ? (axe y,x)
from shapes import read_mask  # noqa
nn = names[:30]; eq = [np.array_equal(downsample_mask(read_mask(n)), gt[n]) for n in nn]
eqT = [np.array_equal(downsample_mask(read_mask(n)).T, gt[n]) for n in nn]
print(f"downsample(mask 512) == masks_128 : {sum(eq)}/30 ; transposé : {sum(eqT)}/30")

print("\n######## 1c. orientation empirique des sources (fold 0 : normal vs transposé vs retourné)")
SRCS = json.loads((TASK / "t1/mgr/sources.json").read_text())
for method in bb.METHODS:
    for s in SRCS[method]:
        f = s["val"][0]
        if not (TASK / f).exists():
            continue
        pr = bb.load_val([f], method, {n: m["method"] for n, m in meta.items()}); pr = {n: p for n, p in pr.items() if n in gt}
        sc = lambda tf: np.mean([miou_image(guard(tf(p), reg_tr[n])[0], gt[n]) for n, p in pr.items()])
        print(f"{method:10s} {s['name']:18s} n={len(pr)} normal={sc(lambda p: p):.4f} T={sc(lambda p: p.T):.4f} flipud={sc(lambda p: p[::-1]):.4f}")

print("\n######## 1d. reconstruction exacte de sub04 / sub03 depuis leur .json")
for tag in ("sub03", "sub04"):
    info = json.loads((SUB / f"{tag}.json").read_text())
    boxdir = Path(info["boxdir"]); boxdir = boxdir if boxdir.is_absolute() else TASK / boxdir
    tot = 0
    for method, c in info["chosen"].items():
        mt = [n for n in tnames if tmeth[n] == method]
        if c["source"] == "constante":
            pred = {n: (np.ones if reg_te[n] else np.zeros)((128, 128), np.uint8) for n in mt}
        else:
            tprobs = bb.load_test(c["test_files"], mt); tb = None if c["params"].get("box") is None else bb.load_boxes(boxdir / f"{method}_test_boxes.json")
            pred = {n: bb.predict(tprobs[n], reg_te[n], c["params"], tb, n) for n in mt}
        same = sum(np.array_equal(pred[n], subs[tag][n]) for n in mt); tot += same
        print(f"{tag} {method:10s} identiques {same}/{len(mt)}")
    print(tag, "total identiques", tot)

print("\n######## 1e. alignement des noms test : accord entre modèles INDÉPENDANTS, vrais noms vs noms permutés")
rng = np.random.default_rng(0)
def agree(A, B, ns, perm=False):
    nb = list(ns);
    if perm: nb = list(rng.permutation(nb))
    return np.mean([miou_image(A[a], B[b]) for a, b in zip(ns, nb)])
pairs = {"trustmark": ("t1/out/dense/login/tm", "t1/out/frozen/trustmark"), "pixelseal": ("t1/out/dense/login/ps", "t1/out/mgr/smoke_gpu/ps"),
         "stegastamp": ("t1/out/train/v0", "t1/out/dense/login/tm")}
for method, (a, b) in pairs.items():
    for split, fn, rg in (("fold0", "fold0_probs128.npz", reg_tr), ("test", "test_probs128.npz", reg_te)):
        za, zb = np.load(TASK / a / fn), np.load(TASK / b / fn)
        A = {str(n): guard(p.astype(np.float32), rg[str(n)])[0] for n, p in zip(za["names"], za["probs"]) if (meta if split == "fold0" else tmeth).get(str(n)) and ((meta[str(n)]["method"] if split == "fold0" else tmeth[str(n)]) == method)}
        B = {str(n): guard(p.astype(np.float32), rg[str(n)])[0] for n, p in zip(zb["names"], zb["probs"]) if str(n) in A}
        ns = sorted(set(A) & set(B))
        if method == "stegastamp":   # pas de 2e modèle indépendant : on compare la source v0 à la soumission sub01 (garde-fou) - test seulement
            continue
        print(f"{method:10s} {split:5s} n={len(ns)} accord vrais noms={agree(A, B, ns):.4f}  noms permutés={agree(A, B, ns, True):.4f}")
# soumission vs refine masks vs signal maps (indépendant du segmenteur)
for method in ("trustmark", "pixelseal"):
    zm = np.load(TASK / f"t1/out/signal/maps_{method}_test.npz"); print(method, "maps keys", zm.files, {k: zm[k].shape for k in zm.files})

print("\n######## 5. masques raffinés vs *_probs128 vs sub04")
for method in ("trustmark", "pixelseal", "stegastamp"):
    for split in ("fold0", "test"):
        a = np.load(TASK / f"t1/out/refine/{method}_{split}_masks.npz"); b = np.load(TASK / f"t1/out/refine/{method}_{split}_probs128.npz")
        A = dict(zip(a["names"].tolist(), a["masks"])); B = dict(zip(b["names"].tolist(), b["probs"]))
        same = sum(np.array_equal(A[n].astype(np.float32), B[n].astype(np.float32)) for n in A if n in B)
        rgx = reg_tr if split == "fold0" else reg_te
        contra = sum(((A[n].mean() >= 0.5) != rgx[n]) for n in A)
        allsame = sum((A[n].all() or not A[n].any()) for n in A)
        dg = sum(not np.array_equal(guard(B[n].astype(np.float32), rgx[n])[0], A[n]) for n in B)
        uniq = np.unique(b["probs"]).tolist()
        print(f"{method:10s} {split:5s} masks n={len(A)} probs n={len(B)} dtype={b['probs'].dtype} valeurs={uniq} identiques={same} "
              f"contredit régime={contra} classe vide={allsame} garde-fou(probs)!=masque={dg}")
        if split == "test" and method != "stegastamp":
            print(f"   sub04 == refine masks : {sum(np.array_equal(subs['sub04'][n], A[n]) for n in A)}/{len(A)}")

print("\n######## différences entre soumissions (images modifiées par méthode)")
for x, y in itertools.combinations(sorted(subs), 2):
    print(x, y, {m: sum(not np.array_equal(subs[x][n], subs[y][n]) for n in tnames if tmeth[n] == m) for m in bb.METHODS},
          "accord mIoU", {m: round(float(np.mean([miou_image(subs[x][n], subs[y][n]) for n in tnames if tmeth[n] == m])), 3) for m in bb.METHODS})
