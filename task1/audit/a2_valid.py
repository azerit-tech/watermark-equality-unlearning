"""Audit 2 : caches de régime (couverture, distributions train/test, zone ambiguë), optimisme de la sélection sur fold 0,
bruit local -> public, fuites (cname des chunks). Depuis TASK : OMP_NUM_THREADS=8 python t1/audit/a2_valid.py"""
import json, sys, glob
from pathlib import Path
import numpy as np

TASK = Path.cwd()
sys.path[:0] = [str(TASK / "t1/mgr"), str(TASK / "t1/train"), str(TASK / "t1/synth"), str(TASK / "t1/common"), str(TASK)]
import build_best as bb  # noqa
from regime_guard import THR, guard  # noqa
from wmloc.utils import miou_image  # noqa

tm = json.loads((TASK / "release/test/metadata.json").read_text()); tmeth = {m["image_name"]: m["method"] for m in tm}
meta, gt = bb.load_train()
folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fold0 = set(folds[0])
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
rng = np.random.default_rng(0)

print("######## 4. caches de régime")
for m, t in THR.items():
    zs = {s: np.load(TASK / f"t1/out/signal/fullacc_{m}_{s}.npz") for s in ("train", "test")}
    zt = zs["test"]; nt = [str(n) for n in zt["names"]]
    exp_t = sorted(n for n, mm in tmeth.items() if mm == m)
    print(f"{m}: test n={len(nt)} uniques={len(set(nt))} == noms test de la méthode : {sorted(nt) == exp_t} ; "
          f"nan={int(np.isnan(zt['acc']).sum())}")
    ztr = zs["train"]; ntr = [str(n) for n in ztr["names"]]; atr = ztr["acc"].astype(float)
    full = np.array([bool(meta[n]["full_watermark"]) for n in ntr]); f0 = np.array([n in fold0 for n in ntr])
    print(f"   train n={len(ntr)} (toutes de la méthode : {all(meta[n]['method'] == m for n in ntr)})")
    for s, a, lab in (("train", atr, full), ("test", zt["acc"].astype(float), None)):
        h, e = np.histogram(a, bins=np.arange(0.40, 1.0001, 0.05))
        print(f"   {s:5s} hist [0.40..1.00 pas 0.05] {h.tolist()}  part >= seuil {np.mean(a >= t):.3f}")
        for w in (0.02, 0.05):
            print(f"         |acc - {t}| < {w} : {int((np.abs(a - t) < w).sum())}")
    err = (atr >= t) != full
    print(f"   erreurs train au seuil {t}: {int(err.sum())}/{len(ntr)} (fold0 {int(err[f0].sum())}/{int(f0.sum())}) ; "
          f"acc des erreurs {np.round(atr[err], 3).tolist()} full={full[err].tolist()}")
    # seuil optimal (intervalle) sur folds 1-4 seulement, et erreurs fold 0 avec ce seuil
    cand = np.round(np.arange(0.5, 1.0, 0.0025), 4)
    e14 = np.array([((atr[~f0] >= c) != full[~f0]).sum() for c in cand]); best = cand[e14 == e14.min()]
    print(f"   seuils optimaux sur folds 1-4 : [{best.min()}, {best.max()}] ({e14.min()} err) ; "
          f"erreurs fold0 au milieu {((atr[f0] >= best.mean()) != full[f0]).sum()}")
    # max de la classe partielle, min de la classe full
    print(f"   train partial: max acc {atr[~full].max():.3f} (q99 {np.quantile(atr[~full], .99):.3f}) ; full: min acc {atr[full].min():.3f} (q01 {np.quantile(atr[full], .01):.3f})")
    print(f"   test : dans la zone [max partiel-0.01, min full+0.01] de train : "
          f"{int(((zt['acc'] > min(atr[~full].max(), atr[full].min()) - .01) & (zt['acc'] < max(atr[~full].max(), atr[full].min()) + .01)).sum())}")
    if (TASK / f"t1/out/signal/fullacc_{m}_test_val.npz").exists():
        zv = np.load(TASK / f"t1/out/signal/fullacc_{m}_test_val.npz")
        d = dict(zip([str(n) for n in zv["names"]], zv["acc"])); same = np.mean([abs(d.get(n, -9) - a) < 1e-6 for n, a in zip(nt, zt["acc"])])
        print(f"   fullacc_{m}_test_val.npz : n={len(zv['names'])} clés={zv.files} identique au cache test : {same:.3f}")

print("\n######## coût d'une erreur de régime (fold 0, sub04 : masque prédit avec le régime décodé vs inverse)")
SUB = json.loads((TASK / "t1/out/mgr/submissions/sub04.json").read_text())["chosen"]
mtr = {n: m["method"] for n, m in meta.items()}
for m in bb.METHODS:
    f = SUB[m]["val_files"]; pr = bb.load_val(f, m, mtr)
    raw = [f.replace("t1/out/refine/", "") for f in f]
    ok = [miou_image(guard(p, bool(meta[n]["full_watermark"]))[0], gt[n]) for n, p in pr.items()]
    bad = [miou_image(guard(p, not bool(meta[n]["full_watermark"]))[0], gt[n]) for n, p in pr.items()]
    print(f"   {m:10s} régime vrai {np.mean(ok):.3f}  régime inversé {np.mean(bad):.3f}  coût {np.mean(ok) - np.mean(bad):.3f} par erreur")

print("\n######## 2. optimisme de sélection sur fold 0")
SRC = json.loads((TASK / "t1/mgr/sources.json").read_text())
reg_tr = {}
from regime_guard import load_regimes  # noqa
reg_tr = load_regimes("train")
B = 2000
for m in bb.METHODS:
    cols, labels = [], []
    for s in SRC[m]:
        pr = bb.load_val(s["val"], m, mtr); pr = {n: p for n, p in pr.items() if n in gt}
        names = sorted(pr)
        cols.append([miou_image(guard(pr[n], reg_tr[n])[0], gt[n]) for n in names]); labels.append(s["name"])
    # configurations de raffinage testées (exp_refine, sets 1 et 2) pour chaque source de base
    for f in sorted(glob.glob(str(TASK / f"t1/out/refine/exp_{m}_*.json"))):
        d = json.loads(Path(f).read_text()); assert d["names"] == names, f
        for c, v in d["scores"].items():
            cols.append(v); labels.append(Path(f).stem[4:] + ":" + c)
    S = np.array(cols).T     # [34, K]
    K = S.shape[1]; full = S.mean(0); kbest = int(np.argmax(full))
    ins, oob, gain = [], [], []
    for _ in range(B):
        ib = rng.integers(0, len(names), len(names)); ob = np.setdiff1d(np.arange(len(names)), ib)
        k = int(np.argmax(S[ib].mean(0)))
        ins.append(S[ib, k].mean()); oob.append(S[ob, k].mean()); gain.append(S[ib, k].mean() - S[ob, k].mean())
    # rangs : la meilleure config parmi K, et le classement si on ne garde que les sources du build_best
    ns = len(SRC[m])
    print(f"{m:10s} K={K} candidats (dont {ns} sources) ; meilleur sur fold0 = {labels[kbest]} {full[kbest]:.4f} ; "
          f"optimisme bootstrap (in-bag - OOB) = {np.mean(gain):+.4f}  [OOB moyen {np.mean(oob):.4f}]")
    Ss = S[:, :ns]; g2 = []
    for _ in range(B):
        ib = rng.integers(0, len(names), len(names)); ob = np.setdiff1d(np.arange(len(names)), ib)
        k = int(np.argmax(Ss[ib].mean(0))); g2.append(Ss[ib, k].mean() - Ss[ob, k].mean())
    print(f"           sources seules (build_best) : optimisme {np.mean(g2):+.4f} ; scores {dict(zip(labels[:ns], np.round(full[:ns], 4)))}")
    sd = S[:, kbest].std(ddof=1); print(f"           écart-type par image du choix {sd:.3f} -> erreur type sur 34 images {sd / np.sqrt(34):.3f}")

print("\n######## 2b. bruit local -> public (sub01..sub04, configurations fold 0)")
def per_image(chosen):
    out = {}
    for m in bb.METHODS:
        c = chosen[m]
        if c["source"] == "constante":
            for n in gt:
                if n in fold0 and mtr[n] == m:
                    out[n] = miou_image((np.ones if reg_tr[n] else np.zeros)((128, 128)), gt[n])
            continue
        pr = bb.load_val(c["val_files"], m, mtr)
        for n, p in pr.items():
            out[n] = miou_image(bb.predict(p, reg_tr[n], c["params"], None, n), gt[n])
    return out
P = {}
for tag in ("sub01_stega_v0", "sub02_candidate", "sub03", "sub04"):
    f = TASK / f"t1/out/mgr/submissions/{tag}.json"
    if f.exists():
        P[tag] = per_image(json.loads(f.read_text())["chosen"])
if "sub01_stega_v0" not in P:   # sub01 : stega v0 + constante
    P["sub01_stega_v0"] = per_image({"stegastamp": dict(source="v0", val_files=["t1/out/train/v0/fold0_probs128.npz"], params=dict(bb.DEFAULT)),
                                     "trustmark": dict(source="constante"), "pixelseal": dict(source="constante")})
pub = {"sub01_stega_v0": 0.5528, "sub02_candidate": 0.62, "sub03": 0.6326}
NPUB = 306
for tag in ("sub01_stega_v0", "sub02_candidate", "sub03", "sub04"):
    v = np.array(list(P[tag].values())); sd = v.std(ddof=1)
    se = np.sqrt(sd ** 2 / len(v) + sd ** 2 / NPUB * (1 - NPUB / 1020))
    s = f"{tag:16s} local {v.mean():.4f} (n={len(v)}) sd/image {sd:.3f}  ET(local - public) ~ {se:.4f}"
    if tag in pub:
        s += f"  public {pub[tag]:.4f}  écart {pub[tag] - v.mean():+.4f} = {(pub[tag] - v.mean()) / se:+.2f} ET"
    print(s)
tags = ["sub01_stega_v0", "sub02_candidate", "sub03", "sub04"]
for a_, b_ in zip(tags, tags[1:]):
    ns = sorted(P[a_]); d = np.array([P[b_][n] - P[a_][n] for n in ns]); sd = d.std(ddof=1)
    se = np.sqrt(sd ** 2 / len(d) + sd ** 2 / NPUB)
    s = f"{a_} -> {b_}: gain local {d.mean():+.4f} sd(diff) {sd:.3f} ET(gain local - gain public) ~ {se:.4f}"
    if a_ in pub and b_ in pub:
        s += f" ; gain public {pub[b_] - pub[a_]:+.4f} -> écart {(pub[b_] - pub[a_] - d.mean()) / se:+.2f} ET"
    print(s)

print("\n######## 3. fuites : cname des chunks synthétiques, folds des bibliothèques")
tnames = set(tmeth)
for m in ("stegastamp", "trustmark", "pixelseal"):
    for kind in ("full", "rect"):
        fs = sorted(d for d in glob.glob(str(TASK / f"t1/out/synth/v0/{m}/{kind}/*")) if Path(d).is_dir())
        cn, nocn = set(), 0
        for f in fs:
            if (Path(f) / "cname.npy").exists():
                cn |= set(str(x) for x in np.load(Path(f) / "cname.npy"))
            else:
                nocn += 1
        print(f"   synth/v0/{m}/{kind}: {len(fs)} chunks, sans cname {nocn}, covers {len(cn)} ; dans fold0 {len(cn & fold0)} ; "
              f"dans test {len(cn & tnames)} ; folds {sorted({i for i, f in enumerate(folds) for n in cn if n in f})}")

import torch
for ck in ("t1/out/train/v0/stegastamp/best.pt", "t1/out/dense/login/tm/best.pt", "t1/out/dense/login/ps/best.pt"):
    a = torch.load(TASK / ck, map_location="cpu", weights_only=False).get("args", {})
    print("  ", ck, {k: a.get(k) for k in ("synth", "val_fold", "p_real", "mix", "it", "transplant_source", "synth_val")}, "it", torch.load(TASK / ck, map_location="cpu", weights_only=False).get("it"))
for lg in sorted(glob.glob(str(TASK / "t1/out/train/v0/logs/gen_*.log")))[:3] + sorted(glob.glob(str(TASK / "t1/out/train/v0/stegastamp/*.log"))):
    print("  ", Path(lg).name, [l.strip()[:200] for l in open(lg) if "fold" in l.lower() or "lib" in l.lower() or "leak" in l.lower()][:4])
