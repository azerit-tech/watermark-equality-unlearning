"""[copie integrun2 : integrate7v7.py = integrate7.py avec base v7 : chaîne chain7v7.py + --tmboxes tmedge/tm_p2,
 contrôle fold 0 face à v7 (TM = tmedge/chain_p2, PS/SG = stegatta2), comparaison avec sub14_v7 en plus]
integrate7 : intègre les modèles TrustMark/PixelSeal entraînés sur les folds 1-4 SEULEMENT (donc mesurables au fold 0)
livrés par Jacques (vague 2 tâche C : t1/out/jacques2/*holes*_{trustmark,pixelseal|tm,ps}/ ; vague 3 :
t1/out/jacques3/informed_{trustmark,pixelseal}/ ; chacun avec fold0_probs128.npz, test_probs128.npz, eval_fold0.json, DONE).
SANS AUCUN RÉGLAGE. CPU seulement. Toutes les écritures sous --root (défaut ~/t1w/integ7) ; entrées LUES dans $TASK.

    cd $TASK && source .venv/bin/activate && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 \\
        python ~/t1w/integ7/code/integrate7.py > ~/t1w/integ7/run.log 2>&1

Étapes
 0. base = v6, ou v6 + membres 5 folds déjà intégrés par ~/t1w/code/integrate.py si <integ-ens>/manifest.json existe
    (défaut ~/t1w/integ/ens). Fold 0 de la base = cartes v6 (mgr/ens/{tm,ps}/fold0) dans les deux cas.
    Chaîne de la base : réutilisée (t1/out/stegatta2 si base = v6 ; <integ-ens>/../chain si DONE) sinon relancée.
 1. pour chaque nouveau modèle complet (DONE + 34 images fold 0 + 340 test, valeurs dans [0,1]) : variante de l'ensemble
    {ens}/<methode>/ (carte MAPS_NEW de chain_stta, partie « ens » ; TM garde 0,7 ens + 0,3 tmdense) où le modèle est
    AJOUTÉ par moyenne simple : fold 0 = moyenne(carte v6, nouveau) ; test = moyenne(membres de la base + nouveau).
 2. chaîne v6 complète (chain7.py = chain_integ.py + --methods/--splits) sur le fold 0 de chaque variante.
 3. comparaison appariée au fold 0 de la base, par méthode x type (bootstrap 10 000). Règle d'adoption, fixée a priori :
    un nouveau membre n'est gardé que si le gain de SA méthode au fold 0 est >= +0,01 ET la borne basse de l'IC95 > 0.
 4. ensemble final = base + membres adoptés (même moyenne simple) ; chaîne fold 0 + test pour les méthodes changées ;
    build_best.py (--min-gain 1 --boxdir t1/out/mgr/nobox, comme integrate.py) -> <root>/sub13_integ7.pt (NON envoyée) ;
    comparaison cellule à cellule avec sub11_v6 (et sub12_integ si présente).
Rapport : <root>/integ7_report.json ; résumé imprimé. Code de sortie 0 si tout s'est déroulé (adoption ou non).
NB : au fold 0, le nouveau modèle pèse 1/2 de la partie « ens » (une seule carte v6 au fold 0), au test 1/(N+1)
(N membres 5 folds) : la mesure fold 0 surestime l'effet test du membre ajouté, dans un sens comme dans l'autre."""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np

TASK = Path(os.environ.get("TASK") or "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
CODE = Path(__file__).resolve().parent
O = "t1/out"
V6_CHAIN = TASK / f"{O}/stegatta2"
V6_SUB = TASK / f"{O}/mgr/submissions/sub11_v6.pt"
V7_SUB = Path.home() / "t1w/v7/sub14_v7.pt"
TMBOXES = f"{O}/tmedge/tm_p2/trustmark_{{s}}_boxes.json"
REF_CHAIN = {"stegastamp": V6_CHAIN, "trustmark": TASK / f"{O}/tmedge/chain_p2", "pixelseal": V6_CHAIN}
METHODS = ("stegastamp", "trustmark", "pixelseal")
NEWM = ("trustmark", "pixelseal")
ABBR = {"trustmark": ("trustmark", "tm"), "pixelseal": ("pixelseal", "ps")}
TYPES = ("hole", "transplant", "outpainting")
V6_MEMBERS = {   # = mgr/ens/<m>/test_probs128.npz (vérifié par integrate.py : sub12 sans Jacques == sub11_v6)
    "trustmark": [f"{O}/tiles2/trustmark/test_probs128.npz"] + [f"{O}/devel/trustmark/fold{k}/pred/test_probs128.npz" for k in (1, 2, 3, 4)],
    "pixelseal": [f"{O}/tiles3/pixelseal/test_probs128.npz"] + [f"{O}/devel/pixelseal/fold{k}/pred/test_probs128.npz" for k in (1, 2, 3)],
}


def P(p):
    return TASK / p   # p absolu -> inchangé


def load(path):
    z = np.load(P(path))
    return {str(n): p for n, p in zip(z["names"], z["probs"])}


def save(path, d, names):
    p = P(path); p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, names=np.array(names), probs=np.stack([d[n] for n in names]).astype(np.float16))


def mean_of(files, names):
    acc = np.zeros((len(names), 128, 128), np.float64)
    for f in files:
        d = load(f); acc += np.stack([d[n] for n in names]).astype(np.float32)
    return {n: (acc[i] / len(files)).astype(np.float32) for i, n in enumerate(names)}


def check(path, names):
    try:
        d = load(path)
    except Exception as e:  # noqa: BLE001
        return f"illisible ({e})"
    miss = [n for n in names if n not in d]
    if miss:
        return f"{len(miss)}/{len(names)} images manquantes"
    a = np.stack([d[n] for n in names]).astype(np.float32)
    if a.shape[1:] != (128, 128) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        return f"forme/valeurs invalides {a.shape}"
    return ""


def same_maps(f1, f2):
    a, b = load(f1), load(f2)
    return sorted(a) == sorted(b) and all(np.array_equal(a[n], b[n]) for n in a)


def boot(d, B=10000, seed=0):
    bm = d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "t1w" / "integ7"))
    ap.add_argument("--j2", default=f"{O}/jacques2", help="sorties vague 2 (relatif à TASK ou absolu)")
    ap.add_argument("--j3", default=f"{O}/jacques3", help="sorties vague 3")
    ap.add_argument("--integ-ens", default=str(Path.home() / "t1w" / "integ" / "ens"),
                    help="ensemble 5 folds de integrate.py (utilisé si manifest.json existe)")
    ap.add_argument("--sub", default=None, help="défaut <root>/sub13_integ7.pt")
    ap.add_argument("--min-gain", type=float, default=0.01, help="règle d'adoption (NE PAS CHANGER hors test)")
    ap.add_argument("--test-adopt-all", action="store_true", help="TEST SEULEMENT : adopte tout candidat valide")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    sub = Path(a.sub).expanduser().resolve() if a.sub else root / "sub13_integ7.pt"
    W = root / "work"
    for w in (root, sub):
        assert not w.is_relative_to(TASK), f"écriture interdite sous TASK : {w}"
    os.chdir(TASK)
    lines = []

    def log(s=""):
        print(s, flush=True); lines.append(s)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

    def chain(ens, out, methods, splits):
        log(f"  chaîne {','.join(methods)} [{splits}] -> {out}")
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        subprocess.run([sys.executable, str(CODE / "chain7v7.py"), "--tmboxes", TMBOXES, "--maps", "new", "--rule", "zone", "--ens", str(ens),
                        "--out", str(out), "--workers", str(a.workers), "--methods", ",".join(methods), "--splits", splits],
                       check=True, env=env, stdout=open(out / "chain.log", "w"), stderr=subprocess.STDOUT)

    sys.path[:0] = [str(TASK / "t1" / x) for x in ("train", "synth", "common")] + [str(TASK)]
    from wmloc.utils import miou_image  # noqa: E402
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    if isinstance(tidx, list):
        tidx = {e["image_name"]: e for e in tidx}

    # ------------------------------------------------------------------ 0. base
    ie = Path(a.integ_ens).expanduser()
    man = json.loads((ie / "manifest.json").read_text()) if (ie / "manifest.json").is_file() else None
    base_members = {m: list(man["members"][m].values()) for m in NEWM} if man else dict(V6_MEMBERS)
    log(f"== 0. base = {'v6 + membres 5 folds de ' + str(ie) if man else 'v6'}")
    for m in NEWM:
        log(f"  {m}: {len(base_members[m])} membres test")
    f0names = {m: sorted(load(f"{O}/mgr/ens/{m}/fold0_probs128.npz")) for m in NEWM}
    tnames = {m: sorted(load(f"{O}/mgr/ens/{m}/test_probs128.npz")) for m in NEWM}
    sg_src = ({"fold0": f"{O}/stega/best/fold0_probs128.npz", "test": str(ie / "stegastamp/test_probs128.npz"),
               "fold0_tta": f"{O}/stegatta/stegastamp/fold0_probs128.npz", "test_tta": str(ie / "stegastamp/test_tta_probs128.npz")}
              if man else
              {"fold0": f"{O}/stega/best/fold0_probs128.npz", "test": f"{O}/stega/best/test_probs128.npz",
               "fold0_tta": f"{O}/stegatta/stegastamp/fold0_probs128.npz", "test_tta": f"{O}/stegatta/stegastamp/test_probs128.npz"})

    def make_ens(d, extra):
        """extra = {methode: [(fold0_npz, test_npz), ...]} ajoutés par moyenne simple."""
        if d.exists():
            shutil.rmtree(d)
        for k, f in sg_src.items():
            sp, tta = k.split("_")[0], "_tta" if k.endswith("_tta") else ""
            (d / "stegastamp").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(P(f), d / "stegastamp" / f"{sp}{tta}_probs128.npz")
        for m in NEWM:
            ex = extra.get(m, [])
            v6f0 = f"{O}/mgr/ens/{m}/fold0_probs128.npz"
            if ex:
                save(d / m / "fold0_probs128.npz", mean_of([v6f0] + [e[0] for e in ex], f0names[m]), f0names[m])
            else:
                (d / m).mkdir(parents=True, exist_ok=True); shutil.copyfile(P(v6f0), d / m / "fold0_probs128.npz")
            save(d / m / "test_probs128.npz", mean_of(base_members[m] + [e[1] for e in ex], tnames[m]), tnames[m])

    base_ens = W / "ens_base"
    make_ens(base_ens, {})
    ic = ie.parent / "chain"
    if not man:
        base_chain = W / "chain_base"; chain(base_ens, base_chain, METHODS, "fold0,test")
    elif (ic / "DONE").is_file() and all((ic / m / f"{s}_probs128.npz").is_file() for m in METHODS for s in ("fold0", "test")):
        base_chain = ic; log(f"  chaîne de base réutilisée : {base_chain} (integrate.py)")
    else:
        base_chain = W / "chain_base"; chain(base_ens, base_chain, METHODS, "fold0,test")
    f0_id = {m: same_maps(base_chain / m / "fold0_probs128.npz", REF_CHAIN[m] / m / "fold0_probs128.npz") for m in METHODS}
    log(f"  fold 0 de la base identique à v7 : {f0_id}")
    assert all(f0_id.values()), "le fold 0 de la base doit être celui de v7"

    # ------------------------------------------------------------------ 1. candidats
    log("== 1. nouveaux modèles (folds 1-4)")
    cands, skipped = {m: [] for m in NEWM}, []
    for kind, jr, pat in (("holes", a.j2, "*holes*"), ("informed", a.j3, "informed*")):
        for d in sorted(P(jr).glob(pat)) if P(jr).is_dir() else []:
            if not d.is_dir() or "smoke" in d.name:
                continue
            m = next((mm for mm in NEWM if d.name.lower().endswith(tuple("_" + x for x in ABBR[mm]))), None)
            if m is None:
                continue
            tag = f"{kind}:{d.name}"
            if not (d / "DONE").is_file():
                skipped.append(f"{tag}: pas de DONE"); log(f"  . {tag} : pas de DONE (ignoré)"); continue
            f0, te = d / "fold0_probs128.npz", d / "test_probs128.npz"
            why = check(f0, f0names[m]) or check(te, tnames[m])
            if why:
                skipped.append(f"{tag}: {why}"); log(f"  ! {tag} ignoré : {why}"); continue
            ev = None
            if (d / "eval_fold0.json").is_file():
                try:
                    ev = json.loads((d / "eval_fold0.json").read_text())
                except Exception:  # noqa: BLE001
                    ev = "illisible"
            cands[m].append({"tag": tag, "dir": str(d), "fold0": str(f0), "test": str(te), "eval_fold0": ev})
            log(f"  + {m} {tag}")

    # ------------------------------------------------------------------ 2-3. variantes au fold 0
    def per_image(chdir, m):
        pr = load(chdir / m / "fold0_probs128.npz")
        names = sorted(pr)
        return names, np.array([miou_image(pr[n].astype(np.uint8), gt[n]) for n in names]), \
            np.array([tidx[n]["edit_type"] for n in names])

    base_per = {m: per_image(base_chain, m) for m in METHODS}
    res = {m: [] for m in NEWM}
    nvar = max([len(v) for v in cands.values()] + [0])
    log(f"== 2-3. {nvar} passe(s) de chaîne fold 0 ; règle : gain méthode >= {a.min_gain:+.3f} et IC95 bas > 0")
    for i in range(nvar):
        ex = {m: [(cands[m][i]["fold0"], cands[m][i]["test"])] for m in NEWM if len(cands[m]) > i}
        d = W / f"var{i}"
        make_ens(d / "ens", ex)
        chain(d / "ens", d / "chain", list(ex), "fold0")
        for m in ex:
            c = cands[m][i]
            nb, sb, et = base_per[m]; nn, sn, _ = per_image(d / "chain", m)
            assert nb == nn
            dd = sn - sb
            r = {"tag": c["tag"], "base": float(sb.mean()), "new": float(sn.mean()), "delta_boot95": boot(dd),
                 "by_type": {t: {"n": int((et == t).sum()), "base": float(sb[et == t].mean()), "new": float(sn[et == t].mean()),
                                 "delta_boot95": boot(dd[et == t])} for t in TYPES},
                 "images_changed": int((dd != 0).sum()), "eval_fold0_jacques": c["eval_fold0"]}
            r["adopt"] = bool(a.test_adopt_all or (r["delta_boot95"][0] >= a.min_gain and r["delta_boot95"][1] > 0))
            res[m].append(r)
            g = r["delta_boot95"]
            log(f"  {m:10s} {c['tag']:32s} base {r['base']:.4f} -> {r['new']:.4f}  delta {g[0]:+.4f} IC95 [{g[1]:+.4f}, {g[2]:+.4f}]"
                f"  ({r['images_changed']} images changées)  => {'ADOPTÉ' if r['adopt'] else 'rejeté'}")
            for t in TYPES:
                b = r["by_type"][t]
                log(f"      {t:11s} n={b['n']:2d}  {b['base']:.4f} -> {b['new']:.4f}  delta {b['delta_boot95'][0]:+.4f} "
                    f"[{b['delta_boot95'][1]:+.4f}, {b['delta_boot95'][2]:+.4f}]")

    # ------------------------------------------------------------------ 4. ensemble final + soumission
    adopted = {m: [cands[m][i] for i, r in enumerate(res[m]) if r["adopt"]] for m in NEWM}
    changed = [m for m in NEWM if adopted[m]]
    log(f"== 4. adoptés : { {m: [c['tag'] for c in adopted[m]] for m in NEWM} }")
    final_chain = {m: base_chain for m in METHODS}
    final_score = {}
    if changed:
        fd = W / "final"
        make_ens(fd / "ens", {m: [(c["fold0"], c["test"]) for c in adopted[m]] for m in changed})
        chain(fd / "ens", fd / "chain", changed, "fold0,test")
        for m in changed:
            final_chain[m] = fd / "chain"
            nb, sb, _ = base_per[m]; _, sn, _ = per_image(fd / "chain", m)
            final_score[m] = {"base": float(sb.mean()), "final": float(sn.mean()), "delta_boot95": boot(sn - sb),
                              "note": "optimiste : membres choisis sur ce même fold 0"}
            log(f"  final {m}: {sb.mean():.4f} -> {sn.mean():.4f}  delta {boot(sn - sb)}")
    glob_b = float(np.mean([base_per[m][1].mean() for m in METHODS]))
    glob_f = float(np.mean([final_score[m]["final"] if m in final_score else base_per[m][1].mean() for m in METHODS]))
    log(f"  fold 0 global (moyenne des 3 méthodes) : base {glob_b:.4f} -> final {glob_f:.4f}")

    src = {"_doc": "integ7 = base + modèles folds 1-4 adoptés, ~/t1w/integ7/code/integrate7.py"}
    for m in METHODS:
        src[m] = [{"name": f"integ7_{m}", "val": [str(final_chain[m] / m / "fold0_probs128.npz")],
                   "test": [str(final_chain[m] / m / "test_probs128.npz")]}]
    sj = root / "sources_integ7.json"; sj.write_text(json.dumps(src, indent=1))
    sub.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(CODE / "build_best.py"), "--sources", str(sj), "--log", str(root / "RESULTS.tsv"),
                    "--boxdir", f"{O}/mgr/nobox", "--min-gain", "1", "--out", str(sub)],
                   check=True, env=env, stdout=open(root / "build.log", "w"), stderr=subprocess.STDOUT)

    import torch
    new = torch.load(sub, weights_only=True)
    tmeth = {x["image_name"]: x["method"] for x in json.loads((TASK / "release/test/metadata.json").read_text())}
    cmp = {}
    refs = [("sub14_v7", V7_SUB), ("sub11_v6", V6_SUB), ("sub12_integ", Path.home() / "t1w/submissions/sub12_integ.pt")]
    for rn, rp in refs:
        if not rp.is_file():
            continue
        old = torch.load(rp, weights_only=True)
        assert new["image_names"] == old["image_names"]
        diff = new["masks"] != old["masks"]
        cmp[rn] = {"total_cells_changed": int(diff.sum())}
        for m in METHODS:
            idx = [i for i, n in enumerate(new["image_names"]) if tmeth[n] == m]
            dm = diff[idx]
            cmp[rn][m] = {"images_changed": int(dm.flatten(1).any(1).sum()), "cells_changed": int(dm.sum()),
                          "cells_changed_pct": round(100 * float(dm.float().mean()), 3)}
            log(f"  vs {rn} {m}: {cmp[rn][m]}")
    jn = json.loads(sub.with_suffix(".json").read_text())
    rep = {"base": "v6+integ" if man else "v6", "base_members": base_members, "base_chain": str(base_chain),
           "candidates": cands, "skipped": skipped, "min_gain": a.min_gain, "results": res,
           "adopted": {m: [c["tag"] for c in adopted[m]] for m in NEWM}, "final_fold0": final_score,
           "fold0_global": {"base": glob_b, "final": glob_f},
           "build_best_val": {m: [jn["chosen"][m]["val"], jn["chosen"][m]["cfg"]] for m in METHODS},
           "compare": cmp, "sub": str(sub), "log": lines}
    (root / "integ7_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"== FIN : adoptés {rep['adopted']} ; soumission {sub} (NON envoyée) ; rapport {root / 'integ7_report.json'}")


if __name__ == "__main__":
    main()
