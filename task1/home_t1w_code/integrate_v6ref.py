"""[copie agent integhome : /p/scratch n'est plus inscriptible -> TOUTES les écritures vont sous --root
 (défaut $T1W ou ~/t1w) : <root>/integ/{ens,chain,sgtta,sources_integ.json,integ_report.json,*.log},
 <root>/submissions/sub12_integ.pt(+.json), <root>/RESULTS.tsv. Les entrées sont LUES dans $TASK (défaut ci-dessous).
 Lancer depuis n'importe où : CUDA_VISIBLE_DEVICES= python ~/t1w/code/integrate.py]
Intègre les sorties GPU de Jacques (t1/out/jacques/<tache>/DONE) dans la chaîne v6, SANS AUCUN RÉGLAGE.
Depuis TASK, CPU seulement (≈ 5 min) :
    CUDA_VISIBLE_DEVICES= python t1/integ/integrate.py --sub t1/out/mgr/submissions/sub12_integ.pt
Étapes :
 1. détecte les tâches Jacques terminées (DONE) et complètes (toutes les images test de la méthode) ;
 2. reconstruit les ensembles test par MOYENNE SIMPLE de tous les membres disponibles -> <out>/ens/ (mgr/ens n'est jamais modifié)
    - trustmark : tiles2 + devel folds 1-4 + tiles_trustmark_s*
    - pixelseal : tiles3 + devel folds 1-3 + tiles_pixelseal_s* ; ps_oof/foldK REMPLACE devel foldK (même entraînement,
      poursuivi de 7 à 14 min de phase b : ce n'est pas un nouveau membre), ps_oof/fold4 s'ajoute
    - stegastamp, régime minoritaire ({split}_probs128) : stega/best + sg_all_s*
      régime majoritaire ({split}_tta_probs128) : stegatta (TTA 4 retournements de stega/best) + sg_all_s* (leur TTA,
      calculée ici sur CPU depuis final.pt par t1/stegatta/flip_tta.py -> <out>/sgtta/, ≈ 3 min par modèle, gardée en
      cache ; à défaut la carte simple) — même logique que chain_stta
    - fold0 : copies exactes des cartes v6 (les modèles 5 folds n'ont pas de carte fold 0)
 3. lance t1/guard2/chain_integ.py (copie de chain_stta.py lisant ces ensembles) -> <out>/chain/
 4. build_best.py --min-gain 1 --boxdir t1/out/mgr/nobox -> --sub ; compare cellule à cellule avec sub11_v6.pt et
    vérifie que le fold 0 est identique à v6 (cartes de la chaîne et scores de validation).
Rapport : <out>/integ_report.json. Sans sortie Jacques, la soumission doit être identique à sub11_v6.pt."""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np

TASK = Path(os.environ.get("TASK") or "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
CODE = Path(__file__).resolve().parent             # copies locales : chain_integ.py, build_best.py, flip_tta.py
ROOT = Path(os.environ.get("T1W", str(Path.home() / "t1w")))
O = "t1/out"
V6_CHAIN = f"{O}/stegatta2"                       # sorties de chain_stta pour v6 (sub11_v6)
V6_SUB = f"{O}/mgr/submissions/sub11_v6.pt"
METHODS = ("stegastamp", "trustmark", "pixelseal")


def load(path):
    z = np.load(TASK / path)
    return {str(n): p for n, p in zip(z["names"], z["probs"])}


def save(path, d, names):
    p = TASK / path; p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, names=np.array(names), probs=np.stack([d[n] for n in names]).astype(np.float16))


def ok_member(path, names):
    """Fichier lisible, toutes les images attendues, 128x128, valeurs finies dans [0,1]."""
    try:
        d = load(path)
    except Exception as e:  # noqa: BLE001
        return False, f"illisible ({e})"
    miss = [n for n in names if n not in d]
    if miss:
        return False, f"{len(miss)}/{len(names)} images manquantes"
    a = np.stack([d[n] for n in names]).astype(np.float32)
    if a.shape[1:] != (128, 128) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        return False, f"forme/valeurs invalides {a.shape} [{a.min()}, {a.max()}]"
    return True, ""


def mean_of(files, names):
    acc = np.zeros((len(names), 128, 128), np.float64)
    for f in files:
        d = load(f); acc += np.stack([d[n] for n in names]).astype(np.float32)
    return {n: (acc[i] / len(files)).astype(np.float32) for i, n in enumerate(names)}


def done_dirs(jroot, pattern):
    return sorted(p for p in (TASK / jroot).glob(pattern) if (p / "DONE").is_file())


def rel(p):
    p = Path(p)
    return str(p.relative_to(TASK)) if p.is_relative_to(TASK) else str(p)


def build_ens(a, log):
    J, E = a.jroot, a.out + "/ens"
    if (TASK / E).exists():
        shutil.rmtree(TASK / E)
    man = {"jroot": J, "members": {}, "skipped": [], "ps_oof_check": {}}
    for pat in ("tiles_*_s*", "sg_all_s*", "ps_oof/fold*"):
        for p in sorted((TASK / J).glob(pat)):
            if p.is_dir() and not (p / "DONE").is_file():
                man["skipped"].append(f"{rel(p)}: pas de DONE (en cours ou échouée)"); log(f"  . pas de DONE : {rel(p)}")

    def accept(tag, path, names):
        good, why = ok_member(path, names)
        if not good:
            man["skipped"].append(f"{tag}: {path} -> {why}"); log(f"  ! ignoré {tag} {path} : {why}")
        return good

    # ---- trustmark / pixelseal
    for m, base, folds, tiles in (("trustmark", "tiles2/trustmark", [1, 2, 3, 4], "tiles_trustmark_s*"),
                                  ("pixelseal", "tiles3/pixelseal", [1, 2, 3], "tiles_pixelseal_s*")):
        names = sorted(load(f"{O}/mgr/ens/{m}/test_probs128.npz"))
        mem = {"base": f"{O}/{base}/test_probs128.npz"}
        mem.update({f"devel_fold{k}": f"{O}/devel/{m}/fold{k}/pred/test_probs128.npz" for k in folds})
        if m == "pixelseal":
            for d in done_dirs(J, "ps_oof/fold*"):
                k = int(d.name[4:]); f = rel(d / "pred" / "test_probs128.npz")
                if accept(f"ps_oof fold{k}", f, names):
                    mem.pop(f"devel_fold{k}", None); mem[f"ps_oof_fold{k}"] = f
                    man["ps_oof_check"][f"fold{k}"] = check_oof(k, d, log)
        for d in done_dirs(J, tiles):
            f = rel(d / "test_probs128.npz")
            if accept(d.name, f, names):
                mem[d.name] = f
        save(f"{E}/{m}/test_probs128.npz", mean_of(list(mem.values()), names), names)
        shutil.copyfile(TASK / f"{O}/mgr/ens/{m}/fold0_probs128.npz", TASK / f"{E}/{m}/fold0_probs128.npz")
        man["members"][m] = mem
        log(f"{m}: {len(mem)} membres test -> {list(mem)}")

    # ---- stegastamp
    names = sorted(load(f"{O}/stega/best/test_probs128.npz"))
    mn = {"best": f"{O}/stega/best/test_probs128.npz"}
    mj = {"best_tta": f"{O}/stegatta/stegastamp/test_probs128.npz"}
    for d in done_dirs(J, "sg_all_s*"):
        f = rel(d / "test_probs128.npz")
        if accept(d.name, f, names):
            mn[d.name] = f
            t = f"{a.out}/sgtta/{d.name}/test_probs128.npz"
            if not (TASK / t).is_file() and (d / "final.pt").is_file() and not a.no_sg_tta:
                log(f"  TTA 4 retournements (CPU, ≈ 3 min) de {d.name}/final.pt -> {t}")
                r = subprocess.run([sys.executable, str(CODE / "flip_tta.py"), "--ckpt", rel(d / "final.pt"), "--split", "test",
                                    "--views", "0,1,2,3", "--threads", "8", "--out", f"{a.out}/sgtta/{d.name}"],
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1"),
                                   stdout=open(TASK / a.out / f"sgtta_{d.name}.log", "w"), stderr=subprocess.STDOUT)
                if r.returncode:
                    log(f"  ! TTA {d.name} en échec (rc={r.returncode}) -> carte simple")
            if (TASK / t).is_file() and accept(d.name + " tta", t, names):
                mj[d.name + "_tta"] = t
            else:
                mj[d.name] = f
    save(f"{E}/stegastamp/test_probs128.npz", mean_of(list(mn.values()), names), names)
    save(f"{E}/stegastamp/test_tta_probs128.npz", mean_of(list(mj.values()), sorted(load(mj["best_tta"]))),
         sorted(load(mj["best_tta"])))
    shutil.copyfile(TASK / f"{O}/stega/best/fold0_probs128.npz", TASK / f"{E}/stegastamp/fold0_probs128.npz")
    shutil.copyfile(TASK / f"{O}/stegatta/stegastamp/fold0_probs128.npz", TASK / f"{E}/stegastamp/fold0_tta_probs128.npz")
    man["members"]["stegastamp_minority"] = mn; man["members"]["stegastamp_majority"] = mj
    log(f"stegastamp: minoritaire {list(mn)} ; majoritaire {list(mj)}")

    # ---- écart à l'ancien ensemble (information)
    man["mean_abs_change_vs_v6_maps"] = {}
    for m, old in (("trustmark", f"{O}/mgr/ens/trustmark/test_probs128.npz"),
                   ("pixelseal", f"{O}/mgr/ens/pixelseal/test_probs128.npz"),
                   ("stegastamp", f"{O}/stega/best/test_probs128.npz"),
                   ("stegastamp_tta", f"{O}/stegatta/stegastamp/test_probs128.npz")):
        new = load(f"{E}/{m.replace('_tta', '')}/test{'_tta' if m.endswith('_tta') else ''}_probs128.npz"); od = load(old)
        man["mean_abs_change_vs_v6_maps"][m] = float(np.mean([np.abs(new[n].astype(np.float32) - od[n].astype(np.float32)).mean()
                                                            for n in od]))
    (TASK / E / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def check_oof(k, d, log):
    """Carte hors-fold ps_oof/foldK face à l'ancienne devel foldK (binarisation 0,5 + garde-fou de régime)."""
    sys.path[:0] = [str(TASK / "t1" / x) for x in ("train", "synth", "common")] + [str(TASK)]
    from regime_guard import guard, load_regimes  # noqa: E402
    from wmloc.utils import miou_image  # noqa: E402
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    reg = load_regimes("train")
    new_f = d / "pred" / f"fold{k}_probs128.npz"; old_f = TASK / f"{O}/devel/pixelseal/fold{k}/pred/fold{k}_probs128.npz"
    r = {}
    for tag, f in (("new", new_f), ("old_devel", old_f)):
        if f.is_file():
            P = load(rel(f))
            s = np.array([miou_image(guard(P[n].astype(np.float32), reg[n])[0].astype(np.uint8), gt[n]) for n in sorted(P)])
            r[tag] = {"n": len(s), "score": float(s.mean()), "_per": s}
    if "new" in r and "old_devel" in r and r["new"]["n"] == r["old_devel"]["n"]:
        dd = r["new"]["_per"] - r["old_devel"]["_per"]
        bm = dd[np.random.default_rng(0).integers(0, len(dd), (10000, len(dd)))].mean(1)
        r["delta_new_minus_old"] = [float(dd.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    for v in r.values():
        if isinstance(v, dict):
            v.pop("_per", None)
    log(f"  ps_oof fold{k} hors-fold : {json.dumps(r)}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jroot", default=f"{O}/jacques", help="dossier des sorties Jacques (relatif à TASK)")
    ap.add_argument("--root", default=str(ROOT), help="racine de TOUTES les écritures (défaut $T1W ou ~/t1w)")
    ap.add_argument("--out", default=None, help="ens/, chain/, sources_integ.json, integ_report.json (défaut <root>/integ)")
    ap.add_argument("--sub", default=None, help="soumission (défaut <root>/submissions/sub12_integ.pt)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-log", action="store_true", help="transmis à build_best (pas de ligne dans t1/RESULTS.tsv)")
    ap.add_argument("--no-sg-tta", action="store_true", help="ne calcule pas la TTA des sg_all_s* (carte simple au majoritaire)")
    a = ap.parse_args()
    a.root = Path(a.root).expanduser().resolve()
    a.out = str(Path(a.out).expanduser().resolve()) if a.out else str(a.root / "integ")
    a.sub = str(Path(a.sub).expanduser().resolve()) if a.sub else str(a.root / "submissions" / "sub12_integ.pt")
    for w in (a.out, a.sub):   # garde-fou : rien n'est écrit sous TASK
        assert not Path(w).is_relative_to(TASK), f"écriture interdite sous TASK : {w}"
    os.chdir(TASK)
    (TASK / a.out).mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    log(f"== 1-2. ensembles ({a.jroot} -> {a.out}/ens)")
    man = build_ens(a, log)

    log(f"== 3. chaîne -> {a.out}/chain")
    subprocess.run([sys.executable, str(CODE / "chain_integ.py"), "--maps", "new", "--rule", "zone", "--ens", f"{a.out}/ens",
                    "--out", f"{a.out}/chain", "--workers", str(a.workers)], check=True, env=env,
                   stdout=open(TASK / a.out / "chain.log", "w"), stderr=subprocess.STDOUT)

    log("== vérification fold 0 (doit être identique à v6)")
    f0 = {}
    for m in METHODS:
        new, old = load(f"{a.out}/chain/{m}/fold0_probs128.npz"), load(f"{V6_CHAIN}/{m}/fold0_probs128.npz")
        f0[m] = bool(sorted(new) == sorted(old) and all(np.array_equal(new[n], old[n]) for n in old))
        log(f"  {m}: fold0 identique = {f0[m]}")

    log(f"== 4. build_best -> {a.sub}")
    src = {"_doc": "integ = v6 + sorties Jacques (moyenne simple), ~/t1w/code/integrate.py (copie integhome)"}
    for m in METHODS:
        src[m] = [{"name": f"integ_{m}", "val": [f"{a.out}/chain/{m}/fold0_probs128.npz"],
                   "test": [f"{a.out}/chain/{m}/test_probs128.npz"]}]
    sj = TASK / a.out / "sources_integ.json"; sj.write_text(json.dumps(src, indent=1))
    cmd = [sys.executable, str(CODE / "build_best.py"), "--sources", str(sj), "--log", str(a.root / "RESULTS.tsv"), "--boxdir", f"{O}/mgr/nobox", "--min-gain", "1",
           "--out", a.sub] + (["--no-log"] if a.no_log else [])
    subprocess.run(cmd, check=True, env=env, stdout=open(TASK / a.out / "build.log", "w"), stderr=subprocess.STDOUT)

    import torch
    new, old = torch.load(TASK / a.sub, weights_only=True), torch.load(TASK / V6_SUB, weights_only=True)
    assert new["image_names"] == old["image_names"]
    tmeth = {x["image_name"]: x["method"] for x in json.loads((TASK / "release/test/metadata.json").read_text())}
    diff = (new["masks"] != old["masks"])
    cmp = {}
    for m in METHODS:
        idx = [i for i, n in enumerate(new["image_names"]) if tmeth[n] == m]
        dm = diff[idx]
        cmp[m] = {"images_changed": int(dm.flatten(1).any(1).sum()), "cells_changed": int(dm.sum()),
                  "cells_changed_pct": round(100 * float(dm.float().mean()), 3)}
        log(f"  vs sub11_v6 {m}: {cmp[m]}")
    jn, jo = json.loads((TASK / a.sub).with_suffix(".json").read_text()), json.loads((TASK / V6_SUB).with_suffix(".json").read_text())
    val = {m: [jn["chosen"][m]["val"], jo["chosen"][m]["val"], jn["chosen"][m]["cfg"]] for m in METHODS}
    val_same = all(abs(v[0] - v[1]) < 1e-12 and v[2] == jo["chosen"][m]["cfg"] for m, v in val.items())
    log(f"  validation fold 0 (integ, v6, réglage) : {val} ; identique = {val_same}")
    rep = {"fold0_maps_identical": f0, "fold0_val_identical": val_same, "val": val, "vs_v6": cmp,
           "total_cells_changed": int(diff.sum()), "manifest": man, "sub": a.sub, "log": lines}
    (TASK / a.out / "integ_report.json").write_text(json.dumps(rep, indent=1))
    ok = all(f0.values()) and val_same
    log(f"== FIN : fold 0 identique à v6 = {ok} ; cellules test changées = {int(diff.sum())} ; soumission {a.sub} (NON envoyée)")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
