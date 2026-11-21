"""[copie agent sub17 de ~/t1w/integ7/code/integrate7v7.py : base = sub16 au lieu de v7]
 - chaîne = chain17.py (= sub16/code/chain_integ.py + --splits), rectangles de sub16 : --tmboxes t1/out/tmedge2,
   --sgboxes ~/sgseed ; PixelSeal : psbox par défaut.
 - ensembles de base = ~/t1w/integ/ens tels quels (manifest : TM 6 membres, PS 6 membres dont ps_oof fold4) ;
   chaîne de base TOUJOURS relancée (fold0 + test).
 - contrôle : fold 0 de la base identique, carte à carte, à sub16 (SG sub16/chain, TM/PS sub12b/chain) ;
   attendu TM 0,7155, SG 0,7979, PS 0,6550, global 0,7228.
 - candidats Jacques (jacques2 holes_*, jacques3 informed_*) : ajoutés par moyenne simple (fold 0 = moyenne(carte v6,
   nouveau) ; test = moyenne(membres base + nouveau)), chaîne fold 0 de SA méthode, bootstrap apparié 10 000,
   adoption si gain méthode >= +0,01 ET IC95 bas > 0 (fixé a priori, inchangé).
 - final = base + adoptés -> build_best.py (--boxdir t1/out/mgr/nobox --min-gain 1) -> <root>/sub17.pt (NON envoyée),
   comparaison cellule à cellule avec sub16 par méthode.
CPU seulement. Écritures sous --root (défaut ~/t1w/sub17) et --log."""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np

TASK = Path(os.environ.get("TASK") or "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
CODE = Path(__file__).resolve().parent
HOME = Path.home()
O = "t1/out"
SUB16 = HOME / "t1w/sub16/sub16_sgseed.pt"
TMBOXES = f"{O}/tmedge2/trustmark_{{s}}_boxes.json"
SGBOXES = str(HOME / "sgseed/stegastamp_{s}_boxes.json")
REF_CHAIN = {"stegastamp": HOME / "t1w/sub16/chain", "trustmark": HOME / "t1w/sub12b/chain", "pixelseal": HOME / "t1w/sub12b/chain"}
METHODS = ("stegastamp", "trustmark", "pixelseal")
NEWM = ("trustmark", "pixelseal")
ABBR = {"trustmark": ("trustmark", "tm"), "pixelseal": ("pixelseal", "ps")}
TYPES = ("hole", "transplant", "outpainting")


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


def ndiff(f1, f2):
    a, b = load(f1), load(f2)
    if sorted(a) != sorted(b):
        return -1
    return sum(int(not np.array_equal(a[n], b[n])) for n in a)


def boot(d, B=10000, seed=0):
    bm = d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HOME / "t1w" / "sub17"))
    ap.add_argument("--j2", default=f"{O}/jacques2")
    ap.add_argument("--j3", default=f"{O}/jacques3")
    ap.add_argument("--integ-ens", default=str(HOME / "t1w" / "integ" / "ens"))
    ap.add_argument("--sub", default=None, help="défaut <root>/sub17.pt")
    ap.add_argument("--log", default=None, help="RESULTS.tsv de build_best (défaut <root>/RESULTS.tsv)")
    ap.add_argument("--min-gain", type=float, default=0.01, help="règle d'adoption (NE PAS CHANGER)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    sub = Path(a.sub).expanduser().resolve() if a.sub else root / "sub17.pt"
    rlog = Path(a.log).expanduser() if a.log else root / "RESULTS.tsv"
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
        subprocess.run([sys.executable, str(CODE / "chain17.py"), "--tmboxes", TMBOXES, "--sgboxes", SGBOXES, "--maps", "new",
                        "--rule", "zone", "--ens", str(ens), "--out", str(out), "--workers", str(a.workers),
                        "--methods", ",".join(methods), "--splits", splits],
                       check=True, env=env, stdout=open(out / "chain.log", "w"), stderr=subprocess.STDOUT)

    sys.path[:0] = [str(TASK / "t1" / x) for x in ("train", "synth", "common")] + [str(TASK)]
    from wmloc.utils import miou_image  # noqa: E402
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    if isinstance(tidx, list):
        tidx = {e["image_name"]: e for e in tidx}

    # ------------------------------------------------------------------ 0. base = ensembles integ à jour + chaîne sub16
    ie = Path(a.integ_ens).expanduser()
    man = json.loads((ie / "manifest.json").read_text())
    base_members = {m: list(man["members"][m].values()) for m in NEWM}
    log(f"== 0. base = {ie} (manifest) + chaîne sub16 (tmedge2, sgseed, psbox)")
    for m in NEWM:
        log(f"  {m}: {len(base_members[m])} membres test : {list(man['members'][m])}")
    f0names = {m: sorted(load(ie / m / "fold0_probs128.npz")) for m in NEWM}
    tnames = {m: sorted(load(ie / m / "test_probs128.npz")) for m in NEWM}
    for m in NEWM:   # l'ensemble test du disque doit être la moyenne des membres du manifest
        dmax = float(np.abs(np.stack([v for _, v in sorted(mean_of(base_members[m], tnames[m]).items())])
                            - np.stack([v for _, v in sorted(load(ie / m / "test_probs128.npz").items())]).astype(np.float32)).max())
        log(f"  {m}: |ens test disque - moyenne manifest| max = {dmax:.4f} (float16)")

    def make_ens(d, extra):
        """extra = {methode: [(fold0_npz, test_npz), ...]} ajoutés par moyenne simple."""
        if d.exists():
            shutil.rmtree(d)
        (d / "stegastamp").mkdir(parents=True)
        for f in ("fold0", "test", "fold0_tta", "test_tta"):
            shutil.copyfile(ie / "stegastamp" / f"{f}_probs128.npz", d / "stegastamp" / f"{f}_probs128.npz")
        for m in NEWM:
            ex = extra.get(m, [])
            (d / m).mkdir(parents=True, exist_ok=True)
            if ex:
                save(d / m / "fold0_probs128.npz", mean_of([ie / m / "fold0_probs128.npz"] + [e[0] for e in ex], f0names[m]), f0names[m])
                save(d / m / "test_probs128.npz", mean_of(base_members[m] + [e[1] for e in ex], tnames[m]), tnames[m])
            else:
                for s in ("fold0", "test"):
                    shutil.copyfile(ie / m / f"{s}_probs128.npz", d / m / f"{s}_probs128.npz")

    base_ens = W / "ens_base"
    make_ens(base_ens, {})
    base_chain = W / "chain_base"
    chain(base_ens, base_chain, METHODS, "fold0,test")
    ctrl = {m: {s: ndiff(base_chain / m / f"{s}_probs128.npz", REF_CHAIN[m] / m / f"{s}_probs128.npz") for s in ("fold0", "test")}
            for m in METHODS}
    log(f"  images différentes de sub16 (chaîne) : {ctrl}")
    assert all(ctrl[m]["fold0"] == 0 for m in METHODS), "le fold 0 de la base doit être celui de sub16"

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
            log(f"  + {m} {tag}  eval_fold0 Jacques : {json.dumps(ev)[:400]}")

    # ------------------------------------------------------------------ 2-3. variantes au fold 0
    def per_image(chdir, m):
        pr = load(chdir / m / "fold0_probs128.npz")
        names = sorted(pr)
        return names, np.array([miou_image(pr[n].astype(np.uint8), gt[n]) for n in names]), \
            np.array([tidx[n]["edit_type"] for n in names])

    base_per = {m: per_image(base_chain, m) for m in METHODS}
    for m in METHODS:
        log(f"  base fold 0 {m}: {base_per[m][1].mean():.4f}")
    log(f"  base fold 0 global : {np.mean([base_per[m][1].mean() for m in METHODS]):.4f}")
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
            r["adopt"] = bool(r["delta_boot95"][0] >= a.min_gain and r["delta_boot95"][1] > 0)
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

    src = {"_doc": "sub17 = sub16 + ps_oof fold4 (ensembles integ à jour) + modèles Jacques adoptés ; ~/t1w/sub17/code/integrate17.py"}
    for m in METHODS:
        src[m] = [{"name": f"sub17_{m}", "val": [str(final_chain[m] / m / "fold0_probs128.npz")],
                   "test": [str(final_chain[m] / m / "test_probs128.npz")]}]
    sj = root / "sources.json"; sj.write_text(json.dumps(src, indent=1))
    sub.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HOME / "t1w/code/build_best.py"), "--sources", str(sj), "--log", str(rlog),
                    "--boxdir", f"{O}/mgr/nobox", "--min-gain", "1", "--out", str(sub)],
                   check=True, env=env, stdout=open(root / "build.log", "w"), stderr=subprocess.STDOUT)

    import torch
    new = torch.load(sub, weights_only=True)
    tmeth = {x["image_name"]: x["method"] for x in json.loads((TASK / "release/test/metadata.json").read_text())}
    cmp = {}
    old = torch.load(SUB16, weights_only=True)
    assert new["image_names"] == old["image_names"]
    diff = new["masks"] != old["masks"]
    cmp["total_cells_changed"] = int(diff.sum())
    log(f"  vs sub16 : {cmp['total_cells_changed']} cellules changées au total")
    for m in METHODS:
        idx = [i for i, n in enumerate(new["image_names"]) if tmeth[n] == m]
        dm = diff[idx]
        cmp[m] = {"images_changed": int(dm.flatten(1).any(1).sum()), "cells_changed": int(dm.sum()),
                  "cells_changed_pct": round(100 * float(dm.float().mean()), 3),
                  "coverage_sub16": round(float(old["masks"][idx].float().mean()), 4),
                  "coverage_sub17": round(float(new["masks"][idx].float().mean()), 4)}
        log(f"  vs sub16 {m}: {cmp[m]}")
    jn = json.loads(sub.with_suffix(".json").read_text())
    rep = {"base": "sub16 (integ/ens à jour)", "base_members": base_members, "base_chain": str(base_chain), "control_vs_sub16": ctrl,
           "candidates": cands, "skipped": skipped, "min_gain": a.min_gain, "results": res,
           "adopted": {m: [c["tag"] for c in adopted[m]] for m in NEWM}, "final_fold0": final_score,
           "fold0_global": {"base": glob_b, "final": glob_f},
           "build_best_val": {m: [jn["chosen"][m]["val"], jn["chosen"][m]["cfg"]] for m in METHODS},
           "compare_sub16": cmp, "sub": str(sub), "log": lines}
    (root / "integ17_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"== FIN : adoptés {rep['adopted']} ; soumission {sub} (NON envoyée) ; rapport {root / 'integ17_report.json'}")


if __name__ == "__main__":
    main()
