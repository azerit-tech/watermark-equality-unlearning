"""integrate20.py (compte mougeot1, 22/11 nuit) = copie de integrate19sg.py rebasée sur sub20 (rien d'autre ne change) :
 - rectangles par défaut = ceux de sub20 (~/t1w/sub20/config.json) : TM train tmedge2 / test ~/tmseed ; SG train ~/sgseed /
   test ~/sgseed2 ; PS train+test ~/psseed ; chaque rectangle se change seul (--{tm,sg,ps}boxes-{train,test}).
 - contrôle de la base : sorties de chaîne = celles de sub20 (~/t1w/sub20/work/chain_<m>/<m>) au fold 0 TOUJOURS, au test
   si les rectangles test de la méthode sont ceux de sub20 ; comparaison finale avec ~/t1w/sub20/sub20.pt.
 - sorties : <root>/sub20i.pt (NON envoyée), <root>/integ20_report.json ; root par défaut ~/t1w/integ20/run.
 Lancer : ~/t1w/run_integ.sh (voir son en-tête).
--- docstring integrate19sg ---
integrate19sg.py = integrate19.py (copie, ~/t1w/integ19/code) + prise en charge des modèles StegaStamp folds 1-4.

[integ19sg] Ajout, rien d'autre ne change pour TM/PS :
1b. StegaStamp (--sg, défaut t1/out/jacques5b/sg_long,t1/out/jacques7/sg_informed) : seulement les dossiers avec DONE,
   fold0_probs128.npz (34 images) et test_probs128.npz (340). Ensemble SG de base reconstruit et contrôlé :
   minoritaire = {split}_probs128 (fold 0 : stega/best ; test : stega/best + sg_all_s1),
   majoritaire = {split}_tta_probs128 (fold 0 : stegatta ; test : stegatta + TTA de sg_all_s1).
   Candidat ajouté SEUL par moyenne simple aux DEUX cartes : minoritaire += sa carte simple ; majoritaire += sa TTA
   4 retournements (flip_tta.py sur <dossier>/final.pt, CPU, cache <root>/sgtta/<tag>/, clé = taille+date du checkpoint),
   à défaut (pas de final.pt, échec, ou vue identité incohérente avec sa carte livrée) sa carte simple — même choix au
   fold 0 et au test. Chaîne fold 0 (stegastamp seul), même règle d'adoption (gain >= +0,01 et IC95 bas > 0), puis test.
   Sans candidat SG adopté, la sortie SG est celle de la base (sub18p).
Sorties : <root>/sub19sg.pt (NON envoyée), <root>/integ20_report.json.

--- docstring d'origine (integrate19) ---
integrate19.py — intégration sans réglage des modèles de Jacques vagues 5 et 6 sur la base sub18 (+ rectangles psseed).

Base (fixée) :
 - ensembles = ~/t1w/sub17/work/final/ens (ceux de sub18 : TM 6 membres integ + informed_trustmark ; PS 6 membres integ
   + holes_pixelseal + informed_pixelseal ; SG = integ/ens) ; membres reconstruits depuis ~/t1w/integ/ens/manifest.json
   + ~/t1w/sub17/integ17_report.json (adoptés) ;
 - chaîne = chain19.py (= chain17 + --psboxes), rectangles : TrustMark train tmedge2 + test v3 (~/t1w/final/...v3.json),
   StegaStamp ~/sgseed, PixelSeal ~/psseed (consigne coordinateur 23:20 ; --psboxes psbox redonne exactement sub18).
 - chaîne de base TOUJOURS relancée (fold 0 + test) ; contrôle : SG et TM identiques à sub18 (fold 0 et test),
   PS identique à sub18 si --psboxes psbox (sinon nombre d'images différentes rapporté).
1. jacques5 (folds 1-4, t1/out/jacques5/<tâche>/) : seulement les tâches avec DONE (état final ; les tranches
   intermédiaires des tâches longues sont ignorées). Chaque candidat est ajouté SEUL par moyenne simple à l'ensemble de
   sa méthode (fold 0 = moyenne(cartes fold 0 des membres, nouveau) ; test = moyenne(membres test, nouveau)), chaîne
   fold 0, gain apparié face à la base (bootstrap 10 000). Adoption si gain méthode >= +0,01 ET IC95 bas > 0 (fixé a priori).
2. jacques6 (t1/out/jacques6/*_all/, 5 folds, test seulement) : DONE -> ajouté comme membre de TEST (fold 0 inchangé),
   sans mesure possible (le fold 0 a servi à l'entraînement).
3. ensemble final = base + adoptés + membres 5-folds -> chaîne (méthodes changées) -> build_best
   (--boxdir t1/out/mgr/nobox --min-gain 1, --no-log) -> <root>/sub19.pt (NON envoyée) ; comparaison cellule à cellule
   avec sub18 (et sub18p s'il existe), par méthode.
4. rapport <root>/integ20_report.json (integ19_report.json dans integrate19). Code de sortie 0 = contrôles OK.
CPU seulement. Écritures sous --root uniquement.

Lancer :
  cd $TASK && source .venv/bin/activate && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 \
    python ~/t1w/integ19sg/code/integrate19sg.py > ~/t1w/integ19sg/run.log 2>&1; echo rc=$?
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np

TASK = Path(os.environ.get("TASK") or "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
CODE = Path(__file__).resolve().parent
HOME = Path.home()
O = "t1/out"
METHODS = ("stegastamp", "trustmark", "pixelseal")
NEWM = ("trustmark", "pixelseal")
CANDM = ("trustmark", "pixelseal", "stegastamp")   # [integ19sg] méthodes pouvant recevoir un candidat folds 1-4
SG = "stegastamp"
SGKINDS = ("", "_tta")   # "" = carte régime minoritaire, "_tta" = carte régime majoritaire (lue par chain19)
SUFFIX = {"trustmark": ("trustmark", "tm"), "pixelseal": ("pixelseal", "ps")}
TYPES = ("hole", "transplant", "outpainting")
SUB20 = HOME / "t1w/sub20/sub20.pt"   # [integ20] référence
# sorties de chaîne de sub18 (sub18/sources.json)
REF20 = {m: HOME / f"t1w/sub20/work/chain_{m}/{m}" for m in ("stegastamp", "trustmark", "pixelseal")}   # [integ20]
BOX20 = {("trustmark", "train"): f"$TASK/{O}/tmedge2/trustmark_train_boxes.json", ("trustmark", "test"): "~/tmseed/trustmark_test_boxes.json",
         ("stegastamp", "train"): "~/sgseed/stegastamp_train_boxes.json", ("stegastamp", "test"): "~/sgseed2/stegastamp_test_boxes.json",
         ("pixelseal", "train"): "~/psseed/pixelseal_train_boxes.json", ("pixelseal", "test"): "~/psseed/pixelseal_test_boxes.json"}
ABB = {"trustmark": "tm", "stegastamp": "sg", "pixelseal": "ps"}
FAILS = []


def P(p):
    p = Path(os.path.expanduser(str(p).replace("$TASK", str(TASK))))
    return p if p.is_absolute() else TASK / p


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


def method_of(name):
    toks = name.lower().split("_")[1:]   # [integ20] méthode = un jeton du nom (ex. divcov_trustmark_s11, informed_tm_all)
    return next((m for m in NEWM if any(t in SUFFIX[m] for t in toks)), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HOME / "t1w" / "integ20" / "run"))
    ap.add_argument("--sg", default=f"{O}/jacques5b/sg_long,{O}/jacques7/sg_informed",
                    help="[integ19sg] dossiers StegaStamp folds 1-4 (virgules ; '' = aucun)")
    ap.add_argument("--no-sg-tta", action="store_true", help="[integ19sg] carte simple au régime majoritaire")
    ap.add_argument("--j5", default=f"{O}/jacques5")
    ap.add_argument("--j5-glob", default="*", help="[integ20] filtre des dossiers jacques5 (ex. 'divcov_*')")
    ap.add_argument("--j5-names", default="", help="[integ20] liste (virgules) des seuls dossiers jacques5 à essayer")
    ap.add_argument("--base-extra", default="", help="[integ20] dossiers (virgules, relatifs à TASK) AJOUTÉS à la base "
                    "(fold 0 + test, moyenne simple), ex. les adoptés de C2 ; contrôle sub20 alors limité aux autres méthodes")
    ap.add_argument("--ref-sub", default=str(SUB20), help="[integ20] soumission de comparaison (défaut sub20)")
    ap.add_argument("--j6", default=f"{O}/jacques6")
    ap.add_argument("--base-ens", default=str(HOME / "t1w/sub17/work/final/ens"))
    ap.add_argument("--integ-ens", default=str(HOME / "t1w/integ/ens"))
    ap.add_argument("--integ17", default=str(HOME / "t1w/sub17/integ17_report.json"))
    for (m_, s_), v_ in BOX20.items():   # [integ20] un fichier par méthode et split, défaut = sub20
        ap.add_argument(f"--{ABB[m_]}boxes-{s_}", default=v_)
    ap.add_argument("--sub", default=None, help="défaut <root>/sub20i.pt")
    ap.add_argument("--min-gain", type=float, default=0.01, help="règle d'adoption (NE PAS CHANGER)")
    ap.add_argument("--rule", default="lumine", choices=["lumine", "member", "all"],
                    help="[integ20] lumine = gain >= --min-gain et IC95 bas > 0 ; member = « membre de moyenne » : accepté "
                         "par défaut, refusé si score SEUL au fold 0 < médiane des membres de sa méthode - 0,02, ou si "
                         "l'ensemble de la méthode perd plus de 0,003 au fold 0")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    sub = Path(a.sub).expanduser().resolve() if a.sub else root / "sub20i.pt"
    W = root / "work"
    for w in (root, sub):
        assert not w.is_relative_to(TASK), f"écriture interdite sous TASK : {w}"
    os.chdir(TASK)
    lines = []

    def log(s=""):
        print(s, flush=True); lines.append(s)

    def fail(s):
        FAILS.append(s); log(f"  ÉCHEC : {s}")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    # rectangles TrustMark : train et test dans deux dossiers différents -> copie sous <root>/boxes (motif {s})
    bdir = root / "boxes"; bdir.mkdir(exist_ok=True)
    box_default = {}
    for m_ in METHODS:   # [integ20] copie sous <root>/boxes (motif {s}) ; défaut = fichiers de sub20
        for s_ in ("train", "test"):
            f_ = getattr(a, f"{ABB[m_]}boxes_{s_}"); shutil.copyfile(P(f_), bdir / f"{m_}_{s_}_boxes.json")
            box_default[(m_, s_)] = P(f_).resolve() == P(BOX20[(m_, s_)]).resolve()
            log(f"  rectangles {m_} {s_}: {f_}{'' if box_default[(m_, s_)] else '  (DIFFÉRENT de sub20)'}")
    TMB, SGB, PSB = (str(bdir / f"{m_}_{{s}}_boxes.json") for m_ in ("trustmark", "stegastamp", "pixelseal"))

    def chain(ens, out, methods, splits):
        log(f"  chaîne {','.join(methods)} [{splits}] -> {out}")
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        subprocess.run([sys.executable, str(CODE / "chain19.py"), "--tmboxes", TMB, "--sgboxes", SGB, "--psboxes", PSB,
                        "--maps", "new", "--rule", "zone", "--ens", str(ens), "--out", str(out), "--workers", str(a.workers),
                        "--methods", ",".join(methods), "--splits", splits],
                       check=True, env=env, stdout=open(out / "chain.log", "w"), stderr=subprocess.STDOUT)

    sys.path[:0] = [str(TASK / "t1" / x) for x in ("train", "synth", "common")] + [str(TASK)]
    from wmloc.utils import miou_image  # noqa: E402
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    if isinstance(tidx, list):
        tidx = {e["image_name"]: e for e in tidx}

    # ------------------------------------------------------------------ 0. base sub20
    be, ie = Path(a.base_ens).expanduser(), Path(a.integ_ens).expanduser()
    man = json.loads((ie / "manifest.json").read_text())
    r17 = json.loads(Path(a.integ17).expanduser().read_text())
    c17 = {c["tag"]: c for m in NEWM for c in r17["candidates"][m]}
    f0_members = {m: [str(ie / m / "fold0_probs128.npz")] + [c17[t]["fold0"] for t in r17["adopted"][m]] for m in NEWM}
    te_members = {m: list(man["members"][m].values()) + [c17[t]["test"] for t in r17["adopted"][m]] for m in NEWM}
    log(f"== 0. base = {be} + chaîne sub20 (rectangles {bdir})")
    f0names = {m: sorted(load(be / m / "fold0_probs128.npz")) for m in NEWM}
    tnames = {m: sorted(load(be / m / "test_probs128.npz")) for m in NEWM}
    for m in NEWM:   # les membres reconstruits doivent redonner l'ensemble du disque
        for s, mem, nm in (("fold0", f0_members[m], f0names[m]), ("test", te_members[m], tnames[m])):
            rec = np.stack([v for _, v in sorted(mean_of(mem, nm).items())])
            disk = np.stack([v for _, v in sorted(load(be / m / f"{s}_probs128.npz").items())]).astype(np.float32)
            dmax = float(np.abs(rec - disk).max())
            log(f"  {m} {s}: {len(mem)} membres ; |moyenne reconstruite - ensemble disque| max = {dmax:.5f}")
            if dmax > 2e-3:
                fail(f"{m} {s}: membres reconstruits incohérents avec {be}")
    # [integ19sg] ensemble StegaStamp de base : 2 cartes par split (minoritaire "", majoritaire "_tta")
    f0names[SG] = sorted(load(be / SG / "fold0_probs128.npz")); tnames[SG] = sorted(load(be / SG / "test_probs128.npz"))
    sg_members = {("fold0", ""): [str(ie / SG / "fold0_probs128.npz")],
                  ("fold0", "_tta"): [str(ie / SG / "fold0_tta_probs128.npz")],
                  ("test", ""): list(man["members"]["stegastamp_minority"].values()),
                  ("test", "_tta"): list(man["members"]["stegastamp_majority"].values())}
    sgnames = {}
    for (s, k), mem in sg_members.items():
        sgnames[(s, k)] = nm = sorted(load(be / SG / f"{s}{k}_probs128.npz"))
        rec = np.stack([v for _, v in sorted(mean_of(mem, nm).items())])
        disk = np.stack([v for _, v in sorted(load(be / SG / f"{s}{k}_probs128.npz").items())]).astype(np.float32)
        dmax = float(np.abs(rec - disk).max())
        log(f"  {SG} {s}{k}: {len(mem)} membres ; |moyenne reconstruite - ensemble disque| max = {dmax:.5f}")
        if dmax > 2e-3:
            fail(f"{SG} {s}{k}: membres reconstruits incohérents avec {be}")

    # [integ20] --base-extra : base = sub20 + ces membres (fold 0 et test), ensembles reconstruits sous W/ens_base
    bx_methods = set()
    bextra = [P(x.strip()) for x in a.base_extra.split(",") if x.strip()]
    if bextra:
        ebx = W / "ens_base"
        if ebx.exists():
            shutil.rmtree(ebx)
        add = {m: [] for m in NEWM}
        for d in bextra:
            m = method_of(d.name)
            assert m in NEWM and (d / "DONE").is_file(), f"--base-extra {d} : méthode inconnue ou pas de DONE"
            add[m].append(d)
        for m in METHODS:
            (ebx / m).mkdir(parents=True)
            if m in NEWM and add[m]:
                bx_methods.add(m)
                f0_members[m] += [str(d / "fold0_probs128.npz") for d in add[m]]
                te_members[m] += [str(d / "test_probs128.npz") for d in add[m]]
                save(ebx / m / "fold0_probs128.npz", mean_of(f0_members[m], f0names[m]), f0names[m])
                save(ebx / m / "test_probs128.npz", mean_of(te_members[m], tnames[m]), tnames[m])
                log(f"  base + {m}: {[d.name for d in add[m]]} -> fold 0 {len(f0_members[m])} membres, test {len(te_members[m])}")
            else:
                for f in (be / m).glob("*_probs128.npz"):
                    shutil.copyfile(f, ebx / m / f.name)
        be = ebx
    base_chain = W / "chain_base"
    chain(be, base_chain, METHODS, "fold0,test")
    ctrl = {m: {s: ndiff(base_chain / m / f"{s}_probs128.npz", REF20[m] / f"{s}_probs128.npz") for s in ("fold0", "test")}
            for m in METHODS}
    log(f"  images différentes de sub20 (sorties de chaîne) : {ctrl}")
    for m in METHODS:   # [integ20] fold 0 : rectangles train de sub20 -> identique exigé ; test : si rectangles test de sub20
        for s, bs in (("fold0", "train"), ("test", "test")):
            if box_default[(m, bs)] and ctrl[m][s] and m not in bx_methods:
                fail(f"{m} {s}: base différente de sub20 alors que ses composants sont ceux de sub20")
    if FAILS:
        return finish(root, {"log": lines}, 2)

    # ------------------------------------------------------------------ 1. jacques5 (folds 1-4)
    log("== 1. jacques5 : modèles folds 1-4")
    cands, skipped = {m: [] for m in CANDM}, []
    j5 = P(a.j5)
    only = {x.strip() for x in a.j5_names.split(",") if x.strip()}
    for d in sorted(j5.glob(a.j5_glob)) if j5.is_dir() else []:
        if (only and d.name not in only) or any(d.resolve() == b.resolve() for b in bextra):
            continue
        if not d.is_dir() or "smoke" in d.name:
            continue
        m, tag = method_of(d.name), f"j5:{d.name}"
        if m is None:
            skipped.append(f"{tag}: méthode inconnue"); log(f"  . {tag} : méthode inconnue (ignoré)"); continue
        if not (d / "DONE").is_file():
            skipped.append(f"{tag}: pas de DONE (état non final)"); log(f"  . {tag} : pas de DONE (ignoré, état non final)"); continue
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
        log(f"  + {m} {tag}  eval_fold0 Jacques : {json.dumps(ev)[:300]}")

    # ------------------------------------------------------------------ 1b. [integ19sg] StegaStamp folds 1-4
    log("== 1b. StegaStamp : modèles folds 1-4 (" + (a.sg or "aucun") + ")")

    def sg_tta(d, tag):
        """TTA 4 retournements de <d>/final.pt, fold 0 et test ; (dict split->fichier, info) ou (None, raison)."""
        ck = d / "final.pt"
        if a.no_sg_tta or not ck.is_file():
            return None, "--no-sg-tta" if a.no_sg_tta else "pas de final.pt"
        st = ck.stat(); key = {"ckpt": str(ck.resolve()), "size": st.st_size, "mtime": st.st_mtime}
        out = HOME / "t1w/integ20/sgtta" / tag.replace(":", "_").replace("/", "_")   # [integ20] cache partagé entre runs
        kf = out / "ckpt.json"
        fresh = kf.is_file() and json.loads(kf.read_text()) == key
        if not fresh:
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            for s, sp in (("fold0", ["--split", "train", "--fold", "0"]), ("test", ["--split", "test"])):
                log(f"    {tag}: TTA 4 retournements {s} (CPU {a.workers} fils) depuis {ck}")
                r = subprocess.run([sys.executable, str(CODE / "flip_tta.py"), "--ckpt", str(ck), *sp, "--views", "0,1,2,3",
                                    "--threads", str(a.workers), "--out", str(out)],
                                   env=dict(env, OMP_NUM_THREADS=str(a.workers)), stdout=open(out / f"tta_{s}.log", "w"),
                                   stderr=subprocess.STDOUT)
                if r.returncode:
                    return None, f"TTA {s} en échec (rc={r.returncode}, {out}/tta_{s}.log)"
            kf.write_text(json.dumps(key))
        info = {"cache": str(out), "reused": fresh}
        for s in ("fold0", "test"):
            why = check(out / f"{s}_probs128.npz", sgnames[(s, "_tta")])
            if why:
                return None, f"TTA {s} : {why}"
            # cohérence : vue identité (CPU) contre la carte livrée par le modèle (même checkpoint attendu)
            v0, sm = load(out / f"{s}_v0_probs128.npz"), load(d / f"{s}_probs128.npz")
            nm = f0names[SG] if s == "fold0" else tnames[SG]
            mad = float(np.mean([np.abs(v0[n].astype(np.float32) - sm[n].astype(np.float32)).mean() for n in nm]))
            info[f"{s}_mad_view0_vs_map"] = mad
            if mad > 0.02:
                return None, f"vue identité {s} incohérente avec la carte livrée (écart moyen {mad:.4f} > 0,02)"
        return {"fold0": str(out / "fold0_probs128.npz"), "test": str(out / "test_probs128.npz")}, info

    for sd in [x for x in a.sg.split(",") if x.strip()]:
        d = P(sd.strip()); tag = f"sg:{d.parent.name}/{d.name}"
        if not (d / "DONE").is_file():
            skipped.append(f"{tag}: pas de DONE (absent ou état non final)"); log(f"  . {tag} : pas de DONE (ignoré)"); continue
        f0, te = d / "fold0_probs128.npz", d / "test_probs128.npz"
        why = check(f0, f0names[SG]) or check(te, tnames[SG])
        if why:
            skipped.append(f"{tag}: {why}"); log(f"  ! {tag} ignoré : {why}"); continue
        tta, info = sg_tta(d, tag)
        if tta is None:
            log(f"    {tag}: régime majoritaire = carte simple ({info})")
            tta = {"fold0": str(f0), "test": str(te)}
        else:
            log(f"    {tag}: régime majoritaire = TTA ({json.dumps(info)})")
        ev = None
        if (d / "eval_fold0.json").is_file():
            try:
                ev = json.loads((d / "eval_fold0.json").read_text())
            except Exception:  # noqa: BLE001
                ev = "illisible"
        cands[SG].append({"tag": tag, "dir": str(d), "fold0": {"": str(f0), "_tta": tta["fold0"]},
                          "test": {"": str(te), "_tta": tta["test"]}, "tta": info, "eval_fold0": ev})
        log(f"  + {SG} {tag}  eval_fold0 Jacques : {json.dumps(ev)[:300]}")

    def per_image(chdir, m):
        pr = load(chdir / m / "fold0_probs128.npz")
        names = sorted(pr)
        return names, np.array([miou_image(pr[n].astype(np.uint8), gt[n]) for n in names]), \
            np.array([tidx[n]["edit_type"] for n in names])

    def make_ens(d, f0extra, teextra):
        """ensembles TM/PS des méthodes touchées : moyenne simple membres base + extra (fold 0 et test séparément)."""
        if d.exists():
            shutil.rmtree(d)
        for m in set(f0extra) | set(teextra):
            (d / m).mkdir(parents=True)
            if m == SG:   # [integ19sg] extra = liste de {kind: fichier}, 2 cartes par split
                for s, ex in (("fold0", f0extra.get(m, [])), ("test", teextra.get(m, []))):
                    for k in SGKINDS:
                        f = f"{s}{k}_probs128.npz"
                        if ex:
                            save(d / m / f, mean_of(sg_members[(s, k)] + [e[k] for e in ex], sgnames[(s, k)]), sgnames[(s, k)])
                        else:
                            shutil.copyfile(be / m / f, d / m / f)
                continue
            fx, tx = f0extra.get(m, []), teextra.get(m, [])
            if fx:
                save(d / m / "fold0_probs128.npz", mean_of(f0_members[m] + fx, f0names[m]), f0names[m])
            else:
                shutil.copyfile(be / m / "fold0_probs128.npz", d / m / "fold0_probs128.npz")
            if tx:
                save(d / m / "test_probs128.npz", mean_of(te_members[m] + tx, tnames[m]), tnames[m])
            else:
                shutil.copyfile(be / m / "test_probs128.npz", d / m / "test_probs128.npz")

    base_per = {m: per_image(base_chain, m) for m in METHODS}
    for m in METHODS:
        log(f"  base fold 0 {m}: {base_per[m][1].mean():.4f}")
    log(f"  base fold 0 global : {np.mean([base_per[m][1].mean() for m in METHODS]):.4f}")
    res = {m: [] for m in CANDM}
    nvar = max([len(v) for v in cands.values()] + [0])
    log(f"== 2. {nvar} passe(s) de chaîne fold 0 ; règle : " + (f"gain méthode >= {a.min_gain:+.3f} et IC95 bas > 0" if a.rule == "lumine"
        else "membre de moyenne (refus si seul < médiane membres - 0,02 ou ensemble < base - 0,003)"))
    solo = {}   # [integ20] règle member : score SEUL (chaîne complète sur la carte du membre seule) au fold 0

    def solo_score(m, f0, tag):
        d = W / "solo" / f"{m}_{len(solo)}"
        if d.exists():
            shutil.rmtree(d)
        (d / "ens" / m).mkdir(parents=True)
        if m == SG:
            for k in SGKINDS:
                save(d / "ens" / m / f"fold0{k}_probs128.npz", mean_of([f0[k]], sgnames[("fold0", k)]), sgnames[("fold0", k)])
        else:
            save(d / "ens" / m / "fold0_probs128.npz", mean_of([f0], f0names[m]), f0names[m])
        chain(d / "ens", d / "chain", [m], "fold0")
        sc = float(per_image(d / "chain", m)[1].mean()); solo[tag] = sc
        log(f"    score seul fold 0 {m} {tag} : {sc:.4f}")
        return sc

    member_med = {}
    if a.rule == "member":
        for m in CANDM:
            if not cands[m]:
                continue
            mem = list(zip(sg_members[("fold0", "")], sg_members[("fold0", "_tta")])) if m == SG else f0_members[m]
            sc = [solo_score(m, ({"": x[0], "_tta": x[1]} if m == SG else x), f"membre:{x if m != SG else x[0]}") for x in mem]
            member_med[m] = float(np.median(sc))
            log(f"  {m}: médiane des scores seuls des {len(sc)} membres fold 0 = {member_med[m]:.4f}")
    for i in range(nvar):
        ex = {m: [cands[m][i]["fold0"]] for m in CANDM if len(cands[m]) > i}
        d = W / f"var{i}"
        make_ens(d / "ens", ex, {})
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
            if a.rule == "all":   # [integ20] tous adoptés (membres de moyenne sans règle), mesure seulement
                r["adopt"] = True
            elif a.rule == "lumine":
                r["adopt"] = bool(r["delta_boot95"][0] >= a.min_gain and r["delta_boot95"][1] > 0)
            else:   # [integ20] règle « membre de moyenne »
                r["solo"] = solo_score(m, c["fold0"], c["tag"]); r["member_median"] = member_med[m]
                r["adopt"] = bool(r["solo"] >= member_med[m] - 0.02 and r["delta_boot95"][0] >= -0.003)
                log(f"    {c['tag']}: seul {r['solo']:.4f} vs médiane membres {member_med[m]:.4f} (seuil -0,02) ; "
                    f"écart ensemble {r['delta_boot95'][0]:+.4f} (seuil -0,003)")
            res[m].append(r)
            g = r["delta_boot95"]
            log(f"  {m:10s} {c['tag']:28s} base {r['base']:.4f} -> {r['new']:.4f}  delta {g[0]:+.4f} IC95 [{g[1]:+.4f}, {g[2]:+.4f}]"
                f"  ({r['images_changed']} images changées)  => {'ADOPTÉ' if r['adopt'] else 'rejeté'}")
            for t in TYPES:
                b = r["by_type"][t]
                log(f"      {t:11s} n={b['n']:2d}  {b['base']:.4f} -> {b['new']:.4f}  delta {b['delta_boot95'][0]:+.4f} "
                    f"[{b['delta_boot95'][1]:+.4f}, {b['delta_boot95'][2]:+.4f}]")
        shutil.rmtree(d / "ens")   # ensembles intermédiaires inutiles ensuite

    # ------------------------------------------------------------------ 3. jacques6 (5 folds, test seulement)
    log("== 3. jacques6 : modèles 5-folds (membres de test, fold 0 inchangé)")
    j6m = {m: [] for m in NEWM}
    j6 = P(a.j6)
    for d in sorted(j6.glob("*_all")) if j6.is_dir() else []:
        if not d.is_dir():
            continue
        m, tag = method_of(d.name), f"j6:{d.name}"
        if m is None:
            skipped.append(f"{tag}: méthode inconnue"); log(f"  . {tag} : méthode inconnue (ignoré)"); continue
        if not (d / "DONE").is_file():
            skipped.append(f"{tag}: pas de DONE"); log(f"  . {tag} : pas de DONE (ignoré)"); continue
        why = check(d / "test_probs128.npz", tnames[m])
        if why:
            skipped.append(f"{tag}: {why}"); log(f"  ! {tag} ignoré : {why}"); continue
        j6m[m].append({"tag": tag, "test": str(d / "test_probs128.npz")})
        log(f"  + {m} {tag} (membre de test)")

    # ------------------------------------------------------------------ 4. ensemble final + soumission
    adopted = {m: [cands[m][i] for i, r in enumerate(res[m]) if r["adopt"]] for m in CANDM}
    j6m[SG] = []   # [integ19sg] pas de membre StegaStamp 5-folds
    f0x = {m: [c["fold0"] for c in adopted[m]] for m in CANDM if adopted[m]}
    tex = {m: [c["test"] for c in adopted[m]] + [c["test"] for c in j6m[m]] for m in CANDM if adopted[m] or j6m[m]}
    changed = [m for m in CANDM if m in tex]
    log(f"== 4. adoptés (folds 1-4) : { {m: [c['tag'] for c in adopted[m]] for m in CANDM} } ; "
        f"membres 5-folds : { {m: [c['tag'] for c in j6m[m]] for m in NEWM} }")
    final_chain = {m: base_chain for m in METHODS}
    final_score = {}
    members = {m: {"fold0": f0_members[m] + f0x.get(m, []), "test": te_members[m] + tex.get(m, [])} for m in NEWM}
    members[SG] = {f"{s}{k}": sg_members[(s, k)] + [e[k] for e in (f0x if s == "fold0" else tex).get(SG, [])]
                   for s in ("fold0", "test") for k in SGKINDS}
    if changed:
        fd = W / "final"
        make_ens(fd / "ens", f0x, tex)
        chain(fd / "ens", fd / "chain", changed, "fold0,test")
        for m in changed:
            final_chain[m] = fd / "chain"
            nb, sb, _ = base_per[m]; _, sn, _ = per_image(fd / "chain", m)
            final_score[m] = {"base": float(sb.mean()), "final": float(sn.mean()), "delta_boot95": boot(sn - sb),
                              "note": "optimiste si adoption (membres choisis sur ce fold 0)" if m in f0x else "fold 0 inchangé"}
            log(f"  final {m}: fold 0 {sb.mean():.4f} -> {sn.mean():.4f}  delta {boot(sn - sb)}")
            if m not in f0x and ndiff(fd / "chain" / m / "fold0_probs128.npz", base_chain / m / "fold0_probs128.npz"):
                fail(f"{m}: fold 0 changé alors qu'aucun membre fold 0 n'a été ajouté")
    glob_b = float(np.mean([base_per[m][1].mean() for m in METHODS]))
    glob_f = float(np.mean([final_score[m]["final"] if m in final_score else base_per[m][1].mean() for m in METHODS]))
    log(f"  fold 0 global (moyenne des 3 méthodes) : base {glob_b:.4f} -> final {glob_f:.4f}")
    # [integ20] bootstrap apparié global, stratifié par méthode (même tirage que chain : 10 000)
    dglob = {m: per_image(final_chain[m], m)[1] - base_per[m][1] for m in METHODS}
    rng = np.random.default_rng(1); bm = np.zeros(10000)
    for m in METHODS:
        bm += dglob[m][rng.integers(0, len(dglob[m]), (10000, len(dglob[m])))].mean(1) / len(METHODS)
    glob_boot = [glob_f - glob_b, float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    log(f"  fold 0 global vs sub20 : écart {glob_boot[0]:+.4f} IC95 [{glob_boot[1]:+.4f}, {glob_boot[2]:+.4f}]")
    for m in METHODS:
        g = boot(dglob[m]); log(f"  fold 0 {m} vs sub20 : {base_per[m][1].mean():.4f} -> {base_per[m][1].mean() + g[0]:.4f}  écart {g[0]:+.4f} IC95 [{g[1]:+.4f}, {g[2]:+.4f}]")

    src = {"_doc": "sub20i = sub20 (+ rectangles donnés) + jacques5 adoptés + jacques6 5-folds + StegaStamp folds 1-4 "
                   "adoptés ; ~/t1w/integ20/code/integrate20.py"}
    for m in METHODS:
        src[m] = [{"name": f"sub19sg_{m}", "val": [str(final_chain[m] / m / "fold0_probs128.npz")],
                   "test": [str(final_chain[m] / m / "test_probs128.npz")]}]
    sj = root / "sources.json"; sj.write_text(json.dumps(src, indent=1))
    subprocess.run([sys.executable, str(HOME / "t1w/code/build_best.py"), "--sources", str(sj), "--no-log",
                    "--boxdir", f"{O}/mgr/nobox", "--min-gain", "1", "--out", str(sub)],
                   check=True, env=dict(env, OMP_NUM_THREADS="8"), stdout=open(root / "build.log", "w"), stderr=subprocess.STDOUT)

    import torch
    new = torch.load(sub, weights_only=True)
    tmeth = {x["image_name"]: x["method"] for x in json.loads((TASK / "release/test/metadata.json").read_text())}
    M = new["masks"]
    if not (M.dtype == torch.bool and tuple(M.shape) == (1020, 128, 128) and sub.stat().st_size < 20e6):
        fail(f"format de {sub} invalide")
    empty = int(sum(1 for k in range(M.shape[0]) if M[k].all() or not M[k].any()))
    if empty:
        fail(f"{empty} masques vides ou pleins")
    bj = json.loads(sub.with_suffix(".json").read_text())
    rj = json.loads(SUB20.with_suffix(".json").read_text())
    for m in METHODS:
        if bj["chosen"][m]["cfg"] != rj["chosen"][m]["cfg"]:
            log(f"  ALERTE : {m}: build_best a choisi {bj['chosen'][m]['cfg']} (sub20 {rj['chosen'][m]['cfg']})")
    cmp = {}
    sys.path.insert(0, str(TASK / "t1/guard2")); from rule import THR as RTHR  # noqa: E402
    REGD = {}
    for m_ in METHODS:
        z_ = np.load(TASK / "t1/out/signal" / f"fullacc_{m_}_test.npz"); REGD.update({str(n): float(x) >= RTHR[m_] for n, x in zip(z_["names"], z_["acc"])})
    for refname, refp in (("ref", Path(a.ref_sub).expanduser()),):
        if not refp.is_file():
            log(f"  ({refname} absent : {refp})"); continue
        old = torch.load(refp, weights_only=True)
        assert new["image_names"] == old["image_names"]
        diff = M != old["masks"]
        cmp[refname] = {"file": str(refp), "total_cells_changed": int(diff.sum())}
        log(f"  vs {refname} : {cmp[refname]['total_cells_changed']} cellules changées au total")
        for m in METHODS:
            idx = [i for i, n in enumerate(new["image_names"]) if tmeth[n] == m]
            dm = diff[idx]
            ch = [new["image_names"][idx[k]] for k in np.nonzero(dm.flatten(1).any(1).numpy())[0]]
            cmp[refname][m] = {"images_changed": len(ch), "cells_changed": int(dm.sum()),
                               "cells_changed_pct": round(100 * float(dm.float().mean()), 3),
                               f"coverage_{refname}": round(float(old["masks"][idx].float().mean()), 4),
                               "coverage_sub19": round(float(M[idx].float().mean()), 4), "images": ch[:60]}
            log(f"  vs {refname} {m}: {len(ch)} images, {int(dm.sum())} cellules ({cmp[refname][m]['cells_changed_pct']} %)")
            # [integ20] contrôles de build_final : couverture, zone majoritaire vs régime décodé, bascules
            cn = M[idx].flatten(1).float().mean(1).numpy(); co = old["masks"][idx].flatten(1).float().mean(1).numpy()
            maj, regf = float((cn >= 0.5).mean()), float(np.mean([REGD[new["image_names"][i]] for i in idx]))
            flips = [new["image_names"][idx[k]] for k in range(len(idx)) if (cn[k] >= 0.5) != (co[k] >= 0.5)]
            cmp[refname][m].update(majority_share=round(maj, 4), decoded_regime=round(regf, 4), regime_flips=flips)
            log(f"     couverture {cn.mean():.4f} (sub20 {co.mean():.4f}) ; zone majoritaire {maj:.3f} (régime décodé {regf:.3f}) ; bascules {len(flips)} {flips[:5]}")
            if abs(cn.mean() - co.mean()) > 0.02:
                log(f"  ALERTE : {m}: couverture moyenne s'écarte de sub20 de plus de 0,02")
            if abs(maj - regf) > 0.03:
                log(f"  ALERTE : {m}: part de zone majoritaire loin du régime décodé")
            if flips:
                log(f"  ALERTE : {m}: {len(flips)} bascules de zone majoritaire vs sub20")
    rep = {"base": "sub20", "boxes": {f"{m_}_{s_}": getattr(a, f"{ABB[m_]}boxes_{s_}") for m_ in METHODS for s_ in ("train", "test")}, "base_ens": str(be), "base_members": members, "base_chain": str(base_chain),
           "control_vs_sub20_chain": ctrl, "candidates_j5": cands, "members_j6": j6m, "skipped": skipped,
           "min_gain": a.min_gain, "results_j5": res, "adopted_j5": {m: [c["tag"] for c in adopted[m]] for m in CANDM},
           "base_fold0": {m: float(base_per[m][1].mean()) for m in METHODS}, "final_fold0": final_score,
           "fold0_global": {"base": glob_b, "final": glob_f, "delta_boot95": glob_boot}, "rule": a.rule, "solo_fold0": solo,
           "build_best_val": {m: [bj["chosen"][m]["val"], bj["chosen"][m]["cfg"]] for m in METHODS},
           "compare": cmp, "sub": str(sub), "fails": FAILS, "log": lines}
    return finish(root, rep, 2 if FAILS else 0)


def finish(root, rep, code):
    rep["code"] = code
    print(f"== FIN code {code} ; échecs {FAILS} ; rapport {root / 'integ20_report.json'} ; soumission NON envoyée", flush=True)
    (root / "integ20_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main() or 0)
