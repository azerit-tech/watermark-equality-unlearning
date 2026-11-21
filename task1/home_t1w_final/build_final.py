"""build_final.py — soumission finale de la tâche 1 en UNE commande, à partir de composants validés (CPU seul).

Base = sub16 (~/t1w/sub16/sub16_sgseed.pt, 0,7132 public). Pour chaque méthode, config.json donne :
  ens          dossier d'ensembles (<ens>/<méthode>/{fold0,test}_probs128.npz ; StegaStamp aussi {fold0,test}_tta_probs128.npz)
  boxes_train  rectangles du train (seules les entrées du fold 0 servent)
  boxes_test   rectangles du test
  conf         seuil de confiance des rectangles (fixé a priori ; sub16 : SG 0,9, TM 0,79, PS 0,90)
Formats de rectangles acceptés : {"boxes": {nom: {"rect"|"box": [y0,y1,x0,x1], "conf": c}}} ou {nom: {...}} (psseed).

Étapes :
 1. empreinte de chaque composant, comparée à celle figée de sub16 (~/t1w/final/ref_sub16.json) ; la chaîne
    (~/t1w/sub16/code/chain_integ.py, inchangée, importée) n'est relancée que pour les méthodes dont un composant a changé
    (--force : toutes) ; sinon on reprend la sortie de chaîne de sub16 ;
 2. fold 0 : composant fold 0 inchangé -> sortie fold 0 IDENTIQUE à sub16 exigée ; changé -> nouveau score par type
    + écart apparié (bootstrap) face à sub16 ; test inchangé mais relancé -> sortie test identique exigée ;
 3. contrôles : noms/ordre = sub16, tenseur bool [1020,128,128], < 20 Mo, aucun masque vide ou plein,
    couverture et part de zone majoritaire plausibles (comparées à sub16 et au régime décodé) ;
 4. build_best (--boxdir t1/out/mgr/nobox --min-gain 1) -> sortie .pt ;
 5. comparaison cellule à cellule avec sub16, par méthode, liste des images changées.
Code de sortie : 0 = tous les contrôles durs passent ; 2 = au moins un échec (ne pas envoyer).

Lancer (CPU, ~1-3 min par méthode relancée) :
  cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python ~/t1w/final/build_final.py [--config ~/t1w/final/config.json] [--force]
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, subprocess, sys
from pathlib import Path

import numpy as np

TASK = Path(os.environ.get("TASK") or "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
HOME = Path.home()
FINAL = HOME / "t1w" / "final"
CHAIN_PY = HOME / "t1w" / "sub16" / "code" / "chain_integ.py"
BUILD_PY = HOME / "t1w" / "code" / "build_best.py"
REF_SUB = HOME / "t1w" / "sub16" / "sub16_sgseed.pt"
METHODS = ["stegastamp", "trustmark", "pixelseal"]
# composants exacts de sub16 et sorties de chaîne correspondantes (sub16/build.log, sub12b/sources.json, integ/sources_integ.json)
REF = {
    "stegastamp": dict(ens="~/t1w/integ/ens", boxes_train="~/sgseed/stegastamp_train_boxes.json",
                       boxes_test="~/sgseed/stegastamp_test_boxes.json", conf=0.9, chain="~/t1w/sub16/chain/stegastamp"),
    "trustmark": dict(ens="~/t1w/integ/ens", boxes_train="$TASK/t1/out/tmedge2/trustmark_train_boxes.json",
                      boxes_test="$TASK/t1/out/tmedge2/trustmark_test_boxes.json", conf=0.79, chain="~/t1w/sub12b/chain/trustmark"),
    "pixelseal": dict(ens="~/t1w/integ/ens", boxes_train="$TASK/t1/out/psbox/pixelseal_train_boxes.json",
                      boxes_test="$TASK/t1/out/psbox/pixelseal_test_boxes.json", conf=0.90, chain="~/t1w/integ/chain/pixelseal"),
}
FAILS, WARNS, LOG = [], [], []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def fail(s):
    FAILS.append(s); log(f"  ÉCHEC : {s}")


def warn(s):
    WARNS.append(s); log(f"  ALERTE : {s}")


def P(p):
    p = str(p).replace("$TASK", str(TASK))
    p = Path(os.path.expanduser(p))
    return p if p.is_absolute() else TASK / p


def sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_npz(path):
    z = np.load(path)
    return [str(n) for n in z["names"]], z["probs"]


def load_boxes(path):
    d = json.loads(P(path).read_text())
    d = d["boxes"] if isinstance(d, dict) and "boxes" in d else d
    out = {}
    for n, b in d.items():
        r = b.get("rect") if b.get("rect") is not None else b.get("box")
        if r is None:
            continue
        r = [int(v) for v in r]
        if not (len(r) == 4 and 0 <= r[0] < r[1] <= 512 and 0 <= r[2] < r[3] <= 512):
            warn(f"rectangle hors image ou dégénéré {n} {r} dans {path}")
        out[str(n)] = {"rect": r, "conf": float(b.get("conf", 0.0))}
    return out


def canon(boxes, names=None):
    keep = sorted(boxes if names is None else [n for n in boxes if n in names])
    return hashlib.sha1(json.dumps([(n, boxes[n]["rect"], round(boxes[n]["conf"], 6)) for n in keep]).encode()).hexdigest()


def fingerprint(method, c, fold0_names):
    ens = P(c["ens"]) / method
    f0 = [ens / "fold0_probs128.npz"] + ([ens / "fold0_tta_probs128.npz"] if method == "stegastamp" else [])
    te = [ens / "test_probs128.npz"] + ([ens / "test_tta_probs128.npz"] if method == "stegastamp" else [])
    return {"fold0": {"ens": [sha(f) for f in f0], "boxes": canon(load_boxes(c["boxes_train"]), fold0_names), "conf": float(c["conf"])},
            "test": {"ens": [sha(f) for f in te], "boxes": canon(load_boxes(c["boxes_test"])), "conf": float(c["conf"])}}


def run_chain_child(method, ens, pattern, conf, out):
    """Mode interne : chain_integ.py de sub16 tel quel, avec BOXES/CONF de la méthode remplacés."""
    sys.path.insert(0, str(CHAIN_PY.parent))
    import chain_integ as C
    C.BOXES[method] = pattern; C.CONF[method] = float(conf)
    sys.argv = ["chain_integ.py", "--maps", "new", "--rule", "zone", "--ens", ens, "--out", out, "--methods", method, "--workers", "8"]
    C.main()


def boot(d, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(FINAL / "config.json"))
    ap.add_argument("--force", action="store_true", help="relance la chaîne pour toutes les méthodes")
    ap.add_argument("--freeze-ref", action="store_true", help="(une fois) fige les empreintes des composants de sub16")
    ap.add_argument("--_chain", nargs=5, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._chain:
        return run_chain_child(*a._chain)

    cfg = json.loads(Path(a.config).read_text())
    out = P(cfg.get("out", "~/t1w/final/sub_final.pt"))
    work = P(cfg.get("work", "~/t1w/final/work")); work.mkdir(parents=True, exist_ok=True)
    ref_fp_path = FINAL / "ref_sub16.json"
    log(f"== build_final {datetime.datetime.now().isoformat(timespec='seconds')} config={a.config} work={work} out={out}")

    ref_f0 = {m: load_npz(P(REF[m]["chain"]) / "fold0_probs128.npz") for m in METHODS}
    f0names = {m: set(ref_f0[m][0]) for m in METHODS}
    if a.freeze_ref:
        assert not ref_fp_path.exists(), f"{ref_fp_path} existe déjà"
        ref_fp_path.write_text(json.dumps({m: fingerprint(m, REF[m], f0names[m]) for m in METHODS}, indent=1))
        log(f"empreintes sub16 figées -> {ref_fp_path}"); return 0
    ref_fp = json.loads(ref_fp_path.read_text())

    # ---------- 1. empreintes et relance de la chaîne
    log("== 1. composants")
    chain_dir, status = {}, {}
    for m in METHODS:
        c = {**{k: REF[m][k] for k in ("ens", "boxes_train", "boxes_test", "conf")}, **cfg["methods"].get(m, {})}
        fp = fingerprint(m, c, f0names[m])
        ch0 = fp["fold0"] != ref_fp[m]["fold0"]; cht = fp["test"] != ref_fp[m]["test"]
        diffs = [f"{s}.{k}" for s in ("fold0", "test") for k in fp[s] if fp[s][k] != ref_fp[m][s][k]]
        status[m] = dict(cfg=c, fold0_changed=ch0, test_changed=cht, changed=diffs)
        log(f"  {m}: ens={c['ens']} boxes_train={c['boxes_train']} boxes_test={c['boxes_test']} conf={c['conf']}")
        log(f"     composants changés vs sub16 : {diffs or 'aucun'}")
        if not (ch0 or cht or a.force):
            chain_dir[m] = P(REF[m]["chain"]); log(f"     -> sortie de chaîne de sub16 reprise : {chain_dir[m]}"); continue
        bdir = work / "boxes"; bdir.mkdir(exist_ok=True)
        for s in ("train", "test"):   # copie normalisée (rect, conf) : chain_integ lit json["boxes"][nom]["rect"|"conf"]
            (bdir / f"{m}_{s}.json").write_text(json.dumps({"boxes": load_boxes(c[f"boxes_{s}"])}))
        od = work / f"chain_{m}"
        env = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = [sys.executable, __file__, "--_chain", m, str(P(c["ens"])), str(bdir / f"{m}_{{s}}.json"), str(c["conf"]), str(od)]
        log(f"     -> relance de la chaîne ({'--force' if not (ch0 or cht) else 'composant changé'}) -> {od}")
        r = subprocess.run(cmd, env=env, cwd=TASK, stdout=open(work / f"chain_{m}.log", "w"), stderr=subprocess.STDOUT)
        if r.returncode:
            fail(f"chaîne {m} code {r.returncode} (voir {work}/chain_{m}.log)"); continue
        chain_dir[m] = od / m

    # ---------- 2. fold 0 et identité test
    log("== 2. fold 0 (34 images par méthode) face à sub16")
    sys.path[:0] = [str(TASK / "t1" / "common"), str(TASK)]
    from wmloc.utils import miou_image
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    tidx = {e["image_name"]: e for e in tidx} if isinstance(tidx, list) else tidx
    fold0 = {}
    for m in METHODS:
        if m not in chain_dir:
            continue
        nn, pn = load_npz(chain_dir[m] / "fold0_probs128.npz"); no, po = ref_f0[m]
        same = nn == no and np.array_equal(pn, po)
        new = np.array([miou_image(p.astype(np.uint8), gt[n]) for n, p in zip(nn, pn)])
        old = np.array([miou_image(p.astype(np.uint8), gt[n]) for n, p in zip(no, po)])
        et = np.array([tidx[n]["edit_type"] for n in nn])
        bytype = {t: [round(float(old[et == t].mean()), 4), round(float(new[et == t].mean()), 4)] for t in ("hole", "transplant", "outpainting")}
        fold0[m] = dict(identical=bool(same), sub16=float(old.mean()), new=float(new.mean()), by_type_sub16_new=bytype,
                        delta_boot95=boot(new - old) if nn == no else None)
        if not status[m]["fold0_changed"]:
            log(f"  {m}: composant fold 0 non mesurable/inchangé -> identique à sub16 : {same}  (score {new.mean():.4f})")
            if not same:
                fail(f"{m}: fold 0 différent de sub16 alors que ses composants fold 0 sont inchangés")
        else:
            d = fold0[m]["delta_boot95"]
            nimg = sum(int(not np.array_equal(pn[i], po[i])) for i in range(len(nn))) if nn == no else -1
            log(f"  {m}: composant fold 0 MESURABLE changé ({nimg} images fold 0 changées) : sub16 {old.mean():.4f} -> nouveau {new.mean():.4f} ; "
                f"écart {d[0]:+.4f} IC95 [{d[1]:+.4f}, {d[2]:+.4f}] ; par type (sub16, nouveau) {bytype}")
            log(f"     règle : adopter seulement si gain ≥ +0,01 (IC95 > 0) ; < 0,03 sur 34 images = bruit probable"
                + (" ; ICI PERTE -> revenir au composant de sub16" if d[0] < 0 else ""))
            if d[0] < 0:
                warn(f"{m}: le nouveau composant fait BAISSER le fold 0 ({d[0]:+.4f})")
        if (status[m]["fold0_changed"] or status[m]["test_changed"] or a.force) and not status[m]["test_changed"]:
            tn, tp = load_npz(chain_dir[m] / "test_probs128.npz"); rn, rp = load_npz(P(REF[m]["chain"]) / "test_probs128.npz")
            ts = tn == rn and np.array_equal(tp, rp)
            log(f"  {m}: composants test inchangés -> sortie test identique à sub16 : {ts}")
            if not ts:
                fail(f"{m}: sortie test de chaîne différente de sub16 alors que ses composants test sont inchangés")

    if FAILS:
        log(f"== ARRÊT avant build_best : {FAILS}"); return finish(work, out, status, fold0, {}, 2)

    # ---------- 4. build_best
    log(f"== 4. build_best -> {out}")
    src = {"_doc": f"build_final {a.config}"}
    for m in METHODS:
        src[m] = [{"name": f"final_{m}", "val": [str(chain_dir[m] / "fold0_probs128.npz")], "test": [str(chain_dir[m] / "test_probs128.npz")]}]
    sj = work / "sources.json"; sj.write_text(json.dumps(src, indent=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(BUILD_PY), "--sources", str(sj), "--boxdir", str(TASK / "t1/out/mgr/nobox"),
                        "--min-gain", "1", "--no-log", "--out", str(out)], cwd=TASK,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8"},
                       stdout=open(work / "build.log", "w"), stderr=subprocess.STDOUT)
    if r.returncode:
        fail(f"build_best code {r.returncode} (voir {work}/build.log)"); return finish(work, out, status, fold0, {}, 2)
    bj = json.loads(out.with_suffix(".json").read_text())
    rj = json.loads(REF_SUB.with_suffix(".json").read_text())
    for m in METHODS:
        cn, co = bj["chosen"][m], rj["chosen"][m]
        log(f"  {m}: réglage {cn['cfg']} (sub16 {co['cfg']}) ; val {cn['val']:.4f} (sub16 {co['val']:.4f})")
        if cn["cfg"] != co["cfg"]:
            warn(f"{m}: build_best a choisi un réglage différent de sub16")

    # ---------- 3. contrôles de format et 5. comparaison avec sub16
    import torch
    log("== 3. contrôles de la soumission")
    new, old = torch.load(out, weights_only=True), torch.load(REF_SUB, weights_only=True)
    size = out.stat().st_size / 1e6
    tm = json.loads((TASK / "release/test/metadata.json").read_text()); tmeth = {x["image_name"]: x["method"] for x in tm}
    M, O = new["masks"], old["masks"]
    ok_fmt = (new["image_names"] == old["image_names"] == [x["image_name"] for x in tm] and M.dtype == torch.bool
              and tuple(M.shape) == (1020, 128, 128) and size < 20)
    log(f"  noms et ordre = sub16 = metadata, bool [1020,128,128], {size:.2f} Mo < 20 : {ok_fmt}")
    if not ok_fmt:
        fail("format ou ordre des noms différent de sub16")
    names = new["image_names"]
    cov_n, cov_o = M.flatten(1).float().mean(1).numpy(), O.flatten(1).float().mean(1).numpy()
    bad = [names[i] for i in range(len(names)) if cov_n[i] in (0.0, 1.0)]
    log(f"  masques vides ou pleins : {len(bad)} {bad[:10]}")
    if bad:
        fail(f"{len(bad)} masques vides ou pleins")
    sys.path.insert(0, str(TASK / "t1" / "guard2"))
    from rule import THR
    reg = {}
    for m in METHODS:
        zz = np.load(TASK / "t1/out/signal" / f"fullacc_{m}_test.npz")
        reg.update({str(n): float(x) >= THR[m] for n, x in zip(zz["names"], zz["acc"])})
    log("== 5. comparaison avec sub16 (cellules 128x128)")
    cmp = {}
    for m in METHODS:
        idx = [i for i, n in enumerate(names) if tmeth[n] == m]
        maj, regf = float((cov_n[idx] >= 0.5).mean()), float(np.mean([reg[names[i]] for i in idx]))
        log(f"  {m}: couverture moy {cov_n[idx].mean():.3f} (sub16 {cov_o[idx].mean():.3f}) ; part zone majoritaire {maj:.3f} "
            f"(sub16 {(cov_o[idx] >= 0.5).mean():.3f}, régime décodé {regf:.3f})")
        if abs(cov_n[idx].mean() - cov_o[idx].mean()) > 0.02:
            warn(f"{m}: couverture moyenne s'écarte de sub16 de plus de 0,02")
        if abs(maj - regf) > 0.03:
            warn(f"{m}: part de zone majoritaire loin du régime décodé")
        d = (M[idx] != O[idx]).flatten(1).sum(1).numpy()
        ch = [(names[idx[k]], int(d[k]), round(float(cov_o[idx[k]]), 3), round(float(cov_n[idx[k]]), 3)) for k in np.nonzero(d)[0]]
        flip = [c[0] for c in ch if (c[2] >= 0.5) != (c[3] >= 0.5)]
        cmp[m] = {"images_changed": len(ch), "cells_changed": int(d.sum()), "regime_flips": flip,
                  "images": [{"name": n, "cells": k, "cov_sub16": a_, "cov_new": b_} for n, k, a_, b_ in ch]}
        log(f"  {m}: {len(ch)} images changées, {int(d.sum())} cellules ; bascules de zone majoritaire : {len(flip)}")
        for n, k, a_, b_ in ch:
            log(f"     {n}  {k:5d} cellules  couverture {a_:.3f} -> {b_:.3f}")
        if flip:
            warn(f"{m}: {len(flip)} images changent de zone majoritaire : {flip[:5]}")
    log(f"  TOTAL cellules différentes de sub16 : {int((M != O).sum())}")
    return finish(work, out, status, fold0, cmp, 2 if FAILS else 0)


def finish(work, out, status, fold0, cmp, code):
    log(f"== FIN : {'OK' if code == 0 else 'ÉCHEC'} ; échecs {len(FAILS)} {FAILS} ; alertes {len(WARNS)} {WARNS}")
    log(f"   soumission : {out if code == 0 else '(ne pas envoyer)'} — NON envoyée")
    rep = dict(code=code, out=str(out), fails=FAILS, warns=WARNS, status=status, fold0=fold0, vs_sub16=cmp, log=LOG)
    (work / "final_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main() or 0)
