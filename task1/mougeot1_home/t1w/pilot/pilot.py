"""pilot.py — assemblage tâche 1 piloté par une config (compte mougeot1, CPU seul, écritures sous ~/t1w uniquement).

Reconstruit la chaîne d'intégration de Lumine sans ses contrôles vers son HOME :
  moyenne simple des membres -> chain19.py INCHANGÉ (importé, BOXES/CONF remplacés) -> override optionnel
  -> build_best.py INCHANGÉ (--boxdir t1/out/mgr/nobox --min-gain 1) -> contrôles (format, masques vides/pleins,
  couverture, bascules de zone majoritaire) -> .pt + rapport JSON.

Config (JSON ; "$TASK" et "~" développés ; chemin relatif = relatif à TASK) :
  {"out": "~/t1w/E1/E1.pt", "ref_sub": "t1/out/mgr/submissions/sub11_v6.pt",
   "gates": {"stegastamp": 0.7979, "trustmark": 0.7155, "pixelseal": 0.6550, "global": 0.7228},   (optionnel, 4 décimales)
   "methods": {"<méthode>": {"fold0": [npz...], "test": [npz...],            (SG en plus : "fold0_tta", "test_tta")
                             "boxes_train": [src...], "boxes_test": [src...], "conf": 0.9,
                             "override": {"fold0": npz|null, "test": npz|null, "names": [..]|null}}}}   (optionnel)
  src de rectangles = "fichier.json" ou {"file": ..., "names": [...]} ; fusion dans l'ordre, l'entrée suivante remplace.
Lancer (depuis n'importe où, venv TASK activé) :
  python ~/t1w/pilot/pilot.py --config ~/t1w/pilot/sub18.json                         # assemble + contrôles
  python ~/t1w/pilot/pilot.py --config BASE.json --try CAND.json [--adopt-out NEW.json] # essai d'un candidat seul
  CAND.json = {"method": m, "fold0": [...], "test": [...] (SG : + "fold0_tta", "test_tta")}  -> membres AJOUTÉS à BASE ;
  ou clés "boxes_train"/"boxes_test"/"conf" qui REMPLACENT celles de BASE. Règle : gain >= +0,01 ET IC95 bas > 0.
Code de sortie : 0 = contrôles durs OK (essai : 0 = adopté, 1 = rejeté) ; 2 = échec (ne pas envoyer)."""
from __future__ import annotations

import argparse, copy, hashlib, json, os, subprocess, sys
from pathlib import Path

import numpy as np

TASK = Path(os.environ.get("TASK") or "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
HOME = Path.home()
CHAIN = HOME / "t1w/integ19sg/code/chain19.py"
BUILD = HOME / "t1w/code/build_best.py"
WORK = HOME / "t1w/pilot/work"
METHODS = ("stegastamp", "trustmark", "pixelseal")
TYPES = ("hole", "transplant", "outpainting")
FAILS, WARNS, LOG = [], [], []


def log(s=""): print(s, flush=True); LOG.append(s)
def fail(s): FAILS.append(s); log(f"  ÉCHEC : {s}")
def warn(s): WARNS.append(s); log(f"  ALERTE : {s}")


def P(p):
    p = Path(os.path.expanduser(str(p).replace("$TASK", str(TASK))))
    return p if p.is_absolute() else TASK / p


def load(p):
    z = np.load(P(p)); return {str(n): v for n, v in zip(z["names"], z["probs"])}


def mean_of(files, names):   # identique à integrate*.py de Lumine (bit à bit après float16)
    acc = np.zeros((len(names), 128, 128), np.float64)
    for f in files:
        d = load(f); acc += np.stack([d[n] for n in names]).astype(np.float32)
    return (acc / len(files)).astype(np.float32)


def check_member(f, names):
    try:
        d = load(f)
    except Exception as e:  # noqa: BLE001
        return f"illisible ({e})"
    miss = [n for n in names if n not in d]
    if miss: return f"{len(miss)}/{len(names)} images manquantes"
    a = np.stack([d[n] for n in names]).astype(np.float32)
    if a.shape[1:] != (128, 128) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1: return f"forme/valeurs invalides {a.shape}"
    return ""


def load_boxes(srcs):
    out = {}
    for s in srcs:
        f, keep = (s["file"], s.get("names")) if isinstance(s, dict) else (s, None)
        d = json.loads(P(f).read_text()); d = d["boxes"] if "boxes" in d else d
        for n, b in d.items():
            r = b.get("rect") if b.get("rect") is not None else b.get("box")
            if r is None or (keep is not None and n not in keep): continue
            r = [int(v) for v in r]
            if not (0 <= r[0] < r[1] <= 512 and 0 <= r[2] < r[3] <= 512): warn(f"rectangle dégénéré {n} {r} ({f})")
            out[str(n)] = {"rect": r, "conf": float(b.get("conf", 0.0))}
        if keep is not None and set(keep) - set(d): warn(f"{f} : noms absents {sorted(set(keep) - set(d))}")
    return out


def fstamp(f):
    st = P(f).stat(); return [str(P(f)), st.st_size, int(st.st_mtime)]


def maps_of(m, c):
    return ("fold0", "test", "fold0_tta", "test_tta") if m == "stegastamp" else ("fold0", "test")


def run_chain(m, c, splits):
    """ensembles + chain19 pour une méthode ; cache dans WORK/chain/<m>_<clé> ; renvoie le dossier <...>/chain/<m>."""
    bx = {s: load_boxes(c[f"boxes_{s}"]) for s in ("train", "test")}
    key = hashlib.sha1(json.dumps([m, splits, c["conf"], bx, {k: [fstamp(f) for f in c[k]] for k in maps_of(m, c)},
                                   hashlib.sha1(CHAIN.read_bytes()).hexdigest()], sort_keys=True).encode()).hexdigest()[:12]
    d = WORK / "chain" / f"{m}_{key}"
    if (d / "chain" / "DONE").is_file():
        log(f"  {m}: chaîne en cache {d}"); return d / "chain" / m
    (d / "ens" / m).mkdir(parents=True, exist_ok=True)
    for k in maps_of(m, c):
        if not c[k]: fail(f"{m}: aucun membre {k}"); return None
        names = sorted(load(c[k][0]))
        for f in c[k]:
            why = check_member(f, names)
            if why: fail(f"{m} {k}: membre {f} : {why}"); return None
        np.savez(d / "ens" / m / f"{k}_probs128.npz", names=np.array(names), probs=mean_of(c[k], names).astype(np.float16))
    for s in ("train", "test"):
        (d / f"boxes_{s}.json").write_text(json.dumps({"boxes": bx[s]}))
    log(f"  {m}: chaîne [{splits}] -> {d} (rectangles conf >= {c['conf']} : train {sum(v['conf'] >= c['conf'] for v in bx['train'].values())},"
        f" test {sum(v['conf'] >= c['conf'] for v in bx['test'].values())})")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, __file__, "--_chain", m, str(d / "ens"), str(d / "boxes_{s}.json"), str(c["conf"]), str(d / "chain"), splits],
                       cwd=TASK, env=env, stdout=open(d / "chain.log", "w"), stderr=subprocess.STDOUT)
    if r.returncode: fail(f"chaîne {m} code {r.returncode} ({d}/chain.log)"); return None
    return d / "chain" / m


def chain_child(m, ens, pattern, conf, out, splits):
    sys.path.insert(0, str(CHAIN.parent))
    import chain19 as C
    C.BOXES[m] = pattern; C.CONF[m] = float(conf)
    sys.argv = ["chain19.py", "--maps", "new", "--rule", "zone", "--ens", ens, "--out", out, "--methods", m, "--splits", splits, "--workers", "8"]
    C.main(); Path(out, "DONE").write_text("ok\n")


def apply_override(m, c, chdir, dst):
    """copie la sortie de chaîne dans dst/<m>/ en remplaçant les images listées par les masques de l'override."""
    ov = c.get("override") or {}
    (dst / m).mkdir(parents=True, exist_ok=True)
    for s in ("fold0", "test"):
        z = np.load(chdir / f"{s}_probs128.npz"); names = [str(n) for n in z["names"]]; pr = z["probs"].copy()
        if ov.get(s):
            o = load(ov[s]); keep = [n for n in (ov.get("names") or o) if n in o and n in names]
            for n in keep: pr[names.index(n)] = (o[n].astype(np.float32) >= 0.5).astype(pr.dtype)
            log(f"  {m} {s}: override {len(keep)} images depuis {ov[s]}")
        np.savez(dst / m / f"{s}_probs128.npz", names=np.array(names), probs=pr)
    return dst / m


def scores(chdir, gt, tidx):
    pr = load(chdir / "fold0_probs128.npz"); names = sorted(pr)
    from wmloc.utils import miou_image
    return names, np.array([miou_image(pr[n].astype(np.uint8), gt[n]) for n in names]), np.array([tidx[n]["edit_type"] for n in names])


def boot(d, B=10000, seed=0):
    bm = d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def gt_tidx():
    sys.path[:0] = [str(TASK / "t1" / x) for x in ("train", "synth", "common")] + [str(TASK)]
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    t = json.loads((TASK / "t1/common/train_index.json").read_text())
    return gt, ({e["image_name"]: e for e in t} if isinstance(t, list) else t)


def trial(cfg, cand, a):
    m = cand["method"]; base = cfg["methods"][m]; new = copy.deepcopy(base)
    for k in maps_of(m, base):
        new[k] = base[k] + cand.get(k, [])
    for k in ("boxes_train", "boxes_test", "conf"):
        if k in cand: new[k] = cand[k]
    gt, tidx = gt_tidx()
    cb, cn = run_chain(m, base, "fold0"), run_chain(m, new, "fold0")
    if FAILS: return 2
    nb, sb, et = scores(cb, gt, tidx); nn, sn, _ = scores(cn, gt, tidx); assert nb == nn
    g = boot(sn - sb); adopt = bool(g[0] >= a.min_gain and g[1] > 0)
    log(f"== ESSAI {m} {a.try_}: base {sb.mean():.4f} -> {sn.mean():.4f} ; écart {g[0]:+.4f} IC95 [{g[1]:+.4f}, {g[2]:+.4f}] ;"
        f" {int((sn != sb).sum())} images changées => {'ADOPTÉ' if adopt else 'rejeté'} (règle >= {a.min_gain:+.2f} et IC95 bas > 0)")
    for t in TYPES:
        k = et == t; log(f"   {t:11s} n={k.sum():2d} {sb[k].mean():.4f} -> {sn[k].mean():.4f} écart {boot((sn - sb)[k])[0]:+.4f}")
    if adopt and a.adopt_out:
        c2 = copy.deepcopy(cfg); c2["methods"][m] = new; c2.pop("gates", None)
        c2["_doc"] = (cfg.get("_doc", "") + f" + essai adopté {a.try_}").strip(); Path(a.adopt_out).expanduser().write_text(json.dumps(c2, indent=1))
        log(f"   config avec le candidat -> {a.adopt_out}")
    (WORK / "trials.jsonl").open("a").write(json.dumps({"config": a.config, "try": a.try_, "method": m, "base": float(sb.mean()),
                                                         "new": float(sn.mean()), "delta_boot95": g, "adopt": adopt}) + "\n")
    return 0 if adopt else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--out", help="remplace cfg['out']")
    ap.add_argument("--try", dest="try_"); ap.add_argument("--adopt-out")
    ap.add_argument("--min-gain", type=float, default=0.01, help="règle d'adoption (NE PAS CHANGER)")
    ap.add_argument("--_chain", nargs=6, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._chain: return chain_child(*a._chain)
    cfg = json.loads(Path(a.config).expanduser().read_text()); os.chdir(TASK)
    if a.try_: return trial(cfg, json.loads(Path(a.try_).expanduser().read_text()), a)
    out = P(a.out or cfg["out"]); rep_path = out.with_suffix(".report.json")
    for w in (out, WORK): assert not w.resolve().is_relative_to(TASK), f"écriture interdite sous TASK : {w}"
    log(f"== pilot {a.config} -> {out}")
    gt, tidx = gt_tidx()
    final_dir = out.parent / (out.stem + "_chain"); res = {}
    for m in METHODS:
        c = cfg["methods"][m]; ch = run_chain(m, c, "fold0,test")
        if ch is None: continue
        fd = apply_override(m, c, ch, final_dir); _, s, et = scores(fd, gt, tidx)
        res[m] = {"fold0": float(s.mean()), "by_type": {t: float(s[et == t].mean()) for t in TYPES}, "members": {k: len(c[k]) for k in maps_of(m, c)}}
        log(f"  {m}: fold 0 = {s.mean():.4f}  {json.dumps({t: round(v, 4) for t, v in res[m]['by_type'].items()})}")
    if FAILS: return finish(rep_path, res, 2)
    glob = float(np.mean([res[m]["fold0"] for m in METHODS])); log(f"  fold 0 global = {glob:.4f}")
    for k, v in (cfg.get("gates") or {}).items():
        got = glob if k == "global" else res[k]["fold0"]
        (log if round(got, 4) == round(v, 4) else fail)(f"  porte {k}: attendu {v:.4f}, obtenu {got:.4f}")
    if FAILS: return finish(rep_path, res, 2)
    src = {m: [{"name": f"pilot_{m}", "val": [str(final_dir / m / "fold0_probs128.npz")], "test": [str(final_dir / m / "test_probs128.npz")]}] for m in METHODS}
    sj = out.parent / (out.stem + "_sources.json"); out.parent.mkdir(parents=True, exist_ok=True); sj.write_text(json.dumps(src, indent=1))
    r = subprocess.run([sys.executable, str(BUILD), "--sources", str(sj), "--boxdir", str(TASK / "t1/out/mgr/nobox"), "--min-gain", "1", "--no-log", "--out", str(out)],
                       cwd=TASK, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1"),
                       stdout=open(out.parent / (out.stem + "_build.log"), "w"), stderr=subprocess.STDOUT)
    if r.returncode: fail(f"build_best code {r.returncode}"); return finish(rep_path, res, 2)
    import torch
    new, ref = torch.load(out, weights_only=True), torch.load(P(cfg.get("ref_sub", "t1/out/mgr/submissions/sub11_v6.pt")), weights_only=True)
    tm = json.loads((TASK / "release/test/metadata.json").read_text()); tmeth = {x["image_name"]: x["method"] for x in tm}
    M, O = new["masks"], ref["masks"]; names = new["image_names"]; size = out.stat().st_size / 1e6
    ok = names == ref["image_names"] == [x["image_name"] for x in tm] and M.dtype == torch.bool and tuple(M.shape) == (1020, 128, 128) and size <= 25
    (log if ok else fail)(f"  format : noms = metadata = référence, bool [1020,128,128], {size:.2f} Mo <= 25 : {ok}")
    cov, covo = M.flatten(1).float().mean(1).numpy(), O.flatten(1).float().mean(1).numpy()
    bad = [names[i] for i in range(len(names)) if cov[i] in (0.0, 1.0)]
    (fail if bad else log)(f"  masques vides ou pleins : {len(bad)} {bad[:10]}")
    sys.path.insert(0, str(TASK / "t1/guard2")); from rule import THR
    reg = {}
    for m in METHODS:
        z = np.load(TASK / "t1/out/signal" / f"fullacc_{m}_test.npz"); reg.update({str(n): float(x) >= THR[m] for n, x in zip(z["names"], z["acc"])})
    cmp = {}
    for m in METHODS:
        idx = [i for i, n in enumerate(names) if tmeth[n] == m]
        maj, regf = float((cov[idx] >= 0.5).mean()), float(np.mean([reg[names[i]] for i in idx]))
        d = (M[idx] != O[idx]).flatten(1).sum(1).numpy(); ch = [names[idx[k]] for k in np.nonzero(d)[0]]
        flips = [names[i] for i in idx if (cov[i] >= 0.5) != (covo[i] >= 0.5)]
        cmp[m] = {"coverage": round(float(cov[idx].mean()), 4), "coverage_ref": round(float(covo[idx].mean()), 4), "majority_share": round(maj, 4),
                  "decoded_regime": round(regf, 4), "images_changed": len(ch), "cells_changed": int(d.sum()), "regime_flips": flips, "changed": ch}
        log(f"  {m}: couverture {cmp[m]['coverage']} (réf {cmp[m]['coverage_ref']}) ; zone majoritaire {maj:.3f} (régime décodé {regf:.3f}) ;"
            f" {len(ch)} images / {int(d.sum())} cellules changées vs réf ; bascules {len(flips)}")
        if abs(maj - regf) > 0.03: warn(f"{m}: part de zone majoritaire loin du régime décodé")
        if flips: warn(f"{m}: {len(flips)} bascules de zone majoritaire vs réf : {flips[:5]}")
    res["_global"] = glob; res["_vs_ref"] = cmp
    return finish(rep_path, res, 2 if FAILS else 0)


def finish(rep_path, res, code):
    log(f"== FIN code {code} ; échecs {FAILS} ; alertes {len(WARNS)} ; soumission NON envoyée")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps({"code": code, "fails": FAILS, "warns": WARNS, "results": res, "log": LOG}, indent=1, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main() or 0)
