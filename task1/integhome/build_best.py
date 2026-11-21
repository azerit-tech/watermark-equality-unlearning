"""[copie agent integhome : TASK fixé (lecture des entrées) indépendamment de l'emplacement du script ;
 --log = chemin du journal (défaut $T1W/RESULTS.tsv ou ~/t1w/RESULTS.tsv) au lieu de t1/RESULTS.tsv ; rien d'autre ne change]
Choisit la meilleure source + post-traitement par méthode sur la validation réelle, écrit la soumission (NON envoyée).
Depuis TASK, CPU (quelques secondes) :
    python t1/mgr/build_best.py --sources t1/mgr/sources.json --out t1/out/mgr/submissions/subNN.pt
Options : --boxdir (défaut t1/out/boxsearch), --no-log (n'écrit pas t1/RESULTS.tsv), --min-gain (défaut 0.02).

sources.json : {"<methode>": [{"name": "...", "val": [npz, ...], "test": [npz, ...]}, ...], ...}
  - val  : fichiers de validation (folds différents) CONCATÉNÉS ; test : fichiers de modèles MOYENNÉS.
  - chemins relatifs à TASK ; clés commençant par "_" ignorées ; source absente/incomplète = ignorée (avertissement).
  - chaque npz : names, probs [N,128,128] ; filtrés par méthode via release/{train,test}/metadata.json.
Candidats par méthode : constante par régime (référence) ; pour chaque source, défaut = seuil 0.5 + garde-fou,
plus seuil global, sans garde-fou, seuil par régime, nettoyage morphologique, garantie de classe minoritaire,
rectangles (<boxdir>/<methode>_{train,test}_boxes.json, {nom: {"box": [y0,y1,x0,x1] (pixels 512, fin exclue),
"conf": c}}). Un réglage autre que le défaut n'est retenu que s'il gagne >= --min-gain ; une source n'est retenue
que si elle bat la constante sur ses propres images de validation."""
import argparse, datetime, itertools, json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage

TASK = Path(os.environ.get("TASK") or "/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1" / "train", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from regime_guard import guard, load_regimes, COV  # noqa: E402
from train import binarize  # noqa: E402
from evaluate import evaluate, load_train  # noqa: E402
from wmloc.utils import miou_image, downsample_mask  # noqa: E402

METHODS = ("stegastamp", "trustmark", "pixelseal")
THRS = (0.3, 0.4, 0.5, 0.6, 0.7)
NS = (0, 8, 32)
CELLS = 128 * 128
K1 = 164                                                     # 1 % de la grille (= binarize)
KMED = {True: int(round((1 - COV[True]) * CELLS)), False: int(round(COV[False] * CELLS))}
DEFAULT = dict(thr=0.5, guard=True, N=0, fill=False, k="1pct", box=None)


# ---------------------------------------------------------------- post-traitement
def binar(p, full, thr, use_guard):
    if use_guard and thr == 0.5:
        return guard(p, full)[0].astype(np.uint8)                # exactement la recette de sub01
    m = binarize(p, thr)
    if use_guard and (m.mean() >= 0.5) != full:
        k = int(round(COV[full] * CELLS))
        m = np.zeros(CELLS, np.uint8); m[np.argsort(-p.reshape(-1), kind="stable")[:k]] = 1
        m = m.reshape(p.shape)
    return m


def ensure_minority(m, p, full, kmode):
    maj = int(m.mean() >= 0.5)
    if (m == 1 - maj).any():
        return m
    k = K1 if kmode == "1pct" else KMED[full]
    m = m.copy().reshape(-1)
    if maj == 1:
        m[np.argsort(p.reshape(-1), kind="stable")[:k]] = 0      # cellules les moins probables -> 0
    else:
        m[np.argsort(-p.reshape(-1), kind="stable")[:k]] = 1     # cellules les plus probables -> 1
    return m.reshape(p.shape)


def morph(m, N, fill):
    if N == 0 and not fill:
        return m
    maj = int(m.mean() >= 0.5)
    mino = (m == 1 - maj)
    if N > 0:
        lab, n = ndimage.label(mino)
        if n:
            sizes = np.bincount(lab.reshape(-1))
            mino = (sizes[lab] >= N) & mino
    if fill:
        mino = ndimage.binary_fill_holes(mino)                   # trous (cellules majoritaires enclavées) comblés
    return np.where(mino, 1 - maj, maj).astype(np.uint8)


def post(p, full, thr, use_guard, N, fill, kmode):
    m = binar(p, full, thr, use_guard)
    m = morph(m, N, fill)
    return ensure_minority(m, p, full, kmode)


def box_mask(b):
    y0, y1, x0, x1 = [int(round(v)) for v in b]
    big = np.zeros((512, 512), np.uint8); big[max(y0, 0):min(y1, 512), max(x0, 0):min(x1, 512)] = 1
    return downsample_mask(big)


def apply_boxes(m, p, full, boxes, name, cthr, kmode):
    if cthr is None or boxes is None or name not in boxes or boxes[name]["conf"] < cthr:
        return m
    return ensure_minority(box_mask(boxes[name]["box"]), p, full, kmode)


# ---------------------------------------------------------------- chargement
def load_boxes(path):
    if not path.exists():
        return None
    raw = json.loads(path.read_text()); out = {}
    if isinstance(raw.get("boxes"), dict):  # format boxsearch {"format","method","boxes":{...}}
        raw = raw["boxes"]
    for n, v in raw.items():
        if isinstance(v, dict):
            b = v.get("box") or v.get("rect") or [v["y0"], v["y1"], v["x0"], v["x1"]]
            out[str(n)] = {"box": [float(x) for x in b], "conf": float(v["conf"])}
        else:                                                    # [y0, y1, x0, x1, conf]
            out[str(n)] = {"box": [float(x) for x in v[:4]], "conf": float(v[4])}
    return out


def load_val(files, method, meth):
    probs = {}
    for f in files:
        z = np.load(TASK / f)
        for n, p in zip(z["names"], z["probs"]):
            n = str(n)
            if meth.get(n) == method and n not in probs:
                probs[n] = p.astype(np.float32)
    return probs


def load_test(files, names):
    acc = None
    for f in files:
        z = np.load(TASK / f); d = dict(zip([str(n) for n in z["names"]], range(len(z["names"]))))
        miss = [n for n in names if n not in d]
        if miss:
            raise ValueError(f"{f}: {len(miss)} images test manquantes")
        arr = z["probs"][[d[n] for n in names]].astype(np.float32)
        acc = arr if acc is None else acc + arr
    return dict(zip(names, acc / len(files)))


# ---------------------------------------------------------------- recherche
def cfg_str(c):
    s = f"thr={c['thr']}" if c["thr"] != "regime" else f"thr=regime(full={c['thr_full']},part={c['thr_part']})"
    s += " guard" if c["guard"] else " noguard"
    if c["N"]: s += f" N={c['N']}"
    if c["fill"]: s += " fill"
    if c["k"] != "1pct": s += f" k={c['k']}"
    if c["box"] is not None: s += f" box>={c['box']:.3g}"
    return s


def search(probs, reg, gt, boxes):
    names = sorted(probs)
    isfull = np.array([reg[n] for n in names])
    # tableau des scores par image pour chaque (thr, guard, N, fill, k)
    S = {}
    for thr, g, N, fill, k in itertools.product(THRS, (True, False), NS, (False, True), ("1pct", "med")):
        S[(thr, g, N, fill, k)] = np.array([miou_image(post(probs[n], reg[n], thr, g, N, fill, k), gt[n])
                                            for n in names])
    cands = []                                                  # (cfg, per-image scores)
    for (thr, g, N, fill, k), v in S.items():
        cands.append((dict(thr=thr, guard=g, N=N, fill=fill, k=k, box=None), v))
    for N, fill, k in itertools.product(NS, (False, True), ("1pct", "med")):   # seuil par régime (avec garde-fou)
        best = {}
        for r in (True, False):
            sel = isfull == r
            best[r] = max(THRS, key=lambda t: (S[(t, True, N, fill, k)][sel].mean() if sel.any() else 0, t == 0.5))
        v = np.where(isfull, S[(best[True], True, N, fill, k)], S[(best[False], True, N, fill, k)])
        cands.append((dict(thr="regime", thr_full=best[True], thr_part=best[False], guard=True, N=N, fill=fill,
                           k=k, box=None), v))
    dflt = next(v for c, v in cands if all(c[x] == DEFAULT[x] for x in DEFAULT))
    combo_c, combo_v = max(cands, key=lambda cv: cv[1].mean())
    if boxes is not None:                                       # rectangles sur le défaut et sur le meilleur combo
        confs = sorted({boxes[n]["conf"] for n in names if n in boxes})
        for base_c in (dict(DEFAULT), dict(combo_c)):
            for ct in confs:
                c = dict(base_c, box=ct)
                v = np.array([miou_image(predict(probs[n], reg[n], c, boxes, n), gt[n]) for n in names])
                cands.append((c, v))
    return names, cands, dflt


def predict(p, full, c, boxes, name):
    thr = c["thr"] if c["thr"] != "regime" else (c["thr_full"] if full else c["thr_part"])
    m = post(p, full, thr, c["guard"], c["N"], c["fill"], c["k"])
    return apply_boxes(m, p, full, boxes, name, c["box"], c["k"])


def one_at_a_time(cands):
    """Variantes qui ne changent qu'une option par rapport au défaut (pour afficher les gains)."""
    out = []
    for c, v in cands:
        diff = [x for x in DEFAULT if c.get(x) != DEFAULT[x]]
        if len(diff) == 1 or (c["thr"] == "regime" and diff == ["thr"]):
            out.append((c, v))
    return out


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(TASK / "t1/mgr/sources.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxdir", default=str(TASK / "t1/out/boxsearch"))
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--log", default=str(Path(os.environ.get("T1W", str(Path.home() / "t1w"))) / "RESULTS.tsv"))
    a = ap.parse_args()
    cfg = {k: v for k, v in json.loads(Path(a.sources).read_text()).items() if not k.startswith("_")}
    meta, gt = load_train()
    meth_tr = {n: m["method"] for n, m in meta.items()}
    tm = json.loads((TASK / "release/test/metadata.json").read_text())
    tnames = [m["image_name"] for m in tm]; tmeth = {m["image_name"]: m["method"] for m in tm}
    assert len(tnames) == len(set(tnames)) == 1020
    reg, treg = load_regimes("train"), load_regimes("test")
    boxdir = Path(a.boxdir)

    chosen, report, tpred = {}, {}, {}
    for method in METHODS:
        print(f"\n=================== {method}")
        mt = [n for n in tnames if tmeth[n] == method]
        vb = load_boxes(boxdir / f"{method}_train_boxes.json"); tb = load_boxes(boxdir / f"{method}_test_boxes.json")
        if (vb is None) != (tb is None):
            print(f"  ! rectangles: un seul des fichiers train/test existe -> ignorés"); vb = tb = None
        elif vb is not None:
            print(f"  rectangles: {len(vb)} val, {len(tb)} test ({boxdir})")
        const = lambda n, r: (np.ones if r[n] else np.zeros)((128, 128), np.uint8)   # noqa: E731
        rows, best = [], None          # best = (delta vs constante, score, n, src, cfg)
        for src in cfg.get(method, []):
            files = list(src["val"]) + list(src["test"])
            miss = [f for f in files if not (TASK / f).exists()]
            if miss:
                print(f"  ! source {src['name']}: fichiers absents {miss} -> ignorée"); continue
            probs = load_val(src["val"], method, meth_tr)
            probs = {n: p for n, p in probs.items() if n in gt}
            if len(probs) == 0:
                print(f"  ! source {src['name']}: aucune image {method} en validation -> ignorée"); continue
            try:
                tprobs = load_test(src["test"], mt)
            except ValueError as e:
                print(f"  ! source {src['name']}: {e} -> ignorée"); continue
            names, cands, dflt = search(probs, reg, gt, vb)
            cst = np.array([miou_image(const(n, reg), gt[n]) for n in names])
            print(f"  -- source {src['name']}  (val n={len(names)}, test: moyenne de {len(src['test'])} fichier(s))")
            print(f"     {'constante par régime':58s} {cst.mean():.4f}  (delta vs défaut {cst.mean()-dflt.mean():+.4f})")
            print(f"     {'DÉFAUT ' + cfg_str(DEFAULT):58s} {dflt.mean():.4f}")
            for c, v in one_at_a_time(cands):
                if any(c[x] != DEFAULT[x] for x in DEFAULT):
                    print(f"     {cfg_str(c):58s} {v.mean():.4f}  gain {v.mean()-dflt.mean():+.4f}")
            bc, bv = max(cands, key=lambda cv: (cv[1].mean(), cv[0] == DEFAULT))
            print(f"     {'MEILLEUR combo ' + cfg_str(bc):58s} {bv.mean():.4f}  gain {bv.mean()-dflt.mean():+.4f}")
            if bv.mean() - dflt.mean() >= a.min_gain:
                sc, sv = bc, bv; why = f"gain {bv.mean()-dflt.mean():+.4f} >= {a.min_gain}"
            else:
                sc, sv = dict(DEFAULT), dflt; why = f"gain {bv.mean()-dflt.mean():+.4f} < {a.min_gain} -> défaut"
            delta = sv.mean() - cst.mean()
            print(f"     => retenu pour cette source : {cfg_str(sc)}  val {sv.mean():.4f} ({why}); "
                  f"vs constante {delta:+.4f}")
            rows.append(dict(source=src["name"], n_val=len(names), const=float(cst.mean()), default=float(dflt.mean()),
                             best_combo=float(bv.mean()), best_combo_cfg=cfg_str(bc), chosen_cfg=cfg_str(sc),
                             chosen=float(sv.mean()), delta_vs_const=float(delta)))
            if best is None or delta > best[0]:
                best = (delta, float(sv.mean()), len(names), src, sc, tprobs)
        # constante sur toutes les images de validation de la méthode disponibles (union des sources)
        if best is None or best[0] <= 0:
            vn = sorted({n for s in cfg.get(method, []) for f in s["val"] if (TASK / f).exists()
                         for n in [str(x) for x in np.load(TASK / f)["names"]] if meth_tr.get(n) == method})
            cs = float(np.mean([miou_image(const(n, reg), gt[n]) for n in vn])) if vn else float("nan")
            chosen[method] = dict(source="constante", cfg="constante par régime", val=cs, n_val=len(vn))
            tpred.update({n: const(n, treg) for n in mt})
            if best is not None:
                print(f"  aucune source ne bat la constante")
        else:
            _, sv, nv, src, sc, tprobs = best
            chosen[method] = dict(source=src["name"], cfg=cfg_str(sc), val=sv, n_val=nv, val_files=src["val"],
                                  test_files=src["test"], params=sc)
            tpred.update({n: predict(tprobs[n], treg[n], sc, tb, n) for n in mt})
        report[method] = rows
        print(f"  CHOIX {method}: {chosen[method]['source']} / {chosen[method]['cfg']}  "
              f"val {chosen[method]['val']:.4f} (n={chosen[method]['n_val']})")

    tot = sum(c["n_val"] for c in chosen.values())
    glob = sum(c["val"] * c["n_val"] for c in chosen.values()) / tot
    print("\n=================== RÉSUMÉ")
    print(f"{'méthode':12s} {'source':28s} {'val':>7s} {'n':>4s}  réglage")
    for m in METHODS:
        c = chosen[m]
        print(f"{m:12s} {c['source']:28s} {c['val']:7.4f} {c['n_val']:4d}  {c['cfg']}")
    print(f"score de validation global attendu (pondéré) : {glob:.4f}")

    # ---------------- écriture + vérifications
    masks = torch.from_numpy(np.stack([tpred[n] for n in tnames]).astype(bool))
    assert masks.dtype == torch.bool and tuple(masks.shape) == (1020, 128, 128)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": tnames, "masks": masks}, out)
    size = os.path.getsize(out); assert size <= 20 * 1000 * 1000, size
    chk = torch.load(out, weights_only=True)
    assert chk["image_names"] == tnames and torch.equal(chk["masks"], masks)
    print("\n=================== garde-fous test")
    for m in METHODS:
        idx = [i for i, n in enumerate(tnames) if tmeth[n] == m]
        cov = masks[idx].float().mean(dim=(1, 2))
        empty = int(((cov == 0) | (cov == 1)).sum())
        print(f"test {m:10s}: n={len(idx)} couverture moy={cov.mean():.3f} part zone majoritaire={float((cov >= 0.5).float().mean()):.3f}"
              f" (régime full décodé {np.mean([treg[tnames[i]] for i in idx]):.3f}) minoritaire vide={empty}"
              + ("  ATTENTION" if empty else ""))
    print(f"écrit {out}: {size/1e6:.2f} Mo (NON envoyé)")
    info = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), file=str(out), sources_json=a.sources,
                boxdir=str(boxdir), min_gain=a.min_gain, chosen=chosen, val_global=glob, table=report)
    out.with_suffix(".json").write_text(json.dumps(info, indent=1, default=str))
    if not a.no_log:
        log = Path(a.log); log.parent.mkdir(parents=True, exist_ok=True)
        cols = ["date", "task", "file", "src_stegastamp", "src_trustmark", "src_pixelseal",
                "val_stegastamp", "val_trustmark", "val_pixelseal", "val_global", "public"]
        if not log.exists():
            log.write_text("\t".join(cols) + "\n")
        row = [info["date"], "t1", out.name] + [f"{chosen[m]['source']} [{chosen[m]['cfg']}]" for m in METHODS] \
            + [f"{chosen[m]['val']:.4f}" for m in METHODS] + [f"{glob:.4f}", ""]
        with log.open("a") as f:
            f.write("\t".join(row) + "\n")
        print(f"ligne ajoutée à {log}")


if __name__ == "__main__":
    main()
