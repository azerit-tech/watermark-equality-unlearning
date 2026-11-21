"""apply_sam.py — remplace GrabCut (TM/PS) / le masque seuillé (SG) par une silhouette SAM sur les images routées
de la soumission sub18p (régime minoritaire décidé par la chaîne, garde-fou non déclenché, masque non rectangulaire
(rectness < 0,9), AUCUN rectangle décodé appliqué : les rectangles priment). Candidats SAM identiques à ~/samtp/exp_sam.py.

  python apply_sam.py compute fold0,test     -> ~/t1w/sam/cands_<split>.npz (reprenable : relancer la même commande)
  python apply_sam.py write  RULE            -> ~/t1w/sam/<methode>/{fold0,test}_probs128.npz
  python apply_sam.py eval   RULE            -> fold 0 : SAM vs sub18p (global, méthode, type, bootstrap)
RULE ex. "trustmark=auto_union,pixelseal=pick,stegastamp=auto_union" (variantes : pick, auto_union, box_main, box_all).
Lancer depuis TASK, .venv actif, CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 [NT=threads]."""
import json, os, sys, time
from pathlib import Path
import numpy as np

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
HOME = Path.home()
sys.path[:0] = [str(HOME / "samtp"), str(HOME / "t1w/sub16/code")]
import chain_integ as C          # chaîne de sub16/sub18p, importée telle quelle (build_map, rect128, read_img)
import refine as R
from rule import decide, THR
from wmloc.utils import miou_image

OUT = HOME / "t1w" / "sam"
METHODS = ["stegastamp", "trustmark", "pixelseal"]
SUB = {"stegastamp": HOME / "t1w/sub16/chain/stegastamp",                       # sorties de chaîne de sub18p (sources.json)
       "trustmark": HOME / "t1w/sub18p/work/chain_trustmark/trustmark",
       "pixelseal": HOME / "t1w/sub18p/work/chain_pixelseal/pixelseal"}
CFG = json.loads((HOME / "t1w/sub18p/config.json").read_text())["methods"]
VARS = ["box_main", "box_all", "auto_union", "pick"]


def P_(p):
    return Path(os.path.expanduser(str(p).replace("$TASK", str(TASK))))


def load_boxes(path):
    d = json.loads(P_(path).read_text()); d = d["boxes"] if "boxes" in d else d
    out = {}
    for n, b in d.items():
        r = b.get("rect") if b.get("rect") is not None else b.get("box")
        if r is not None:
            out[str(n)] = {"rect": [int(v) for v in r], "conf": float(b.get("conf", 0.0))}
    return out


def load_npz(p):
    z = np.load(p); return [str(n) for n in z["names"]], z["probs"]


def regs():
    reg = {}
    for sp, tag in (("fold0", "train"), ("test", "test")):
        reg[sp] = {}
        for mm in THR:
            zz = np.load(TASK / "t1/out/signal" / f"fullacc_{mm}_{tag}.npz")
            reg[sp].update({str(n): float(x) for n, x in zip(zz["names"], zz["acc"])})
    return reg


def route(method, split, reg):
    """-> names, sub18p masks, map P, list of routed indices ; vérifie la reconstruction de la chaîne."""
    ens = str(P_(CFG[method]["ens"]))
    C.MAPS = {m: [(w, [f.replace("{ens}", ens) for f in fs]) for w, fs in v] for m, v in C.MAPS_NEW.items()}
    C.STTA = "{ens}/stegastamp/{split}_tta_probs128.npz".replace("{ens}", ens)
    C.REG.update(reg)
    names, sub = load_npz(SUB[method] / f"{split}_probs128.npz")
    assert names == sorted(C.load(C.MAPS[method][0][1][0].format(split=split))), (method, split)
    P = C.build_map(method, split, names)
    bx = load_boxes(CFG[method]["boxes_train" if split == "fold0" else "boxes_test"]); conf = float(CFG[method]["conf"])
    routed, chk = [], {"full_ok": 0, "full_bad": 0, "rect_ok": 0, "rect_bad": 0}
    for i, n in enumerate(names):
        mg, full = decide(P[i], reg[split][n], method)
        if full:
            k = "full_ok" if np.array_equal(sub[i].astype(np.uint8), mg) else "full_bad"; chk[k] += 1; continue
        b = bx.get(n)
        if b is not None and b["conf"] >= conf:
            r = C.rect128(b["rect"])
            if r.any() and r.mean() < 0.5:
                k = "rect_ok" if np.array_equal(sub[i].astype(np.uint8), r) else "rect_bad"; chk[k] += 1; continue
        m0, fixed = R.guard(P[i], False)
        if fixed or R.rectness(m0) >= 0.9:
            continue
        routed.append(i)
    print(f"{method} {split}: {len(names)} images, routées SAM {len(routed)}, contrôle chaîne {chk}", flush=True)
    return names, sub, P, routed


def sam_cands(sam, grid, p, img):
    import torch, cv2
    up512 = lambda a: cv2.resize(a, (512, 512), interpolation=cv2.INTER_LINEAR)
    to128 = lambda m: (R.pool128(m.astype(np.float32)) >= 0.5).astype(np.uint8)
    nn512 = lambda a: cv2.resize(a.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)

    def greedy_union(cands, target):
        prec = [(c & target).sum() / max(c.sum(), 1) for c in cands]
        cur = np.zeros_like(target); best = 0.0
        for i in np.argsort(prec)[::-1]:
            nxt = cur | cands[i]; v = iou(nxt, target)
            if v > best:
                cur, best = nxt, v
        return cur
    m0, _ = R.guard(p, False)
    tb = up512(p) >= 0.5
    if not tb.any():
        tb = nn512(m0)
    mc = nn512(R.main_component(m0))
    e = sam.embed(img); cand = {}
    for nm, src in (("box_main", mc), ("box_all", tb)):
        ys, xs = np.nonzero(src)
        lg, _ = sam.predict(e, boxes=[[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]])
        ms = (lg[0] > 0).numpy(); cand[nm] = ms[int(np.argmax([iou(a, tb) for a in ms]))]
    lg, pi = sam.predict(e, points=grid[:, None, :], labels=np.ones((len(grid), 1)), size=256)
    lg = lg.reshape(-1, 256, 256); pi = pi.numpy().reshape(-1)
    sel = [i for i in range(len(pi)) if pi[i] >= 0.88 and 0 < (lg[i] > 0).float().mean() < 0.5]
    ok = list((torch.nn.functional.interpolate(lg[sel][:, None], size=(512, 512), mode="bilinear")[:, 0] > 0).numpy()) if sel else []
    cand["auto_union"] = greedy_union(ok, tb) if ok else tb
    allc = [cand["box_main"], cand["box_all"], cand["auto_union"]]
    cand["pick"] = allc[int(np.argmax([iou(a, tb) for a in allc]))]
    out = {}
    for k in VARS:
        mk = to128(cand[k]); out[k] = mk if (mk.any() and mk.mean() < 0.5) else np.full((128, 128), 255, np.uint8)
    return out


def compute(splits):
    import torch, cv2
    from minisam import MiniSAM
    nt = int(os.environ.get("NT", 8)); torch.set_num_threads(nt); cv2.setNumThreads(nt)
    reg = regs(); OUT.mkdir(exist_ok=True)
    sam = MiniSAM()
    gy, gx = np.mgrid[16:512:32, 16:512:32]
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    for split in splits:
        f = OUT / f"cands_{split}.npz"
        done = {}
        if f.exists():
            z = np.load(f); done = {k: {v: z[v][i] for v in VARS} for i, k in enumerate(z["keys"].tolist())}
        todo = []
        for m in METHODS:
            names, _, P, routed = route(m, split, reg)
            todo += [(m, names[i], P[i]) for i in routed]
        allk = [f"{m}|{n}" for m, n, _ in todo]
        print(f"{split}: {len(todo)} routées, {sum(k in done for k in allk)} déjà faites", flush=True)
        t0 = time.time(); j = 0

        def save():
            ks = sorted(done); tmp = OUT / f"cands_{split}.tmp.npz"
            np.savez_compressed(tmp, keys=np.array(ks), all_routed=np.array(allk),
                                **{v: np.stack([done[k][v] for k in ks]) if ks else np.zeros((0, 128, 128), np.uint8) for v in VARS})
            os.replace(tmp, f)
        for m, n, p in todo:
            k = f"{m}|{n}"
            if k in done:
                continue
            done[k] = sam_cands(sam, grid, p, C.read_img(split, n)); j += 1
            if j % 10 == 0:
                save(); print(f"{split} {len(done)}/{len(allk)} {time.time() - t0:.0f}s", flush=True)
        save(); (OUT / f"DONE_{split}").write_text(f"{len(done)}/{len(allk)}\n")
        print(f"{split} fini {len(done)}/{len(allk)} {time.time() - t0:.0f}s", flush=True)


def parse_rule(s):
    r = dict(x.split("=") for x in s.split(","))
    assert set(r) == set(METHODS) and set(r.values()) <= set(VARS), r
    return r


def build(split, rule, reg):
    """-> {method: (names, sub18p, new, routed_idx, replaced_idx)}"""
    z = np.load(OUT / f"cands_{split}.npz"); keys = z["keys"].tolist(); allk = z["all_routed"].tolist()
    pos = {k: i for i, k in enumerate(keys)}; res = {}
    for m in METHODS:
        names, sub, _, routed = route(m, split, reg)
        new = sub.copy(); rep, miss = [], []
        for i in routed:
            k = f"{m}|{names[i]}"; assert k in allk
            if k not in pos:
                miss.append(i); continue
            c = z[rule[m]][pos[k]]
            if c.max() <= 1:
                new[i] = c.astype(new.dtype); rep.append(i)
        print(f"  {m} {split}: routées {len(routed)}, remplacées {len(rep)}, SAM invalide {len(routed) - len(rep) - len(miss)}, "
              f"NON CALCULÉES {len(miss)}", flush=True)
        res[m] = (names, sub, new, routed, rep, miss)
    return res


def write(rule):
    reg = regs()
    for split in ("fold0", "test"):
        if not (OUT / f"cands_{split}.npz").exists():
            print("pas de candidats", split); continue
        for m, (names, _, new, _, _, miss) in build(split, rule, reg).items():
            d = OUT / m; d.mkdir(exist_ok=True)
            np.savez(d / f"{split}_probs128.npz", names=np.array(names), probs=new.astype(np.float16))
            if miss:
                print(f"  ATTENTION {m} {split}: {len(miss)} images routées non calculées -> sortie sub18p gardée", flush=True)
    (OUT / "rule.json").write_text(json.dumps(rule))


def boot(d, B=10000, seed=0):
    bm = d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)


def evaluate(rule):
    reg = regs()
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
    tidx = json.loads((TASK / "t1/common/train_index.json").read_text())
    tidx = {e["image_name"]: e for e in tidx} if isinstance(tidx, list) else tidx
    res = build("fold0", rule, reg); D = {}
    for m, (names, sub, new, routed, rep, _) in res.items():
        o = np.array([miou_image(p.astype(np.uint8), gt[n]) for n, p in zip(names, sub)])
        w = np.array([miou_image(p.astype(np.uint8), gt[n]) for n, p in zip(names, new)])
        et = np.array([tidx[n]["edit_type"] for n in names]); D[m] = w - o
        bt = " ".join(f"{t}={o[et == t].mean():.3f}->{w[et == t].mean():.3f}" for t in ("hole", "transplant", "outpainting"))
        rt = {t: int(sum(et[i] == t for i in rep)) for t in ("transplant", "outpainting", "hole")}
        print(f"{m:10s} sub18p {o.mean():.4f} -> SAM {w.mean():.4f}  écart {(w - o).mean():+.4f} IC95 [{boot(w - o)[1]:+.4f}, {boot(w - o)[2]:+.4f}] "
              f"| {bt} | remplacées {len(rep)} {rt}")
        tr = et == "transplant"
        print(f"{'':10s} transplants : écart {boot((w - o)[tr])[0]:+.4f} IC95 [{boot((w - o)[tr])[1]:+.4f}, {boot((w - o)[tr])[2]:+.4f}]")
    rng = np.random.default_rng(1); B = 10000; bm = np.zeros(B)
    for m in D:
        bm += D[m][rng.integers(0, len(D[m]), (B, len(D[m])))].mean(1)
    bm /= len(D); g = np.mean([D[m].mean() for m in D])
    so = np.mean([miou_image(p.astype(np.uint8), gt[n]) for m in res for n, p in zip(res[m][0], res[m][1])])
    print(f"GLOBAL fold 0 : sub18p {so:.4f} -> SAM {so + g:.4f}  écart {g:+.4f} IC95 [{np.percentile(bm, 2.5):+.4f}, {np.percentile(bm, 97.5):+.4f}]")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compute":
        compute(sys.argv[2].split(","))
    elif cmd == "write":
        write(parse_rule(sys.argv[2]))
    elif cmd == "eval":
        evaluate(parse_rule(sys.argv[2]))
