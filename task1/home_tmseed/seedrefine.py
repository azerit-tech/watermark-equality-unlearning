"""tmseed: départs multi-cartes pour la recherche de rectangles TrustMark (même idée que ~/sgseed, ~/psseed) :
boîtes englobantes des 2 composantes principales de nombreuses cartes de probabilité (seuils 0.5/0.35), dédoublonnées,
décodées une fois (crop R+3, 245 bilinéaire, décodeur boxsearch2), top-K raffinés (rectsearch.refine local, budget PER
décodages par départ), seuil 0.79 inchangé ; puis bords raffinés tmedge (2 passes, courbe de marge, alpha 0.95, biais -1),
SANS le décalage fixe tmedge2.
usage (depuis TASK) : python ~/tmseed/seedrefine.py fold0|test -> ~/tmseed/seeds_{tag}.json"""
import sys, os, json, time
from pathlib import Path
import numpy as np, torch
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK / "t1/boxsearch2"), str(TASK / "t1/tmedge")]
import rectsearch as RS  # noqa
from cv import apply as edge_apply  # noqa  (tmedge)
from scipy.ndimage import label
from PIL import Image
OUT = Path.home() / "tmseed"; T = TASK / "t1/out"
PER, K, MINA = int(os.environ.get("PER", 60)), int(os.environ.get("K", 3)), 16
DEADLINE = float(os.environ.get("DEADLINE", 1e18)); A = 2; THR = 0.79
RULE = ("mar", 0.95, -1, 99999); R_EDGE = 12


def maps(tag):
    fs = [T / f"tiles2/trustmark/{tag}{s}_probs128.npz" for s in ("", "_s192", "_s256", "_s320", "_s384", "_smean")]
    fs += [T / f"tta/trustmark/{tag}{s}_probs128.npz" for s in ("", "_base", "_t320_g0", "_t320_g1", "_t320_g2", "_t320_g3")]
    fs += [T / f"jacques3/informed_trustmark/{tag}_probs128.npz", T / f"tmdense/best/{tag}_probs128.npz"]
    if tag == "test":
        fs += [T / f"devel/trustmark/fold{k}/pred/test{s}_probs128.npz" for k in (1, 2, 3, 4) for s in ("", "_t320")]
        fs += [T / f"jacques/tiles_trustmark_s1/test{s}_probs128.npz" for s in ("", "_t256", "_t320")]
    out = []
    for f in fs:
        if f.exists():
            z = np.load(f); out.append(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
    print(tag, "maps", len(out), [str(f.relative_to(T)) for f in fs if f.exists()], flush=True)
    return out


def boxes_of(p, thr):
    lab, n = label(p >= thr, structure=np.ones((3, 3)))
    if n == 0: return []
    ar = np.bincount(lab.ravel())[1:]; order = np.argsort(-ar); bs = []
    for j in order[:2]:
        if ar[j] < MINA or ar[j] < 0.1 * ar[order[0]]: continue
        ys, xs = np.nonzero(lab == j + 1)
        bs.append((4 * int(ys.min()), 4 * int(ys.max() + 1), 4 * int(xs.min()), 4 * int(xs.max() + 1)))
    return bs


def starts_of(nm, M):
    S = []
    for thr in (0.5, 0.35):
        for m in M:
            if nm not in m: continue
            for b in boxes_of(m[nm], thr):
                if all(max(abs(b[i] - s[i]) for i in range(4)) > 4 for s in S): S.append(b)
    return [s for s in S if s[1] - s[0] >= 8 and s[3] - s[2] >= 8 and (s[1] - s[0]) * (s[3] - s[2]) < 0.6 * 512 * 512]


def refine(D, x, msg, r, obj, stop, max_it=12):   # = rectsearch.refine + budget de décodages
    Aeff = A if RS.ok(r, A) else 999
    for step in (16, 8, 4, 2, 1):
        for _ in range(max_it):
            if D.n >= stop: return r, obj
            c = []
            for e in range(4):
                for d in (-step, step):
                    q = list(r); q[e] += d; c.append(tuple(q))
            for (a, b) in ((0, 1), (2, 3)):
                for d in (-step, step):
                    q = list(r); q[a] += d; q[b] += d; c.append(tuple(q))
            c = [q for q in c if RS.ok(q, Aeff)]
            if not c: break
            o, _, _ = D.score(x, c, msg); j = int(np.argmax(o))
            if o[j] <= obj: break
            r, obj = c[j], o[j]
    return r, obj


def moved(r, e, d):   # = tmedge/sweep.moved
    q = list(r); q[e] = q[e] - d if e in (0, 2) else q[e] + d
    q[0] = max(0, q[0]); q[2] = max(0, q[2]); q[1] = min(512, q[1]); q[3] = min(512, q[3]); return tuple(q)


def edge_pass(D, x, msg, r):   # = tmedge/sweep.py (une image) puis mkrects (règle mar/0.95/-1)
    ds = list(range(-R_EDGE, R_EDGE + 1))
    cand = [moved(r, e, d) for e in range(4) for d in ds]
    ok = [q[1] - q[0] >= 8 and q[3] - q[2] >= 8 for q in cand]
    obj, acc, mar = D.score(x, [q if o else tuple(r) for q, o in zip(cand, ok)], msg)
    acc = np.where(ok, acc, -1); mar = np.where(ok, mar, -99); obj = np.where(ok, obj, -1)
    v = {"rect": list(map(int, r)), "ds": ds, "acc": acc.reshape(4, -1).tolist(), "obj": obj.reshape(4, -1).tolist(),
         "mar": mar.reshape(4, -1).tolist()}
    return [int(t) for t in edge_apply(v, *RULE)]


if __name__ == "__main__":
    tag = sys.argv[1]; split = "test" if tag == "test" else "train"
    torch.set_num_threads(int(os.environ.get("NT", 8)))
    D = RS.Dec(RS.load_watermark("trustmark", device="cpu"), "cpu"); D.bs = 128
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text())}
    M = maps(tag)
    if tag == "fold0":
        old = json.loads((T / "tmbox4/trustmark_train_boxes.json").read_text())["boxes"]
        f0 = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0]
        f0 = [n for n in f0 if meta[n]["method"] == "trustmark"]
        names = [n for n in f0 if idx[n]["edit_type"] == "outpainting" and old.get(n, {}).get("conf", 0) < THR]
        names += [n for n in f0 if idx[n]["edit_type"] == "transplant"]
        names += [n for n in f0 if idx[n]["edit_type"] == "outpainting" and n not in names] if os.environ.get("ALLOUT") else []
    else:
        old = json.loads((Path.home() / "t1w/final/trustmark_test_boxes_v3.json").read_text())["boxes"]
        z = np.load(T / "signal/fullacc_trustmark_test.npz"); fa = dict(zip(z["names"].tolist(), z["acc"].tolist()))
        names = [n for n, m in meta.items() if m["method"] == "trustmark" and fa[n] < 0.76
                 and not (old.get(n, {}).get("rect") and old[n].get("conf", 0) >= THR)]
        print("test todo", len(names), flush=True)
    names = names[int(os.environ.get("SKIP", 0)):]
    pi, pn = map(int, os.environ.get("PART", "0/1").split("/")); names = names[pi::pn]
    fout = OUT / f"seeds_{tag}{os.environ.get('SUF', '')}.json"; res = json.loads(fout.read_text()) if fout.exists() else {}
    with torch.inference_mode():
        for nm in names:
            if nm in res: continue
            if time.time() > DEADLINE: print("DEADLINE", flush=True); break
            t0 = time.time(); n0 = D.n
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
            S = starts_of(nm, M); r_ = {"n_starts": len(S)}
            if tag == "fold0": r_["edit_type"] = idx[nm]["edit_type"]
            if S:
                o, acc, _ = D.score(x, S, msg); order = np.argsort(-o); best = None
                for j in order[:K]:
                    r, ob = refine(D, x, msg, S[j], o[j], D.n + PER)
                    _, ac, _ = D.score(x, [r], msg)
                    if best is None or ac[0] > best[1]: best = (r, float(ac[0]), S[j])
                    if ac[0] >= 0.9: break
                r_.update(rect_seed=list(map(int, best[0])), conf=best[1], start=list(map(int, best[2])), start_acc_max=float(acc.max()))
                if best[1] >= THR:   # raffinement de bords tmedge, 2 passes, sans décalage tmedge2
                    r1 = edge_pass(D, x, msg, best[0]); r2 = edge_pass(D, x, msg, r1)
                    _, ac2, mar2 = D.score(x, [tuple(r2)], msg)
                    r_.update(rect_p1=r1, rect=r2, conf_edge=float(ac2[0]), margin_edge=float(mar2[0]))
                if tag == "fold0" and idx[nm].get("bbox512"):
                    r_["iou_seed"] = RS.iou(best[0], idx[nm]["bbox512"])
                    if "rect" in r_: r_["iou"] = RS.iou(r_["rect"], idx[nm]["bbox512"])
            r_.update(n_decodes=D.n - n0, sec=round(time.time() - t0, 1)); res[nm] = r_
            print(tag, nm, r_, flush=True); fout.write_text(json.dumps(res))
    (OUT / f"DONE_{tag}{os.environ.get('SUF', '')}").write_text("ok"); print("DONE", flush=True)
