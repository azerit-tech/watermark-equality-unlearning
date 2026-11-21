"""StegaStamp rectangle search with segmenter-map starts (from TASK).
python t1/boxsearch3/stega_seg.py --sets train,test --device cuda:0 [--folds 0,1] [--limit N]
Starts: bboxes of connected components (top 3 by prob mass) + union bbox of the StegaStamp segmenter maps at
thresholds 0.2/0.35/0.5/0.7, plus the boxsearch rect when it exists; each start expanded into a fine local grid
(each edge +-5%,+-10% of side, uniform grow/shrink), top-K distinct candidates refined by rectsearch.refine.
Images: minority regime (fullacc < 0.755) + all outpaintings of the given train folds; test: minority regime.
Writes t1/out/boxsearch3/stegastamp_<set>_seg.json"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rectsearch as RS  # noqa
from PIL import Image
TASK = Path(os.getcwd()); OUT = TASK / "t1/out/boxsearch3"; OUT.mkdir(parents=True, exist_ok=True)
M = "stegastamp"; A = RS.ASPECT[M]


def load_maps(files):
    out = []
    for f in files:
        f = TASK / f
        if f.exists():
            z = np.load(f); out.append(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))); RS.log("map", f, len(out[-1]))
    return out


def seg_starts(maps):
    st = []
    for p in maps:
        for t in (0.5, 0.35, 0.7, 0.2):
            m = p > t
            if m.sum() < 2 or m.mean() > 0.6: continue
            lab, n = ndimage.label(m, structure=np.ones((3, 3)))
            sl = ndimage.find_objects(lab); mass = ndimage.sum(p, lab, range(1, n + 1))
            cand = [sl[i] for i in np.argsort(-np.atleast_1d(mass))[:3]]
            ys, xs = np.nonzero(m); cand.append((slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)))
            for s in cand:
                st.append((s[0].start * 4, s[0].stop * 4, s[1].start * 4, s[1].stop * 4))
    return st


def local_grid(r):
    h, w = r[1] - r[0], r[3] - r[2]; dy, dx = max(2, round(0.05 * h)), max(2, round(0.05 * w)); out = [tuple(r)]
    for e, d in ((0, dy), (1, dy), (2, dx), (3, dx)):
        for k in (-2, -1, 1, 2):
            q = list(r); q[e] += k * d; out.append(tuple(q))
    for k in (-1, 1):
        out.append((r[0] - k * dy, r[1] + k * dy, r[2] - k * dx, r[3] + k * dx))
    return [q for q in out if RS.ok(q, 999)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sets", default="train,test"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--folds", default="0,1"); ap.add_argument("--K", type=int, default=4); ap.add_argument("--maxstarts", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--skipconf", type=float, default=1.01); ap.add_argument("--tag", default="")
    a = ap.parse_args(); torch.set_num_threads(8)
    wm = RS.load_watermark(M, device=a.device); RS.log("providers", wm.sess.get_providers()); D = RS.Dec(wm, a.device); D.bs = 96
    idx = json.loads((TASK / "t1/common/train_index.json").read_text()); folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    for S in a.sets.split(","):
        meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{S}/metadata.json").read_text()) if m["method"] == M}
        fa = np.load(TASK / f"t1/out/signal/fullacc_{M}_{S}.npz"); facc = dict(zip(fa["names"].tolist(), fa["acc"].tolist()))
        old = json.loads((TASK / f"t1/out/boxsearch/{M}_{S}_boxes.json").read_text())["boxes"]
        if S == "train":
            todo, mapsets = [], {}
            for k in map(int, a.folds.split(",")):
                files = [f"t1/out/train/v1/stegastamp_f{k}/fold_probs128.npz"] + (["t1/out/stega/best/fold0_probs128.npz", "t1/out/train/v0/fold0_probs128.npz"] if k == 0 else [])
                ms = load_maps(files)
                for n in folds[k]:
                    if n in meta and (facc[n] < RS.THR[M] or idx[n]["edit_type"] == "outpainting"): todo.append(n); mapsets[n] = ms
        else:
            ms = load_maps(["t1/out/stega/best/test_probs128.npz", "t1/out/train/v0/test_probs128.npz"])
            todo = [n for n in meta if facc[n] < RS.THR[M]]; mapsets = {n: ms for n in todo}
        if a.limit: todo = todo[:a.limit]
        fout = OUT / f"{M}_{S}_seg{a.tag}.json"; res = json.loads(fout.read_text()) if fout.exists() else {}
        RS.log(S, len(todo), "images")
        for i, nm in enumerate(todo):
            if nm in res: continue
            if nm in old and old[nm]["conf"] >= a.skipconf: continue
            t0 = time.time(); n0 = D.n
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{S}/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
            st = seg_starts([m[nm] for m in mapsets[nm] if nm in m])
            if nm in old: st = [tuple(old[nm]["rect"])] + st
            uniq = []
            for r in st:
                if RS.ok(r, 999) and all(RS.iou(r, u) < 0.92 for u in uniq): uniq.append(r)
            uniq = uniq[:a.maxstarts] or [(192, 320, 192, 320)]
            C = list(dict.fromkeys(q for r in uniq for q in local_grid(r)))
            o = D.score(x, C, msg)[0]; best = None; sel = []
            for j in np.argsort(-o):
                if all(RS.iou(C[j], s) < 0.5 for s, _ in sel): sel.append((C[j], o[j]))
                if len(sel) >= a.K: break
            for s, o0 in sel:
                r, ob = RS.refine(D, x, msg, s, o0, 999)
                if best is None or ob > best[1]: best = (r, ob)
            r = best[0]; _, acc, mar = D.score(x, [r], msg)
            v = {"rect": list(map(int, r)), "conf": float(acc[0]), "margin": float(mar[0]), "n_starts": len(uniq), "n_decodes": D.n - n0, "sec": time.time() - t0, "fullacc": facc[nm]}
            if S == "train":
                v["edit_type"] = idx[nm]["edit_type"]; v["bbox512"] = idx[nm]["bbox512"]; v["iou"] = RS.iou(r, idx[nm]["bbox512"]) if idx[nm]["bbox512"] else 0.0
                v["fold"] = next(k for k, f in enumerate(folds) if nm in f)
            if nm in old: v["old_conf"] = old[nm]["conf"]; v["old_rect"] = old[nm]["rect"]
            res[nm] = v
            RS.log(S, i, nm, v.get("edit_type"), r, f"acc {v['conf']:.3f} (old {v.get('old_conf', -1):.3f})", f"iou {v.get('iou', -1):.3f}", f"{v['sec']:.1f}s {v['n_decodes']} starts {len(uniq)}")
            if i % 5 == 0: fout.write_text(json.dumps(res))
        fout.write_text(json.dumps(res))
    (OUT / f"{M}_seg{a.tag}.done").write_text("ok"); RS.log("DONE")
