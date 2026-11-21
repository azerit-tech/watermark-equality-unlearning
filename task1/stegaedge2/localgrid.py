"""StegaStamp: local grid around the top-K perimeter candidates of t1/out/stegaedge/edge_<set>.json, then refine, fixed thr 0.9.
From TASK: python t1/stegaedge2/localgrid.py --set train --device cuda:1 [--K 10] [--limit N]
Grid per candidate (h,w = its sides): joint grids of the 4 edges at offsets {-12,0,+12}% and {-6,0,+6}% of the side (2x81, includes
translations; the full 5^4 grid costs 36 s/image), plus single-edge moves at +-9% and pair moves (both edges of one axis) at +-9%; 'ext' = single-edge moves at
+-18,+-24,+-30% (reported separately). Best grid point (all candidates) and the best 'spec'-only point are refined
(steps 4,2,1 px). conf = bit acc of the crop R+3. Writes t1/out/stegaedge2/grid_<set>.json"""
import torch  # before onnxruntime
import argparse, itertools, json, os, sys, time
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch3")); sys.path.insert(0, str(TASK / "t1/stegaedge2"))
from PIL import Image
import edgecand as EC  # copied refine/iou/ok
OUT = TASK / "t1/out/stegaedge2"; OUT.mkdir(parents=True, exist_ok=True)


def clipr(q):
    q = (max(0, q[0]), min(512, q[1]), max(0, q[2]), min(512, q[3]))
    return q if EC.ok(q) else None


def local_grid(c):
    h, w = c[1] - c[0], c[3] - c[2]; spec, ext = set(), set()
    for O in ((-0.12, 0, 0.12), (-0.06, 0, 0.06)):   # two joint 3^4 grids (81 each; include translations)
        for a, b, d, e in itertools.product(O, O, O, O):
            q = clipr((c[0] + round(a * h), c[1] + round(b * h), c[2] + round(d * w), c[3] + round(e * w)))
            if q: spec.add(q)
    for f in (-0.09, 0.09):
        for k in range(4):
            q = list(c); q[k] += round(f * (h if k < 2 else w)); q = clipr(q)
            if q: spec.add(q)
        for (i, j, s) in ((0, 1, h), (2, 3, w)):
            q = list(c); q[i] += round(f * s); q[j] += round(f * s); q = clipr(q)
            if q: spec.add(q)
    for f in (-0.30, -0.24, -0.18, 0.18, 0.24, 0.30):
        for k in range(4):
            q = list(c); q[k] += round(f * (h if k < 2 else w)); q = clipr(q)
            if q and q not in spec: ext.add(q)
    return sorted(spec), sorted(ext)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="train"); ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--K", type=int, default=10); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--tag", default=""); ap.add_argument("--skipfound", action="store_true")  # skip images already >= 0.9 in t1/out/stegaedge
    a = ap.parse_args(); torch.set_num_threads(8)
    split = "test" if a.set == "test" else "train"
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text()) if m["method"] == "stegastamp"}
    ed = json.loads((TASK / f"t1/out/stegaedge/edge_{a.set}.json").read_text())["boxes"]
    idx = json.loads((TASK / "t1/common/train_index.json").read_text()) if split == "train" else {}
    todo = list(ed)
    if a.skipfound:
        pv = json.loads((TASK / f"t1/out/stegaedge/stegastamp_{split}_boxes.json").read_text())["boxes"]
        todo = [n for n in todo if pv.get(n, {}).get("conf", 0) < 0.9]
    todo = todo[a.start:]
    if a.limit: todo = todo[:a.limit]
    import rectsearch as RS
    wm = RS.load_watermark("stegastamp", device=a.device); D = RS.Dec(wm, a.device); D.bs = 128
    res = {}; fout = OUT / f"grid_{a.set}{a.tag}.json"; t00 = time.time()
    for k, nm in enumerate(todo):
        t0 = time.time(); img = np.asarray(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))
        x = torch.from_numpy(np.array(img)).to(a.device).permute(2, 0, 1)[None].float() / 255
        msg = np.array([int(ch) for ch in meta[nm]["message"]], np.float64); n0 = D.n
        C = [tuple(c) for c in ed[nm]["cands"][:a.K]]
        S, E = set(C), set()
        for c in C:
            s_, e_ = local_grid(c); S |= set(s_); E |= set(e_)
        E -= S; S, E = sorted(S), sorted(E)
        o, acc, _ = D.score(x, S + E, msg); oS, oE = o[:len(S)], o[len(S):]
        r_ = {"n_spec": len(S), "n_ext": len(E), "grid_acc_spec": float(acc[:len(S)].max()), "grid_acc_ext": float(acc[len(S):].max()) if len(E) else 0}
        jS = int(np.argmax(oS)); rS, obS = EC.refine(D, x, msg, S[jS], oS[jS], steps=(4, 2, 1))
        _, aS, _ = D.score(x, [rS], msg); r_.update(rect_spec=list(map(int, rS)), conf_spec=float(aS[0]))
        best = (rS, float(aS[0]))
        if len(E) and oE.max() > oS[jS]:
            jE = int(np.argmax(oE)); rE, obE = EC.refine(D, x, msg, E[jE], oE[jE], steps=(4, 2, 1))
            _, aE, _ = D.score(x, [rE], msg); r_.update(rect_ext=list(map(int, rE)), conf_ext=float(aE[0]))
            if aE[0] > best[1]: best = (rE, float(aE[0]))
        r_.update(rect=list(map(int, best[0])), conf=best[1], n_decodes=D.n - n0, sec=time.time() - t0)
        if split == "train":
            et = idx[nm]["edit_type"]; r_["edit_type"] = et
            if et == "outpainting":
                g = idx[nm]["bbox512"]; r_["gt"] = g; r_["iou"] = EC.iou(best[0], g); r_["iou_spec"] = EC.iou(rS, g)
        res[nm] = r_
        print(k, nm, r_.get("edit_type"), r_.get("gt"), "spec %.2f->%.2f" % (r_["grid_acc_spec"], r_["conf_spec"]),
              "ext %.2f->%.2f" % (r_["grid_acc_ext"], r_.get("conf_ext", -1)), r_["rect"], "iou %.2f" % r_.get("iou", -1),
              r_["n_decodes"], "%.1fs" % r_["sec"], flush=True)
        if k % 5 == 0 or k == len(todo) - 1:
            fout.write_text(json.dumps({"format": "rect=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px", "method": "stegastamp", "boxes": res}))
    (OUT / f"grid_{a.set}{a.tag}.done").write_text("ok %.0fs" % (time.time() - t00)); print("DONE", time.time() - t00)
