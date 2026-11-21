"""StegaStamp outpainting rectangle candidates from the HF-residual energy map (no segmenter), verified by decoding.
From TASK:  python t1/stegaedge/edgecand.py --set train --device cuda:1 [--diag] [--K 40]
--cand perim (default): every cell-aligned rect (side >= 2 cells, aspect <= 16) scored by the perimeter sign test of
perim.py (sign of log(E_in/E_out) across each side, E = sum_rgb |img - median3x3|); --cand area: old in-vs-ring contrast
(useless: GT median rank 1e7). Top-K after NMS (IoU < 0.5). Verification: crop R+3 decoded for each candidate,
top-3 distinct refined (steps 8,4,2,1 px per edge), kept if acc >= 0.9 (fixed).
Writes t1/out/stegaedge/edge_<set>.json"""
import torch  # before onnxruntime
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch3")); sys.path.insert(0, str(TASK / "t1/stegaedge"))
import perim as PM  # noqa
OUT = TASK / "t1/out/stegaedge"; OUT.mkdir(parents=True, exist_ok=True)
RING, MINS, ASP = 2, 2, 16


def emap(img):
    x = img.astype(np.float32)
    e = np.abs(x - median_filter(x, size=(3, 3, 1))).sum(2)
    e = gaussian_filter(e, 1.0)
    return np.log(e.reshape(128, 4, 128, 4).mean((1, 3)) + 0.5)


def score_all(L, dev, pw):
    """returns score matrix over (y-pair, x-pair) and pair lists"""
    Lt = torch.tensor(L, device=dev, dtype=torch.float64); s = Lt.std()
    II = torch.zeros(129, 129, device=dev, dtype=torch.float64); II[1:, 1:] = Lt.cumsum(0).cumsum(1)
    pr = [(a, b) for a in range(128) for b in range(a + MINS, 129)]
    P = torch.tensor(pr, device=dev)
    a, b = P[:, 0], P[:, 1]; ao, bo = (a - RING).clamp(min=0), (b + RING).clamp(max=128)
    Ain = II[b] - II[a]; Aout = II[bo] - II[ao]                       # [P,129]
    c, d = P[:, 0], P[:, 1]; co, do = (c - RING).clamp(min=0), (d + RING).clamp(max=128)
    hin = (b - a).double(); hout = (bo - ao).double(); win = (d - c).double(); wout = (do - co).double()
    S = torch.empty(len(pr), len(pr), device=dev, dtype=torch.float32)
    for i in range(0, len(pr), 1024):
        sl = slice(i, i + 1024)
        sin = Ain[sl][:, d] - Ain[sl][:, c]; sout = Aout[sl][:, do] - Aout[sl][:, co]
        nin = hin[sl, None] * win[None]; nout = hout[sl, None] * wout[None]
        nr = nout - nin; ring = (sout - sin) / nr.clamp(min=1)
        t = (sin / nin - ring) / s * torch.sqrt(nin * nr / (nin + nr).clamp(min=1)) ** pw
        asp = torch.maximum(hin[sl, None] / win[None], win[None] / hin[sl, None])
        t[(nr < 1) | (asp > ASP)] = -1e9
        S[sl] = t.float()
    return S, pr


def topk(S, pr, K, nms=0.5):
    v, ix = torch.topk(S.flatten(), 4000); ix = ix.cpu().numpy(); n = len(pr); out = []
    for j in ix:
        (a, b), (c, d) = pr[j // n], pr[j % n]; r = (4 * a, 4 * b, 4 * c, 4 * d)
        if all(iou(r, q) < nms for q in out): out.append(r)
        if len(out) >= K: break
    return out


def iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    i = iy * ix; return i / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - i)


def ok(r):
    return r[0] >= 0 and r[2] >= 0 and r[1] <= 512 and r[3] <= 512 and r[1] - r[0] >= 6 and r[3] - r[2] >= 6


def moves(r, step):
    c = []
    for e in range(4):
        for dd in (-step, step):
            q = list(r); q[e] += dd; c.append(tuple(q))
    return [q for q in c if ok(q)]


def refine(D, x, msg, r, obj, steps=(8, 4, 2, 1), max_it=10):
    for st in steps:
        for _ in range(max_it):
            c = moves(r, st)
            for (p, q_) in ((0, 1), (2, 3)):
                for dd in (-st, st):
                    q = list(r); q[p] += dd; q[q_] += dd
                    if ok(q): c.append(tuple(q))
            o, _, _ = D.score(x, c, msg); j = int(np.argmax(o))
            if o[j] <= obj: break
            r, obj = c[j], o[j]
    return r, obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="train"); ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--K", type=int, default=40); ap.add_argument("--pow", type=float, default=1.0)
    ap.add_argument("--diag", action="store_true"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--cand", default="perim")
    a = ap.parse_args(); torch.set_num_threads(8)
    split = "test" if a.set == "test" else "train"
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text()) if m["method"] == "stegastamp"}
    fz = np.load(TASK / f"t1/out/signal/fullacc_stegastamp_{split}.npz"); facc = dict(zip(fz["names"].tolist(), fz["acc"].tolist()))
    todo = sorted(n for n in meta if facc[n] < 0.755)
    if split == "train":  # all outpaintings first, then the other minority-regime images (false-positive check)
        todo = sorted(n for n in meta if idx[n]["edit_type"] == "outpainting") + [n for n in todo if idx[n]["edit_type"] != "outpainting"]
    if a.limit: todo = todo[:a.limit]
    if not a.diag:
        import rectsearch as RS
        wm = RS.load_watermark("stegastamp", device=a.device); D = RS.Dec(wm, a.device); D.bs = 96
    res = {}; fout = OUT / f"edge_{a.set}{a.tag}.json"; t00 = time.time()
    for k, nm in enumerate(todo):
        t0 = time.time(); img = np.asarray(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))
        S, pr = (PM.score_all(PM.side_maps(img, sign=True), a.device) if a.cand == "perim" else score_all(emap(img), a.device, a.pow)); C = topk(S, pr, a.K)
        r_ = {"cands": [list(map(int, c)) for c in C], "fullacc": facc[nm]}
        if split == "train":
            g = idx[nm]["bbox512"]; r_["edit_type"] = idx[nm]["edit_type"]; r_["gt"] = g
            ious = [iou(c, g) for c in C] if r_["edit_type"] == "outpainting" else [0]
            r_["cand_best_iou"] = max(ious); r_["cand_rank"] = int(np.argmax(ious))
        if not a.diag:
            msg = np.array([int(ch) for ch in meta[nm]["message"]], np.float64); n0 = D.n
            x = torch.from_numpy(np.array(img)).to(a.device).permute(2, 0, 1)[None].float() / 255
            ext = []
            for c in C: ext += [tuple(c)]
            o, acc, _ = D.score(x, ext, msg); order = np.argsort(-o)
            starts = []
            for j in order:
                if all(iou(ext[j], s_[0]) < 0.5 for s_ in starts): starts.append((ext[j], o[j]))
                if len(starts) >= 3: break
            best = None
            for s_, o0 in starts:
                r, ob = refine(D, x, msg, s_, o0)
                if best is None or ob > best[1]: best = (r, ob)
                if ob >= 0.9: break
            _, ac, mar = D.score(x, [best[0]], msg)
            r_.update(rect=list(map(int, best[0])), conf=float(ac[0]), margin=float(mar[0]), n_decodes=D.n - n0)
            if split == "train": r_["iou"] = iou(best[0], idx[nm]["bbox512"])
        r_["sec"] = time.time() - t0; res[nm] = r_
        print(k, nm, r_.get("edit_type"), r_.get("gt"), "cand iou %.2f rank %s" % (r_.get("cand_best_iou", -1), r_.get("cand_rank")),
              r_.get("rect"), r_.get("conf"), "iou %.2f" % r_.get("iou", -1), "%.1fs" % r_["sec"], flush=True)
        if k % 5 == 0 or k == len(todo) - 1:
            fout.write_text(json.dumps({"format": "rect=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px", "method": "stegastamp", "boxes": res}))
    (OUT / f"edge_{a.set}{a.tag}.done").write_text("ok %.0fs" % (time.time() - t00)); print("DONE", time.time() - t00)
