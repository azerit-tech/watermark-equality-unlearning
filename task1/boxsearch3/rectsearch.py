"""Rectangle search for outpainting images: find R such that the crop R+PAD decodes the known message best.
usage (from TASK): python t1/boxsearch/rectsearch.py --method trustmark --device cuda:1 --sets evaltrain,test,train
Writes t1/out/boxsearch/<method>_{evaltrain,test,train}_boxes.json incrementally.
Box format: [y0, y1, x0, x1] in 512 px, exclusive ends (R = predicted watermarked rectangle, NOT padded)."""
import argparse, json, os, sys, time
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK))
from wmloc.watermarks import load_watermark  # noqa
OUT = TASK / "t1/out/boxsearch"; OUT.mkdir(parents=True, exist_ok=True)
RES = {"pixelseal": (256, "bilinear"), "trustmark": (245, "bilinear"), "stegastamp": (400, "bicubic")}
ASPECT = {"pixelseal": 16, "trustmark": 2, "stegastamp": 16}
THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}
SEG = {"stegastamp": "t1/out/train/v0/{}_probs128.npz", "trustmark": "t1/out/dense/login/tm/{}_probs128.npz",
       "pixelseal": "t1/out/dense/login/ps/{}_probs128.npz"}
SEG_FALLBACK = {"trustmark": "t1/out/mgr/smoke_gpu/tm/{}_probs128.npz", "pixelseal": "t1/out/mgr/smoke_gpu/ps/{}_probs128.npz"}
PAD = 3


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


class Dec:
    def __init__(self, wm, dev):
        self.wm, self.dev = wm, dev; self.res, self.mode = RES[wm.name]; self.n = 0
        self.bs = 48 if wm.name == "stegastamp" else 128

    def prep(self, c):
        y = F.interpolate(c, size=(self.res, self.res), mode=self.mode, align_corners=False, antialias=True)
        return y.clamp(0, 1) if self.mode == "bicubic" else y

    def logits(self, x, boxes):
        out = []
        for i in range(0, len(boxes), self.bs):
            xs = torch.cat([self.prep(x[:, :, b[0]:b[1], b[2]:b[3]]) for b in boxes[i:i + self.bs]])
            l = self.wm.logits(xs, bs=self.bs) if self.wm.name == "stegastamp" else self.wm.logits(xs)
            out.append(l.float().cpu())
        self.n += len(boxes)
        return torch.cat(out).numpy()

    def score(self, x, rects, msg):
        """rects R -> decode crop R+PAD. returns (objective, acc, margin)"""
        boxes = [pad(r) for r in rects]
        L = self.logits(x, boxes); s = (2.0 * msg - 1)
        acc = ((L > 0) == msg.astype(bool)).mean(1)
        sm = L * s; sd = L.std(1, keepdims=True) + 1e-6
        soft = np.clip(sm / sd, -1, 1).mean(1)
        return acc + 0.1 * soft, acc, sm.mean(1)


def pad(r):
    return (max(0, r[0] - PAD), min(512, r[1] + PAD), max(0, r[2] - PAD), min(512, r[3] + PAD))


def ok(r, A):
    h, w = r[1] - r[0], r[3] - r[2]
    return r[0] >= 0 and r[2] >= 0 and r[1] <= 512 and r[3] <= 512 and h >= 8 and w >= 8 and max(h / w, w / h) <= A


def grid(A, sizes=(64, 128, 192, 288, 416)):
    pos = {}
    for s in sizes:
        st = s // 2; p = list(range(0, 512 - s + 1, st))
        if p[-1] != 512 - s: p.append(512 - s)
        pos[s] = p
    out = []
    for h in sizes:
        for w in sizes:
            if max(h / w, w / h) > A: continue
            out += [(y, y + h, x, x + w) for y in pos[h] for x in pos[w]]
    return out


def refine(D, x, msg, r, obj, A, max_it=12):
    for step in (16, 8, 4, 2, 1):
        for _ in range(max_it):
            c = []
            for e in range(4):
                for d in (-step, step):
                    q = list(r); q[e] += d; c.append(tuple(q))
            for (a, b) in ((0, 1), (2, 3)):
                for d in (-step, step):
                    q = list(r); q[a] += d; q[b] += d; c.append(tuple(q))
            c = [q for q in c if ok(q, A)]
            if not c: break
            o, _, _ = D.score(x, c, msg); j = int(np.argmax(o))
            if o[j] <= obj: break
            r, obj = c[j], o[j]
    return r, obj


def iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    i = iy * ix; return i / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - i)


def search(D, x, msg, A, G, seg_rect=None, topk=2):
    starts = []
    order = np.argsort(-D.score(x, G, msg)[0]) if G else []
    for j in order:
        if all(iou(G[j], s) < 0.5 for s in starts): starts.append(G[j])
        if len(starts) >= topk: break
    if seg_rect is not None:
        starts = [r for r in seg_rect if ok(r, 999)] + starts
    if not starts: starts = [(192, 320, 192, 320)]
    best = None
    for s in starts:
        o0 = D.score(x, [s], msg)[0][0]
        r, ob = refine(D, x, msg, s, o0, A if ok(s, A) else 999)
        if best is None or ob > best[1]: best = (r, ob)
    r = best[0]; _, a, m = D.score(x, [r], msg)
    return r, float(a[0]), float(m[0]), [list(map(int, s)) for s in starts]


def seg_maps(method, split):
    out = {}
    for tmpl in (SEG.get(method), SEG_FALLBACK.get(method)):
        if not tmpl: continue
        f = TASK / tmpl.format(split)
        if f.exists():
            z = np.load(f); out = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))); log("seg", f, len(out)); break
    return out


def seg_rect(p):
    if p is None: return None
    out = []
    for t in (0.5, 0.3, 0.7):
        m = p > t
        if m.sum() < 4 or m.mean() > 0.9: continue
        ys, xs = np.nonzero(m)
        r = (int(ys.min()) * 4, int(ys.max() + 1) * 4, int(xs.min()) * 4, int(xs.max() + 1) * 4)
        if r not in out: out.append(r)
    return out or None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--sets", default="evaltrain,test,train"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--tag", default=""); ap.add_argument("--sizes", default="64,128,192,288,416"); ap.add_argument("--segonly", action="store_true"); ap.add_argument("--fold0", action="store_true")
    a = ap.parse_args(); torch.set_num_threads(2)
    M = a.method; A = ASPECT[M]; G = [] if a.segonly else grid(A, [int(v) for v in a.sizes.split(",")]); log(M, "grid", len(G))
    F0 = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0])
    wm = load_watermark(M, device=a.device); D = Dec(wm, a.device)
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    if a.check:   # equality of our batched preprocessing with the wrapper
        meta = json.loads((TASK / "release/train/metadata.json").read_text())
        nm = [m["image_name"] for m in meta if m["method"] == M][0]
        x = torch.from_numpy(np.asarray(Image.open(TASK / f"release/train/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
        bx = [(0, 100, 0, 37), (50, 300, 60, 200), (10, 512, 0, 512)]
        L1 = D.logits(x, bx); L2 = np.concatenate([wm.logits(x[:, :, b[0]:b[1], b[2]:b[3]]).float().cpu().numpy() for b in bx])
        log("CHECK max|dL|", float(np.abs(L1 - L2).max()), "max|L|", float(np.abs(L2).max())); sys.exit(0)
    cache = {}
    for S in a.sets.split(","):
        split = "test" if S == "test" else "train"
        meta = [m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text()) if m["method"] == M]
        fa = np.load(TASK / f"t1/out/signal/fullacc_{M}_{split}.npz"); facc = dict(zip(fa["names"].tolist(), fa["acc"].tolist()))
        if S == "evaltrain":
            meta = [m for m in meta if idx[m["image_name"]]["edit_type"] == "outpainting"] + [m for m in meta if idx[m["image_name"]]["edit_type"] == "transplant"][:(999 if a.fold0 else 20)]
        else:
            meta = [m for m in meta if facc[m["image_name"]] < THR[M]]
        if a.fold0 and split == "train": meta = [m for m in meta if m["image_name"] in F0]
        if a.limit: meta = meta[:a.limit]
        segs = seg_maps(M, "test" if split == "test" else "fold0")
        res = {}; fout = OUT / f"{M}_{S}{a.tag}_boxes.json"
        log(S, len(meta), "images")
        for k, m in enumerate(meta):
            nm = m["image_name"]; t0 = time.time(); n0 = D.n
            if split == "train" and nm in cache:
                res[nm] = cache[nm]; continue
            msg = np.array([int(c) for c in m["message"]], np.float64)
            x = torch.from_numpy(np.asarray(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
            sr = seg_rect(segs.get(nm))
            r, acc, mar, starts = search(D, x, msg, A, G, sr)
            res[nm] = {"rect": list(map(int, r)), "conf": acc, "margin": mar, "n_decodes": D.n - n0, "sec": time.time() - t0,
                       "seg_rect": sr, "fullacc": facc[nm]}
            if split == "train":
                cache[nm] = res[nm]
                res[nm]["edit_type"] = idx[nm]["edit_type"]; res[nm]["bbox512"] = idx[nm]["bbox512"]
                res[nm]["iou"] = iou(r, idx[nm]["bbox512"])
            if k % 5 == 0 or k == len(meta) - 1:
                fout.write_text(json.dumps({"format": "rect=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px", "method": M, "boxes": res}, indent=0))
                log(S, k, nm, res[nm].get("edit_type"), r, f"acc {acc:.3f}", f"iou {res[nm].get('iou', -1):.3f}", f"{res[nm]['sec']:.1f}s", res[nm]["n_decodes"])
        fout.write_text(json.dumps({"format": "rect=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px", "method": M, "boxes": res}, indent=0))
    (OUT / f"{M}{a.tag}.done").write_text("ok")
    log("DONE")
