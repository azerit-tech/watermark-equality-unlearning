"""psdecode: does another decoding preprocessing raise the PixelSeal/TrustMark ceiling at the TRUE outpainting rect?
usage (from TASK): python ~/psdecode/psdecode.py --method pixelseal --threads 2 --out ~/psdecode/ps.json"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK))
from wmloc.watermarks import load_watermark  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--method"); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out"); ap.add_argument("--ntp", type=int, default=15); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); torch.set_num_threads(a.threads); M = a.method
wm = load_watermark(M, device="cpu")
if M == "pixelseal":
    RES = 256; NET = lambda z: wm.det(z)[:, 1:]
else:
    RES = wm.resolution; NET = lambda z: wm.net(z * 2 - 1)
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
meta = [m for m in json.loads((TASK / "release/train/metadata.json").read_text()) if m["method"] == M]
outp = [m for m in meta if idx[m["image_name"]]["edit_type"] == "outpainting"]
tps = [m for m in meta if idx[m["image_name"]]["edit_type"] == "transplant"][:a.ntp]
best = {}
for f in ["t1/out/boxsearch/%s_evaltrain_boxes.json" % M, "t1/out/boxsearch/%s_evaltrain_grid1_boxes.json" % M,
          "t1/out/boxsearch/%s_train_boxes.json" % M]:
    p = TASK / f
    if p.exists():
        for k, v in json.loads(p.read_text())["boxes"].items():
            best.setdefault(k, tuple(v["rect"]))


def prep(c, mode):
    lum = mode.endswith("_Y")
    if lum: mode = mode[:-2]
    if lum:
        y = 0.299 * c[:, 0:1] + 0.587 * c[:, 1:2] + 0.114 * c[:, 2:3]; c = y.expand(-1, 3, -1, -1)
    if mode == "lanczos":
        arr = c[0].numpy()
        out = np.stack([np.asarray(Image.fromarray(arr[i], mode="F").resize((RES, RES), Image.Resampling.LANCZOS)) for i in range(3)])
        return torch.from_numpy(out)[None].clamp(0, 1)
    if mode == "area": return F.interpolate(c, size=(RES, RES), mode="area")
    if mode == "nearest": return F.interpolate(c, size=(RES, RES), mode="nearest")
    md, aa = mode.split("-")
    y = F.interpolate(c, size=(RES, RES), mode=md, align_corners=False, antialias=(aa == "aa"))
    return y.clamp(0, 1) if md == "bicubic" else y


def crop(x, r, m):
    return x[:, :, max(0, r[0] - m[0]):min(512, r[1] + m[1]), max(0, r[2] - m[2]):min(512, r[3] + m[3])]


@torch.no_grad()
def logits(x, jobs):
    """jobs = list of (rect, margins4, mode) -> logits [n, nbits]"""
    out = []
    for i in range(0, len(jobs), 32):
        out.append(NET(torch.cat([prep(crop(x, r, m), md) for r, m, md in jobs[i:i + 32]])).float())
    return torch.cat(out).numpy()


def acc(L, msg): return ((L > 0) == msg.astype(bool)).mean(-1)


def scaled(r, s):
    h, w = r[1] - r[0], r[3] - r[2]; dy, dx = (s - 1) * h / 2, (s - 1) * w / 2
    return (int(round(r[0] - dy)), int(round(r[1] + dy)), int(round(r[2] - dx)), int(round(r[3] + dx)))


def evaluate(x, r, msg, rng):
    res = {}
    U = lambda m: (m, m, m, m)
    # A: bilinear-aa uniform margins
    jobs = [(r, U(m), "bilinear-aa") for m in range(-2, 7)]
    # B: interpolations x margins 2,3,4 ; C: luminance
    modes = ["bilinear-noaa", "bicubic-aa", "area", "lanczos", "bilinear-aa_Y", "bicubic-aa_Y", "lanczos_Y"]
    jobs += [(r, U(m), md) for md in modes for m in (2, 3, 4)]
    L = logits(x, jobs); A = acc(L, msg)
    for (rr, m, md), v in zip(jobs, A): res[f"{md}|m{m[0]}"] = float(v)
    Lb = {f"{md}|m{m[0]}": l for (rr, m, md), l in zip(jobs, L)}
    # D: TTA (logit averaging)
    for base in ("bilinear-aa", "bilinear-aa_Y"):
        jit = [(r, tuple(int(v) for v in 3 + rng.integers(-2, 3, 4)), base) for _ in range(10)]
        jit += [(scaled(r, s), U(3), base) for s in (0.98, 1.02)]
        Lj = logits(x, jit)
        L0 = Lb[f"{base}|m3"]
        res[f"tta_jit12+c|{base}"] = float(acc(np.vstack([Lj, L0[None]]).mean(0), msg))
        res[f"tta_marg2-5|{base}"] = float(acc(np.mean([Lb.get(f"{base}|m{m}", None) if f"{base}|m{m}" in Lb else logits(x, [(r, U(m), base)])[0] for m in (2, 3, 4, 5)], 0), msg))
    # E: per-side oracle margins (coordinate ascent, bilinear-aa), values 0..7
    for base in ("bilinear-aa",):
        cur = [3, 3, 3, 3]; ca = res[f"{base}|m3"]
        for _ in range(1):
            for e in range(4):
                cand = []
                for v in range(1, 7):
                    q = list(cur); q[e] = v; cand.append(q)
                Ac = acc(logits(x, [(r, tuple(q), base) for q in cand]), msg); j = int(np.argmax(Ac))
                if Ac[j] > ca: ca, cur = float(Ac[j]), cand[j]
        res[f"oracle_side|{base}"] = ca; res[f"oracle_side_m|{base}"] = cur
    return res


fout = Path(a.out); R = json.loads(fout.read_text()) if fout.exists() else {}
items = [(m, "outpainting", tuple(idx[m["image_name"]]["bbox512"])) for m in outp]
for m in tps:
    nm = m["image_name"]; items.append((m, "tp_bbox", tuple(idx[nm]["bbox512"])))
    if nm in best: items.append((m, "tp_best", best[nm]))
if a.limit: items = items[:a.limit]
t0 = time.time()
for k, (m, kind, r) in enumerate(items):
    key = f"{m['image_name']}|{kind}"
    if key in R: continue
    msg = np.array([int(c) for c in m["message"]], np.float64)
    x = torch.from_numpy(np.array(Image.open(TASK / f"release/train/images/{m['image_name']}.png").convert("RGB"))).permute(2, 0, 1)[None].float().div(255)
    res = evaluate(x, r, msg, np.random.default_rng(k)); res["rect"] = list(r); res["kind"] = kind
    R[key] = res; fout.write_text(json.dumps(R))
    print(time.strftime("%H:%M:%S"), k, key, r, "base", round(res["bilinear-aa|m3"], 3), "oracle", round(res["oracle_side|bilinear-aa"], 3),
          "tta", round(res["tta_jit12+c|bilinear-aa"], 3), "Y", round(res["bilinear-aa_Y|m3"], 3), f"{time.time()-t0:.0f}s", flush=True)
Path(str(fout) + ".done").write_text("ok"); print("DONE", flush=True)
