"""Edge profile of the watermark residual energy vs signed distance to the official mask boundary.
Real images (per method / edit_type) vs synthetic composites (clean C / watermarked Cw pairs from
t1/out/synth/v0/<method>/full) built with the SAME real masks, shifted by s px (s>0: watermarked zone larger
than the label) or blended with a gaussian fade. Run from TASK: python t1/edges/edge_profile.py stegastamp"""
import os, sys, json, glob
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, cv2
from scipy import ndimage as ndi
from PIL import Image

cv2.setNumThreads(8)
METHOD = sys.argv[1] if len(sys.argv) > 1 else "stegastamp"
TWO = len(sys.argv) > 2 and sys.argv[2] == "two"  # synthetic: wm object from cover k pasted on clean cover j (content seam)
TAG = METHOD + ("_two" if TWO else "")
R = 24; NB = 2 * R; CEN = np.arange(NB) - R + 0.5
OUT = "t1/out/edges"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
idx = json.load(open("t1/common/train_index.json"))


def coords(M):
    din = ndi.distance_transform_edt(M); dout = ndi.distance_transform_edt(~M)
    return np.where(M, din - 0.5, -(dout - 0.5))


def energies(img):
    f = img.astype(np.float32)
    med = np.abs(f - cv2.medianBlur(img, 3).astype(np.float32)).mean(2)
    dog = np.abs(cv2.GaussianBlur(f, (0, 0), 0.7) - cv2.GaussianBlur(f, (0, 0), 2.0)).mean(2)
    g = cv2.GaussianBlur(f.mean(2), (0, 0), 2.0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    keep = mag <= np.percentile(mag, 80)
    return {"med": med, "dog": dog}, keep


def stats(img, x):
    """per-config (sum, count) histograms over the 48 distance bins"""
    E, keep = energies(img)
    border = np.zeros(x.shape, bool); border[3:-3, 3:-3] = True
    b = np.floor(x).astype(int) + R
    ok = (b >= 0) & (b < NB) & border
    out = {}
    for fk, e in E.items():
        for ek, sel in (("all", ok), ("noedge", ok & keep)):
            out[f"{fk}_{ek}"] = (np.bincount(b[sel], e[sel], NB), np.bincount(b[sel], minlength=NB).astype(float))
    return out


def plateau(s, c, side):
    m = (CEN <= -6) if side < 0 else (CEN >= 6)
    n = c[m].sum()
    return (s[m].sum() / n if n >= 50 else np.nan)


def per_image(S):
    """-> (A num (n,NB), cnt (n,NB), den (n,)) for the ratio estimator, dropping images without both plateaus"""
    rows = []
    for s, c in S:
        lo, hi = plateau(s, c, -1), plateau(s, c, +1)
        if np.isnan(lo) or np.isnan(hi): continue
        mean = np.where(c > 0, s / np.maximum(c, 1), 0)
        rows.append(((mean - lo) * (c > 0), (c > 0).astype(float), hi - lo))
    if not rows: return None
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def profile(A, C, D):
    num = A.sum(0); den = (C * D[:, None]).sum(0)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def crossing(P, lev):
    """crossing of level lev closest to 0 (linear interp), restricted to |x|<=12"""
    best = np.nan
    for i in range(NB - 1):
        a, b = P[i], P[i + 1]
        if np.isnan(a) or np.isnan(b) or abs(CEN[i]) > 12.5: continue
        if (a - lev) * (b - lev) <= 0 and a != b:
            xc = CEN[i] + (lev - a) / (b - a)
            if np.isnan(best) or abs(xc) < abs(best): best = xc
    return best


def summarize(S, nboot=300):
    pi = per_image(S)
    if pi is None: return None
    A, C, D = pi
    P = profile(A, C, D)
    f = lambda P: (crossing(P, .5), crossing(P, .9) - crossing(P, .1))
    mid, wid = f(P)
    bs = []
    for _ in range(nboot):
        k = rng.integers(len(D), size=len(D)); bs.append(f(profile(A[k], C[k], D[k])))
    bs = np.array(bs)
    pim = np.array([crossing(np.where(C[i] > 0, A[i] / D[i], np.nan), .5) for i in range(len(D))])
    good = pim[~np.isnan(pim)]
    pmed = float(np.median(good)) if len(good) else np.nan
    pbs = [np.median(good[rng.integers(len(good), size=len(good))]) for _ in range(nboot)] if len(good) else [np.nan]
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"n": int(len(D)), "mid": float(mid), "mid_ci": q(bs[:, 0]), "w1090": float(wid), "pimed": pmed, "pimed_ci": q(np.array(pbs)), "w_ci": q(bs[:, 1]),
            "P": [None if np.isnan(v) else round(float(v), 3) for v in P]}


# ---------------- real images
names = {et: sorted(n for n, v in idx.items() if v["method"] == METHOD and v["edit_type"] == et)
         for et in ("hole", "transplant", "outpainting")}
masks, real = {}, {}
for et, ns in names.items():
    masks[et] = []; acc = []
    for n in ns:
        M = np.asarray(Image.open(f"release/train/masks/{n}.png").convert("L")) > 127
        if M.all() or not M.any(): continue
        img = np.asarray(Image.open(f"release/train/images/{n}.png").convert("RGB"))
        x = coords(M); masks[et].append(M); acc.append(stats(img, x))
    real[et] = acc
    print(f"[{METHOD}] {et}: {len(acc)} real images", flush=True)

# ---------------- synthetic covers (C clean, Cw = embed(C) full)
shards = sorted(glob.glob(f"t1/out/synth/v0/{METHOD}/full/s*"))
Cs, Cws = [], []
for d in shards:
    C = np.load(f"{d}/C.npy", mmap_mode="r"); Cw = np.load(f"{d}/Cw.npy", mmap_mode="r")
    for i in range(len(C)):
        Cs.append((C, i)); Cws.append((Cw, i))
    if len(Cs) >= 200: break
print(f"[{METHOD}] synthetic pairs available: {len(Cs)}", flush=True)

CONDS = [("s-4", -4), ("s-2", -2), ("s0", 0), ("s+2", 2), ("s+4", 4), ("blur2", None)]
synth = {}
if Cs:
    for et, ms in masks.items():
        acc = {c: [] for c, _ in CONDS}
        for j, M in enumerate(ms):
            k = int(rng.integers(len(Cs)))
            C = np.asarray(Cs[k][0][Cs[k][1]]); Cw = np.asarray(Cws[k][0][Cws[k][1]])
            if TWO:
                k2 = int(rng.integers(len(Cs))); C = np.asarray(Cs[k2][0][Cs[k2][1]])
            x = coords(M)
            for cn, s in CONDS:
                if s is None:
                    a = ndi.gaussian_filter(M.astype(np.float32), 2.0)[..., None]
                    Y = np.clip(np.round(a * Cw + (1 - a) * C), 0, 255).astype(np.uint8)
                else:
                    Y = np.where((x > -s)[..., None], Cw, C)
                acc[cn].append(stats(Y, x))
        synth[et] = acc
        print(f"[{METHOD}] synth {et} done", flush=True)

# ---------------- summaries
res = {"method": METHOD, "real": {}, "synth": {}}
cfgs = ["med_all", "med_noedge", "dog_all", "dog_noedge"]
for et in real:
    res["real"][et] = {cf: summarize([s[cf] for s in real[et]]) for cf in cfgs}
    if et in synth:
        res["synth"][et] = {cn: {cf: summarize([s[cf] for s in synth[et][cn]], nboot=100) for cf in cfgs}
                            for cn, _ in CONDS}


def est_shift(et, cf):
    """invert the calibration line mid(s) (s = -4..4) at the real mid crossing"""
    if et not in res["synth"]: return None
    ss = np.array([-4, -2, 0, 2, 4.]); mids = np.array([res["synth"][et][f"s{'+' if s > 0 else ''}{int(s)}" if s != 0 else "s0"][cf]["mid"] for s in ss])
    ok = ~np.isnan(mids)
    if ok.sum() < 3: return None
    a, b = np.polyfit(ss[ok], mids[ok], 1)
    r = res["real"][et][cf]
    if r is None: return None
    inv = lambda m: (m - b) / a
    lo, hi = sorted([inv(r["mid_ci"][0]), inv(r["mid_ci"][1])])
    return {"slope": float(a), "icpt": float(b), "shift": float(inv(r["mid"])), "shift_ci": [float(lo), float(hi)]}


res["estimate"] = {et: {cf: est_shift(et, cf) for cf in cfgs} for et in real}
json.dump(res, open(f"{OUT}/profile_{TAG}.json", "w"), indent=1)

lines = []
for et in real:
    lines.append(f"=== {METHOD} / {et}")
    for cf in cfgs:
        r = res["real"][et][cf]
        if r is None: lines.append(f"  {cf}: no data"); continue
        row = f"  {cf:11s} REAL n={r['n']:3d} mid={r['mid']:+.2f} [{r['mid_ci'][0]:+.2f},{r['mid_ci'][1]:+.2f}] w={r['w1090']:.2f} [{r['w_ci'][0]:.2f},{r['w_ci'][1]:.2f}] per-img-median-mid={r['pimed']:+.2f} [{r['pimed_ci'][0]:+.2f},{r['pimed_ci'][1]:+.2f}]"
        lines.append(row)
        if et in res["synth"]:
            cal = "   CAL " + " ".join(f"{cn}:{res['synth'][et][cn][cf]['mid']:+.2f}/w{res['synth'][et][cn][cf]['w1090']:.1f}/pm{res['synth'][et][cn][cf]['pimed']:+.2f}"
                                     for cn, _ in CONDS if res['synth'][et][cn][cf])
            lines.append(cal)
            e = res["estimate"][et][cf]
            if e: lines.append(f"   => shift(wm zone - mask) = {e['shift']:+.2f} px  CI [{e['shift_ci'][0]:+.2f},{e['shift_ci'][1]:+.2f}]  (slope {e['slope']:+.2f})")
    for cf in ("med_noedge",):
        r = res["real"][et][cf]
        if r: lines.append("   profile " + cf + " x=-11.5..11.5: " + " ".join(f"{v:.2f}" if v is not None else "nan" for v in r["P"][R - 12:R + 12]))
        if et in res["synth"]:
            for cn in ("s0", "blur2"):
                r2 = res["synth"][et][cn][cf]
                if r2: lines.append(f"   synth {cn:5s}           : " + " ".join(f"{v:.2f}" if v is not None else "nan" for v in r2["P"][R - 12:R + 12]))
txt = "\n".join(lines); print(txt)
open(f"{OUT}/summary_{TAG}.txt", "w").write(txt + "\n")
open(f"{OUT}/DONE_{TAG}", "w").write("ok\n")
