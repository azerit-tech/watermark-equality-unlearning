"""Hypotheses 1, 3, 4: global decoding accuracy train vs test, regime proportion, valley position, message consistency.
Run from TASK: python t1/shift/a_signal.py"""
import json
import numpy as np

THR = {"pixelseal": 0.80, "trustmark": 0.76, "stegastamp": 0.755}
idx = json.load(open("t1/common/train_index.json"))
meta = {s: {d["image_name"]: d for d in json.load(open(f"release/{s}/metadata.json"))} for s in ("train", "test")}


def hist(a, bins):
    h, _ = np.histogram(a, bins)
    return " ".join(f"{x:3d}" for x in h)


def otsu(a):
    best, bt = -1, None
    for t in np.unique(a):
        lo, hi = a[a < t], a[a >= t]
        if len(lo) == 0 or len(hi) == 0:
            continue
        v = len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2
        if v > best:
            best, bt = v, t
    return bt


bins = np.round(np.arange(0.3, 1.0001, 0.05), 3)
for m, thr in THR.items():
    print(f"\n===== {m}  thr {thr}")
    z = {s: np.load(f"t1/out/signal/fullacc_{m}_{s}.npz") for s in ("train", "test")}
    tr, te = z["train"], z["test"]
    fw = np.array([idx[n]["full_watermark"] for n in tr["names"]])
    et = np.array([idx[n]["edit_type"] for n in tr["names"]])
    print("bins      ", " ".join(f"{b:.2f}"[1:] for b in bins[:-1]))
    print("train full", hist(tr["acc"][fw], bins))
    print("train part", hist(tr["acc"][~fw], bins))
    print("test      ", hist(te["acc"], bins))
    for s, a in (("train", tr["acc"]), ("test", te["acc"])):
        print(f"{s}: frac>=thr {np.mean(a >= thr):.3f} ({(a >= thr).sum()}/{len(a)})  otsu {otsu(a):.3f} "
              f" n in [0.62,thr) {np.sum((a >= .62) & (a < thr))}  n in [thr,0.88) {np.sum((a >= thr) & (a < .88))}"
              f"  min {a.min():.3f} max {a.max():.3f}  n<0.40 {np.sum(a < .4)}  n>=0.99 {np.sum(a >= .99)}")
    for e in ("hole", "transplant", "outpainting"):
        a = tr["acc"][et == e]
        print(f"   train {e:11s} acc q05/50/95 {np.quantile(a, .05):.3f} {np.quantile(a, .5):.3f} {np.quantile(a, .95):.3f}"
              f"  wrong-regime {np.mean((a >= thr) != fw[et == e]):.3f}")
    # sorted low-margin cases near threshold
    a = te["acc"]; near = np.sort(a[np.abs(a - thr) < 0.06])
    print("test acc within +-0.06 of thr:", np.round(near, 3))
    a = tr["acc"]; near = np.sort(a[np.abs(a - thr) < 0.06])
    print("train acc within +-0.06 of thr:", np.round(near, 3))
    # margin distribution
    for s in ("train", "test"):
        mg = z[s]["margin"]
        print(f"{s} margin q05/25/50/75/95", np.round(np.quantile(mg, [.05, .25, .5, .75, .95]), 3))
    # message consistency: for low-acc test images, accuracy vs every other message of same method
    for s in ("train", "test"):
        names = [str(n) for n in z[s]["names"]]
        L = z[s]["logits"]
        msgs = np.stack([np.array([int(c) for c in meta[s][n]["message"]], np.uint8) for n in names]).astype(bool)
        A = ((L[:, None, :] > 0) == msgs[None]).mean(-1)            # [img, msg]
        own = np.diag(A).copy(); np.fill_diagonal(A, 0)
        oth = A.max(1)
        swapped = np.sum((oth >= 0.75) & (oth > own + 0.1))
        # also against complement (bit-flipped message)
        inv = np.sum(own <= 0.25)
        print(f"{s}: max acc vs OTHER messages: q50 {np.median(oth):.3f} max {oth.max():.3f}; "
              f"images better explained by another message: {swapped}; own<=0.25 (inverted): {inv}; "
              f"duplicate messages: {len(names) - len(set(meta[s][n]['message'] for n in names))}")
    # sliding-window max accuracy (local presence of the message) for low-regime images
    for s in ("train", "test"):
        try:
            sl = np.load(f"t1/out/signal/slide_{m}_{s}.npz")
        except FileNotFoundError:
            continue
        names = [str(n) for n in sl["names"]]
        fa = dict(zip([str(n) for n in z[s]["names"]], z[s]["acc"]))
        a = np.array([fa[n] for n in names])
        mx = sl["acc"].astype(np.float32).reshape(len(names), -1).max(1)
        low = a < thr
        print(f"{s} slide-max acc for minority-regime imgs: q05/25/50 {np.round(np.quantile(mx[low], [.05, .25, .5]), 3)}"
              f"  n(slide-max<0.6) {np.sum(mx[low] < .6)}/{low.sum()}; majority q05 {np.quantile(mx[~low], .05):.3f}")
        if s == "train":
            fwn = np.array([idx[n]["full_watermark"] for n in names])
            ee = np.array([idx[n]["edit_type"] for n in names])
            for e in ("transplant", "outpainting"):
                k = ee == e
                print(f"    train {e} slide-max q05/50 {np.round(np.quantile(mx[k], [.05, .5]), 3)} min {mx[k].min():.3f}")
