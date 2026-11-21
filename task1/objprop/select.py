"""Pick, per minority-regime image, the Mask R-CNN proposal (or pairwise union) that best agrees with our probability
map; keep the original prediction when agreement < tau. Run from TASK (CPU):
    python t1/objprop/select.py --eval            # fold 0 grid over agreement / tau, per method x edit_type
    python t1/objprop/select.py --write CRIT TAU METHODS   # writes fold0 + test npz for chosen methods
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "train", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from regime_guard import load_regimes, guard  # noqa
from wmloc.utils import downsample_mask, miou_image  # noqa

SRC = {"stegastamp": "t1/out/train/v0", "trustmark": "t1/out/frozen/trustmark", "pixelseal": "t1/out/dense/login/ps"}
PROPS = TASK / "t1/out/objprop/props"
OUT = TASK / "t1/out/objprop"


def load_probs(split):
    fn = "fold0_probs128.npz" if split == "fold0" else "test_probs128.npz"
    P = {}
    for m, d in SRC.items():
        z = np.load(TASK / d / fn)
        P[m] = {str(n): p.astype(np.float32) for n, p in zip(z["names"], z["probs"])}
    return P


def candidates(f, kpair=15, mthr=128):
    z = np.load(f)
    bm = z["masks"] >= mthr
    if len(bm) == 0:
        return np.zeros((0, 128, 128), np.uint8)
    bm = bm[bm.reshape(len(bm), -1).sum(1) > 0]
    c = [downsample_mask(b) for b in bm]
    k = min(kpair, len(bm))
    for i in range(k):
        for j in range(i + 1, k):
            c.append(downsample_mask(bm[i] | bm[j]))
    c = [x for x in c if 0 < x.sum() < x.size]
    return np.array(c, np.uint8) if c else np.zeros((0, 128, 128), np.uint8)


def agree(C, p, crit):
    if crit == "iou":
        b = p >= 0.5
        inter = (C & b).reshape(len(C), -1).sum(1); uni = (C | b).reshape(len(C), -1).sum(1)
        return inter / np.maximum(uni, 1)
    if crit == "softiou":
        Cf = C.astype(np.float32)
        return np.minimum(Cf, p).reshape(len(C), -1).sum(1) / np.maximum(Cf, p).reshape(len(C), -1).sum(1)
    if crit == "ll":   # mean log-likelihood of the candidate under p (per cell), in [-inf, 0] -> exp
        q = np.clip(p, 1e-3, 1 - 1e-3)
        ll = np.where(C == 1, np.log(q), np.log(1 - q)).reshape(len(C), -1).mean(1)
        return np.exp(ll)
    raise ValueError(crit)


def prepare(split):
    P = load_probs(split)
    reg = load_regimes("train" if split == "fold0" else "test")
    meta = {m["image_name"]: m["method"] for m in json.load(open(TASK / "release" / ("train" if split == "fold0" else "test") / "metadata.json"))}
    items = []
    for f in sorted((PROPS / split).glob("*.npz")):
        n = f.stem; m = meta[n]
        p = P[m][n]
        orig, _ = guard(p, reg[n])
        items.append((n, m, p, orig, candidates(f)))
    return items


def decide(items, crit, tau, methods):
    out = {}
    for n, m, p, orig, C in items:
        if m not in methods or len(C) == 0:
            continue
        a = agree(C, p, crit); i = int(np.argmax(a))
        if a[i] >= tau:
            out[n] = C[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--write", nargs=3)
    a = ap.parse_args()
    items = prepare("fold0")
    from evaluate import load_train
    _, gt = load_train()
    meta = json.load(open(TASK / "t1/common/train_index.json"))
    if a.eval:
        # oracle: best candidate vs GT
        print("method/edit n orig oracle")
        rows = {}
        for n, m, p, orig, C in items:
            e = meta[n]["edit_type"]
            o = miou_image(orig, gt[n])
            orc = max([miou_image(c, gt[n]) for c in C], default=o)
            rows.setdefault((m, e), []).append((o, orc))
        for k, v in sorted(rows.items()):
            v = np.array(v); print(k, len(v), v.mean(0).round(3))
        for crit in ("iou", "softiou", "ll"):
            for tau in ((0.0, 0.3, 0.4, 0.5, 0.6, 0.7) if crit != "ll" else (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)):
                ch = decide(items, crit, tau, SRC)
                res = {}
                for n, m, p, orig, C in items:
                    e = meta[n]["edit_type"]
                    o = miou_image(orig, gt[n]); nw = miou_image(ch.get(n, orig), gt[n])
                    res.setdefault((m, e), []).append((o, nw, n in ch))
                s = " | ".join(f"{m[:2]}/{e[:2]} {np.mean([r[1]-r[0] for r in v]):+.3f} ({sum(r[2] for r in v)}/{len(v)})"
                               for (m, e), v in sorted(res.items()))
                tot = {m: np.sum([r[1]-r[0] for (mm, e), v in res.items() if mm == m for r in v]) for m in SRC}
                print(f"{crit:7s} tau={tau:.2f} | {s} | gain over 34/method: " + " ".join(f"{m[:2]} {tot[m]/34:+.3f}" for m in SRC))
    if a.write:
        crit, tau, methods = a.write[0], float(a.write[1]), a.write[2].split(",")
        test_items = prepare("test")
        for m in methods:
            for split, its in (("fold0", items), ("test", test_items)):
                ch = decide(its, crit, tau, [m])
                names = sorted(ch)
                np.savez(OUT / f"{m}_{split}_masks.npz", names=np.array(names),
                         masks=np.array([ch[n] for n in names], np.uint8).reshape(-1, 128, 128))
                (OUT / f"{m}_{split}_names.txt").write_text("\n".join(names) + "\n")
                print(m, split, len(names), "replaced")


if __name__ == "__main__":
    main()
