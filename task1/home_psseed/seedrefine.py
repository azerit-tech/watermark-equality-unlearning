"""psseed: extra starts for the PixelSeal rectangle search (same idea as ~/sgseed): bounding boxes of the main components of
several probability maps (TTA views, tiles3 scales, devel folds, jacques s1), dedup, scored once (crop R+3, rectsearch.Dec CPU),
top-K refined with the boxsearch rectsearch.refine local search (A=16) under a budget of PER decodes per start. Threshold 0.90 fixed.
usage (from TASK): python ~/psseed/seedrefine.py fold0|test   -> ~/psseed/seeds_{fold0,test}.json ; merge.py writes the boxes files."""
import sys, os, json, time
from pathlib import Path
import numpy as np, torch
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch3"))
import rectsearch as RS  # noqa
from scipy.ndimage import label
from PIL import Image
OUT = Path.home() / "psseed"; T = TASK / "t1/out"
PER, K, MINA = int(os.environ.get("PER", 60)), int(os.environ.get("K", 3)), 16
DEADLINE = float(os.environ.get("DEADLINE", 1e18)); A = 16


def maps(tag):
    fs = [T / f"tiles3/pixelseal/s2_22min/{tag}_t{t}_probs128.npz" for t in (192, 256, 320)]
    fs += [T / f"tta/pixelseal/{tag}_base_probs128.npz"] + [T / f"tta/pixelseal/{tag}_t{t}_g{g}_probs128.npz" for t in (256, 320) for g in range(4)]
    fs += sorted((T / "pstta/pixelseal").glob(f"views_{tag}_*.npz"))
    if tag == "test":
        fs += [T / f"devel/pixelseal/fold{k}/pred/test{s}_probs128.npz" for k in (1, 2, 3) for s in ("", "_t256", "_t320")]
        fs += [T / f"jacques/tiles_pixelseal_s1/test{s}_probs128.npz" for s in ("", "_t256", "_t320")]
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


def compact(nm, M):
    ps = [m[nm] for m in M if nm in m]
    if not ps: return 0.0
    p = np.mean(ps, 0); lab, n = label(p >= 0.5, structure=np.ones((3, 3)))
    if n == 0: return 0.0
    ar = np.bincount(lab.ravel())[1:]; j = int(np.argmax(ar)); ys, xs = np.nonzero(lab == j + 1)
    if ar[j] < MINA: return 0.0
    return float(ar[j] / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)))


def refine(D, x, msg, r, obj, stop, max_it=12):   # = rectsearch.refine + decode budget
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


if __name__ == "__main__":
    tag = sys.argv[1]; split = "test" if tag == "test" else "train"
    torch.set_num_threads(int(os.environ.get("NT", 8)))
    D = RS.Dec(RS.load_watermark("pixelseal", device="cpu"), "cpu"); D.bs = 128
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text())}
    M = maps(tag)
    if tag == "fold0":
        f0 = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0]
        f0 = [n for n in f0 if meta[n]["method"] == "pixelseal"]
        names = [n for n in f0 if idx[n]["edit_type"] == "outpainting"] + [n for n in f0 if idx[n]["edit_type"] == "transplant"]
    else:
        z = np.load(T / "signal/fullacc_pixelseal_test.npz"); fa = dict(zip(z["names"].tolist(), z["acc"].tolist()))
        old = json.loads((T / "psbox/pixelseal_test_boxes.json").read_text())["boxes"]
        names = [n for n, m in meta.items() if m["method"] == "pixelseal" and fa[n] < 0.80 and old.get(n, {}).get("conf", 0) < 0.9]
        cp = {n: compact(n, M) for n in names}; names = sorted(names, key=lambda n: -cp[n])
        print("test todo", len(names), "compactness>0:", sum(cp[n] > 0 for n in names), flush=True)
    names = names[int(os.environ.get("SKIP", 0)):]
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
                r_.update(rect=list(map(int, best[0])), conf=best[1], start=list(map(int, best[2])), start_acc_max=float(acc.max()))
                if tag == "fold0" and idx[nm].get("bbox512"): r_["iou"] = RS.iou(best[0], idx[nm]["bbox512"])
            r_.update(n_decodes=D.n - n0, sec=round(time.time() - t0, 1)); res[nm] = r_
            print(tag, nm, r_, flush=True); fout.write_text(json.dumps(res))
    (OUT / f"DONE_{tag}{os.environ.get('SUF', '')}").write_text("ok"); print("DONE", flush=True)
