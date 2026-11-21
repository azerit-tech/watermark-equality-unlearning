"""jacques4 (tasks A = StegaStamp, B = PixelSeal): GPU version of the CPU 'seed' rectangle search (~/sgseed, ~/psseed).
Starts = bounding boxes of the components of many probability maps (TTA views, shifts, ensemble members, tiles scales,
devel folds, jacques s1) at thresholds 0.3/0.5/0.7, up to 4 components per map + bbox of their union, + the current best rect.
All starts decoded once (crop R+3, t1/boxsearch3/rectsearch.Dec, ONNX CUDA for StegaStamp), the K=10 best distinct starts
refined with PER=600 decodes each (local edge search steps 16..1 + scale moves, then random edge perturbations + local search
until the budget is spent). Threshold 0.9 on conf (bit acc of crop R+3) is NOT changed here: applied by the consumer (merge.py).
Usage (from TASK): python t1/jacques4/seedgpu.py --method stegastamp --set test --shard 0 --nshard 4 --out DIR [--device cuda] [--smoke]
Resumes from DIR/res.json; env DEADLINE (epoch s) -> clean stop, exit 3. Single output file per task: DIR/res.json."""
import argparse, json, os, sys, time
from pathlib import Path
import torch  # before onnxruntime
import numpy as np
from PIL import Image
from scipy.ndimage import label
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK / "t1/boxsearch3"), str(TASK)]
import rectsearch as RS  # noqa: E402
T = TASK / "t1/out"; J4 = TASK / "t1/jacques4"
A = 16; CONF = 0.9; MINA = 16; THRS = (0.3, 0.5, 0.7); NCOMP = 4


def map_files(m, tag):
    if m == "stegastamp":   # ~/sgseed list + mean maps + view 0 of stegashift
        fs = [T / f"stegatta/stegastamp/{tag}_v{i}_probs128.npz" for i in range(4)] + [T / f"stegatta/stegastamp/{tag}_probs128.npz"]
        fs += [T / f"stegashift/maps/{tag}_v{i}_probs128.npz" for i in range(8)] + [T / f"stegashift/maps/{tag}_probs128.npz"]
        fs += sorted((T / "stegaens/models").glob(f"*/{tag}_probs128.npz"))
        fs += [T / f"jacques/sg_all_s1/{tag}_probs128.npz"]
    else:                   # ~/psseed list + tiles3 t384 + devel/jacques means
        fs = [T / f"tiles3/pixelseal/s2_22min/{tag}_t{t}_probs128.npz" for t in (192, 256, 320, 384)]
        fs += [T / f"tta/pixelseal/{tag}_base_probs128.npz", T / f"tta/pixelseal/{tag}_probs128.npz"]
        fs += [T / f"tta/pixelseal/{tag}_t{t}_g{g}_probs128.npz" for t in (256, 320) for g in range(4)]
        fs += sorted((T / "pstta/pixelseal").glob(f"views_{tag}_*.npz")) + [T / f"pstta/pixelseal/{tag}_probs128.npz"]
        if tag == "test":
            fs += [T / f"devel/pixelseal/fold{k}/pred/test{s}_probs128.npz" for k in (1, 2, 3) for s in ("", "_t256", "_t320")]
            fs += [T / f"jacques/tiles_pixelseal_s1/test{s}_probs128.npz" for s in ("", "_t256", "_t320")]
    return [f for f in fs if f.exists()]


def load_maps(fs, names):
    out = []
    for f in fs:
        z = np.load(f); nm = z["names"].tolist(); P = z["probs"]
        out.append({n: P[i].astype(np.float32) for i, n in enumerate(nm) if n in names})
    return out


def boxes_of(p, thr):
    lab, n = label(p >= thr, structure=np.ones((3, 3)))
    if n == 0: return []
    ar = np.bincount(lab.ravel())[1:]; order = np.argsort(-ar); bs = []; keep = np.zeros_like(p, bool)
    for j in order[:NCOMP]:
        if ar[j] < MINA or ar[j] < 0.1 * ar[order[0]]: continue
        ys, xs = np.nonzero(lab == j + 1); keep[ys, xs] = True
        bs.append((4 * int(ys.min()), 4 * int(ys.max() + 1), 4 * int(xs.min()), 4 * int(xs.max() + 1)))
    if len(bs) > 1:
        ys, xs = np.nonzero(keep); bs.append((4 * int(ys.min()), 4 * int(ys.max() + 1), 4 * int(xs.min()), 4 * int(xs.max() + 1)))
    return bs


def starts_of(nm, Ms, extra):
    S = []
    for b in [tuple(map(int, e)) for e in extra] + [b for thr in THRS for m in Ms if nm in m for b in boxes_of(m[nm], thr)]:
        if b[1] - b[0] < 8 or b[3] - b[2] < 8 or (b[1] - b[0]) * (b[3] - b[2]) >= 0.6 * 512 * 512: continue
        if all(max(abs(b[i] - s[i]) for i in range(4)) > 4 for s in S): S.append(b)
    return S


def okA(q, Aeff):
    return RS.ok(q, Aeff)


def moves(r, st):
    c = []
    for e in range(4):
        for d in (-st, st):
            q = list(r); q[e] += d; c.append(tuple(q))
    for (p, q_) in ((0, 1), (2, 3)):
        for d in (-st, st):
            q = list(r); q[p] += d; q[q_] += d; c.append(tuple(q))
    for d in (-st, st):
        c.append((r[0] - d, r[1] + d, r[2] - d, r[3] + d))
    return c


def local(D, x, msg, r, obj, stop, Aeff, steps=(16, 8, 4, 2, 1), max_it=12):
    for st in steps:
        for _ in range(max_it):
            if D.n >= stop: return r, obj
            c = [q for q in moves(r, st) if okA(q, Aeff)]
            if not c: break
            o, _, _ = D.score(x, c, msg); j = int(np.argmax(o))
            if o[j] <= obj: break
            r, obj = c[j], o[j]
    return r, obj


def refine(D, x, msg, r, obj, per, rng, nb=32):
    """local search, then random 4-edge perturbations (amplitude 12 -> 2 px) + local search until `per` decodes are spent."""
    stop = D.n + per; Aeff = A if RS.ok(r, A) else 999
    r, obj = local(D, x, msg, r, obj, stop, Aeff)
    J, fails = 12, 0
    while D.n + nb <= stop:
        c = [tuple(int(r[e] + rng.integers(-J, J + 1)) for e in range(4)) for _ in range(nb)]
        c = list(dict.fromkeys(q for q in c if okA(q, Aeff)))
        if not c: break
        o, _, _ = D.score(x, c, msg); j = int(np.argmax(o))
        if o[j] > obj:
            r, obj = local(D, x, msg, c[j], o[j], stop, Aeff, steps=(4, 2, 1)); fails = 0
        else:
            fails += 1
            if fails >= 3: J, fails = max(2, J // 2), 0
    return r, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--set", required=True, choices=["test", "fold0"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--K", type=int, default=10); ap.add_argument("--PER", type=int, default=600)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    M, tag = a.method, a.set; split = "test" if tag == "test" else "train"
    if a.smoke: a.K, a.PER = 2, 40
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); fres = out / "res.json"
    deadline = float(os.environ.get("DEADLINE", 1e18))
    todo = json.loads((J4 / f"todo_{M}.json").read_text())[tag]
    names = sorted(todo, key=lambda n: (-todo[n]["conf"], n))[a.shard::a.nshard]   # balanced shards, most promising first
    if a.smoke: names = names[:2]
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text())}
    idx = json.loads((TASK / "t1/common/train_index.json").read_text()) if tag == "fold0" else {}
    res = json.loads(fres.read_text()) if fres.exists() else {}
    fs = map_files(M, tag); Ms = load_maps(fs, set(names))
    print(M, tag, f"shard {a.shard}/{a.nshard}: {len(names)} images, done {sum(n in res for n in names)}, K={a.K} PER={a.PER}, "
          f"{len(fs)} maps: {[str(f.relative_to(T)) for f in fs]}", flush=True)
    wm = RS.load_watermark(M, device=a.device); D = RS.Dec(wm, a.device)
    D.bs = (96 if M == "stegastamp" else 256) if a.device != "cpu" else (16 if M == "stegastamp" else 128)
    if hasattr(wm, "sess"): print("onnx providers", wm.sess.get_providers(), flush=True)
    t00 = time.time(); n00 = D.n; complete = True
    with torch.inference_mode():
        for k, nm in enumerate(names):
            if nm in res: continue
            if time.time() > deadline: complete = False; break
            t0 = time.time(); n0 = D.n; old = todo[nm]
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            S = starts_of(nm, Ms, [old["rect"]] if old["conf"] > 0 else [])
            rr = {"n_starts": len(S), "old_conf": old["conf"], "old_rect": old["rect"]}
            if tag == "fold0": rr["edit_type"] = old["edit_type"]
            if S:
                o, acc, _ = D.score(x, S, msg); order = np.argsort(-o); sel = []
                for j in order:
                    if all(RS.iou(S[j], S[i]) < 0.8 for i in sel): sel.append(int(j))
                    if len(sel) >= a.K: break
                rng = np.random.default_rng(int(nm[:8], 16) % (2 ** 31)); best = None
                for j in sel:
                    r, ob = refine(D, x, msg, S[j], o[j], a.PER, rng)
                    _, ac, mar = D.score(x, [r], msg)
                    if best is None or (ac[0], ob) > (best[1], best[3]): best = (r, float(ac[0]), S[j], float(ob), float(mar[0]))
                    if best[1] >= 0.97: break
                rr.update(rect=list(map(int, best[0])), conf=best[1], margin=best[4], start=list(map(int, best[2])),
                          start_acc_max=float(acc.max()), n_refined=len(sel))
                if tag == "fold0" and idx[nm].get("bbox512") and old["edit_type"] == "outpainting":
                    rr["iou"] = RS.iou(best[0], idx[nm]["bbox512"])
            rr.update(n_decodes=D.n - n0, sec=round(time.time() - t0, 1)); res[nm] = rr
            fres.write_text(json.dumps(res))
            print(k, nm, rr.get("edit_type", ""), "old %.3f -> %.3f" % (old["conf"], rr.get("conf", 0)), rr.get("rect"),
                  "starts", len(S), "dec", rr["n_decodes"], rr["sec"], "s", flush=True)
    dt = time.time() - t00
    print(f"elapsed {dt:.0f}s, {(D.n - n00) / max(dt, 1e-9):.1f} decodes/s, done {sum(n in res for n in names)}/{len(names)}", flush=True)
    if not complete:
        print("DEADLINE reached: relaunch the same task to resume", flush=True); sys.exit(3)
    new = sorted(n for n in names if res[n].get("conf", 0) >= CONF and res[n]["old_conf"] < CONF)
    print(f"SUMMARY {M} {tag} shard {a.shard}/{a.nshard}: searched {len(names)}, newly conf >= {CONF}: {len(new)} {new}")
    if tag == "fold0":
        fp = [(n, res[n]["edit_type"], round(res[n]["conf"], 3)) for n in names if res[n]["edit_type"] != "outpainting" and res[n].get("conf", 0) >= CONF]
        tc = [res[n].get("conf", 0) for n in names if res[n]["edit_type"] == "transplant"]
        rec = [(n, round(res[n]["conf"], 3), round(res[n].get("iou", -1), 3)) for n in new if res[n]["edit_type"] == "outpainting"]
        print(f"FOLD0 outpaintings recovered (conf, IoU): {rec}")
        print(f"FOLD0 transplants {len(tc)}, max conf {max(tc or [0]):.3f}, FALSE POSITIVES conf >= {CONF}: {fp}")
        if fp: print("WARNING: TRANSPLANT FALSE POSITIVE -> do not merge the test results without raising the threshold")


if __name__ == "__main__":
    main()
