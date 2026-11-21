"""Tasks A/B (jacques2): GPU rectangle 'cloud' search for minority-regime images whose best rectangle conf is < 0.9.
For each image: ~40 candidates (stegaedge perimeter candidates, recomputed if absent, + the current best rect + seg rects),
around each candidate a cloud of N rects whose 4 edges are shifted independently by U{-J..J} px (candidate included),
all decoded (crop R+3, same preprocessing/objective as t1/boxsearch3/rectsearch.Dec = t1/sgbox2/common.Dec), then the
3 best distinct (IoU < 0.5) are refined edge by edge (steps 4,2,1 px, + translations) and the best kept.
Usage (from TASK):  python t1/jacques2/cloud.py --method stegastamp --split test --out DIR [--device cuda] [--smoke]
Env DEADLINE (epoch s): stops cleanly before it -> exit 3 (resumes from DIR/cloud_<split>.json when relaunched).
At the end writes DIR/<method>_<split>_boxes.json = base boxes, entry replaced when the cloud conf is higher
(src 'cloud'); threshold 0.9 unchanged (applied by the consumer). Train: prints outpaintings found / IoU / false positives."""
import argparse, json, os, sys, time
from pathlib import Path
import torch  # before onnxruntime
import numpy as np
from PIL import Image
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK / "t1/boxsearch3"), str(TASK / "t1/stegaedge"), str(TASK / "t1/common"), str(TASK)]
import rectsearch as RS  # noqa: E402
import perim as PM  # noqa: E402
THR = {"pixelseal": 0.80, "stegastamp": 0.755}   # regime thresholds (full-image acc)
CONF = 0.9                                        # rectangle acceptance threshold (unchanged)
A = 16                                            # max aspect ratio (both methods)
FMT = "rect=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px"


def iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    i = iy * ix; return i / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - i)


def ok(r):
    h, w = r[1] - r[0], r[3] - r[2]
    return r[0] >= 0 and r[2] >= 0 and r[1] <= 512 and r[3] <= 512 and h >= 6 and w >= 6 and max(h / w, w / h) <= A


def clip(r):
    return (max(0, r[0]), min(512, r[1]), max(0, r[2]), min(512, r[3]))


def base_boxes(M, split):
    """current best rect per image (same files as the wave-1 consumers)."""
    if M == "stegastamp":
        return json.loads((TASK / f"t1/out/stegaedge/stegastamp_{split}_boxes.json").read_text())["boxes"]
    if split == "test":
        b = json.loads((TASK / "t1/out/psbox/pixelseal_test_boxes.json").read_text())["boxes"]
        b0 = json.loads((TASK / "t1/out/boxsearch/pixelseal_test_boxes.json").read_text())["boxes"]   # for seg_rect only
        return {n: {"rect": v["rect"], "conf": v["conf"], "src": v.get("src", "pass1"), "seg_rect": b0.get(n, {}).get("seg_rect")}
                for n, v in b.items()}
    b1 = json.loads((TASK / "t1/out/boxsearch/pixelseal_train_boxes.json").read_text())["boxes"]   # same merge as psbox/cv_build
    p3 = json.loads((TASK / "t1/out/boxsearch3/pixelseal_train_pass3.json").read_text())
    out = {}
    for n, v in b1.items():
        r, c, s = v["rect"], v["conf"], "pass1"
        if n in p3 and p3[n]["conf"] > c: r, c, s = p3[n]["rect"], p3[n]["conf"], "pass3"
        out[n] = {"rect": list(map(int, r)), "conf": float(c), "src": s, "seg_rect": v.get("seg_rect")}
    return out


def refine(D, x, msg, r, obj, steps=(4, 2, 1), max_it=8):
    for st in steps:
        for _ in range(max_it):
            c = []
            for e in range(4):
                for d in (-st, st):
                    q = list(r); q[e] += d; c.append(tuple(q))
            for (p, q_) in ((0, 1), (2, 3)):
                for d in (-st, st):
                    q = list(r); q[p] += d; q[q_] += d; c.append(tuple(q))
            c = [q for q in c if ok(q)]
            if not c: break
            o, _, _ = D.score(x, c, msg); j = int(np.argmax(o))
            if o[j] <= obj: break
            r, obj = c[j], o[j]
    return r, obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--K", type=int, default=40); ap.add_argument("--N", type=int, default=50); ap.add_argument("--J", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    M, S = a.method, a.split; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.smoke: a.K, a.N = 4, 5
    deadline = float(os.environ.get("DEADLINE", 1e12))
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{S}/metadata.json").read_text()) if m["method"] == M}
    fz = np.load(TASK / f"t1/out/signal/fullacc_{M}_{S}.npz"); facc = dict(zip(fz["names"].tolist(), fz["acc"].tolist()))
    base = base_boxes(M, S)
    edge = {}
    ef = TASK / f"t1/out/stegaedge/edge_{S}.json"
    if M == "stegastamp" and ef.exists():
        edge = json.loads(ef.read_text())["boxes"]
    if S == "test":
        todo = sorted(n for n, v in base.items() if v["conf"] < CONF and facc[n] < THR[M])
    else:   # train: every base entry below 0.9 (missed outpaintings + other minority images = false-positive check)
        todo = sorted(n for n, v in base.items() if v["conf"] < CONF)
    if a.smoke: todo = todo[:2]
    fres = out / f"cloud_{S}.json"
    res = json.loads(fres.read_text()) if fres.exists() else {}
    wm = RS.load_watermark(M, device=a.device); D = RS.Dec(wm, a.device)
    D.bs = 96 if M == "stegastamp" else 256
    if hasattr(wm, "sess"): print("onnx providers", wm.sess.get_providers(), flush=True)
    print(f"{M} {S}: todo {len(todo)} (base conf<{CONF}), already done {sum(n in res for n in todo)}, K={a.K} N={a.N} J={a.J}", flush=True)
    t00 = time.time(); complete = True
    for k, nm in enumerate(todo):
        if nm in res: continue
        if time.time() > deadline: complete = False; break
        t0 = time.time(); n0 = D.n
        img = np.asarray(Image.open(TASK / f"release/{S}/images/{nm}.png").convert("RGB"))
        x = torch.from_numpy(np.array(img)).to(a.device).permute(2, 0, 1)[None].float() / 255
        msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
        if nm in edge and edge[nm].get("cands"):
            C = [tuple(c) for c in edge[nm]["cands"]][:a.K]
        else:
            Sc, pr = PM.score_all(PM.side_maps(img, sign=True), a.device)
            v, ix = torch.topk(Sc.flatten(), 4000); ix = ix.cpu().numpy(); n = len(pr); C = []; del Sc
            for j in ix:
                (y0, y1), (x0, x1) = pr[j // n], pr[j % n]; r = (4 * y0, 4 * y1, 4 * x0, 4 * x1)
                if all(iou(r, q) < 0.5 for q in C): C.append(r)
                if len(C) >= a.K: break
        extra = [tuple(base[nm]["rect"])] + [tuple(r) for r in (base[nm].get("seg_rect") or [])]
        C = [tuple(map(int, c)) for c in extra + C]
        C = list(dict.fromkeys(c for c in C if ok(c)))
        rng = np.random.default_rng(int(nm[:8], 16) % (2 ** 31))
        cloud = []
        for c in C:
            cloud.append(c)
            for _ in range(a.N - 1):
                q = clip(tuple(int(c[e] + rng.integers(-a.J, a.J + 1)) for e in range(4)))
                if ok(q): cloud.append(q)
        cloud = list(dict.fromkeys(cloud))
        o, acc, _ = D.score(x, cloud, msg); order = np.argsort(-o); starts = []
        for j in order:
            if all(iou(cloud[j], s[0]) < 0.5 for s in starts): starts.append((cloud[j], o[j]))
            if len(starts) >= 3: break
        best = None
        for s, o0 in starts:
            r, ob = refine(D, x, msg, s, o0)
            if best is None or ob > best[1]: best = (r, ob)
        _, ac, mar = D.score(x, [best[0]], msg)
        rr = {"rect": list(map(int, best[0])), "conf": float(ac[0]), "margin": float(mar[0]), "cloud_max_acc": float(acc.max()),
              "old_conf": float(base[nm]["conf"]), "old_rect": list(map(int, base[nm]["rect"])), "n_cands": len(C),
              "n_decodes": D.n - n0, "sec": round(time.time() - t0, 2)}
        if S == "train":
            rr["edit_type"] = idx[nm]["edit_type"]
            if rr["edit_type"] == "outpainting":
                rr["iou"] = iou(best[0], idx[nm]["bbox512"]); rr["old_iou"] = iou(base[nm]["rect"], idx[nm]["bbox512"])
        res[nm] = rr; fres.write_text(json.dumps(res))
        print(k, nm, rr.get("edit_type", ""), "old %.2f -> %.2f" % (rr["old_conf"], rr["conf"]), rr["rect"],
              "iou %.3f (old %.3f)" % (rr["iou"], rr["old_iou"]) if "iou" in rr else "", rr["n_decodes"], "dec", rr["sec"], "s", flush=True)
    print(f"elapsed {time.time() - t00:.0f}s, done {sum(n in res for n in todo)}/{len(todo)}", flush=True)
    if not complete:
        print("DEADLINE reached: relaunch to resume", flush=True); sys.exit(3)
    # ---- final boxes file (base format; new rect only where the cloud is better)
    boxes = {}
    for n, v in base.items():
        b = {"rect": list(map(int, v["rect"])), "conf": float(v["conf"]), "src": v.get("src", "base")}
        if n in res and res[n]["conf"] > b["conf"]:
            b = {"rect": res[n]["rect"], "conf": res[n]["conf"], "src": "cloud"}
        if M == "pixelseal": b.update(box=b["rect"], selected=b["conf"] >= CONF)
        boxes[n] = b
    d = {"format": FMT, "method": M, "boxes": boxes}
    if M == "pixelseal": d["thr"] = CONF
    (out / f"{M}_{S}_boxes.json").write_text(json.dumps(d, indent=0))
    new = [n for n in res if res[n]["conf"] >= CONF and res[n]["old_conf"] < CONF]
    print(f"{S}: searched {len(res)}, newly >= {CONF}: {len(new)} {new} | total conf>={CONF}: {sum(b['conf'] >= CONF for b in boxes.values())}")
    if S == "train":
        sys.path.insert(0, str(TASK)); from wmloc.utils import miou_image, downsample_mask  # noqa
        from evaluate import load_train  # noqa
        _, gt = load_train()
        def r128(r):
            big = np.zeros((512, 512), np.uint8); big[r[0]:r[1], r[2]:r[3]] = 1; return downsample_mask(big).astype(np.uint8)
        op = [n for n in res if res[n]["edit_type"] == "outpainting"]; ot = [n for n in res if res[n]["edit_type"] != "outpainting"]
        fo = {n: i for i, f in enumerate(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]) for n in f}
        print(f"train outpaintings searched {len(op)}: newly >= {CONF}: {sum(n in new for n in op)}, "
              f"mean IoU of those {np.mean([res[n]['iou'] for n in op if n in new]) if any(n in new for n in op) else float('nan'):.3f}, "
              f"rect score (miou) of those {np.mean([miou_image(r128(res[n]['rect']), gt[n]) for n in op if n in new]) if any(n in new for n in op) else float('nan'):.3f}")
        print(f"train non-outpaintings searched {len(ot)}: FALSE POSITIVES conf >= {CONF}: {[(n, res[n]['edit_type'], round(res[n]['conf'], 3)) for n in ot if res[n]['conf'] >= CONF]}")
        print("by fold (newly found outpaintings):", {f: sum(fo[n] == f for n in op if n in new) for f in range(5)})
    (out / f"DONE_{S}").write_text("ok %.0fs" % (time.time() - t00))


if __name__ == "__main__":
    main()
