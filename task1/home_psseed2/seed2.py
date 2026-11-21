"""psseed2: high-budget version of ~/psseed/seedrefine.py for PixelSeal (same maps/seeds/objective/refine/decoder), like ~/sgseed2:
K=10 starts (seeds topped up with +-8 px jitter variants, scored once, dedup), PER decodes per start (env PER, default 100).
Each worker handles its share of the val (transplant, whole-image acc < 0.80 like the test list) names first, then its share of the test names.
If any val (transplant) reaches conf >= 0.90 -> writes ~/psseed2/STOP and every worker stops (no test merge).
Usage (from TASK): THREADS=1 python ~/psseed2/seed2.py <wid> <nw>   (names from ~/psseed2/val_names.txt, test_order.txt)
Outputs ~/psseed2/{val,test}_w<wid>.json ; merge: python ~/psseed2/merge.py"""
import sys, os
TH = int(os.environ.get("THREADS", 1)); os.environ["OMP_NUM_THREADS"] = str(TH); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path.home() / "psseed"))
import seedrefine as SR  # noqa  (inserts t1/boxsearch3 in path; TASK = cwd)
RS = SR.RS
from PIL import Image
TASK, T, H = SR.TASK, SR.T, Path.home() / "psseed2"
PER, K, THR = int(os.environ.get("PER", 100)), int(os.environ.get("K", 10)), 0.90
torch.set_num_threads(TH)


def maps(tag):
    if tag in ("fold0", "test"): return SR.maps(tag)
    out = []
    for s in ("", "_t256", "_t320"):
        f = T / f"devel/pixelseal/{tag}/pred/{tag}{s}_probs128.npz"
        if f.exists(): z = np.load(f); out.append(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
    print(tag, "maps", len(out), flush=True); return out


def far(b, S): return all(max(abs(b[i] - s[i]) for i in range(4)) > 4 for s in S)


def jitter(s):
    y0, y1, x0, x1 = s; d = 8
    c = [(y0 - d, y1 + d, x0 - d, x1 + d), (y0 + d, y1 - d, x0 + d, x1 - d), (y0 - d, y1 - d, x0, x1), (y0 + d, y1 + d, x0, x1),
         (y0, y1, x0 - d, x1 - d), (y0, y1, x0 + d, x1 + d)]
    return [tuple(min(512, max(0, v)) for v in q) for q in c]


def run(D, mode, nm, M, meta, idx):
    split = "train" if mode == "val" else "test"; t0 = time.time(); n0 = D.n
    msg = np.array([int(c) for c in meta[split][nm]["message"]], np.float64)
    x = torch.from_numpy(np.array(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
    S = SR.starts_of(nm, M); P = list(S)
    for s in S[:10]:
        for q in jitter(s):
            if RS.ok(q, SR.A) and far(q, P): P.append(q)
    r_ = {"n_seeds": len(S), "n_pool": len(P)}
    if mode == "val": r_["edit_type"] = idx[nm]["edit_type"]
    if P:
        o, acc, _ = D.score(x, P, msg); order = np.argsort(-o); best = None; per = []
        for j in order[:K]:
            r, ob = SR.refine(D, x, msg, P[j], o[j], D.n + PER)
            _, ac, _ = D.score(x, [r], msg); per.append(round(float(ac[0]), 3))
            if best is None or ac[0] > best[1]: best = (r, float(ac[0]), P[j])
            if ac[0] >= THR: break
        r_.update(rect=list(map(int, best[0])), conf=best[1], start=list(map(int, best[2])), pool_acc_max=float(acc.max()), per_start=per)
    r_.update(n_decodes=D.n - n0, sec=round(time.time() - t0, 1)); return r_


if __name__ == "__main__":
    wid, nw = int(sys.argv[1]), int(sys.argv[2])
    D = RS.Dec(RS.load_watermark("pixelseal", device="cpu"), "cpu"); D.bs = 128
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    meta = {s: {m["image_name"]: m for m in json.loads((TASK / f"release/{s}/metadata.json").read_text())} for s in ("train", "test")}
    folds = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]; fold_of = {n: k for k, f in enumerate(folds) for n in f}
    z = np.load(T / "signal/fullacc_pixelseal_train.npz"); fa = dict(zip(z["names"].tolist(), z["acc"].tolist()))
    vn = [n for n in (H / "val_names.txt").read_text().split() if fa[n] < 0.80]  # same minority filter as the test list
    jobs = [("val", n) for n in vn[wid::nw]]
    jobs += [("test", n) for n in (H / "test_order.txt").read_text().split()[wid::nw]]
    res = {m: (json.loads(f.read_text()) if (f := H / f"{m}_w{wid}.json").exists() else {}) for m in ("val", "test")}
    Mc = {}
    with torch.inference_mode():
        for mode, nm in jobs:
            if (H / "STOP").exists(): print("STOP file", flush=True); break
            if nm in res[mode]: continue
            tag = "test" if mode == "test" else f"fold{fold_of[nm]}"
            if tag not in Mc: Mc[tag] = maps(tag)
            r_ = run(D, mode, nm, Mc[tag], meta, idx); res[mode][nm] = r_
            print(mode, nm, r_, flush=True); (H / f"{mode}_w{wid}.json").write_text(json.dumps(res[mode]))
            if mode == "val" and r_.get("conf", 0) >= THR: (H / "STOP").write_text(f"{nm} {r_['conf']}"); print("TRANSPLANT FP -> STOP", flush=True); break
    print("DONE", flush=True)
