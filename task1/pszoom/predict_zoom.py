"""pszoom inference + fusion + evaluation (all choices fixed a priori, nothing tuned on fold 0):
- zoom map Z = mean of predict_tiled at T = 96, 128, 160 (windows resized to 256 in the model), computed ONLY for the
  images of the minority regime (fullacc cache, load_regimes);
- fusion: on those images, if a 4-connected component of {Z >= 0.5} of >= 4 cells has mean Z > 0.8, the map becomes Z,
  otherwise the base map (t1/out/tiles3/pixelseal/{fold0,test}_probs128.npz) is kept.
Reports fold-0 scores by edit type, switches per type, paired bootstrap, and on the held-out REAL outpaintings of
folds 1-4 (never used in training): how many are switched and the IoU of the rectangle.
python t1/pszoom/predict_zoom.py t1/out/pszoom/m1/zoom.pt t1/out/pszoom/m1 [--test 1]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
HERE = Path(__file__).resolve().parent; TASK = HERE.parents[1]
for p in (TASK / "t1" / "tiles3", TASK / "t1" / "tmdense", TASK / "t1" / "train", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from train_tiles import TileSeg, predict_tiled  # noqa
from train_tm import binarize, bits, test_names  # noqa
from regime_guard import load_regimes  # noqa
from shapes import meta, names_in_folds, read_img, read_mask  # noqa
from evaluate import evaluate  # noqa
from wmloc.watermarks import load_watermark  # noqa
from wmloc.utils import downsample_mask  # noqa

TS, THR_CONF = (96, 128, 160), 0.8


def conf(z):
    lab, n = ndimage.label(z >= 0.5)
    best = 0.0
    for k in range(1, n + 1):
        c = lab == k
        if c.sum() >= 4:
            best = max(best, float(z[c].mean()))
    return best


def zmaps(model, imgs, msgs):
    return np.mean([predict_tiled(model, imgs, msgs, T) for T in TS], 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--test", type=int, default=0)
    a = ap.parse_args(); out = TASK / a.out; out.mkdir(parents=True, exist_ok=True)
    wm = load_watermark("pixelseal", device="cpu"); model = TileSeg(wm, "pixelseal", 256).cuda(); del wm
    model.load_state_dict(torch.load(TASK / a.ckpt, map_location="cuda", weights_only=False)["model"]); model.eval()
    ti = json.loads((TASK / "t1/common/train_index.json").read_text())
    ti = {str(d["image_name"]).rsplit(".", 1)[0]: d for d in ti} if isinstance(ti, list) else ti
    reg = load_regimes("train")
    # ---- fold 0
    z0 = np.load(TASK / "t1/out/tiles3/pixelseal/fold0_probs128.npz"); names = [str(n) for n in z0["names"]]
    base = dict(zip(names, z0["probs"].astype(np.float32)))
    part = [n for n in names if not reg[n]]
    Z = dict(zip(part, zmaps(model, [read_img(n) for n in part], [bits(meta()[n]["message"]) for n in part])))
    fused, info = {}, {}
    for n in names:
        fused[n] = base[n]
        if n in Z:
            c = conf(Z[n]); info[n] = c
            if c > THR_CONF:
                fused[n] = Z[n]
    pb = evaluate({n: binarize(base[n]) for n in names}, names, return_per_image=True)["per_image"]
    pf = evaluate({n: binarize(fused[n]) for n in names}, names, return_per_image=True)["per_image"]
    pz = evaluate({n: binarize(Z[n]) for n in part}, part, return_per_image=True)["per_image"]
    et = {n: ti[n]["edit_type"] for n in names}
    area = {n: float(min(downsample_mask(read_mask(n)).mean(), 1 - downsample_mask(read_mask(n)).mean())) for n in names}
    print("fold0 minority-regime images:", len(part))
    for n in sorted(part, key=lambda n: (et[n], area[n])):
        print(f"  {n} {et[n]:11s} area={area[n]:.3f} conf={info[n]:.2f} switch={int(info[n] > THR_CONF)} base={pb[n]:.3f} "
              f"zoom={pz[n]:.3f} fused={pf[n]:.3f}")
    res = {}
    for k, P in (("base", pb), ("fused", pf)):
        res[k] = {"all": round(float(np.mean([P[n] for n in names])), 4),
                  **{t: round(float(np.mean([P[n] for n in names if et[n] == t])), 4) for t in ("hole", "transplant", "outpainting")}}
        print(k, json.dumps(res[k]))
    d = np.array([pf[n] - pb[n] for n in names]); rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    boot = {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    print("paired bootstrap fused-base (34 images):", boot)
    np.savez_compressed(out / "fold0_probs128.npz", names=np.array(names), probs=np.stack([fused[n] for n in names]).astype(np.float16))
    # ---- held-out real outpaintings of folds 1-4
    ho = [n for n in names_in_folds([1, 2, 3, 4], "pixelseal") if ti[n]["edit_type"] == "outpainting" and not reg[n]]
    Zh = zmaps(model, [read_img(n) for n in ho], [bits(meta()[n]["message"]) for n in ho])
    ph = evaluate({n: binarize(z) for n, z in zip(ho, Zh)}, ho, return_per_image=True)["per_image"]
    rows = []
    for n, z in zip(ho, Zh):
        g = downsample_mask(read_mask(n)).astype(bool); m = binarize(z).astype(bool)
        iou = float((g & m).sum() / max((g | m).sum(), 1)); c = conf(z)
        rows.append((n, float(g.mean()), c, iou, ph[n]))
    print(f"folds1-4 held-out real outpaintings (minority regime): {len(ho)}")
    for r in sorted(rows, key=lambda r: r[1]):
        print(f"  {r[0]} area={r[1]:.3f} conf={r[2]:.2f} switch={int(r[2] > THR_CONF)} iou_rect={r[3]:.2f} score_zoom={r[4]:.3f}")
    sw = [r for r in rows if r[2] > THR_CONF]
    print(f"  switched {len(sw)}/{len(rows)}; IoU>0.5 among all: {sum(r[3] > 0.5 for r in rows)}; mean zoom score switched "
          f"{np.mean([r[4] for r in sw]) if sw else float('nan'):.3f}")
    (out / "eval_zoom.json").write_text(json.dumps({"res": res, "boot": boot, "conf": info, "base": pb, "fused": pf,
                                                    "heldout": rows}, indent=1))
    if a.test:
        regt = load_regimes("test"); zt = np.load(TASK / "t1/out/tiles3/pixelseal/test_probs128.npz")
        tn = [str(n) for n in zt["names"]]; tb = zt["probs"].astype(np.float32); _, tmsg = test_names("pixelseal")
        tp = [n for n in tn if not regt[n]]
        Zt = dict(zip(tp, zmaps(model, [read_img(n, "test") for n in tp], [tmsg[n] for n in tp])))
        fo, nsw = [], 0
        for n, p in zip(tn, tb):
            if n in Zt and conf(Zt[n]) > THR_CONF:
                fo.append(Zt[n]); nsw += 1
            else:
                fo.append(p)
        np.savez_compressed(out / "test_probs128.npz", names=np.array(tn), probs=np.stack(fo).astype(np.float16))
        print(f"test: minority {len(tp)}/{len(tn)}, switched {nsw}", flush=True)


if __name__ == "__main__":
    main()
