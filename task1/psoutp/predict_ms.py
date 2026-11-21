"""Multi-scale tiled inference of a tile-mode PixelSeal/TrustMark segmenter (EMA weights of a LAST checkpoint).
python t1/tiles3/predict_ms.py <method> <ckpt> <out_dir> [--scales 192,256,320,384] [--test 1]
Writes <out_dir>/{fold0,test}_t<T>_probs128.npz, prints fold-0 scores per edit type for every scale and for
fixed scale ensembles (mean of probabilities), writes <out_dir>/eval_ms.json."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; TASK = HERE.parents[1]
for p in (HERE, TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from train_tiles import TileSeg, predict_tiled  # noqa: E402
from train_tm import binarize, bits, group_scores, test_names  # noqa: E402
from shapes import meta, names_in_folds, read_img  # noqa: E402
from evaluate import evaluate  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

ENS = {"e256_320": [256, 320], "e192_384": [192, 256, 320, 384], "e256_384": [256, 320, 384], "e192_320": [192, 256, 320]}


def score(d, names):
    r = evaluate({n: binarize(d[n]) for n in names}, names, return_per_image=True)
    g = group_scores(r["per_image"], names)
    return {"all": round(r["all"][0], 4), **{k[5:]: round(v[0], 4) for k, v in g.items() if k.startswith("edit/")}}, r["per_image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method"); ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--scales", default="192,256,320,384"); ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--res", type=int, default=256)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    scales = [int(s) for s in a.scales.split(",")]
    wm = load_watermark(a.method, device="cpu")
    model = TileSeg(wm, a.method, a.res, ppl=False).cuda(); del wm
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()
    vn = names_in_folds([0], a.method); vm = [bits(meta()[n]["message"]) for n in vn]; vi = [read_img(n) for n in vn]
    P = {}
    for T in scales:
        pr = predict_tiled(model, vi, vm, T)
        P[T] = dict(zip(vn, pr))
        np.savez_compressed(out / f"fold0_t{T}_probs128.npz", names=np.array(vn), probs=pr.astype(np.float16))
    res, per = {}, {}
    for T in scales:
        res[f"t{T}"], per[f"t{T}"] = score(P[T], vn)
    for k, ss in ENS.items():
        if all(s in P for s in ss):
            res[k], per[k] = score({n: np.mean([P[s][n] for s in ss], 0) for n in vn}, vn)
    for k, v in res.items():
        print(f"{k:10s}", json.dumps(v), flush=True)
    (out / "eval_ms.json").write_text(json.dumps({"res": res, "per_image": per}, indent=1))
    if a.test:
        tn, tmsg = test_names(a.method); ti = [read_img(n, "test") for n in tn]
        for T in scales:
            pr = predict_tiled(model, ti, [tmsg[n] for n in tn], T)
            np.savez_compressed(out / f"test_t{T}_probs128.npz", names=np.array(tn), probs=pr.astype(np.float16))
        print(f"wrote test maps ({len(tn)} images)", flush=True)


if __name__ == "__main__":
    main()
