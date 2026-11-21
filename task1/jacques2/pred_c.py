"""Task C prediction: EMA of the fine-tuned LAST state (never best.pt) -> fold 0 + test maps, and an honest fold-0 comparison
with the starting checkpoint (same inference, paired bootstrap). Adapted from t1/holes/eval_ft.py (always writes test).
python t1/jacques2/pred_c.py <method> <base_ckpt> <new_ckpt> <out_dir> --scales 256,320 [--device cuda] [--smoke]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
TASK = Path(__file__).resolve().parents[2]; HOLES = TASK / "t1" / "holes"
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK, HOLES):
    sys.path.insert(0, str(p))
from train_tiles import TileSeg, predict_tiled  # noqa: E402  (t1/holes/train_tiles.py)
from train_tm import binarize, bits, test_names, train_index  # noqa: E402
from shapes import meta, names_in_folds, read_img  # noqa: E402
from evaluate import evaluate  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def load(method, ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False); a = ck["args"]
    wm = load_watermark(method, device="cpu")
    m = TileSeg(wm, method, a["res"], ppl=bool(a["ppl"])).to(dev); del wm
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); m.eval(); return m


def infer(model, imgs, msgs, scales, dev):
    return np.mean([predict_tiled(model, imgs, msgs, T, device=dev) for T in scales], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method"); ap.add_argument("base"); ap.add_argument("new"); ap.add_argument("out")
    ap.add_argument("--scales", default="320"); ap.add_argument("--device", default="cuda"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); out = Path(a.out); dev = a.device; scales = [int(s) for s in a.scales.split(",")]
    vn = names_in_folds([0], a.method)
    tn, tmsg = test_names(a.method)
    if a.smoke: vn, tn = vn[:2], tn[:2]
    vm = [bits(meta()[n]["message"]) for n in vn]; vi = [read_img(n) for n in vn]; ti = train_index()
    per, P = {}, {}
    for tag, ck in (("base", a.base), ("new", a.new)):
        model = load(a.method, ck, dev); P[tag] = infer(model, vi, vm, scales, dev)
        per[tag] = evaluate({n: binarize(p) for n, p in zip(vn, P[tag])}, vn, return_per_image=True)["per_image"]
        if tag == "new":
            np.savez_compressed(out / "fold0_probs128.npz", names=np.array(vn), probs=P["new"].astype(np.float16))
            pr = infer(model, [read_img(n, "test") for n in tn], [tmsg[n] for n in tn], scales, dev)
            np.savez_compressed(out / "test_probs128.npz", names=np.array(tn), probs=pr.astype(np.float16))
            print(f"wrote fold0_probs128.npz ({len(vn)}) and test_probs128.npz ({len(tn)}) mean {pr.mean():.3f}", flush=True)
        del model
        if dev.startswith("cuda"): torch.cuda.empty_cache()
    groups = {"all": vn}
    for et in ("hole", "transplant", "outpainting"):
        groups[et] = [n for n in vn if ti[n]["edit_type"] == et]
    rng = np.random.default_rng(0); res = {"scales": scales}
    for g, ns in groups.items():
        if not ns: continue
        b = np.array([per["base"][n] for n in ns]); w = np.array([per["new"][n] for n in ns]); d = w - b
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        res[g] = {"n": len(ns), "base": round(b.mean(), 4), "new": round(w.mean(), 4), "diff": round(d.mean(), 4),
                  "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        print(f"fold0 {g:12s}", json.dumps(res[g]), flush=True)
    (out / "eval_fold0.json").write_text(json.dumps({"res": res, "per_image": per}, indent=1))


if __name__ == "__main__":
    main()
