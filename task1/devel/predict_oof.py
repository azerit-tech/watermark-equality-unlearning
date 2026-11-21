"""devel: tiled inference of an OOF tile segmenter (EMA of the LAST state, no selection) on held-out fold k + test.
python t1/devel/predict_oof.py --method M --ckpt ema_x.pt --fold k --scales 256,320 --out DIR [--smoke]
Writes DIR/fold{k}_t{T}_probs128.npz, DIR/test_t{T}_probs128.npz for every scale, and the delivered map
DIR/fold{k}_probs128.npz + DIR/test_probs128.npz = mean of the given scales (fixed a priori), plus DIR/eval_fold.json."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; TASK = HERE.parents[1]
for p in (HERE, TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from train_oof import TileSeg, predict_tiled  # noqa: E402
from train_tm import binarize, bits, group_scores, test_names  # noqa: E402
from shapes import meta, names_in_folds, read_img  # noqa: E402
from evaluate import evaluate  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, required=True); ap.add_argument("--scales", default="256,320")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = a.device
    scales = [int(s) for s in a.scales.split(",")]
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); args = ck["args"]
    assert args["val_fold"] == a.fold and args["method"] == a.method, (args["val_fold"], args["method"])
    wm = load_watermark(a.method, device="cpu")
    model = TileSeg(wm, a.method, args["res"], ppl=bool(args["ppl"])).to(dev); del wm
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()
    vn = names_in_folds([a.fold], a.method); tn, tmsg = test_names(a.method)
    if a.smoke:
        vn, tn = vn[:3], tn[:3]
    res = {"ckpt": a.ckpt, "it": ck.get("it"), "scales": scales}
    for tag, names, split, msgs in ((f"fold{a.fold}", vn, "train", [bits(meta()[n]["message"]) for n in vn]),
                                    ("test", tn, "test", [tmsg[n] for n in tn])):
        imgs = [read_img(n, split) for n in names]; maps = []
        for T in scales:
            pr = predict_tiled(model, imgs, msgs, T, device=dev); maps.append(pr)
            np.savez_compressed(out / f"{tag}_t{T}_probs128.npz", names=np.array(names), probs=pr.astype(np.float16))
        m = np.mean(maps, 0)
        np.savez_compressed(out / f"{tag}_probs128.npz", names=np.array(names), probs=m.astype(np.float16))
        if tag.startswith("fold"):
            for lab, p in [(f"t{T}", q) for T, q in zip(scales, maps)] + [("deliver", m)]:
                r = evaluate({n: binarize(x) for n, x in zip(names, p)}, names, return_per_image=True)
                g = group_scores(r["per_image"], names)
                res[lab] = {"all": round(r["all"][0], 4), **{kk[5:]: round(v[0], 4) for kk, v in g.items() if kk.startswith("edit/")}}
                print(lab, json.dumps(res[lab]), flush=True)
        print(f"wrote {tag} ({len(names)} images)", flush=True)
    (out / "eval_fold.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
