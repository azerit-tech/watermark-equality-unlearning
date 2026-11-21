"""CPU tiled inference of the CURRENT tile segmenters at a smaller tile size (zoom view), same overlap rule (positions(T)).
TM: t1/out/tiles2/trustmark/last.pt (EMA), PS: t1/out/tiles3/pixelseal/last.pt (EMA).
python ~/zoomtta/infer_small.py <method> <split fold0|test> <T> [--limit N] -> ~/zoomtta/out/<method>/<split>_s<T>_probs128.npz"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
ap = argparse.ArgumentParser(); ap.add_argument("method"); ap.add_argument("split"); ap.add_argument("T", type=int)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args(); torch.set_num_threads(a.threads)
if a.method == "trustmark":
    sys.path.insert(0, str(TASK / "t1" / "tiles2")); sys.path.insert(0, str(TASK / "t1" / "train"))
    from train_tiles import TileSeg, predict_tiled, bits, test_names, meta, names_in_folds, read_img  # noqa
    ck = torch.load(TASK / "t1/out/tiles2/trustmark/last.pt", map_location="cpu", weights_only=False); args = ck["args"]
    from wmloc.watermarks import load_watermark
    wm = load_watermark("trustmark", device="cpu"); model = TileSeg(wm, "trustmark", args["res"], ppl=bool(args["ppl"]))
else:
    for p in (TASK / "t1" / "tiles3", TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK / "t1" / "train", TASK):
        sys.path.insert(0, str(p))
    from train_tiles import TileSeg, predict_tiled  # noqa
    from train_tm import bits, test_names  # noqa
    from shapes import meta, names_in_folds, read_img  # noqa
    from wmloc.watermarks import load_watermark
    ck = torch.load(TASK / "t1/out/tiles3/pixelseal/last.pt", map_location="cpu", weights_only=False)
    wm = load_watermark("pixelseal", device="cpu"); model = TileSeg(wm, "pixelseal", ck.get("args", {}).get("res", 256), ppl=False)
del wm
model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()
if a.split == "fold0":
    names = names_in_folds([0], a.method); msgs = [bits(meta()[n]["message"]) for n in names]; sp = "train"
else:
    names, tm = test_names(a.method); msgs = [tm[n] for n in names]; sp = "test"
if a.limit: names, msgs = names[:a.limit], msgs[:a.limit]
imgs = [read_img(n, sp) for n in names]
out = Path.home() / "zoomtta" / "out" / a.method; out.mkdir(parents=True, exist_ok=True)
t0 = time.time(); P = []
with torch.no_grad():
    for k in range(0, len(names), 17):
        P.append(predict_tiled(model, imgs[k:k + 17], msgs[k:k + 17], a.T, bs=16, device="cpu"))
        print(f"{k + len(P[-1])}/{len(names)} {time.time() - t0:.0f}s", flush=True)
P = np.concatenate(P)
tag = f"{a.split}_s{a.T}" + (f"_lim{a.limit}" if a.limit else "")
np.savez_compressed(out / f"{tag}_probs128.npz", names=np.array(names), probs=P.astype(np.float16))
print(f"DONE {tag} {time.time() - t0:.1f}s {(time.time() - t0) / len(names):.2f} s/img", flush=True)
