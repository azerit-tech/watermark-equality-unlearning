"""Leak-free synthetic TrustMark cache for out-of-fold training. Run from TASK (CPU):
  OMP_NUM_THREADS=8 python t1/oof/cache_oof.py --nsyn 8000
Each synthetic sample is built from ONE source fold c in {1,2,3,4}: cover image from fold c AND shape library
(holes, silhouettes, rect sizes, size stats) from fold c only. syn_meta.json stores 'src_fold' so the model of
fold k can drop every sample with src_fold == k. Real/test features are reused from t1/out/frozen/cache9 (read only).
Writes t1/out/oof/cache/syn_*.npy + syn_meta.json"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, cv2

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/train"), str(TASK / "t1/common"), str(TASK / "t1/frozen")]
os.environ["FCACHE"] = "../oof/cache"           # cache.py writes into t1/out/frozen/../oof/cache = t1/out/oof/cache
import cache as FC  # noqa
from wmloc.watermarks import load_watermark  # noqa
from dataset import SynthDS, TYPES  # noqa
from shapes import Library, names_in_folds, folds  # noqa
from wmloc.utils import downsample_mask  # noqa


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nsyn", type=int, default=8000); ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args(); torch.set_num_threads(8)
    print("cache dir", FC.OUT.resolve(), flush=True)
    wm = load_watermark("trustmark", device="cpu")
    ds = SynthDS(TASK / "t1/out/synth/v0", "trustmark", seed=0, lib_folds=(1, 2, 3, 4))
    cov = names_in_folds((1, 2, 3, 4), regime="partial"); fold_of = {n: f for f, ns in enumerate(folds()) for n in ns}
    cf = np.array([fold_of[n] for n in cov])
    def fold_item(arr, i):
        if "cname" in arr: return fold_of[str(arr["cname"][i])]
        ci = int(arr["cover"][i]); assert ci < len(cov); return int(cf[ci])
    full = {c: [it for it in ds.full if fold_item(*it) == c] for c in (1, 2, 3, 4)}
    rect = {c: [it for it in ds.rect if fold_item(*it) == c] for c in (1, 2, 3, 4)}
    libs = {c: Library((c,)) for c in (1, 2, 3, 4)}
    for c in (1, 2, 3, 4):
        L = libs[c]; print(f"fold {c}: full={len(full[c])} rect={len(rect[c])} holes={len(L.holes)} patches={len(L.patches)} "
                           f"rects_tm={sum(r[2] == 'trustmark' for r in L.rects)} rects={len(L.rects)}", flush=True)
    rng = np.random.default_rng(a.seed); types, src = [], []
    def gs(j):
        c = int(rng.integers(1, 5)); ds.full, ds.rect, ds.lib = full[c], rect[c], libs[c]
        x, lab, ign, t = ds.sample(rng); types.append(TYPES[t]); src.append(c)
        ig = (cv2.resize(ign.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        return np.asarray(x), downsample_mask(lab).astype(np.uint8), ig
    try:
        FC.run("syn", a.nsyn, gs, wm)
    finally:
        json.dump({"type": types, "src_fold": src}, open(FC.OUT / "syn_meta.json", "w"))


if __name__ == "__main__":
    main()
