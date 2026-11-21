"""jacques6 helper (no fold-0 validation anywhere, EMA of the LAST state).
  python t1/jacques6/run6.py train <script.py> <args...>
      runs a training script (t1/jacques3/train_informed.py | t1/holes/train_tiles.py) with the synthetic shape library
      built from ALL 5 real folds (defaults lib_folds/fs (1,2,3,4) -> (0,1,2,3,4)); real folds = all 5 via --val_fold -1.
  python t1/jacques6/run6.py pred <method> <ckpt> <out.npz> --scales 256,320 [--device cuda] [--smoke]
      test maps only (TileSeg or InformedTileSeg, EMA weights) -> names + float16 probs [N,128,128]."""
import argparse
import runpy
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK, TASK / "t1" / "holes", TASK / "t1" / "jacques3"):
    sys.path.insert(0, str(p))
ALL = (0, 1, 2, 3, 4)


def train(argv):
    import holes_dataset, train_tm  # noqa: E401  same module objects as the ones the training scripts import
    for f in (train_tm.MsgSynth.__init__, holes_dataset.MsgHoleSynth.__init__, holes_dataset.Library.__init__):
        f.__defaults__ = tuple(ALL if d == (1, 2, 3, 4) else d for d in f.__defaults__)
    print(f"jacques6: shape library folds {ALL}; argv {argv}", flush=True)
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")


def pred(argv):
    import torch
    from pred_informed import load  # t1/jacques3: loads TileSeg or InformedTileSeg (EMA)
    from informed import Embedder, predict_tiled_aux
    from train_tiles import predict_tiled
    from train_tm import test_names
    from shapes import read_img
    ap = argparse.ArgumentParser()
    ap.add_argument("method"); ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--scales", default="320"); ap.add_argument("--device", default="cuda"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv); dev = a.device; scales = [int(s) for s in a.scales.split(",")]
    tn, tmsg = test_names(a.method)
    if a.smoke: tn = tn[:2]
    imgs = [read_img(n, "test") for n in tn]; msgs = [tmsg[n] for n in tn]
    model, inf = load(a.method, a.ckpt, dev)
    emb = Embedder(a.method, dev) if inf else None
    with torch.no_grad():
        pr = np.mean([predict_tiled_aux(model, emb, imgs, msgs, T, device=dev) if inf else
                      predict_tiled(model, imgs, msgs, T, device=dev) for T in scales], 0)
    assert np.isfinite(pr).all() and pr.shape == (len(tn), 128, 128), pr.shape
    np.savez_compressed(a.out, names=np.array(tn), probs=pr.astype(np.float16))
    print(f"wrote {a.out} ({len(tn)}) informed={inf} scales={scales} mean {pr.mean():.3f} frac>0.5 {(pr > 0.5).mean():.3f}", flush=True)


if __name__ == "__main__":
    {"train": train, "pred": pred}[sys.argv[1]](sys.argv[2:])
