"""Equality test: the stage maps computed by DenseSeg.encode (own torch preprocessing) must equal those of the official
wrapper (wmloc.watermarks) on 4 real images (max abs diff < 1e-4). Run from TASK (CPU is fine):
    python t1/dense/check_equal.py --method trustmark | --method pixelseal --res 256 | --method pixelseal --res 512
Reference: TrustMark -> wm.features(x)[layer1..4]; PixelSeal 256 -> hooks on convnext.stages[i] during wm.features(x)
(the wrapper's own resize + det.preprocess); PixelSeal 512 -> hooks during det.convnext(det.preprocess(x)) without
resize (the e2.py path, the wrapper only works at 256)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (HERE, HERE.parent / "synth", TASK):
    sys.path.insert(0, str(p))
from dense_model import DenseSeg  # noqa: E402
from shapes import folds, read_img  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--res", type=int, default=256)
a = ap.parse_args(); torch.set_num_threads(8)
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
wm = load_watermark(a.method, device="cpu")
x = torch.from_numpy(np.stack([read_img(n) for n in folds()[0][:4]])).permute(0, 3, 1, 2).float().div(255)
model = DenseSeg(wm, a.method, a.res).eval()
with torch.no_grad():
    if a.method == "trustmark":
        f = wm.features(x); ref = [f[k] for k in ("layer1", "layer2", "layer3", "layer4")]
    else:
        det = wm.det; ref = [None] * 4
        print("preprocess:", det.preprocess, "| convnext children:", [n for n, _ in det.convnext.named_children()])
        hs = [det.convnext.stages[i].register_forward_hook(lambda m, i_, o, i=i: ref.__setitem__(i, o.clone())) for i in range(4)]
        if a.res == 256:
            wm.features(x)
        else:
            det.convnext(det.preprocess(x))
        for h in hs:
            h.remove()
    mine = model.encode(x)
    # full forward (decoder) works and has the right shape
    lg = model(x)
ok = True
for i, (r, m) in enumerate(zip(ref, mine)):
    d = (r - m).abs().max().item(); ok &= d < 1e-4
    print(f"stage {i}: shape {tuple(m.shape)} max|diff|={d:.3e} (ref max|.|={r.abs().max().item():.2f})")
print("logits", tuple(lg.shape), "EQUAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
