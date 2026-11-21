"""CPU inference of several StegaStamp U-Net checkpoints (none trained on fold-0 real images), identity view only.
  CUDA_VISIBLE_DEVICES= python t1/stegaens/infer.py   (from TASK)
Writes t1/out/stegaens/models/<tag>/{fold0,test}_probs128.npz (stegastamp images only)."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK):
    sys.path.insert(0, str(p))
from unet import UNet  # noqa: E402
from shapes import read_img  # noqa: E402

torch.set_num_threads(8)
O = TASK / "t1/out"
MODELS = {  # tag: (path, state key)
    "v0": (O / "train/v0/stegastamp/best.pt", "model"),
    "v1f0_last_ema": (O / "train/v1/stegastamp_f0/last.pt", "ema"),
    "v1f0_best": (O / "train/v1/stegastamp_f0/best.pt", "model"),
    "ft1_last": (O / "stega/ft1/last.pt", "model"),
}


def load(path, key):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    m = UNet(3 + (a.get("extra_ch", 0) or 0), hp=a.get("hp", 0.0) or 0.0,
             fp32_stem=not a.get("stem_bf16", "hp" not in a), stem_depth=a.get("stem_depth", 2) or 2)
    m.load_state_dict(ck[key]); return m.eval()


def names_for(split, fold=None):
    md = json.loads((TASK / "release" / split / "metadata.json").read_text())
    n = [m["image_name"] for m in md if m["method"] == "stegastamp"]
    if fold is not None:
        keep = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][fold])
        n = [x for x in n if x in keep]
    return n


SETS = {"fold0": ("train", names_for("train", 0)), "test": ("test", names_for("test"))}
IMGS = {k: [torch.from_numpy(read_img(n, s)).permute(2, 0, 1)[None].float().div(255) for n in ns]
        for k, (s, ns) in SETS.items()}
t0 = time.time()
for tag, (path, key) in MODELS.items():
    model = load(path, key)
    for k, (s, ns) in SETS.items():
        out = O / "stegaens/models" / tag / f"{k}_probs128.npz"
        if out.exists():
            continue
        P = []
        with torch.inference_mode():
            for x in IMGS[k]:
                pr = torch.sigmoid(model(x.contiguous(memory_format=torch.channels_last)).float())
                P.append(F.avg_pool2d(pr[:, None], 4)[0, 0].numpy())
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, names=np.array(ns), probs=np.stack(P).astype(np.float16))
        print(f"{tag} {k} {len(ns)} {time.time() - t0:.0f}s", flush=True)
(O / "stegaens/models/DONE").write_text("ok")
print("DONE", flush=True)
