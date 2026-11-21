"""CPU StegaStamp decoder (ONNX CPUExecutionProvider, 8 threads), same preprocessing as rectsearch.Dec (crop R+3, bicubic antialias 400)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "8")
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import onnxruntime as ort
from PIL import Image
torch.set_num_threads(8)
TASK = Path(os.environ.get("TASK", os.getcwd()))
PAD = 3
LOG = "gen_decoder/decoder_out/BiasAdd:0"


def iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    i = iy * ix; return i / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - i)


def pad(r):
    return (max(0, r[0] - PAD), min(512, r[1] + PAD), max(0, r[2] - PAD), min(512, r[3] + PAD))


class Dec:
    def __init__(self, bs=16):
        so = ort.SessionOptions(); so.intra_op_num_threads = 8; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(TASK / "models/stegastamp/stega_stamp_logits.onnx"), so, providers=["CPUExecutionProvider"])
        self.bs = bs; self.n = 0; self.t = 0.0

    def logits(self, x, boxes):
        t0 = time.time(); out = []
        for i in range(0, len(boxes), self.bs):
            xs = torch.cat([F.interpolate(x[:, :, b[0]:b[1], b[2]:b[3]], size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
                            for b in boxes[i:i + self.bs]]).permute(0, 2, 3, 1).numpy().astype(np.float32)
            out.append(self.sess.run([LOG], {"image": xs, "secret": np.zeros((len(xs), 100), np.float32)})[0])
        self.n += len(boxes); self.t += time.time() - t0
        return np.concatenate(out)

    def score(self, x, rects, msg):
        L = self.logits(x, [pad(r) for r in rects]); s = 2.0 * msg - 1
        acc = ((L > 0) == msg.astype(bool)).mean(1)
        sm = L * s; sd = L.std(1, keepdims=True) + 1e-6
        return acc + 0.1 * np.clip(sm / sd, -1, 1).mean(1), acc, sm.mean(1)


def load(split, nm):
    img = np.asarray(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float() / 255


def meta(split):
    return {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text()) if m["method"] == "stegastamp"}


def msgof(m):
    return np.array([int(c) for c in m["message"]], np.float64)
