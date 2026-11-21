"""For each real outpainting image, search the crop (per-side margins around the mask rect) that decodes best.
    python t1/fidelity/rect_search.py --method stegastamp --out t1/out/fidelity/search_stegastamp.json [--sheet path.png]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch  # noqa: E402
torch.set_num_threads(4)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fid_sim import real_items, read_img, S  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def accs(wm, x, boxes, msg):
    ts = [F.interpolate(torch.from_numpy(np.ascontiguousarray(x[a:b, c:d])).permute(2, 0, 1)[None].float().div(255),
                        size=(wm.resolution,) * 2, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1) for a, b, c, d in boxes]
    lg = wm.logits(torch.cat(ts)).numpy()
    return ((lg > 0).astype(int) == np.asarray(msg)[None]).mean(1)


def clipb(b):
    y0, y1, x0, x1 = b
    return (max(y0, 0), min(y1, S), max(x0, 0), min(x1, S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="stegastamp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", default=None)
    a = ap.parse_args()
    wm = load_watermark(a.method, device="cpu")
    rows = []
    for n, box, msg in real_items(a.method):
        x = read_img(n)
        sym = [clipb((box[0] - d, box[1] + d, box[2] - d, box[3] + d)) for d in range(0, 11)]
        acc_sym = accs(wm, x, sym, msg); d0 = int(np.argmax(acc_sym))
        m = [d0] * 4; best = float(acc_sym[d0])
        for side in range(4):
            cands = []
            for v in range(-2, 11):
                mm = list(m); mm[side] = v
                cands.append(clipb((box[0] - mm[0], box[1] + mm[1], box[2] - mm[2], box[3] + mm[3])))
            ac = accs(wm, x, cands, msg); j = int(np.argmax(ac))
            if ac[j] > best:
                best = float(ac[j]); m[side] = j - 2
        rows.append({"name": n, "box": box, "acc_sym": [float(v) for v in acc_sym], "best_margins": m, "best_acc": best,
                     "short": min(box[1] - box[0], box[3] - box[2]), "long": max(box[1] - box[0], box[3] - box[2])})
        print(n, rows[-1]["short"], rows[-1]["long"], round(float(acc_sym[0]), 2), "sym best d", d0, round(float(acc_sym[d0]), 2), "margins", m, round(best, 2), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    if a.sheet:
        cells = []
        for r in sorted(rows, key=lambda r: r["short"])[::max(len(rows) // 12, 1)][:12]:
            x = read_img(r["name"]).copy(); y0, y1, x0, x1 = r["box"]; m = r["best_margins"]
            crop = x[max(y0 - 12, 0):min(y1 + 12, S), max(x0 - 12, 0):min(x1 + 12, S)]
            f = 256 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
            pad = np.full((256, 256, 3), 255, np.uint8); pad[:crop.shape[0], :crop.shape[1]] = crop
            cv2.rectangle(x, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
            cv2.rectangle(x, (x0 - m[2], y0 - m[0]), (x1 - 1 + m[3], y1 - 1 + m[1]), (255, 0, 0), 1)
            th = cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)
            cell = np.concatenate([th, pad], 1)
            cv2.putText(cell, f"{r['short']}x{r['long']} m={m} {r['acc_sym'][0]:.2f}->{r['best_acc']:.2f}", (4, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cells.append(cell)
        while len(cells) % 3:
            cells.append(np.zeros_like(cells[0]))
        grid = np.concatenate([np.concatenate(cells[i:i + 3], 1) for i in range(0, len(cells), 3)], 0)
        cv2.imwrite(a.sheet, grid[..., ::-1])


if __name__ == "__main__":
    main()
