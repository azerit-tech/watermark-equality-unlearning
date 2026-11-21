"""Step 1: scale hypothesis on REAL PixelSeal outpaintings of the train set (all folds, rectangle from the mask).
(a) official decoder bit accuracy on: exact crop R+3 (resized to 256 by the decoder), a 256 tile containing it, full image.
(b) tiles3 model (last.pt EMA) applied to sliding windows of T px (resized to 256 inside the model): mean prob
inside / outside the rectangle for T in 64,96,128,192,320.
python t1/pszoom/check_scale.py"""
import sys, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; TASK = HERE.parents[1]
for p in (TASK / "t1" / "tiles3", TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from train_tiles import TileSeg, predict_tiled  # noqa
from train_tm import bits  # noqa
from shapes import meta, read_img, read_mask, names_in_folds  # noqa
from wmloc.watermarks import load_watermark  # noqa

ti = json.loads((TASK / "t1/common/train_index.json").read_text())
ti = {str(d["image_name"]).rsplit(".", 1)[0]: d for d in ti} if isinstance(ti, list) else ti
names = [n for n in names_in_folds(range(5), "pixelseal") if ti[n]["edit_type"] == "outpainting"]
wm = load_watermark("pixelseal", device="cuda")
def acc(img, m):
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255).cuda()
    return float(((wm.logits(x)[0].cpu() > 0).int().numpy() == m).mean())
model = TileSeg(wm, "pixelseal", 256).cuda()
ck = torch.load(TASK / "t1/out/tiles3/pixelseal/last.pt", map_location="cuda", weights_only=False)
model.load_state_dict(ck["ema"]); model.eval()
imgs = [read_img(n) for n in names]; msgs = [bits(meta()[n]["message"]) for n in names]
Ts = [64, 96, 128, 192, 320]
P = {T: predict_tiled(model, imgs, msgs, T) for T in Ts}
rows = []
for i, n in enumerate(names):
    mk = read_mask(n); ys, xs = np.nonzero(mk); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    im = imgs[i]; m = msgs[i]
    c = im[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3]
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; S = max(256, y1 - y0 + 8, x1 - x0 + 8)
    ty, tx = int(np.clip(cy - S // 2, 0, 512 - S)), int(np.clip(cx - S // 2, 0, 512 - S))
    r = {"name": n, "area": round(mk.mean(), 4), "h": int(y1 - y0), "w": int(x1 - x0),
         "acc_crop": acc(c, m), "acc_tile": acc(im[ty:ty + S, tx:tx + S], m), "acc_full": acc(im, m)}
    g = mk.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
    for T in Ts:
        r[f"in{T}"] = round(float(P[T][i][g].mean()), 3); r[f"out{T}"] = round(float(P[T][i][~g].mean()), 3)
    rows.append(r)
rows.sort(key=lambda r: r["area"])
for r in rows:
    print(f"{r['name'][:14]:14s} a={r['area']:.3f} {r['h']:3d}x{r['w']:3d} crop={r['acc_crop']:.2f} tile={r['acc_tile']:.2f} "
          f"full={r['acc_full']:.2f} | " + " ".join(f"{T}:{r[f'in{T}']:.2f}/{r[f'out{T}']:.2f}" for T in Ts))
for lab, sel in (("small<3%", [r for r in rows if r["area"] < .03]), ("large>=3%", [r for r in rows if r["area"] >= .03])):
    print(lab, len(sel), {k: round(float(np.mean([r[k] for r in sel])), 3) for k in
                          ["acc_crop", "acc_tile", "acc_full"] + [f"in{T}" for T in Ts] + [f"out{T}" for T in Ts]})
out = TASK / "t1/out/pszoom"; out.mkdir(parents=True, exist_ok=True)
(out / "check_scale.json").write_text(json.dumps(rows, indent=1))
