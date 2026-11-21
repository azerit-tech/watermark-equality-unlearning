"""Shifted-grid TTA for the tile segmenters (TrustMark tiles2, PixelSeal tiles3), EMA weights of a LAST checkpoint.
The image stays in place; only the GRID of overlapping windows moves. Fixed a priori: 4 grids shifted along the
diagonal by 0, 1/4, 1/2, 3/4 of the baseline stride (stride = positions(T)[1] - positions(T)[0]: 96 px at T=320,
128 px at T=256), window starts = shift + k*stride clamped to [0, 512-T] (shift 0 == baseline grid exactly),
pyramid-weighted mean inside a grid, plain mean of the 4 grid maps. Tile sizes: --scales (mean over scales).
Run from TASK on one GPU:
  python t1/tta/tta_infer.py --code tiles2 --method trustmark --ckpt t1/out/tiles2/trustmark/last.pt --scales 320 \
      --out t1/out/tta/trustmark --fold 0 [--test 1]
  python t1/tta/tta_infer.py --code devel --method trustmark --ckpt t1/out/devel/trustmark/fold1/t/last.pt --scales 320 \
      --out t1/out/tta/devel/trustmark/fold1 --fold 1 [--test 1]      (PixelSeal: .../pixelseal/fold1/b/last.pt, 256,320)
  python t1/tta/tta_infer.py --code tiles3 --method pixelseal --ckpt t1/out/tiles3/pixelseal/last.pt --scales 256,320 \
      --out t1/out/tta/pixelseal --fold 0 [--test 1]
Writes in --out: {fold<k>,test}_t<T>_g<j>_probs128.npz (grid j, scale T), {fold<k>,test}_base_probs128.npz (grid 0,
mean of scales = current inference) and {fold<k>,test}_probs128.npz (TTA: mean over grids and scales)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True, choices=["tiles2", "tiles3", "devel"])
ap.add_argument("--method", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scales", default="320")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--test", type=int, default=0)
ap.add_argument("--skip_fold", type=int, default=0)
a = ap.parse_args()

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "t1" / a.code))
if a.code == "tiles2":
    sys.path.insert(1, str(TASK / "t1" / "train"))
if a.code == "devel":   # out-of-fold models of t1/devel (train_oof.py)
    from train_oof import TileSeg, positions, pyramid  # noqa: E402
else:
    from train_tiles import TileSeg, positions, pyramid  # noqa: E402
from train_tm import bits, test_names  # noqa: E402
from shapes import meta, names_in_folds, read_img  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

FRACS = (0.0, 0.25, 0.5, 0.75)


def grid(T, frac, S=512):
    base = positions(T, S); st = int(base[1] - base[0]); d = int(round(frac * st / 4)) * 4
    ks = range(-2, (S // st) + 3)
    return np.unique(np.clip([d + k * st for k in ks], 0, S - T))


@torch.no_grad()
def predict_grid(model, imgs, msgs, T, pos, bs=32, device="cuda"):
    """Same as train_tiles.predict_tiled, with window starts `pos` instead of positions(T)."""
    model.eval(); g = T // 4; w = pyramid(g).to(device)
    wins = [(i, y, x) for i in range(len(imgs)) for y in pos for x in pos]
    acc = torch.zeros(len(imgs), 128, 128, device=device); den = torch.zeros_like(acc)
    X = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)
    M = torch.from_numpy(np.stack(msgs))
    for k in range(0, len(wins), bs):
        ch = wins[k:k + bs]
        x = torch.stack([X[i, :, y:y + T, xx:xx + T] for i, y, xx in ch]).to(device).float().div(255)
        m = M[[i for i, _, _ in ch]].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = torch.sigmoid(model(x, m).float())
        for (i, y, xx), p in zip(ch, pr):
            acc[i, y // 4:y // 4 + g, xx // 4:xx // 4 + g] += p * w
            den[i, y // 4:y // 4 + g, xx // 4:xx // 4 + g] += w
    return (acc / den).cpu().numpy()


def main():
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); scales = [int(s) for s in a.scales.split(",")]
    torch.backends.cudnn.benchmark = True
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False); args = ck.get("args", {})
    wm = load_watermark(a.method, device="cpu")
    if a.code in ("tiles2", "devel"):
        model = TileSeg(wm, a.method, args["res"], ppl=bool(args["ppl"])).cuda()
    else:
        model = TileSeg(wm, a.method, args.get("res", 256), ppl=False).cuda()
    del wm
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"]); model.eval()
    print(f"ckpt {a.ckpt} it={ck.get('it')}", flush=True)
    for T in scales:
        print(T, [list(grid(T, f)) for f in FRACS], flush=True)
    sets = []
    if not a.skip_fold:
        vn = names_in_folds([a.fold], a.method)
        sets.append((f"fold{a.fold}", vn, "train", [bits(meta()[n]["message"]) for n in vn]))
    if a.test:
        tn, tm = test_names(a.method); sets.append(("test", tn, "test", [tm[n] for n in tn]))
    for tag, names, split, msgs in sets:
        imgs = [read_img(n, split) for n in names]; maps = {}
        for T in scales:
            for j, f in enumerate(FRACS):
                maps[T, j] = predict_grid(model, imgs, msgs, T, grid(T, f))
                np.savez_compressed(out / f"{tag}_t{T}_g{j}_probs128.npz", names=np.array(names),
                                    probs=maps[T, j].astype(np.float16))
                print(f"{tag} T={T} grid {j} done", flush=True)
        base = np.mean([maps[T, 0] for T in scales], 0)
        tta = np.mean([maps[T, j] for T in scales for j in range(len(FRACS))], 0)
        np.savez_compressed(out / f"{tag}_base_probs128.npz", names=np.array(names), probs=base.astype(np.float16))
        np.savez_compressed(out / f"{tag}_probs128.npz", names=np.array(names), probs=tta.astype(np.float16))
        print(f"wrote {tag} ({len(names)} images)", flush=True)


if __name__ == "__main__":
    main()
