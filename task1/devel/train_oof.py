"""[devel OOF copy of t1/tiles3/train_tiles.py, fold k = --val_fold excluded everywhere:
  real images: folds != k ; shape library (lib_folds): folds != k ;
  synthetic 'full' items: v0 chunks restricted to covers NOT in fold k (cover index -> names_in_folds((1,2,3,4), 'partial'),
  as t1/oof/cache_oof.py and t1/tilesoof) ; synthetic 'rect' items: freshly generated for this fold by t1/synth/gen.py
  (--cover_folds/--lib_folds = folds != k, --outpaint_margin 2,5) in --rect_root, checked by cname ; tbbox: none.
  No checkpoint selection: ema_<tag>.pt = EMA of the LAST state.]
[tiles3: copy of t1/tiles/train_tiles.py + --init (warm restart from a last.pt), --stop_min (chunk length,
cosine kept over --minutes), --tag (EMA snapshot ema_<tag>.pt of the LAST state); no best-on-fold0 inference here,
see predict_ms.py]
TILE mode of the dense segmenter on the official extractor (TrustMark / PixelSeal), from t1/tmdense/train_tm.py.
Training: random T x T tiles (T in --tiles, offsets multiple of 4) cut from the 512 composites (SynthDS) and real folds
1-4; the model resizes each tile to the extractor's native resolution (TM 245, PS --res); target = mask of the tile
on the T/4 grid. With prob --p_focus the tile is centred near a cell of the minority class (small rectangles).
Inference: overlapping windows (stride T/2), probabilities averaged with a pyramid weight (low on the tile border)
-> 128 x 128 map. Validation on fold 0 real images (tiled inference, size --tile_inf).
Run from TASK:
  python t1/tiles/train_tiles.py --method trustmark --ppl 1 --minutes 8 --out t1/out/tiles/trustmark
Outputs in --out: best.pt, last.pt, log.jsonl, fold0_probs128.npz, test_probs128.npz (names, float16 [N,128,128]).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from tm_model import DenseSeg  # noqa: E402
from train_tm import MsgReal, MsgSynth, binarize, bits, group_scores, loss_fn, test_names, to128  # noqa: E402
import train_tm  # noqa: E402
from dataset import MixDS  # noqa: E402
from shapes import folds, meta, names_in_folds, read_img  # noqa: E402
from evaluate import evaluate  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


class TileSeg(DenseSeg):
    """Same network; output grid = input size / 4 (tile T -> T/4 x T/4) instead of a fixed 128."""

    def forward(self, x, msg=None):
        g = x.shape[-1] // 4
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.frozen:
                with torch.no_grad():
                    feats = self.encode(x)
            else:
                feats = self.encode(x)
        p = self.lat[3](feats[3])
        if self.ppl:
            with torch.autocast(device_type=x.device.type, enabled=False):
                sp, sm = self.signed_ppl(feats[3].float(), msg)
            p = p + self.lat_ppl(torch.cat([sp, sm], 1))
        for i in (2, 1, 0):
            p = self.lat[i](feats[i]) + F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p)
        p = F.interpolate(p, size=(g, g), mode="bilinear", align_corners=False)
        rgb = F.avg_pool2d(x.float(), 4) * 2 - 1
        hin = [p, rgb.to(p.dtype)]
        if self.ppl:
            hin.append(F.interpolate(sm, size=(g, g), mode="bilinear", align_corners=False).to(p.dtype))
        return self.head(torch.cat(hin, 1))[:, 0]


def positions(T, S=512):
    n = math.ceil((S - T) / (T // 2)) + 1
    return np.round(np.linspace(0, S - T, n) / 4).astype(int) * 4


def pyramid(g, floor=0.05):
    u = np.arange(g) + 0.5
    w = np.clip(np.minimum(u, g - u) / (g / 2), floor, 1)
    return torch.from_numpy(np.outer(w, w).astype(np.float32))


@torch.no_grad()
def predict_tiled(model, imgs, msgs, T, bs=32, device="cuda"):
    """imgs: list of uint8 HWC 512 -> probs128 float32 [N,128,128] from overlapping T-windows (pyramid-weighted mean)."""
    model.eval(); pos = positions(T); g = T // 4; w = pyramid(g).to(device); out = []
    wins = [(i, y, x) for i in range(len(imgs)) for y in pos for x in pos]
    acc = torch.zeros(len(imgs), 128, 128, device=device); den = torch.zeros_like(acc)
    X = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)
    M = torch.from_numpy(np.stack(msgs))
    for k in range(0, len(wins), bs):
        ch = wins[k:k + bs]
        x = torch.stack([X[i, :, y:y + T, xx:xx + T] for i, y, xx in ch]).to(device).float().div(255)
        m = M[[i for i, _, _ in ch]].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            pr = torch.sigmoid(model(x, m).float())
        for (i, y, xx), p in zip(ch, pr):
            acc[i, y // 4:y // 4 + g, xx // 4:xx // 4 + g] += p * w
            den[i, y // 4:y // 4 + g, xx // 4:xx // 4 + g] += w
    return (acc / den).cpu().numpy()


def crop_tiles(x, lab, ign, T, p_focus, rng):
    """x [B,3,512,512], lab/ign [B,128,128] -> tiles [B,3,T,T], [B,T/4,T/4] (offsets multiple of 4)."""
    g = T // 4; G = 128 - g; xs, ls, is_ = [], [], []
    lab_np = lab.cpu().numpy()
    for b in range(x.shape[0]):
        cy, cx = rng.integers(0, G + 1), rng.integers(0, G + 1)
        if rng.random() < p_focus:
            lb = lab_np[b]; mino = 1 if lb.mean() < 0.5 else 0
            ys, xs_ = np.nonzero(lb == mino)
            if len(ys):
                j = rng.integers(len(ys))
                cy = int(np.clip(ys[j] - rng.integers(0, g), 0, G)); cx = int(np.clip(xs_[j] - rng.integers(0, g), 0, G))
        xs.append(x[b, :, cy * 4:cy * 4 + T, cx * 4:cx * 4 + T]); ls.append(lab[b, cy:cy + g, cx:cx + g])
        is_.append(ign[b, cy:cy + g, cx:cx + g])
    return torch.stack(xs), torch.stack(ls), torch.stack(is_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--res", type=int, default=256, help="pixelseal encoder resolution (TM: 245)")
    ap.add_argument("--synth", default="t1/out/synth/v0")
    ap.add_argument("--p_real", type=float, default=0.25)
    ap.add_argument("--mix", default="outpainting=1,transplant=1,hole=2")
    ap.add_argument("--tiles", default="256,320")
    ap.add_argument("--tile_inf", type=int, default=320)
    ap.add_argument("--p_focus", type=float, default=0.5)
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--unfreeze_min", type=float, default=2)
    ap.add_argument("--eval_min", type=float, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_enc", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--ppl", type=int, default=0)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default="")
    ap.add_argument("--stop_min", type=float, default=1e9)
    ap.add_argument("--tag", default="last")
    ap.add_argument("--iters", type=int, default=0, help=">0: iteration-matched schedule (cosine over N iterations of THIS session, stop after N), replaces --minutes")
    ap.add_argument("--unfreeze_it", type=int, default=-1, help=">=0 with --iters: unfreeze the encoder at this global iteration")
    ap.add_argument("--rect_root", required=True, help="gen.py --out of the fold-k rect chunks (contains <method>/rect)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train_tm.DICE_W = a.dice_w
    dev = a.device; tiles = [int(t) for t in a.tiles.split(",")]
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(0)

    tr_folds = [f for f in range(5) if f != a.val_fold]
    real_names = names_in_folds(tr_folds, a.method)
    val_names = names_in_folds([a.val_fold], a.method)
    tst_names, tst_msg = test_names(a.method)
    if a.smoke:
        val_names, tst_names, real_names = val_names[:4], tst_names[:4], real_names[:20]
    real = MsgReal(real_names)
    k = a.val_fold; lib_f = tuple(f for f in range(5) if f != k)
    synth = MsgSynth(a.synth, a.method, mix=a.mix, lib_folds=lib_f)
    fold_of = {n: f for f, ns in enumerate(folds()) for n in ns}; cov = names_in_folds((1, 2, 3, 4), regime="partial")
    def cf(arr, i):
        if "cname" in arr: return fold_of[str(arr["cname"][i])]
        ci = int(arr["cover"][i]); assert ci < len(cov); return fold_of[cov[ci]]
    n0 = len(synth.full)
    synth.full = [it_ for it_ in synth.full if cf(*it_) != k]
    synth.rect = MsgSynth._load(Path(a.rect_root) / a.method / "rect"); synth.tbbox = []
    nmin = 5 if a.smoke else 1000
    assert len(synth.full) > nmin and len(synth.rect) > nmin, (len(synth.full), len(synth.rect))
    assert all("cname" in arr and "ebox" in arr for arr, _ in synth.rect)
    assert all(cf(*it_) != k for it_ in synth.full + synth.rect)
    assert not set(real_names) & set(val_names) and all(fold_of[n] != k for n in real_names)
    print(f"OOF fold {k}: synth full {n0}->{len(synth.full)} rect(gen)={len(synth.rect)} lib_folds={lib_f} "
          f"real={len(real_names)} mix={synth.mix}", flush=True)
    ds = MixDS(synth, real, a.p_real if synth else 1.0, length=10 ** 8)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=(dev == "cuda"), drop_last=True,
                    persistent_workers=a.workers > 0, prefetch_factor=(4 if a.workers > 0 else None))
    val_imgs = [read_img(n) for n in val_names]; val_msgs = [bits(meta()[n]["message"]) for n in val_names]
    print(f"method={a.method} tiles={tiles} tile_inf={a.tile_inf} real_train={len(real_names)} val={len(val_names)}", flush=True)

    wm = load_watermark(a.method, device="cpu")
    model = TileSeg(wm, a.method, a.res, ppl=bool(a.ppl)).to(dev); del wm
    model.set_frozen(True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc_ids = {id(p) for p in model.enc.parameters()}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": dec_params, "lr": a.lr, "weight_decay": a.wd},
                             {"params": list(model.enc.parameters()), "lr": 0.0, "weight_decay": 0.0}])
    it, elapsed0, best, it_unf = 0, 0.0, -1.0, None
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        it, elapsed0, best, it_unf = ck["it"], ck["elapsed"], ck["best"], ck.get("it_unf")
        if it_unf is not None:
            model.set_frozen(False)
    elif a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        it, it_unf = ck["it"], ck.get("it_unf")
        if it_unf is not None:
            model.set_frozen(False)
        print(f"INIT from {a.init} it={it} it_unf={it_unf}", flush=True)
    t_start = time.time(); it0 = it
    budget = a.minutes * 60; t0 = time.time() - elapsed0
    logf = open(out / "log.jsonl", "a")
    run_loss = defaultdict(float); nl = 0; tl = time.time(); t_eval = time.time()

    def validate():
        nonlocal best
        probs = predict_tiled(ema, val_imgs, val_msgs, a.tile_inf, device=dev)
        res = evaluate({n: binarize(p) for n, p in zip(val_names, probs)}, val_names, return_per_image=True)
        rec = {"it": it, "time_min": round((time.time() - t0) / 60, 2), "val": round(res["all"][0], 4),
               "groups": {k: v for k, v in group_scores(res["per_image"], val_names).items() if k.startswith("edit/")},
               "enc_frozen": model.frozen, "loss": {k: round(v / max(nl, 1), 4) for k, v in run_loss.items()}}
        if res["all"][0] > best:
            best = res["all"][0]
            torch.save({"model": ema.state_dict(), "args": vars(a), "it": it, "val": best}, out / "best.pt")
        rec["best"] = round(best, 4)
        logf.write(json.dumps(rec) + "\n"); logf.flush(); print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "it": it, "it_unf": it_unf,
                    "elapsed": time.time() - t0, "best": best, "args": vars(a)}, out / "last.pt")

    model.train()
    for x, lab, ign, typ, msg in dl:
        el = time.time() - t0; frac = (it - it0) / a.iters if a.iters > 0 else el / budget
        if frac >= 1 or (a.smoke and it >= 2) or time.time() - t_start >= a.stop_min * 60:
            break
        if model.frozen and ((a.iters > 0 and 0 <= a.unfreeze_it <= it) or (a.iters <= 0 and a.unfreeze_min >= 0 and el >= a.unfreeze_min * 60)):
            model.set_frozen(False); it_unf = it if it_unf is None else it_unf
            print(f"UNFREEZE encoder at it {it} ({el / 60:.1f} min)", flush=True)
        cos = 0.5 * (1 + math.cos(math.pi * frac)) * 0.97 + 0.03
        lr = a.lr * min(1.0, (it + 1) / 200) * cos
        lr_enc = 0.0 if model.frozen else a.lr_enc * min(1.0, (it - it_unf + 1) / 200) * cos
        opt.param_groups[0]["lr"] = lr; opt.param_groups[1]["lr"] = lr_enc
        x = x.to(dev, non_blocking=True).float().div_(255)
        lab, ign = to128(lab.to(dev, non_blocking=True), ign.to(dev, non_blocking=True))
        T = tiles[it % len(tiles)]
        x, lab, ign = crop_tiles(x, lab, ign, T, a.p_focus, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(x, msg.to(dev, non_blocking=True))
        loss, lb, ld = loss_fn(lg, lab, ign)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0); opt.step()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        it += 1; run_loss["bce"] += lb; run_loss["dice"] += ld; nl += 1
        if it % 50 == 0 or a.smoke:
            print(f"it {it} {(time.time() - t0) / 60:.1f}min bce={run_loss['bce'] / nl:.4f} dice={run_loss['dice'] / nl:.4f} "
                  f"lr={lr:.2e} lr_enc={lr_enc:.2e} {(50 if not a.smoke else 1) * a.bs / (time.time() - tl):.1f} img/s", flush=True)
            tl = time.time()
        if time.time() - t_eval >= a.eval_min * 60:
            validate(); run_loss.clear(); nl = 0; model.train(); t_eval = time.time()
    validate()
    torch.save({"model": ema.state_dict(), "args": vars(a), "it": it, "elapsed": time.time() - t0}, out / f"ema_{a.tag}.pt")
    print(f"done it={it} cum_minutes_this_schedule={(time.time() - t0) / 60:.1f} saved ema_{a.tag}.pt", flush=True)

if __name__ == "__main__":
    main()
