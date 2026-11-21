"""pszoom: ZOOMED PixelSeal tile segmenter for small rectangles (fine-tune of t1/out/tiles3/pixelseal/last.pt).
Tiles of T in --tiles (64..160 px, one T per iteration) are cut from 512 images and resized to 256 inside the model
(TileSeg), so a small outpainting is seen at a scale close to the one of its residual. Data: --p_synth of the items
are faithful synthetic outpaintings (t1/out/psoutp/synth/pixelseal/rect, label = box), the rest REAL PixelSeal train
images of folds != val_fold that are NOT outpaintings (hole / transplant: real outpaintings are kept for evaluation).
With prob --p_focus the tile is centred on a minority cell (tiles3.crop_tiles). Saves EMA weights to <out>/zoom.pt.
python t1/pszoom/train_zoom.py --out t1/out/pszoom/m1 --minutes 8"""
import argparse, copy, glob, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
HERE = Path(__file__).resolve().parent; TASK = HERE.parents[1]
for p in (TASK / "t1" / "tiles3", TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK):
    sys.path.insert(0, str(p))
from train_tiles import TileSeg, crop_tiles  # noqa
from train_tm import bits, loss_fn, to128  # noqa
from shapes import meta, names_in_folds, read_img, read_mask  # noqa
from wmloc.watermarks import load_watermark  # noqa


class DS(Dataset):
    def __init__(self, real_names, p_synth, root="t1/out/psoutp/synth/pixelseal/rect"):
        self.chunks = sorted(glob.glob(str(TASK / root / "*")))
        self.n = [len(np.load(c + "/box.npy")) for c in self.chunks]
        self.real = [(read_img(n), read_mask(n), bits(meta()[n]["message"])) for n in real_names]
        self.p = p_synth; self.mm = {}

    def __len__(self):
        return 10 ** 8

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        if rng.random() < self.p:
            c = int(rng.integers(len(self.chunks))); j = int(rng.integers(self.n[c]))
            if c not in self.mm:
                d = self.chunks[c]
                self.mm[c] = (np.load(d + "/x.npy", mmap_mode="r"), np.load(d + "/box.npy"), np.load(d + "/msg.npy"))
            X, B, M = self.mm[c]
            x = np.array(X[j]); y0, y1, x0, x1 = B[j]; lab = np.zeros((512, 512), np.uint8); lab[y0:y1, x0:x1] = 1
            m = M[j]
        else:
            x, lab, m = self.real[int(rng.integers(len(self.real)))]
        if rng.random() < 0.5:                          # flips (the residual pattern is not flip-invariant, but the
            pass                                        # task images are never flipped: no augmentation)
        return (torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1), torch.from_numpy(np.ascontiguousarray(lab)),
                torch.from_numpy(np.asarray(m, np.uint8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--tiles", default="64,96,128,160"); ap.add_argument("--p_synth", type=float, default=0.5)
    ap.add_argument("--p_focus", type=float, default=0.5); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lr_enc", type=float, default=3e-5)
    ap.add_argument("--ema", type=float, default=0.995); ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--init", default="t1/out/tiles3/pixelseal/last.pt")
    a = ap.parse_args(); out = TASK / a.out; out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True; dev = "cuda"; rng = np.random.default_rng(0)
    ti = json.loads((TASK / "t1/common/train_index.json").read_text())
    ti = {str(d["image_name"]).rsplit(".", 1)[0]: d for d in ti} if isinstance(ti, list) else ti
    real = [n for n in names_in_folds([f for f in range(5) if f != a.val_fold], "pixelseal") if ti[n]["edit_type"] != "outpainting"]
    ds = DS(real, a.p_synth)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=7, drop_last=True, persistent_workers=True, prefetch_factor=4,
                    worker_init_fn=lambda w: np.random.seed(w))
    wm = load_watermark("pixelseal", device="cpu")
    model = TileSeg(wm, "pixelseal", 256).to(dev); del wm
    ck = torch.load(TASK / a.init, map_location=dev, weights_only=False)
    model.load_state_dict(ck["ema"]); model.set_frozen(False)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc = {id(p) for p in model.enc.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in enc], "lr": a.lr, "weight_decay": 1e-4},
                             {"params": list(model.enc.parameters()), "lr": a.lr_enc, "weight_decay": 0.0}])
    tiles = [int(t) for t in a.tiles.split(",")]
    print(f"real={len(real)} synth_chunks={len(ds.chunks)} tiles={tiles}", flush=True)
    t0 = time.time(); it = 0; lb_s = ld_s = 0.0; model.train()
    for x, lab, msg in dl:
        frac = (time.time() - t0) / (a.minutes * 60)
        if frac >= 1:
            break
        cos = 0.5 * (1 + math.cos(math.pi * frac)) * 0.95 + 0.05; w = min(1.0, (it + 1) / 100)
        opt.param_groups[0]["lr"] = a.lr * cos * w; opt.param_groups[1]["lr"] = a.lr_enc * cos * w
        x = x.to(dev, non_blocking=True).float().div_(255)
        l128, i128 = to128(lab.to(dev), torch.zeros_like(lab).to(dev))
        T = tiles[it % len(tiles)]
        xt, lt, it_ = crop_tiles(x, l128, i128, T, a.p_focus, rng)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(xt, msg.to(dev))
        loss, lb, ld = loss_fn(lg, lt, it_)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        it += 1; lb_s += lb; ld_s += ld
        if it % 100 == 0:
            print(f"it {it} {(time.time() - t0) / 60:.1f}min bce={lb_s / 100:.4f} dice={ld_s / 100:.4f}", flush=True); lb_s = ld_s = 0.0
    torch.save({"model": ema.state_dict(), "args": vars(a), "it": it}, out / "zoom.pt")
    print(f"done it={it} saved {out / 'zoom.pt'}", flush=True)


if __name__ == "__main__":
    main()
