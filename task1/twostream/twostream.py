"""Two-stream refiner (cheap variant): voie 1 = tile segmenter on the official extractor (t1/tiles, frozen, its 128x128
map precomputed); voie 2 = light full-resolution U-Net (t1/train/unet.py) that gets RGB 512 + the voie-1 logit map
upsampled to 512 and outputs a CORRECTION added to the voie-1 logits (head initialised to 0 -> starts = voie 1).
Phase A (cached): synthetic composites (SynthDS, same generator as voie 1) + real folds 1-4, voie-1 map by tiled
inference (T=320) with t1/out/tiles/<m>/best.pt (= last state of that run). Phase B: train U-Net --minutes.
Eval: LAST checkpoint, fold 0 real (voie-1 maps = t1/out/tiles/<m>/fold0_probs128.npz), by edit type and minority size.
Run from TASK:  python t1/twostream/twostream.py --method pixelseal --minutes 6 --out t1/out/twostream/pixelseal
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
for p in (TASK / "t1" / "tiles", TASK / "t1" / "tmdense", TASK / "t1" / "synth", TASK / "t1" / "common", TASK / "t1" / "train", TASK):
    sys.path.insert(0, str(p))
from unet import UNet  # noqa: E402
from train_tm import MsgSynth, bits, loss_fn, to128, test_names, binarize  # noqa: E402
from shapes import meta, names_in_folds, read_img, read_mask, train_index  # noqa: E402
from evaluate import evaluate  # noqa: E402


def logit(p):
    p = torch.as_tensor(p).float().clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def build_cache(a, cache):
    from train_tiles import TileSeg, predict_tiled
    from wmloc.watermarks import load_watermark
    dev = a.device
    wm = load_watermark(a.method, device="cpu")
    ck = torch.load(TASK / "t1/out/tiles" / a.method / "best.pt", map_location=dev, weights_only=False)
    ca = ck.get("args", {}); print("voie1 args", {k: ca.get(k) for k in ("res", "ppl", "tile_inf")}, flush=True)
    v1 = TileSeg(wm, a.method, ca.get("res", a.res), ppl=bool(ca.get("ppl", 0))).to(dev); del wm
    v1.load_state_dict(ck["model"]); v1.eval()
    ds = MsgSynth(a.synth, a.method, mix=a.mix); ds.seed = 4242
    dl = DataLoader(ds, batch_size=16, sampler=list(range(a.n_synth)), num_workers=7)
    X, L, I, P, T = [], [], [], [], []
    t0 = time.time()
    for x, lab, ign, t, msg in dl:
        imgs = list(x.permute(0, 2, 3, 1).numpy())
        P.append(predict_tiled(v1, imgs, list(msg.numpy()), 320, device=dev))
        l, i = to128(lab, ign); X.append(x.numpy()); L.append(l.numpy().astype(np.uint8)); I.append(i.numpy().astype(np.uint8))
        T.append(t.numpy())
        print(f"synth {sum(len(p) for p in P)} {time.time() - t0:.0f}s", flush=True)
    real = names_in_folds([1, 2, 3, 4], a.method)
    imgs = [read_img(n) for n in real]
    P.append(predict_tiled(v1, imgs, [bits(meta()[n]["message"]) for n in real], 320, device=dev))
    X.append(np.stack(imgs).transpose(0, 3, 1, 2))
    l, i = to128(torch.from_numpy(np.stack([read_mask(n) for n in real])), torch.zeros(len(real), 512, 512))
    L.append(l.numpy().astype(np.uint8)); I.append(i.numpy().astype(np.uint8)); T.append(np.full(len(real), 4))
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "x.npy", np.concatenate(X))
    np.savez(cache / "meta.npz", lab=np.concatenate(L), ign=np.concatenate(I), probs=np.concatenate(P).astype(np.float16),
             typ=np.concatenate(T))
    print(f"cache done {time.time() - t0:.0f}s", flush=True)


class Refiner(nn.Module):
    def __init__(self, chans):
        super().__init__()
        self.unet = UNet(in_ch=4, chans=chans, stem_depth=1, fp32_stem=True)
        nn.init.zeros_(self.unet.head.weight); nn.init.zeros_(self.unet.head.bias)

    def forward(self, x, lg128):
        """x [B,3,512,512] in [0,1]; lg128 voie-1 logits [B,128,128] -> refined logits [B,128,128]."""
        m = F.interpolate(lg128[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False) / 4
        d = self.unet(torch.cat([x * 2 - 1, m], 1))
        return lg128 + F.avg_pool2d(d[:, None], 4)[:, 0]


def aug(x, lg, lab, ign, k, fl):
    if fl:
        x, lg, lab, ign = (t.flip(-1) for t in (x, lg, lab, ign))
    return tuple(torch.rot90(t, k, (-2, -1)) for t in (x, lg, lab, ign))


@torch.no_grad()
def predict(model, imgs, probs, dev, bs=8):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).to(dev).float().div(255)
        lg = logit(probs[i:i + bs]).to(dev)
        out.append(torch.sigmoid(model(x, lg)).cpu().numpy())
    return np.concatenate(out)


def report(names, base, new, path):
    pb = evaluate({n: binarize(p) for n, p in zip(names, base)}, names=names, return_per_image=True)
    pn = evaluate({n: binarize(p) for n, p in zip(names, new)}, names=names, return_per_image=True)
    pb, pn = pb["per_image"], pn["per_image"]
    ti = train_index(); g = {}
    for n in names:
        mask = read_mask(n); f = (F.avg_pool2d(torch.from_numpy(mask)[None, None].float(), 4) >= 0.5).float().mean().item()
        mino = min(f, 1 - f); sz = "min<5%" if mino < 0.05 else ("min5-15%" if mino < 0.15 else "min>15%")
        for k in ("all", ti[n]["edit_type"], sz):
            g.setdefault(k, []).append((pb[n], pn[n]))
    rng = np.random.default_rng(0); res = {}
    for k, v in g.items():
        v = np.array(v); d = v[:, 1] - v[:, 0]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        res[k] = dict(n=len(v), base=round(v[:, 0].mean(), 4), twostream=round(v[:, 1].mean(), 4), gain=round(d.mean(), 4),
                      ci90=[round(float(np.percentile(bs, 5)), 4), round(float(np.percentile(bs, 95)), 4)])
        print(k, res[k], flush=True)
    path.write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--synth", default="t1/out/synth/v0")
    ap.add_argument("--mix", default="outpainting=1,transplant=1,hole=2")
    ap.add_argument("--n_synth", type=int, default=1024)
    ap.add_argument("--p_real", type=float, default=0.2)
    ap.add_argument("--chans", default="16,24,32,48,64,96")
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(8); dev = a.device
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.benchmark = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); cache = out / "cache"
    if not (cache / "meta.npz").exists():
        build_cache(a, cache)
    X = np.load(cache / "x.npy", mmap_mode="r"); md = np.load(cache / "meta.npz")
    LAB, IGN, PR, TYP = (torch.from_numpy(md[k]) for k in ("lab", "ign", "probs", "typ"))
    real_idx = np.nonzero(TYP.numpy() == 4)[0]; syn_idx = np.nonzero(TYP.numpy() != 4)[0]
    print(f"cache: synth={len(syn_idx)} real={len(real_idx)}", flush=True)
    model = Refiner(tuple(int(c) for c in a.chans.split(","))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0); t0 = time.time(); it = 0; rl = []
    model.train()
    while time.time() - t0 < a.minutes * 60:
        frac = (time.time() - t0) / (a.minutes * 60)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1, (it + 1) / 100) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.97 + 0.03)
        idx = np.where(rng.random(a.bs) < a.p_real, rng.choice(real_idx, a.bs), rng.choice(syn_idx, a.bs))
        x = torch.from_numpy(np.stack([X[i] for i in idx])).to(dev).float().div(255)
        lg = logit(PR[idx].float()).to(dev); lab = LAB[idx].to(dev).float(); ign = IGN[idx].to(dev).float()
        x, lg, lab, ign = aug(x, lg, lab, ign, int(rng.integers(4)), bool(rng.integers(2)))
        loss, lb, ld = loss_fn(model(x, lg), lab, ign)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); it += 1; rl.append(float(loss))
        if it % 50 == 0:
            print(f"it {it} {(time.time() - t0) / 60:.1f}min loss={np.mean(rl):.4f}", flush=True); rl = []
    torch.save({"model": model.state_dict(), "args": vars(a), "it": it}, out / "last.pt")
    print(f"trained it={it}", flush=True)
    src = TASK / "t1/out/tiles" / a.method
    f0 = np.load(src / "fold0_probs128.npz"); names = [str(n) for n in f0["names"]]
    assert set(names) == set(names_in_folds([0], a.method))
    new = predict(model, [read_img(n) for n in names], f0["probs"], dev)
    np.savez_compressed(out / "cand_fold0_probs128.npz", names=np.array(names), probs=new.astype(np.float16))
    report(names, f0["probs"].astype(np.float32), new, out / "eval.json")
    te = np.load(src / "test_probs128.npz"); tn = [str(n) for n in te["names"]]
    newt = predict(model, [read_img(n, "test") for n in tn], te["probs"], dev)
    np.savez_compressed(out / "cand_test_probs128.npz", names=np.array(tn), probs=newt.astype(np.float16))
    (out / "DONE").write_text("ok")
    print("done", flush=True)


if __name__ == "__main__":
    main()
