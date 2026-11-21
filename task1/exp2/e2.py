"""E2: linear probe (1x1 conv logistic regression) on frozen official extractors' stages.
Run from TASK on a GPU:  python t1/exp2/e2.py --method pixelseal --res 512   (or --method trustmark)
Writes t1/out/exp2/e2_<method>_<res>.json"""
import argparse, json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.stats import rankdata

TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / "t1" / "synth"))
from wmloc.watermarks import load_watermark  # noqa
from dataset import SynthDS, TYPES  # noqa
from shapes import folds, train_index, read_img, read_mask, dilate, sample_minority  # noqa

OUT = TASK / "t1" / "out" / "exp2"; OUT.mkdir(parents=True, exist_ok=True)


def auc(s, g):
    s = np.asarray(s, np.float64).ravel(); g = np.asarray(g).ravel().astype(bool)
    p = g.sum(); n = g.size - p
    if p == 0 or n == 0: return np.nan
    r = rankdata(s); return float((r[g].sum() - p * (p + 1) / 2) / (p * n))


class Ext:
    def __init__(self, method, res, dev):
        self.wm = load_watermark(method, device=dev); self.method, self.res, self.dev = method, res, dev
        if method == "pixelseal":
            det = self.wm.det; self.feats = {}
            stages = [m for n, m in det.convnext.named_modules() if n.endswith("stages")][0]
            self.stage_names = [n + f".{i}" for n, m in det.convnext.named_modules() if n.endswith("stages") for i in range(4)]
            for i, st in enumerate(stages):
                st.register_forward_hook(lambda mod, inp, out, i=i: self.feats.__setitem__(f"stage{i}", out))

    @torch.no_grad()
    def __call__(self, x):  # x B x 3 x 512 x 512 float [0,1] on dev
        if self.method == "pixelseal":
            det = self.wm.det
            xr = F.interpolate(x, size=(self.res,) * 2, mode="bilinear", align_corners=False, antialias=True) if self.res != 512 else x
            lat = det.convnext(det.preprocess(xr))
            emb = det.pixel_decoder.output_upscaling(lat); lin = det.pixel_decoder.linear
            pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
            out = {k: v for k, v in self.feats.items()}; out["ppl"] = pp[:, 1:]
            out["bitlogits"] = pp[:, 1:].mean((2, 3))
            return out
        f = self.wm.features(x)
        out = {k: f[k] for k in ("layer1", "layer2", "layer3", "layer4")}; out["ppl"] = f["per_position_logits"]
        out["bitlogits"] = self.wm.logits(x)
        return out


def collect(ext, items, bs=16):
    """items: list of (x uint8 HxWx3, lab 512, ign 512, type, msg). returns dict of CPU fp16 tensors"""
    store = {}; t0 = time.time()
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        x = torch.from_numpy(np.stack([it[0] for it in b])).permute(0, 3, 1, 2).float().div(255).to(ext.dev)
        f = ext(x); sgn = torch.from_numpy(np.stack([it[4] for it in b]).astype(np.float32) * 2 - 1).to(ext.dev)
        f["sppl"] = f.pop("ppl") * sgn[:, :, None, None]
        f["bitacc_sign"] = ((f.pop("bitlogits") * sgn) > 0).float().mean(1)
        for k, v in f.items(): store.setdefault(k, []).append(v.half().cpu())
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = (time.time() - t0) / len(items)
    return {k: torch.cat(v) for k, v in store.items()}, dt


def down(lab, h):
    return torch.from_numpy(np.stack([cv2.resize(l.astype(np.float32), (h, h), interpolation=cv2.INTER_AREA) for l in lab]))


def train_probe(F_tr, S_tr, lab, ign, use_feat, use_sppl, dev, steps=300, bs=32):
    h = F_tr.shape[-1] if use_feat else S_tr.shape[-1]
    def inp(idx, Fs, Ss):
        parts = []
        if use_feat: parts.append(Fs[idx].to(dev).float())
        if use_sppl: parts.append(F.interpolate(Ss[idx].to(dev).float(), size=(h, h), mode="bilinear", align_corners=False))
        return torch.cat(parts, 1)
    z = inp(np.arange(min(64, len(lab))), F_tr, S_tr)
    mu = z.mean((0, 2, 3), keepdim=True); sd = z.std((0, 2, 3), keepdim=True) + 1e-4
    conv = torch.nn.Conv2d(z.shape[1], 1, 1).to(dev); opt = torch.optim.Adam(conv.parameters(), lr=3e-3, weight_decay=1e-4)
    y = down(lab, h).to(dev); wgt = (1 - down(ign, h).clamp(0, 1)).to(dev)
    g = torch.Generator().manual_seed(0)
    for s in range(steps):
        idx = torch.randint(len(lab), (bs,), generator=g).numpy()
        o = conv((inp(idx, F_tr, S_tr) - mu) / sd)[:, 0]
        loss = (F.binary_cross_entropy_with_logits(o, y[idx], reduction="none") * wgt[idx]).sum() / wgt[idx].sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    @torch.no_grad()
    def predict(Fs, Ss):
        outs = []
        for i in range(0, (len(Fs) if use_feat else len(Ss)), 32):
            idx = np.arange(i, min(i + 32, len(Fs) if use_feat else len(Ss)))
            o = conv((inp(idx, Fs, Ss) - mu) / sd)
            outs.append(F.interpolate(o, size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy())
        return np.concatenate(outs)
    return predict, float(loss)


def eval_maps(P, lab, types):
    res = {}
    for t in sorted(set(types)):
        a = [auc(P[i], cv2.resize(lab[i].astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) > 0.5) for i in range(len(P)) if types[i] == t]
        res[t] = {"auc_mean": float(np.nanmean(a)) if np.any(~np.isnan(a)) else None, "n": int(np.sum(~np.isnan(a)))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--ntr", type=int, default=300); ap.add_argument("--nva", type=int, default=100)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(0); torch.set_num_threads(4)
    ext = Ext(a.method, a.res, dev)
    def synth(root, n, seed):
        """returns watermarked composed items and CONTROL items (clean cover C, same mask: content prior only)"""
        ds = SynthDS(root, a.method, seed=seed); rng = np.random.default_rng(seed); items = []; ctrl = []
        for i in range(n):
            info = {}; x, lab, ign, t = ds.sample(rng, info=info); items.append((np.asarray(x), lab, ign, TYPES[t], info["msg"].astype(np.uint8)))
            if TYPES[t] in ("hole", "transplant"):  # same recipe as SynthDS, but keeping C for the control
                A, j = ds.full[rng.integers(len(ds.full))]; C = np.asarray(A["C"][j])
                M = (1 - sample_minority(ds.lib, rng, "full")) if TYPES[t] == "hole" else sample_minority(ds.lib, rng, "patch")
                ctrl.append((C, M.astype(np.uint8), np.zeros_like(M, dtype=np.uint8), TYPES[t], A["msg"][j].astype(np.uint8)))
        return items, ctrl
    t0 = time.time()
    (tr, ctr), (va, cva) = synth(TASK / "t1/out/synth/v0", a.ntr, 1), synth(TASK / "t1/out/synth/v0/val", a.nva, 2)
    idx = train_index(); meta = {e["image_name"]: e for e in json.loads((TASK / "release/train/metadata.json").read_text())}
    names = [n for n in folds()[0] if idx[n]["method"] == a.method and idx[n]["edit_type"] in ("hole", "transplant", "outpainting")]
    re = [(read_img(n), read_mask(n), np.zeros((512, 512), np.uint8), idx[n]["edit_type"], np.array([int(c) for c in meta[n]["message"]], np.uint8)) for n in names]
    print(f"data {time.time() - t0:.0f}s tr={len(tr)} va={len(va)} real={len(re)}", flush=True)
    Ftr, dt = collect(ext, tr); Fva, _ = collect(ext, va); Fre, _ = collect(ext, re)
    Fctr, _ = collect(ext, ctr); Fcva, _ = collect(ext, cva)
    stages = [k for k in Ftr if k not in ("sppl", "bitacc_sign")]
    shapes = {k: list(Ftr[k].shape[1:]) for k in Ftr}
    print("feature extraction s/img", dt, shapes, flush=True)
    lab_tr = np.stack([it[1] for it in tr]); ign_tr = np.stack([it[2] for it in tr])
    out = {"method": a.method, "res": a.res, "shapes": shapes, "feat_s_per_img_gpu": dt,
           "stage_modules": getattr(ext, "stage_names", ["net.layer1", "net.layer2", "net.layer3", "net.layer4"]),
           "bitacc_full_hole_val": float(Fva["bitacc_sign"].float()[[i for i, it in enumerate(va) if it[3] == "hole"]].mean()),
           "probes": {}}
    configs = [(s, True, False) for s in stages] + [(s, True, True) for s in stages] + [("sppl_only", False, True)]
    for st, uf, us in configs:
        t1 = time.time()
        Fs_tr = Ftr[st] if uf else None
        pred, loss = train_probe(Fs_tr, Ftr["sppl"], lab_tr, ign_tr, uf, us, dev, steps=a.steps)
        Pva = pred(Fva[st] if uf else None, Fva["sppl"]); Pre = pred(Fre[st] if uf else None, Fre["sppl"])
        key = f"{st}{'+sppl' if (us and uf) else ''}"
        r = {"loss": loss, "synth_val": eval_maps(Pva, [it[1] for it in va], [it[3] for it in va]),
             "real_fold0": eval_maps(Pre, [it[1] for it in re], [it[3] for it in re]), "train_s": time.time() - t1}
        if uf and not us:  # control: same probe trained/evaluated on clean covers with the same kind of masks
            pc, _ = train_probe(Fctr[st], Fctr["sppl"], np.stack([it[1] for it in ctr]), np.stack([it[2] for it in ctr]), True, False, dev, steps=a.steps)
            r["control_clean_val"] = eval_maps(pc(Fcva[st], Fcva["sppl"]), [it[1] for it in cva], [it[3] for it in cva])
        out["probes"][key] = r
        print(key, json.dumps(r), flush=True)
    (OUT / f"e2_{a.method}_{a.res}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
