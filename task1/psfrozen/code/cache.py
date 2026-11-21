"""Cache frozen PixelSeal ConvNeXt-v2 T stages (s1,s2,s3 fp16 memmaps) + signed per-position logits map, synth + real + test.
Run from TASK (CPU): OMP_NUM_THREADS=4 python t1/psfrozen/cache.py --res 256 --nsyn 4000 --threads 4"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn.functional as F

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/train"), str(TASK / "t1/common")]
from wmloc.watermarks import load_watermark  # noqa
from dataset import SynthDS, TYPES  # noqa
from shapes import folds, train_index, read_img  # noqa
from wmloc.utils import downsample_mask  # noqa


def bits(m):
    return np.array([int(c) for c in m], np.float32) if isinstance(m, str) else np.asarray(m, np.float32).reshape(-1)


def feats(wm, x, res, msg):
    """x [B,3,512,512] in [0,1]; msg [B,256] 0/1 -> dict s1,s2,s3, pp [B,2,64,64] (mean signed logit, mean signed sign)."""
    if res != 512:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
    c = wm.det.convnext; h = wm.det.preprocess(x); out = {}
    for i in range(4):
        h = c.stages[i](c.downsample_layers[i](h))
        if i: out[f"s{i}"] = h
    emb = wm.det.pixel_decoder.output_upscaling(h); lin = wm.det.pixel_decoder.linear
    pp = torch.einsum("bchw,oc->bohw", emb, lin.weight[1:]) + lin.bias[1:][None, :, None, None]
    sg = (2 * msg - 1)[:, :, None, None]
    s = pp * sg
    out["pp"] = F.adaptive_avg_pool2d(torch.stack([s.mean(1), torch.sign(s).mean(1)], 1), 64)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nsyn", type=int, default=4000); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--seed", type=int, default=12345); ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    OUT = TASK / f"t1/out/psfrozen/cache{a.res}"; OUT.mkdir(parents=True, exist_ok=True)
    wm = load_watermark("pixelseal", device="cpu")
    # check that our stage loop reproduces det.convnext
    with torch.no_grad():
        x = torch.rand(1, 3, 512, 512); xr = F.interpolate(x, size=(a.res,) * 2, mode="bilinear", align_corners=False, antialias=True) if a.res != 512 else x
        ref = wm.det.convnext(wm.det.preprocess(xr)); f = feats(wm, x, a.res, torch.zeros(1, 256))
        print("check s3 vs convnext:", ref.shape, f["s3"].shape, float((ref - f["s3"]).abs().max()), flush=True)
    SH = {k: tuple(v.shape[1:]) for k, v in f.items()}; print(SH, flush=True)

    def run(tag, n, get):
        arr = {k: np.lib.format.open_memmap(OUT / f"{tag}_{k}.npy", mode="w+", dtype=np.float16, shape=(n,) + s) for k, s in SH.items()}
        rgb = np.lib.format.open_memmap(OUT / f"{tag}_rgb.npy", mode="w+", dtype=np.uint8, shape=(n, 3, 64, 64))
        lab = np.lib.format.open_memmap(OUT / f"{tag}_lab.npy", mode="w+", dtype=np.uint8, shape=(n, 128, 128))
        ign = np.lib.format.open_memmap(OUT / f"{tag}_ign.npy", mode="w+", dtype=np.uint8, shape=(n, 128, 128))
        t0 = time.time()
        for i in range(0, n, a.bs):
            items = [get(j) for j in range(i, min(n, i + a.bs))]
            xs = np.stack([it[0] for it in items]); m = torch.from_numpy(np.stack([it[3] for it in items]))
            x = torch.from_numpy(xs).permute(0, 3, 1, 2).float().div(255)
            with torch.no_grad():
                f = feats(wm, x, a.res, m)
            for k in SH: arr[k][i:i + len(items)] = f[k].half().numpy()
            rgb[i:i + len(items)] = np.stack([cv2.resize(q, (64, 64), interpolation=cv2.INTER_AREA) for q in xs]).transpose(0, 3, 1, 2)
            lab[i:i + len(items)] = np.stack([it[1] for it in items]); ign[i:i + len(items)] = np.stack([it[2] for it in items])
            if (i // a.bs) % 10 == 0:
                print(f"{tag} {i + len(items)}/{n} {time.time() - t0:.0f}s", flush=True)
        for q in list(arr.values()) + [rgb, lab, ign]: q.flush()
        dt = time.time() - t0; (OUT / f"{tag}_done.txt").write_text(f"{n} {dt:.1f} {dt / n:.4f}\n")
        print(f"{tag} done {n} in {dt:.0f}s ({dt / n:.3f}s/img)", flush=True)

    ti = train_index(); fl = folds(); fold_of = {n: k for k, ns in enumerate(fl) for n in ns}
    msgs = {m["image_name"]: bits(m["message"]) for sp in ("train", "test") for m in json.loads((TASK / f"release/{sp}/metadata.json").read_text())}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    rn = sorted(n for n, v in ti.items() if v["method"] == "pixelseal")
    json.dump({"names": rn, "fold": [fold_of[n] for n in rn], "type": [ti[n]["edit_type"] for n in rn]}, open(OUT / "real_meta.json", "w"))
    zero = np.zeros((128, 128), np.uint8)
    run("real", len(rn), lambda j: (read_img(rn[j]), gt[rn[j]].astype(np.uint8), zero, msgs[rn[j]]))
    md = json.loads((TASK / "release/test/metadata.json").read_text())
    tn = [m["image_name"] for m in md if m["method"] == "pixelseal"]
    json.dump({"names": tn}, open(OUT / "test_meta.json", "w"))
    run("test", len(tn), lambda j: (read_img(tn[j], "test"), zero, zero, msgs[tn[j]]))
    ds = SynthDS(TASK / "t1/out/synth/v0", "pixelseal", seed=0, lib_folds=(1, 2, 3, 4))
    rng = np.random.default_rng(a.seed); types = []

    def gs(j):
        info = {}; x, lab, ign, t = ds.sample(rng, info=info); types.append(TYPES[t])
        ig = (cv2.resize(ign.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        return np.asarray(x), downsample_mask(lab).astype(np.uint8), ig, bits(info["msg"])
    try:
        run("syn", a.nsyn, gs)
    finally:
        json.dump({"type": types}, open(OUT / "syn_meta.json", "w"))


if __name__ == "__main__":
    main()
