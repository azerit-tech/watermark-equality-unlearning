"""Step 2a: cache StegaStamp decoder intermediate maps (STN bypassed: dec_in = resized image, spatially aligned) + a
known-message matched-filter map (encoder residual for the KNOWN message, correlated with the image high-pass),
for synthetic composites (SynthDS v0), real train (170) and test (340) StegaStamp images.
Run from TASK: python t1/stegaonnx/cache.py --dev 1 --nsyn 400   (GPU, ~minutes)   | --dev -1 for CPU"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, cv2, torch.nn.functional as F
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/common"), str(TASK / "t1/stegaonnx")]
from mkonnx import sess, prep, LOG, OUT  # noqa
from shapes import folds, train_index, read_img  # noqa
from wmloc.utils import downsample_mask  # noqa
C = OUT / "cache"; C.mkdir(parents=True, exist_ok=True)
L = {"c2": ("gen_decoder/dec_conv_2/Relu:0", (32, 100, 100)), "c3": ("gen_decoder/dec_conv_3/Relu:0", (64, 100, 100)),
     "c4": ("gen_decoder/dec_conv_4/Relu:0", (64, 100, 100)), "c5": ("gen_decoder/dec_conv_5/Relu:0", (64, 50, 50)),
     "c6": ("gen_decoder/dec_conv_6/Relu:0", (128, 25, 25)), "c7": ("gen_decoder/dec_conv_7/Relu:0", (128, 13, 13))}
SH = {**{k: v[1] for k, v in L.items()}, "mf": (3, 100, 100)}


def hp(x):  # x [B,3,H,W]
    k = cv2.getGaussianKernel(9, 2.0).astype(np.float32); k = torch.from_numpy(k @ k.T)[None, None].repeat(3, 1, 1, 1)
    return x - F.conv2d(F.pad(x, (4, 4, 4, 4), mode="reflect"), k, groups=3)


def run(tag, n, get, S, bs=16):
    arr = {k: np.lib.format.open_memmap(C / f"{tag}_{k}.npy", mode="w+", dtype=np.float16, shape=(n,) + s) for k, s in SH.items()}
    lab = np.lib.format.open_memmap(C / f"{tag}_lab.npy", mode="w+", dtype=np.uint8, shape=(n, 128, 128))
    ign = np.lib.format.open_memmap(C / f"{tag}_ign.npy", mode="w+", dtype=np.uint8, shape=(n, 128, 128))
    lg = np.zeros((n, 100), np.float32); t0 = time.time()
    for i in range(0, n, bs):
        it = [get(j) for j in range(i, min(n, i + bs))]; b = len(it)
        x = prep([t[0] for t in it]); msg = np.stack([t[3] for t in it]).astype(np.float32)
        o = S.run([LOG, "residual"] + [v[0] for v in L.values()], {"dec_in": x, "image": x, "secret": msg})
        lg[i:i + b] = o[0]
        for k, f in zip(L, o[2:]):
            f = torch.from_numpy(f)
            if f.shape[-1] == 200: f = F.avg_pool2d(f, 2)
            arr[k][i:i + b] = f.half().numpy()
        xt = torch.from_numpy(x).permute(0, 3, 1, 2); rt = torch.from_numpy(o[1]).permute(0, 3, 1, 2)
        hx, hr = hp(xt), hp(rt / 255.)
        mf = torch.stack([(hx * hr).sum(1), (hr * hr).sum(1), (hx * hx).sum(1)], 1)
        mf = F.avg_pool2d(mf, 4) * torch.tensor([1e4, 1e4, 1e2])[None, :, None, None]
        arr["mf"][i:i + b] = mf.half().numpy()
        lab[i:i + b] = np.stack([t[1] for t in it]); ign[i:i + b] = np.stack([t[2] for t in it])
        if (i // bs) % 5 == 0: print(f"{tag} {i + b}/{n} {time.time() - t0:.0f}s", flush=True)
    for a in list(arr.values()) + [lab, ign]: a.flush()
    np.save(C / f"{tag}_logits.npy", lg); (C / f"{tag}_done.txt").write_text(f"{n} {time.time() - t0:.1f}\n")
    print(f"{tag} done {n} {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", type=int, default=1); ap.add_argument("--nsyn", type=int, default=400)
    ap.add_argument("--seed", type=int, default=12345); a = ap.parse_args(); torch.set_num_threads(8)
    S = sess(OUT / "dec.onnx", a.dev); zero = np.zeros((128, 128), np.uint8)
    bits = lambda s: np.array([int(c) for c in s], np.float32)
    ti = train_index(); fold_of = {n: f for f, ns in enumerate(folds()) for n in ns}
    md = {d["image_name"]: d for d in json.load(open(TASK / "release/train/metadata.json"))}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    rn = sorted(n for n, v in ti.items() if v["method"] == "stegastamp")
    json.dump({"names": rn, "fold": [fold_of[n] for n in rn], "type": [ti[n]["edit_type"] for n in rn]}, open(C / "real_meta.json", "w"))
    run("real", len(rn), lambda j: (read_img(rn[j]), gt[rn[j]].astype(np.uint8), zero, bits(md[rn[j]]["message"])), S)
    tmd = [d for d in json.load(open(TASK / "release/test/metadata.json")) if d["method"] == "stegastamp"]
    json.dump({"names": [d["image_name"] for d in tmd]}, open(C / "test_meta.json", "w"))
    run("test", len(tmd), lambda j: (read_img(tmd[j]["image_name"], "test"), zero, zero, bits(tmd[j]["message"])), S)
    from dataset import SynthDS, TYPES
    ds = SynthDS(TASK / "t1/out/synth/v0", "stegastamp", seed=0, lib_folds=(1, 2, 3, 4))
    rng = np.random.default_rng(a.seed); types = []
    def gs(j):
        info = {}; x, lab, ig, t = ds.sample(rng, info=info); types.append(TYPES[t])
        ig = (cv2.resize(ig.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        return np.asarray(x), downsample_mask(lab).astype(np.uint8), ig, np.asarray(info["msg"], np.float32)
    try:
        run("syn", a.nsyn, gs, S)
    finally:
        json.dump({"type": types}, open(C / "syn_meta.json", "w"))


if __name__ == "__main__":
    main()
