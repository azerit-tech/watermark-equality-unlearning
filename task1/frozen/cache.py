"""Cache frozen TrustMark ResNet-50 stages (layer2/3/4, fp16 memmaps) for synthetic + real train + test images.
Run from TASK (CPU): OMP_NUM_THREADS=8 python t1/frozen/cache.py --nsyn 3000"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn.functional as F

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/train"), str(TASK / "t1/common")]
from wmloc.watermarks import load_watermark  # noqa
from dataset import SynthDS, TYPES  # noqa
from shapes import folds, train_index, read_img  # noqa
from wmloc.utils import downsample_mask  # noqa

OUT = TASK / "t1/out/frozen" / os.environ.get("FCACHE", "cache"); OUT.mkdir(parents=True, exist_ok=True)
SH = {"l2": (512, 31, 31), "l3": (1024, 16, 16), "l4": (2048, 8, 8)}


def mm(name, n, shape, dt):
    return np.lib.format.open_memmap(OUT / f"{name}.npy", mode="w+", dtype=dt, shape=(n,) + shape)


def run(tag, n, get, wm, bs=16):
    arr = {k: mm(f"{tag}_{k}", n, s, np.float16) for k, s in SH.items()}
    rgb = mm(f"{tag}_rgb", n, (3, 62, 62), np.uint8); lab = mm(f"{tag}_lab", n, (128, 128), np.uint8)
    ign = mm(f"{tag}_ign", n, (128, 128), np.uint8)
    t0 = time.time()
    for i in range(0, n, bs):
        items = [get(j) for j in range(i, min(n, i + bs))]
        xs = np.stack([it[0] for it in items])
        x = torch.from_numpy(xs).permute(0, 3, 1, 2).float().div(255)
        with torch.no_grad():
            f = wm.features(x)
        for k, kk in (("l2", "layer2"), ("l3", "layer3"), ("l4", "layer4")):
            assert tuple(f[kk].shape[1:]) == SH[k], (kk, f[kk].shape)
            arr[k][i:i + len(items)] = f[kk].half().numpy()
        rgb[i:i + len(items)] = np.stack([cv2.resize(a, (62, 62), interpolation=cv2.INTER_AREA) for a in xs]).transpose(0, 3, 1, 2)
        lab[i:i + len(items)] = np.stack([it[1] for it in items]); ign[i:i + len(items)] = np.stack([it[2] for it in items])
        if (i // bs) % 10 == 0:
            print(f"{tag} {i + len(items)}/{n} {time.time() - t0:.0f}s", flush=True)
            (OUT / f"{tag}_progress.txt").write_text(f"{i + len(items)} {time.time() - t0:.1f}\n")
    for a in list(arr.values()) + [rgb, lab, ign]: a.flush()
    dt = time.time() - t0; (OUT / f"{tag}_done.txt").write_text(f"{n} {dt:.1f} {dt / n:.4f}\n")
    print(f"{tag} done {n} in {dt:.0f}s ({dt / n:.3f}s/img)", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nsyn", type=int, default=3000); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only_syn", type=int, default=0); ap.add_argument("--tag", default="syn"); ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    wm = load_watermark("trustmark", device="cpu")
    if not a.only_syn:
        real_test(wm)
    synth(wm, a)


def real_test(wm):
    ti = train_index(); fl = folds(); fold_of = {n: f for f, ns in enumerate(fl) for n in ns}
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    rn = sorted(n for n, v in ti.items() if v["method"] == "trustmark")
    json.dump({"names": rn, "fold": [fold_of[n] for n in rn], "type": [ti[n]["edit_type"] for n in rn]}, open(OUT / "real_meta.json", "w"))
    zero = np.zeros((128, 128), np.uint8)
    run("real", len(rn), lambda j: (read_img(rn[j]), gt[rn[j]].astype(np.uint8), zero), wm)
    # test
    md = json.loads((TASK / "release/test/metadata.json").read_text())
    tn = [m["image_name"] for m in md if m["method"] == "trustmark"]
    json.dump({"names": tn}, open(OUT / "test_meta.json", "w"))
    run("test", len(tn), lambda j: (read_img(tn[j], "test"), zero, zero), wm)


def synth(wm, a):
    ds = SynthDS(TASK / "t1/out/synth/v0", "trustmark", seed=0, lib_folds=(1, 2, 3, 4))
    rng = np.random.default_rng(a.seed); types = []
    def gs(j):
        x, lab, ign, t = ds.sample(rng); types.append(TYPES[t])
        ig = (cv2.resize(ign.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        return np.asarray(x), downsample_mask(lab).astype(np.uint8), ig
    try:
        run(a.tag, a.nsyn, gs, wm)
    finally:
        json.dump({"type": types}, open(OUT / f"{a.tag}_meta.json", "w"))


if __name__ == "__main__":
    main()
