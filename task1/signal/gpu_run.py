"""All GPU measurements for ONE method (run one process per method, one GPU each).
  1. full-image decode (train + test): logits, bit accuracy, signed margin vs known message
  2. native spatial maps via features (pixelseal, trustmark): agreement with message per position
  3. sliding-window sweep (train subset, balanced) : sizes 64..256, stride s/4 (s/2 derived), + occlusion
  4. full sliding run (train + test) with the best sweep config that fits the time budget
Outputs in t1/out/signal/*.npz."""
import argparse
import time

import torch  # noqa: F401  (before onnxruntime)

import numpy as np
import torch
import torch.nn.functional as F

from lib import OUT, load_meta, select, load_images, masks128, positions, reproject, auc
from wmloc.watermarks import load_watermark

SIZES = [64, 96, 128, 192, 256]
OCC_SIZES = [64, 128]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_logits(wm, x, bs):
    out = []
    for i in range(0, len(x), bs):
        l = wm.logits(x[i:i + bs]) if wm.name != "stegastamp" else wm.logits(x[i:i + bs], bs=bs)
        out.append(l.float().cpu())
    return torch.cat(out).numpy()


def acc_margin(L, msg):
    sgn = 2.0 * msg.astype(np.float32) - 1
    return ((L > 0) == msg.astype(bool)).mean(-1), (L * sgn).mean(-1)


def to_gpu(imgs, dev):
    return torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float().div(255)


def step_full(wm, method, split, limit, dev, bs, tag):
    meta = select(load_meta(split, method), limit, balanced=(split == "train"))
    names, msgs, fw, imgs = load_images(split, meta)
    t0 = time.time()
    L = np.concatenate([get_logits(wm, to_gpu(imgs[i:i + bs], dev), bs) for i in range(0, len(imgs), bs)])
    dt = (time.time() - t0) / len(imgs)
    acc = ((L > 0) == msgs.astype(bool)).mean(1); mar = ((2.0 * msgs - 1) * L).mean(1)
    np.savez(OUT / f"fullacc_{method}_{split}{tag}.npz", names=np.array(names), fw=fw, acc=acc, margin=mar,
             logits=L.astype(np.float32), sec_per_img=dt)
    log(f"[full] {method} {split}: n={len(names)} acc mean {acc.mean():.4f} L shape {L.shape} "
        f"range [{L.min():.2f},{L.max():.2f}] {dt*1000:.1f} ms/img")
    if split == "train":
        for r in (1, 0):
            a = acc[fw == r]
            log(f"   regime full={r}: acc mean {a.mean():.4f} min {a.min():.4f} max {a.max():.4f}")
    return names, msgs, fw, imgs


def step_native(wm, method, split, names, msgs, fw, imgs, dev, bs, tag):
    sig, frac, det = [], [], []
    maxdiff = 0.0
    t0 = time.time()
    for i in range(0, len(imgs), bs):
        x = to_gpu(imgs[i:i + bs], dev)
        with torch.no_grad():
            f = wm.features(x)
            pp = f["per_position_logits"].float()                   # B x nbits x h x w
            glob = wm.logits(x).float()
            maxdiff = max(maxdiff, (pp.mean((2, 3)) - glob).abs().max().item())
            m = torch.from_numpy(msgs[i:i + bs]).to(dev).float()
            sgn = (2 * m - 1)[:, :, None, None]
            if i == 0:
                log(f"[native] {method} features: " + ", ".join(f"{k}:{tuple(v.shape)}" for k, v in f.items()))
            sig.append((pp * sgn).mean(1).cpu().numpy())
            frac.append(((pp > 0).float() == m[:, :, None, None]).float().mean(1).cpu().numpy())
            if method == "pixelseal":
                lin = wm.det.pixel_decoder.linear
                d = torch.einsum("bchw,c->bhw", f["embedding"].float(), lin.weight[0].float()) + lin.bias[0].float()
                det.append(d.cpu().numpy())
    dt = (time.time() - t0) / len(imgs)
    d = dict(names=np.array(names), fw=fw, signed=np.concatenate(sig).astype(np.float32),
             frac=np.concatenate(frac).astype(np.float32), sec_per_img=dt)
    if det:
        d["det"] = np.concatenate(det).astype(np.float32)
    np.savez(OUT / f"native_{method}_{split}{tag}.npz", **d)
    log(f"[native] {method} {split}: per_pos shape {tuple(pp.shape)} map {d['signed'].shape[1:]} "
        f"| max |mean(per_pos) - logits| = {maxdiff:.2e} | {dt*1000:.1f} ms/img")


def window_eval(wm, x, msg, s, pos, bs):
    crops = torch.cat([x[:, :, y:y + s, xx:xx + s] for y in pos for xx in pos])
    L = get_logits(wm, crops, bs)
    a, m = acc_margin(L, msg)
    n = len(pos)
    return a.reshape(n, n), m.reshape(n, n)


def blur(x, k=9, reps=3):
    for _ in range(reps):
        x = F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)
    return x


def step_sweep(wm, method, n, dev, bs, tag):
    meta = select(load_meta("train", method), n, balanced=True)
    names, msgs, fw, imgs = load_images("train", meta)
    M = masks128()
    res = {"names": np.array(names), "fw": fw}
    timing = {}
    for s in SIZES:
        pos = positions(s, s // 4)
        A, Mg = [], []
        t0 = time.time()
        for k in range(len(names)):
            a, m = window_eval(wm, to_gpu(imgs[k:k + 1], dev), msgs[k], s, pos, bs)
            A.append(a); Mg.append(m)
        timing[s] = (time.time() - t0) / len(names)
        res[f"win_acc_{s}"] = np.stack(A).astype(np.float32); res[f"win_mar_{s}"] = np.stack(Mg).astype(np.float32)
        res[f"pos_{s}"] = np.array(pos)
        res[f"sec_{s}"] = timing[s]
        aucs = [auc(reproject(a, pos, s), M[nm]) for a, nm in zip(A, names)]
        log(f"[sweep] {method} s={s} stride={s//4} windows/img={len(pos)**2} {timing[s]:.2f} s/img "
            f"| AUC(acc map) mean {np.nanmean(aucs):.3f}")
    # occlusion: replace a window by gray / blur, measure the drop of the full-image margin
    for s in OCC_SIZES:
        pos = positions(s, s // 2)
        for kind in ("gray", "blur"):
            D = []
            t0 = time.time()
            for k in range(len(names)):
                x = to_gpu(imgs[k:k + 1], dev)
                base = acc_margin(get_logits(wm, x, 1), msgs[k])[1][0]
                xb = blur(x) if kind == "blur" else torch.full_like(x, 0.5)
                occ = []
                for y in pos:
                    for xx in pos:
                        z = x.clone(); z[:, :, y:y + s, xx:xx + s] = xb[:, :, y:y + s, xx:xx + s]; occ.append(z)
                occ = torch.cat(occ)
                mg = acc_margin(get_logits(wm, occ, bs), msgs[k])[1]
                D.append((base - mg).reshape(len(pos), len(pos)))
            res[f"occ_{kind}_{s}"] = np.stack(D).astype(np.float32); res[f"occpos_{s}"] = np.array(pos)
            aucs = [auc(reproject(d, pos, s), M[nm]) for d, nm in zip(D, names)]
            log(f"[occl] {method} {kind} s={s} stride={s//2} {(time.time()-t0)/len(names):.2f} s/img "
                f"| AUC(drop map) mean {np.nanmean(aucs):.3f}")
    np.savez(OUT / f"sweep_{method}{tag}.npz", **res)
    return res, M


def choose_config(res, M, n_total, budget_s):
    """Candidates: single (size, stride s/4 or s/2) and multi-scale mean at s/2; score = mean AUC (acc map)."""
    names = res["names"].tolist()
    cands = []
    sub_maps = {}
    for s in SIZES:
        pos4 = res[f"pos_{s}"].tolist(); A = res[f"win_acc_{s}"]
        idx2 = [i for i, p in enumerate(pos4) if p % (s // 2) == 0 or p == 512 - s]
        pos2 = [pos4[i] for i in idx2]
        for st, pos, sel in ((s // 4, pos4, slice(None)), (s // 2, pos2, idx2)):
            maps = [reproject(a[np.ix_(sel, sel)] if st == s // 2 else a, pos, s) for a in A]
            if st == s // 2:
                sub_maps[s] = maps
            cost = res[f"sec_{s}"] * (len(pos) / len(pos4)) ** 2
            cands.append(((s, st), np.nanmean([auc(m, M[nm]) for m, nm in zip(maps, names)]), cost))
    multi = [np.mean([sub_maps[s][k] for s in SIZES], 0) for k in range(len(names))]
    cost = sum(c for (s, st), _, c in cands if st == s // 2)
    cands.append((("multi", "s/2"), np.nanmean([auc(m, M[nm]) for m, nm in zip(multi, names)]), cost))
    for c in sorted(cands, key=lambda c: -c[1]):
        log(f"   cand {c[0]} AUC {c[1]:.3f} est {c[2]:.2f} s/img -> {c[2]*n_total/60:.1f} min for {n_total} imgs")
    ok = [c for c in cands if c[2] * n_total <= budget_s] or [min(cands, key=lambda c: c[2])]
    best = max(ok, key=lambda c: c[1])
    log(f"[choose] best within budget: {best[0]} AUC {best[1]:.3f}")
    return best[0]


def step_slide_full(wm, method, cfg, limit, dev, bs, tag):
    sizes = SIZES if cfg[0] == "multi" else [cfg[0]]
    for split in ("train", "test"):
        meta = select(load_meta(split, method), limit, balanced=(split == "train"))
        names, msgs, fw, imgs = load_images(split, meta)
        acc_maps, mar_maps = [], []
        t0 = time.time()
        for k in range(len(names)):
            x = to_gpu(imgs[k:k + 1], dev)
            am, mm = [], []
            for s in sizes:
                st = s // 2 if cfg[0] == "multi" else cfg[1]
                pos = positions(s, st)
                a, m = window_eval(wm, x, msgs[k], s, pos, bs)
                am.append(reproject(a, pos, s)); mm.append(reproject(m, pos, s))
            acc_maps.append(np.mean(am, 0)); mar_maps.append(np.mean(mm, 0))
        dt = (time.time() - t0) / len(names)
        np.savez(OUT / f"slide_{method}_{split}{tag}.npz", names=np.array(names), fw=fw, cfg=np.array([str(c) for c in cfg]),
                 acc=np.stack(acc_maps).astype(np.float16), margin=np.stack(mar_maps).astype(np.float16), sec_per_img=dt)
        log(f"[slide] {method} {split} cfg={cfg} n={len(names)} {dt:.2f} s/img")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--limit", type=int, default=0, help="images per split (0 = all)")
    ap.add_argument("--sweep-n", type=int, default=40)
    ap.add_argument("--budget-min", type=float, default=10.0, help="time budget for the full sliding run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sizes", default="64,96,128,192,256")
    ap.add_argument("--skip", default="", help="comma list of steps to skip: full,native,sweep,slide")
    a = ap.parse_args()
    global SIZES
    SIZES = [int(v) for v in a.sizes.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    skip = set(a.skip.split(",")) - {""}
    dev = "cuda"
    assert torch.cuda.is_available(), "no CUDA"
    log(f"method={a.method} gpu={torch.cuda.get_device_name(0)} limit={a.limit} sweep_n={a.sweep_n}")
    wm = load_watermark(a.method, device=dev)
    if a.method == "stegastamp":
        log("onnx providers:", wm.sess.get_providers())
    bs = {"pixelseal": 128, "trustmark": 128, "stegastamp": 64}[a.method]
    for split in ("train", "test"):
        if "full" in skip and "native" in skip:
            break
        names, msgs, fw, imgs = step_full(wm, a.method, split, a.limit, dev, 32, a.tag) if "full" not in skip else (None,) * 4
        if a.method != "stegastamp" and "native" not in skip:
            if names is None:
                names, msgs, fw, imgs = load_images(split, select(load_meta(split, a.method), a.limit, split == "train"))
            step_native(wm, a.method, split, names, msgs, fw, imgs, dev, 16, a.tag)
    if "sweep" not in skip:
        res, M = step_sweep(wm, a.method, a.sweep_n, dev, bs, a.tag)
        n_total = a.limit * 2 if a.limit else len(load_meta("train", a.method)) + len(load_meta("test", a.method))
        cfg = choose_config(res, M, n_total, a.budget_min * 60)
        if "slide" not in skip:
            step_slide_full(wm, a.method, cfg, a.limit, dev, bs, a.tag)
    log("DONE", a.method)


if __name__ == "__main__":
    main()
