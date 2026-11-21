"""Shift-and-stitch + multi-window TrustMark maps (frozen extractor). Run from TASK on one GPU:
    python t1/tmshift/shift.py [--check 1] [--dev cuda:1]
1) --check: global bit accuracy of the extractor under the shifts used, on 10 real images.
2) Linear probes (logistic, per stage l2/l3/l4) fitted on UNSHIFTED features of real folds 1-4.
3) For every TrustMark image (train 170 + test 340): per stage, 4x4 integer shifts with step = stage cell / 4
   (cell = stride * 512/245), reflect padding, probe score maps interleaved 4x finer, then linearly interpolated to 128.
   Also the message-signed per-position logits (mean of tanh and raw mean) from the l4 shift set.
   Same maps without shift (shift 0 only, interpolated) for comparison, and a 2x2 window variant (320 px windows).
Writes t1/out/tmshift/maps.npz: names, split, keys <src>_<map> float16 [N,128,128]."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/synth"), str(TASK / "t1/common")]
from wmloc.watermarks import load_watermark  # noqa
from shapes import folds, train_index, read_img, meta  # noqa

OUT = TASK / "t1/out/tmshift"; OUT.mkdir(parents=True, exist_ok=True)
SC = 512 / 245
STRIDE = {"l2": 8, "l3": 16, "l4": 32}
KEYS = {"l2": "layer2", "l3": "layer3", "l4": "layer4"}


def bits(s): return np.array([int(c) for c in s], np.float32)


def shifts(stage):
    step = STRIDE[stage] * SC / 4
    return [int(round(a * step)) for a in range(4)]


def shifted(x, dy, dx, pad=56):
    """x [B,3,512,512]; returns xs with xs(p) = x(p + d) (reflect padding at the bottom/right)."""
    xp = F.pad(x, (0, pad, 0, pad), mode="reflect")
    return xp[:, :, dy:dy + 512, dx:dx + 512]


def interp_mat(pos, tgt):
    """Linear interpolation matrix [len(tgt), len(pos)] (clamped at the ends); pos increasing."""
    I = np.eye(len(pos)); return np.stack([np.interp(tgt, pos, I[k]) for k in range(len(pos))], 1)


TGT = np.arange(128) * 4 + 2.0


def to128(m, pos):
    """m [B,n,n] on rectilinear positions pos -> [B,128,128]."""
    W = torch.from_numpy(interp_mat(pos, TGT)).float().to(m.device)
    return torch.einsum("yi,bij,xj->byx", W, m, W)


def cell_pos(stage, n, scale=SC, off=0.0):
    return off + (np.arange(n) * STRIDE[stage] + 0.5) * scale


def feats(wm, x):
    f = wm.features(x)
    return {k: f[KEYS[k]].float() for k in STRIDE}, f["per_position_logits"].float()


def ppl_maps(ppl, msg):
    s = ppl * (2 * msg - 1)[:, :, None, None]
    return torch.tanh(s).mean(1), s.mean(1)


def probe_maps(fd, probes):
    return {k: torch.einsum("bchw,c->bhw", (fd[k] - probes[k][0][None, :, None, None]) / probes[k][1][None, :, None, None], probes[k][2]) + probes[k][3]
            for k in STRIDE}


def fit_probes(wm, names, dev, steps=300):
    z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    F_ = {k: [] for k in STRIDE}; Y = {k: [] for k in STRIDE}
    for i in range(0, len(names), 16):
        nn_ = names[i:i + 16]
        x = torch.from_numpy(np.stack([read_img(n) for n in nn_])).permute(0, 3, 1, 2).float().div(255).to(dev)
        fd, _ = feats(wm, x)
        y = torch.from_numpy(np.stack([gt[n] for n in nn_]).astype(np.float32)).to(dev)[:, None]
        for k in STRIDE:
            n = fd[k].shape[-1]
            # label = mean mask over the 512-px region of each cell centre +- half cell (area interpolation approximation)
            pos = cell_pos(k, n); W = torch.from_numpy(interp_mat(TGT, pos)).float().to(dev)   # [n,128]
            yl = torch.einsum("iy,byx,jx->bij", W, y[:, 0], W)
            F_[k].append(fd[k].permute(0, 2, 3, 1).reshape(-1, fd[k].shape[1]).cpu()); Y[k].append(yl.reshape(-1).cpu())
    probes = {}
    for k in STRIDE:
        X = torch.cat(F_[k]).to(dev); y = torch.cat(Y[k]).to(dev)
        mu, sd = X.mean(0), X.std(0) + 1e-3; Xn = (X - mu) / sd
        w = torch.zeros(X.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=1e-2)
        for _ in range(steps):
            idx = torch.randint(0, len(Xn), (min(65536, len(Xn)),), device=dev)
            loss = F.binary_cross_entropy_with_logits(Xn[idx] @ w + b, y[idx]) + 1e-3 * (w ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        probes[k] = (mu, sd, w.detach(), b.detach()); print(f"probe {k}: n={len(X)} loss={float(loss):.4f}", flush=True)
    return probes


@torch.no_grad()
def maps_for(wm, x, msg, probes):
    """x [B,3,512,512] -> dict of [B,128,128] maps."""
    out = {}
    # no shift
    fd, ppl = feats(wm, x); pm = probe_maps(fd, probes); t, r = ppl_maps(ppl, msg)
    for k in STRIDE: out[f"s0_{k}"] = to128(pm[k], cell_pos(k, pm[k].shape[-1]))
    out["s0_ppt"] = to128(t, cell_pos("l4", 8)); out["s0_ppr"] = to128(r, cell_pos("l4", 8))
    # shift-and-stitch, one 4x4 shift set per stage
    for k in STRIDE:
        sh = shifts(k); acc = {}
        for a, dy in enumerate(sh):
            for b2, dx in enumerate(sh):
                fd, ppl = feats(wm, shifted(x, dy, dx)); pm = probe_maps(fd, probes)
                acc[(a, b2)] = [pm[k]]
                if k == "l4": acc[(a, b2)] += list(ppl_maps(ppl, msg))
        n = acc[(0, 0)][0].shape[-1]
        for j, nm in enumerate([k] + (["ppt", "ppr"] if k == "l4" else [])):
            fine = torch.zeros(x.shape[0], 4 * n, 4 * n, device=x.device)
            for (a, b2), v in acc.items(): fine[:, a::4, b2::4] = v[j]
            pos = (cell_pos(k, n)[:, None] + np.array(sh)[None]).reshape(-1)       # index 4i+a -> centre + shift a
            out[f"ss_{nm}"] = to128(fine, pos)
    # 2x2 windows of 320 px (offsets 0 / 192), each resized to 245 by the extractor
    num = {}; den = torch.zeros(1, 128, 128, device=x.device)
    for oy in (0, 192):
        for ox in (0, 192):
            fd, ppl = feats(wm, x[:, :, oy:oy + 320, ox:ox + 320]); pm = probe_maps(fd, probes); t, r = ppl_maps(ppl, msg)
            sc = 320 / 245; vals = {**pm, "ppt": t, "ppr": r}
            wy = ((TGT >= oy) & (TGT < oy + 320)).astype(np.float32); wx = ((TGT >= ox) & (TGT < ox + 320)).astype(np.float32)
            wm2 = torch.from_numpy(wy[:, None] * wx[None]).to(x.device)[None]
            for nm, v in vals.items():
                st = "l4" if nm in ("ppt", "ppr") else nm; n = v.shape[-1]
                py = cell_pos(st, n, sc, oy); px = cell_pos(st, n, sc, ox)
                Wy = torch.from_numpy(interp_mat(py, TGT)).float().to(x.device); Wx = torch.from_numpy(interp_mat(px, TGT)).float().to(x.device)
                num[nm] = num.get(nm, 0) + torch.einsum("yi,bij,xj->byx", Wy, v, Wx) * wm2
            den = den + wm2
    for nm, v in num.items(): out[f"mw_{nm}"] = v / den
    return out


def check(wm, names, dev):
    x = torch.from_numpy(np.stack([read_img(n) for n in names])).permute(0, 3, 1, 2).float().div(255).to(dev)
    m = torch.from_numpy(np.stack([bits(meta()[n]["message"]) for n in names])).to(dev)
    res = {}
    with torch.no_grad():
        for k in STRIDE:
            for d in shifts(k):
                lg = wm.logits(shifted(x, d, d)); res[f"{k}:{d}"] = round(float(((lg > 0).float() == m).float().mean()), 4)
        for oy in (0, 192):
            for ox in (0, 192):
                lg = wm.logits(x[:, :, oy:oy + 320, ox:ox + 320]); res[f"win{oy},{ox}"] = round(float(((lg > 0).float() == m).float().mean()), 4)
    print("bitacc", json.dumps(res), flush=True); json.dump(res, open(OUT / "check.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", default="cuda:1"); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--check", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); dev = a.dev; t0 = time.time()
    wm = load_watermark("trustmark", device=dev)
    ti = train_index(); fold_of = {n: f for f, ns in enumerate(folds()) for n in ns}
    rn = sorted(n for n, v in ti.items() if v["method"] == "trustmark")
    if a.check:
        check(wm, [n for n in rn if ti[n]["edit_type"] == "hole"][:5] + [n for n in rn if ti[n]["edit_type"] != "hole"][:5], dev)
    probes = fit_probes(wm, [n for n in rn if fold_of[n] != 0], dev)
    torch.save({k: [t.cpu() for t in v] for k, v in probes.items()}, OUT / "probes.pt")
    md = json.loads((TASK / "release/test/metadata.json").read_text())
    tn = [m["image_name"] for m in md if m["method"] == "trustmark"]; tmsg = {m["image_name"]: m["message"] for m in md}
    items = [(n, "train", meta()[n]["message"]) for n in rn] + [(n, "test", tmsg[n]) for n in tn]
    if a.limit: items = items[:a.limit]
    res = {}
    for i in range(0, len(items), a.bs):
        it = items[i:i + a.bs]
        x = torch.from_numpy(np.stack([read_img(n, s) for n, s, _ in it])).permute(0, 3, 1, 2).float().div(255).to(dev)
        msg = torch.from_numpy(np.stack([bits(m) for _, _, m in it])).to(dev)
        o = maps_for(wm, x, msg, probes)
        for k, v in o.items(): res.setdefault(k, []).append(v.half().cpu().numpy())
        if (i // a.bs) % 5 == 0: print(f"{i + len(it)}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    np.savez(OUT / "maps.npz", names=np.array([n for n, _, _ in items]), split=np.array([s for _, s, _ in items]),
             **{k: np.concatenate(v) for k, v in res.items()})
    (OUT / "maps_done.txt").write_text(f"{len(items)} {time.time() - t0:.0f}s\n"); print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
