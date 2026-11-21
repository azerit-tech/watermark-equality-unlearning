"""Second wave (coordinator criteria), TRAIN set, one method per process:
  A  transplant geometry: transplant fragment (out-of-mask gray / mean) decoded in place, shifted by 8 / 128 px;
     control = full-regime image, borrowed transplant shape, same coords and shifted by 8 / 128 px
  B  rectangle module: exact crop vs boxes moved by +-4 / +-8 px (4 translations + 8 single-edge moves), signed margin
  C  finer native maps (pixelseal, trustmark): extractor on the 512 input (no resize) and shift-and-stitch
     (16 shifts of 16 px at 512 of the 256/245 input) -> AUC per image vs mask128, by regime; saved for train + test
Writes t1/out/signal/probe2_<method>.json and nat2_<method>_<split>.npz (128x128 float16 maps)."""
import argparse
import json
import time

import torch  # noqa: F401
import numpy as np
import torch.nn.functional as F
from PIL import Image

from lib import OUT, ROOT, load_meta, masks128, auc
from box_probe import Dec, bbox, clampbox, crop, load_item, stats, DEV
from wmloc.watermarks import load_watermark


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def shifted(t, d):  # shift a 1xCxHxW tensor by d px (right and down), gray fill
    out = torch.full_like(t, 0.5)
    out[:, :, d:, d:] = t[:, :, :512 - d, :512 - d]
    return out


def part_A(D, meta, msg, masks, nonrect, full):
    rows, ctrl = [], []
    for nm in nonrect:
        img, mask, x = load_item(nm); mk = torch.from_numpy(mask).to(DEV)[None, None].float()
        g = x * mk + 0.5 * (1 - mk); mn = x * mk + x.mean((2, 3), keepdim=True) * (1 - mk)
        acc, _ = D.acc_mar([x, g, mn, shifted(g, 8), shifted(g, 128)], msg[nm])
        rows.append({"name": nm, "area": int(mask.sum()), **dict(zip(["asis", "gray", "mean", "gray_shift8", "gray_shift128"], map(float, acc)))})
    for i, nm in enumerate(full):
        shape = masks[nonrect[i % len(nonrect)]]
        img, mask, x = load_item(nm); mk = torch.from_numpy(shape).to(DEV)[None, None].float()
        g = x * mk + 0.5 * (1 - mk); mn = x * mk + x.mean((2, 3), keepdim=True) * (1 - mk)
        acc, _ = D.acc_mar([x, g, mn, shifted(g, 8), shifted(g, 128)], msg[nm])
        ctrl.append({"name": nm, "area": int(shape.sum()), "wm_in_shape": float(mask[shape].mean()),
                     **dict(zip(["asis", "gray", "mean", "gray_shift8", "gray_shift128"], map(float, acc)))})
    out = {"transplant": {k: stats([r[k] for r in rows]) for k in rows[0] if k != "name"} if rows else {},
           "control": {k: stats([r[k] for r in ctrl]) for k in ctrl[0] if k != "name"} if ctrl else {}}
    for lab, lo, hi in (("area<15k", 0, 15000), ("15k-40k", 15000, 40000), (">=40k", 40000, 1e9)):
        out[f"transplant_gray_{lab}"] = stats([r["gray"] for r in rows if lo <= r["area"] < hi])
        out[f"control_gray_{lab}"] = stats([r["gray"] for r in ctrl if lo <= r["area"] < hi])
    log("[A]", json.dumps(out))
    return out, rows, ctrl


def part_B(D, msg, rect):
    rows = []
    for nm in rect:
        img, mask, x = load_item(nm); b = tuple(int(v) for v in bbox(mask))
        if min(b[1] - b[0], b[3] - b[2]) < 40:
            continue
        r = {"name": nm, "minside": int(min(b[1] - b[0], b[3] - b[2]))}
        for d in (4, 8):
            vs = []
            for dy, dx in ((d, 0), (-d, 0), (0, d), (0, -d)):
                vs.append(clampbox(b[0] + dy, b[1] + dy, b[2] + dx, b[3] + dx))
            for e in range(4):
                for s in (-d, d):
                    c = list(b); c[e] += s; vs.append(clampbox(*c))
            vs = [v for v in vs if v != b]
            acc, mar = D.acc_mar([crop(x, b)] + [crop(x, v) for v in vs], msg[nm])
            r[f"wins_{d}"] = bool(mar[0] > mar[1:].max()); r[f"exact_acc"] = float(acc[0])
            r[f"best_other_acc_{d}"] = float(acc[1:].max()); r[f"margin_gap_{d}"] = float(mar[0] - mar[1:].max())
        rows.append(r)
    out = {"n": len(rows), "exact_wins_4": float(np.mean([r["wins_4"] for r in rows])) if rows else None,
           "exact_wins_8": float(np.mean([r["wins_8"] for r in rows])) if rows else None,
           "exact_acc": stats([r["exact_acc"] for r in rows]), "best_other_acc_4": stats([r["best_other_acc_4"] for r in rows]),
           "best_other_acc_8": stats([r["best_other_acc_8"] for r in rows])}
    log("[B]", json.dumps(out))
    return out, rows


def native_maps(wm, method, x, msg):
    """x: Bx3x512x512. Returns dict of 128x128 maps (B x 128 x 128): agreement sum(per_pos * (2m-1))/nbits."""
    sgn = (2 * torch.from_numpy(msg).to(DEV).float() - 1)[:, :, None, None]
    res = {}
    with torch.no_grad():
        # (1) extractor on the raw 512 input
        if method == "pixelseal":
            det = wm.det; lat = det.convnext(det.preprocess(x)); emb = det.pixel_decoder.output_upscaling(lat)
            lin = det.pixel_decoder.linear
            pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]; pp = pp[:, 1:]
        else:
            n = wm.net; z = x * 2 - 1
            z = n.maxpool(n.relu(n.bn1(n.conv1(z))))
            for k in ("layer1", "layer2", "layer3", "layer4"):
                z = getattr(n, k)(z)
            pp = torch.einsum("bchw,oc->bohw", z, n.fc.weight) + n.fc.bias[None, :, None, None]
        a512 = (pp * sgn).mean(1, keepdim=True)
        res["at512"] = F.interpolate(a512, size=(128, 128), mode="bilinear", align_corners=False)[:, 0]
        res["acc512"] = ((pp.mean((2, 3)) > 0).float() == (sgn[:, :, 0, 0] > 0).float()).float().mean(1)
        res["hw512"] = torch.tensor(pp.shape[2:])
        # (2) shift-and-stitch on the standard resized input: shifts of 0/16/32/48 px at 512
        acc = torch.zeros(x.shape[0], 128, 128, device=DEV); cnt = torch.zeros(128, 128, device=DEV)
        for dy in (0, 16, 32, 48):
            for dx in (0, 16, 32, 48):
                xs = F.pad(x[:, :, dy:, dx:], (0, dx, 0, dy), mode="reflect")
                pp = wm.features(xs)["per_position_logits"].float()
                a = (pp * sgn).mean(1)                       # B x h x w, cell (i,j) covers dy + i*512/h ...
                h = a.shape[1]; cs = 128 // h                # grid cells per map cell
                up = a.repeat_interleave(cs, 1).repeat_interleave(cs, 2)    # B x 128 x 128 in shifted coords
                oy, ox = dy // 4, dx // 4
                acc[:, oy:, ox:] += up[:, :128 - oy, :128 - ox]; cnt[oy:, ox:] += 1
        res["stitch"] = acc / cnt.clamp(min=1)
    return res


def part_C(wm, method, M):
    out = {}
    for split in ("train", "test"):
        meta = load_meta(split, method)
        names = [m["image_name"] for m in meta]
        msgs = np.stack([np.array([int(c) for c in m["message"]], np.uint8) for m in meta])
        maps = {"at512": [], "stitch": []}; acc512 = []
        t0 = time.time()
        for i in range(0, len(names), 8):
            x = torch.stack([torch.from_numpy(np.asarray(Image.open(ROOT / "release" / split / "images" / f"{n}.png").convert("RGB")))
                             for n in names[i:i + 8]]).to(DEV).permute(0, 3, 1, 2).float() / 255
            r = native_maps(wm, method, x, msgs[i:i + 8])
            for k in maps: maps[k].append(r[k].cpu().numpy())
            acc512.append(r["acc512"].cpu().numpy())
        dt = (time.time() - t0) / len(names)
        maps = {k: np.concatenate(v) for k, v in maps.items()}
        np.savez(OUT / f"nat2_{method}_{split}.npz", names=np.array(names), acc512=np.concatenate(acc512).astype(np.float32),
                 **{k: v.astype(np.float16) for k, v in maps.items()}, sec_per_img=dt)
        log(f"[C] {method} {split}: native map at 512 = {tuple(r['hw512'].tolist())} | {dt*1000:.0f} ms/img")
        if split == "train":
            fw = np.array([m["full_watermark"] for m in meta])
            acc512 = np.concatenate(acc512)
            out["acc512_full"] = stats(acc512[fw]); out["acc512_patch"] = stats(acc512[~fw])
            for k, v in maps.items():
                a = np.array([auc(m, M[n]) for m, n in zip(v, names)])
                out[f"auc_{k}"] = {"all": float(np.nanmean(a)), "full": float(np.nanmean(a[fw])), "patch": float(np.nanmean(a[~fw]))}
            log("[C]", json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    wm = load_watermark(a.method, device=DEV); D = Dec(wm)
    meta = load_meta("train", a.method)
    msg = {m["image_name"]: np.array([int(c) for c in m["message"]], np.uint8) for m in meta}
    full = [m["image_name"] for m in meta if m["full_watermark"]]
    patch = [m["image_name"] for m in meta if not m["full_watermark"]]
    masks = {nm: np.asarray(Image.open(ROOT / "release/train/masks" / f"{nm}.png")) > 127 for nm in patch + full}
    rect, nonrect = [], []
    for nm in patch:
        b = bbox(masks[nm]); (rect if masks[nm][b[0]:b[1], b[2]:b[3]].all() else nonrect).append(nm)
    if a.n:
        rect, nonrect, full = rect[:a.n], nonrect[:a.n], full[:a.n]
    R = {}
    if a.method != "stegastamp":
        R["A"], R["A_rows"], R["A_ctrl_rows"] = part_A(D, meta, msg, masks, nonrect, full)
    R["B"], R["B_rows"] = part_B(D, msg, rect)
    if a.method != "stegastamp" and not a.n:
        R["C"] = part_C(wm, a.method, masks128())
    (OUT / f"probe2_{a.method}{'_val' if a.n else ''}.json").write_text(json.dumps(R, indent=1, default=float))
    log("DONE", a.method)
