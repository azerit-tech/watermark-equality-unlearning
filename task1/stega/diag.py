"""Diagnostic StegaStamp fold 0 (run from TASK, GPU): pixel probs 512 of a checkpoint, decision rules, error analysis,
montage of 12 cases.  python t1/stega/diag.py --ckpt t1/out/train/v0/stegastamp/best.pt --out t1/out/stega/diag_v0"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, cv2
from PIL import Image, ImageDraw
TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK):
    sys.path.insert(0, str(p))
from predict import load_model
from train import binarize
from regime_guard import load_regimes, guard
from shapes import names_in_folds, read_img, read_mask, meta, train_index
from wmloc.utils import downsample_mask, miou_image

@torch.no_grad()
def pix_probs(model, imgs, bs=4, dev="cuda", tta=False):
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).to(dev).float().div(255).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x).float()
        out.append(torch.sigmoid(lg).cpu().numpy().astype(np.float16))
    return np.concatenate(out)

def rules(pp, full=None):
    """pp float [512,512] -> dict of 128 masks"""
    p = pp.astype(np.float32)
    mean = p.reshape(128, 4, 128, 4).mean((1, 3))
    frac = (p >= 0.5).reshape(128, 4, 128, 4).mean((1, 3))
    r = {"mean": binarize(mean), "frac": binarize(frac + 1e-4 * mean)}
    if full is not None:
        r["mean+guard"] = guard(mean, full)[0]; r["frac+guard"] = guard(frac + 1e-4 * mean, full)[0]
    return r, mean

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train"); ap.add_argument("--montage", default="")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.ckpt)
    if a.split == "test":
        md = json.loads((TASK / "release/test/metadata.json").read_text())
        names = sorted(m["image_name"] for m in md if m["method"] == "stegastamp")
    else:
        names = names_in_folds([0], "stegastamp")
    imgs = [read_img(n, a.split) for n in names]
    pp = pix_probs(model, imgs)
    np.savez_compressed(out / f"{a.split}_pix512.npz", names=np.array(names), probs=pp)
    reg = load_regimes(a.split)
    m128 = np.stack([rules(p)[1] for p in pp])
    np.savez_compressed(out / f"{'fold0' if a.split=='train' else 'test'}_probs128.npz", names=np.array(names), probs=m128.astype(np.float16))
    if a.split != "train":
        print("wrote test", len(names)); return
    ti = train_index(); sc = defaultdict(lambda: defaultdict(list)); rows = []
    for n, p, img in zip(names, pp, imgs):
        gt = read_mask(n); g128 = downsample_mask(gt); full = bool(meta()[n]["full_watermark"])
        et = ti[n]["edit_type"]; r, mean = rules(p, reg[n])
        for k, m in r.items():
            s = miou_image(m, g128); sc[k][et].append(s); sc[k]["all"].append(s)
        # oracle regime guard
        s = miou_image(guard(mean, full)[0], g128); sc["mean+oracle"][et].append(s); sc["mean+oracle"]["all"].append(s)
        # error analysis at pixel level with mean rule
        pm = (p.astype(np.float32) >= 0.5).astype(np.uint8)
        err = pm != gt
        d_in = cv2.distanceTransform(gt, cv2.DIST_L2, 3); d_out = cv2.distanceTransform(1 - gt, cv2.DIST_L2, 3)
        dist = np.where(gt == 1, d_in, d_out)
        minority = gt if gt.mean() < 0.5 else 1 - gt
        fp = err & (pm == 1) & (dist > 6); fn = err & (pm == 0) & (dist > 6); band = err & (dist <= 6)
        n_min = minority.sum()
        # signed boundary bias: mean predicted prob in rings outside/inside
        ring_out = (gt == 0) & (d_out <= 3); ring_in = (gt == 1) & (d_in <= 3)
        rows.append(dict(name=n, et=et, full=full, regime_ok=reg[n] == full, score=round(miou_image(r["mean"], g128), 4),
                         score_guard=round(miou_image(r["mean+guard"], g128), 4),
                         min_frac=round(float(minority.mean()), 4), pred_frac1=round(float(pm.mean()), 4), gt_frac1=round(float(gt.mean()), 4),
                         err_band=int(band.sum()), err_fp_far=int(fp.sum()), err_fn_far=int(fn.sum()), n_minority=int(n_min),
                         p_ring_out=round(float(p[ring_out].astype(np.float32).mean()) if ring_out.any() else -1, 3),
                         p_ring_in=round(float(p[ring_in].astype(np.float32).mean()) if ring_in.any() else -1, 3)))
    summ = {k: {e: [round(float(np.mean(v)), 4), len(v)] for e, v in d.items()} for k, d in sc.items()}
    print(json.dumps(summ, indent=1))
    rows.sort(key=lambda r: r["score"])
    for r in rows:
        print(json.dumps(r))
    (out / "diag.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    # montage of 12 worst: image | GT | prob | errors (red FP, blue FN) ; 256 px tiles
    T = 192; sel = rows[:12]; W = Image.new("RGB", (4 * T, len(sel) * (T + 14)), "white"); dr = ImageDraw.Draw(W)
    idx = {n: i for i, n in enumerate(names)}
    for k, r in enumerate(sel):
        i = idx[r["name"]]; gt = read_mask(r["name"]); p = pp[i].astype(np.float32); pm = p >= 0.5
        e = np.zeros((512, 512, 3), np.uint8); e[(pm == 1) & (gt == 0)] = (255, 0, 0); e[(pm == 0) & (gt == 1)] = (0, 90, 255); e[(pm == gt)] = (gt[pm == gt, None] * 200)
        tiles = [imgs[i], np.repeat(gt[..., None] * 255, 3, 2), np.repeat((p[..., None] * 255).astype(np.uint8), 3, 2), e]
        y = k * (T + 14)
        dr.text((2, y), f"{r['name']} {r['et']} {'full' if r['full'] else 'patch'} score={r['score']} regOK={r['regime_ok']}", fill="black")
        for j, t in enumerate(tiles):
            W.paste(Image.fromarray(t).resize((T, T), Image.NEAREST), (j * T, y + 14))
    W.save(out / "montage_worst12.png")

if __name__ == "__main__":
    main()
