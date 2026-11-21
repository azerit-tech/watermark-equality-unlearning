"""Diagnostic (from TASK): python t1/boxsearch2/diag.py --method trustmark --device cuda:1
- oracle: decode GT rect + p px (p=0..6), resize mode, and pad-to-native mode (reflect / gray) for small crops
- edge profiles around the boxsearch rect: per edge offset d in [-W, W], acc and signed margin of crop R+3.
Writes t1/out/boxsearch2/diag_<method>.json"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rectsearch as RS  # noqa
TASK = Path(os.getcwd())

def padnative(D, x, b, mode):
    c = x[:, :, b[0]:b[1], b[2]:b[3]]; h, w = c.shape[2:]; R = D.res
    if max(h, w) > R: return None
    t, l = (R - h) // 2, (R - w) // 2
    if mode == "gray":
        out = torch.full((1, 3, R, R), 0.5, device=x.device); out[:, :, t:t + h, l:l + w] = c; return out
    out = c
    while out.shape[2] < R or out.shape[3] < R:   # reflect may need several rounds
        ph, pw = min(out.shape[2] - 1, R - out.shape[2]), min(out.shape[3] - 1, R - out.shape[3])
        out = F.pad(out, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="reflect")
    return out[:, :, :R, :R]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method"); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--W", type=int, default=16)
    a = ap.parse_args(); torch.set_num_threads(4); M = a.method
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    meta = {m["image_name"]: m for m in json.loads((TASK / "release/train/metadata.json").read_text()) if m["method"] == M}
    B = json.loads((TASK / f"t1/out/boxsearch/{M}_train_boxes.json").read_text())["boxes"]
    wm = RS.load_watermark(M, device=a.device); D = RS.Dec(wm, a.device)
    res = {}
    for nm, v in B.items():
        msg = np.array([int(c) for c in meta[nm]["message"]], np.float64); s = 2 * msg - 1
        x = torch.from_numpy(np.array(Image.open(TASK / f"release/train/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
        g = idx[nm]["bbox512"]; r = {"edit_type": v["edit_type"], "gt": g, "rect": v["rect"], "conf": v["conf"]}
        if v["edit_type"] == "outpainting":
            bx = [(max(0, g[0] - p), min(512, g[1] + p), max(0, g[2] - p), min(512, g[3] + p)) for p in range(7)]
            L = D.logits(x, bx); r["oracle_resize"] = ((L > 0) == msg.astype(bool)).mean(1).tolist()
            for mode in ("reflect", "gray"):
                cs = [padnative(D, x, b, mode) for b in bx]
                if cs[3] is not None:
                    xs = torch.cat([c for c in cs if c is not None])
                    L = (wm.logits(xs)).float().cpu().numpy(); r["oracle_" + mode] = ((L > 0) == msg.astype(bool)).mean(1).tolist()
        # edge profiles around boxsearch rect, for every image (FP analysis needs the negatives too)
        R0 = list(v["rect"]); prof = {}
        for e in range(4):
            rects = []
            for d in range(-a.W, a.W + 1):
                q = list(R0); q[e] += d; rects.append(q)
            L = D.logits(x, [RS.pad(q) for q in rects]); acc = ((L > 0) == msg.astype(bool)).mean(1); mar = (L * s).mean(1)
            prof[e] = {"acc": acc.tolist(), "margin": mar.tolist()}
        # consistency R+2 vs R+4, shift test
        for p in (0, 1, 2, 4, 5, 6):
            b = (max(0, R0[0] - p), min(512, R0[1] + p), max(0, R0[2] - p), min(512, R0[3] + p))
            L = D.logits(x, [b]); r[f"acc_p{p}"] = float(((L > 0) == msg.astype(bool)).mean())
        r["prof"] = prof; res[nm] = r
    (TASK / f"t1/out/boxsearch2/diag_{M}.json").write_text(json.dumps(res)); print("done", len(res), D.n)
