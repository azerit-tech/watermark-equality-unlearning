"""Polish found rectangles (conf >= cmin): coordinate descent on edges, steps 4,2,1, with an unsaturated objective.
usage (from TASK): python t1/boxsearch/polish.py --method trustmark --set evaltrain --device cuda:0 [--write]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rectsearch as RS  # noqa
from PIL import Image

def objective(D, x, rects, msg, kind, padv):
    boxes = [(max(0, r[0] - padv), min(512, r[1] + padv), max(0, r[2] - padv), min(512, r[3] + padv)) for r in rects]
    L = D.logits(x, boxes); s = 2.0 * msg - 1; sm = L * s
    acc = ((L > 0) == msg.astype(bool)).mean(1)
    if kind == "margin": return sm.mean(1), acc
    if kind == "acc_margin": return acc + 0.002 * sm.mean(1), acc
    if kind == "minq": return np.quantile(sm, 0.1, axis=1), acc   # weakest bits
    raise ValueError(kind)

def polish(D, x, msg, r, A, kind, padv):
    obj = objective(D, x, [r], msg, kind, padv)[0][0]
    for step in (4, 2, 1):
        for _ in range(12):
            c = []
            for e in range(4):
                for d in (-step, step):
                    q = list(r); q[e] += d; c.append(tuple(q))
            c = [q for q in c if RS.ok(q, A)]
            o, _ = objective(D, x, c, msg, kind, padv); j = int(np.argmax(o))
            if o[j] <= obj: break
            r, obj = c[j], o[j]
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method"); ap.add_argument("--set", default="evaltrain")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--cmin", type=float, default=0.85)
    ap.add_argument("--kinds", default="margin,acc_margin,minq"); ap.add_argument("--pads", default="3")
    ap.add_argument("--write", default="")   # e.g. "margin:3" -> writes rect_polished into the json
    a = ap.parse_args(); torch.set_num_threads(2); M = a.method; A = RS.ASPECT[M]
    TASK = Path(os.getcwd()); f = TASK / f"t1/out/boxsearch/{M}_{a.set}_boxes.json"
    J = json.loads(f.read_text()); B = J["boxes"]
    D = RS.Dec(RS.load_watermark(M, device=a.device), a.device)
    split = "test" if a.set.startswith("test") else "train"
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text())}
    combos = [(k, int(p)) for k in a.kinds.split(",") for p in a.pads.split(",")]
    res = {c: [] for c in combos}; base = []
    for nm, v in B.items():
        if v["conf"] < a.cmin: continue
        msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
        x = torch.from_numpy(np.array(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
        for c in combos:
            r = polish(D, x, msg, tuple(v["rect"]), A, *c)
            if "bbox512" in v: res[c].append(RS.iou(r, v["bbox512"]))
            if a.write == f"{c[0]}:{c[1]}":
                v["rect_coarse"] = v.get("rect_coarse", v["rect"]); v["rect"] = list(map(int, r))
                o = objective(D, x, [r], msg, "margin", c[1]); v["conf_polished"] = float(o[1][0])
        if "bbox512" in v: base.append(RS.iou(tuple(v["rect"] if not a.write else v["rect_coarse"]), v["bbox512"]))
    if base:
        print(M, "n", len(base), "IoU before", np.mean(base).round(3), "succ", np.mean(np.array(base) > 0.9).round(3))
        for c in combos: print(M, c, "IoU", np.mean(res[c]).round(3), "succ(IoU>0.9)", np.mean(np.array(res[c]) > 0.9).round(3))
    if a.write:
        J["polish"] = a.write; f.write_text(json.dumps(J, indent=0)); print("written", f)
