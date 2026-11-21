"""Second pass for small rectangles: images whose conf < cmax get a finer grid of small sizes (32-96 px);
the result replaces the first one if its conf is higher. usage (from TASK):
python t1/boxsearch/pass2.py --method trustmark --sets evaltrain,test,train --device cuda:0"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rectsearch as RS  # noqa
from PIL import Image


def small_grid(A):
    sizes = [32, 48, 64, 96]; pos = {}
    for s in sizes:
        p = list(range(0, 512 - s + 1, s // 2))
        if p[-1] != 512 - s: p.append(512 - s)
        pos[s] = p
    return [(y, y + h, x, x + w) for h in sizes for w in sizes if max(h / w, w / h) <= A for y in pos[h] for x in pos[w]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method"); ap.add_argument("--sets", default="evaltrain,test,train")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--cmax", type=float, default=0.9)
    a = ap.parse_args(); torch.set_num_threads(2); M = a.method; A = RS.ASPECT[M]; G = small_grid(A)
    TASK = Path(os.getcwd()); D = RS.Dec(RS.load_watermark(M, device=a.device), a.device); RS.log("small grid", len(G))
    done = {}
    for S in a.sets.split(","):
        f = TASK / f"t1/out/boxsearch/{M}_{S}_boxes.json"; J = json.loads(f.read_text()); B = J["boxes"]
        split = "test" if S == "test" else "train"
        meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text())}
        for k, (nm, v) in enumerate(B.items()):
            if v["conf"] >= a.cmax or "pass2" in v: continue
            if split == "train" and nm in done: B[nm] = done[nm]; continue
            t0 = time.time(); n0 = D.n
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
            r, acc, mar, _ = RS.search(D, x, msg, A, G, None)
            v["pass2"] = {"rect": list(map(int, r)), "conf": acc, "n_decodes": D.n - n0, "sec": time.time() - t0}
            v["n_decodes"] += D.n - n0; v["sec"] += time.time() - t0
            if acc > v["conf"]:
                v["rect_pass1"], v["conf_pass1"] = v["rect"], v["conf"]
                v["rect"], v["conf"], v["margin"] = list(map(int, r)), acc, mar
                if "bbox512" in v: v["iou"] = RS.iou(r, v["bbox512"])
            if split == "train": done[nm] = v
            RS.log(S, k, nm, v.get("edit_type"), r, f"acc {acc:.3f}", f"iou {v.get('iou', -1):.3f}", f"{time.time() - t0:.1f}s")
            if k % 5 == 0: f.write_text(json.dumps(J, indent=0))
        f.write_text(json.dumps(J, indent=0)); RS.log("pass2 written", f)
    (TASK / f"t1/out/boxsearch/{M}_pass2.done").write_text("ok"); RS.log("DONE")
