"""Denser multi-start search for images not found by boxsearch (conf < cmax). (from TASK)
python t1/boxsearch2/pass3.py --method trustmark --sets train,test --device cuda:1 [--only outpainting] [--limit N]
Starts: finer multi-scale grid (stride s/2, 10 sizes), top-K distinct boxes (IoU<0.3) by acc+0.1*soft, each refined
by rectsearch.refine. fp16 autocast for speed. Writes t1/out/boxsearch2/<method>_<set>_pass3.json"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rectsearch as RS  # noqa
from PIL import Image
TASK = Path(os.getcwd()); OUT = TASK / "t1/out/boxsearch3"


class Dec16(RS.Dec):
    def logits(self, x, boxes):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.wm.name != "stegastamp"):
            return super().logits(x, boxes)


def topk_starts(D, x, msg, G, K):
    o = D.score(x, G, msg)[0]; st = []
    for j in np.argsort(-o):
        if all(RS.iou(G[j], s) < 0.3 for s, _ in st): st.append((G[j], o[j]))
        if len(st) >= K: break
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method"); ap.add_argument("--sets", default="train,test")
    ap.add_argument("--device", default="cuda:1"); ap.add_argument("--cmax", type=float, default=0.9); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--sizes", default="48,64,96,128,160,192,240,288,352,416"); ap.add_argument("--only", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=""); ap.add_argument("--aspect", type=float, default=0); ap.add_argument("--folds", default="")
    a = ap.parse_args(); torch.set_num_threads(8); M = a.method; A = a.aspect or RS.ASPECT[M]
    FS = set(n for k in a.folds.split(",") if k for n in json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][int(k)])
    G = RS.grid(A, [int(s) for s in a.sizes.split(",")]); RS.log("grid", len(G))
    D = Dec16(RS.load_watermark(M, device=a.device), a.device)
    for S in a.sets.split(","):
        B = json.loads((TASK / f"t1/out/boxsearch/{M}_{S}_boxes.json").read_text())["boxes"]
        meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{S}/metadata.json").read_text())}
        todo = [n for n, v in B.items() if v["conf"] < a.cmax and (not a.only or v.get("edit_type") == a.only)]
        if FS and S == "train": todo = [n for n in todo if n in FS]
        if a.limit: todo = todo[:a.limit]
        fout = OUT / f"{M}_{S}_pass3{a.tag}.json"; res = json.loads(fout.read_text()) if fout.exists() else {}
        RS.log(S, len(todo), "images")
        for k, nm in enumerate(todo):
            if nm in res: continue
            t0 = time.time(); n0 = D.n
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{S}/images/{nm}.png").convert("RGB"))).to(a.device).permute(2, 0, 1)[None].float() / 255
            best = None
            for s, o0 in topk_starts(D, x, msg, G, a.K):
                r, ob = RS.refine(D, x, msg, s, o0, A)
                if best is None or ob > best[1]: best = (r, ob)
            r = best[0]; _, acc, mar = D.score(x, [r], msg)
            v = {"rect": list(map(int, r)), "conf": float(acc[0]), "margin": float(mar[0]), "n_decodes": D.n - n0, "sec": time.time() - t0}
            if "bbox512" in B[nm]: v["iou"] = RS.iou(r, B[nm]["bbox512"]); v["edit_type"] = B[nm]["edit_type"]; v["bbox512"] = B[nm]["bbox512"]
            res[nm] = v
            RS.log(S, k, nm, v.get("edit_type"), r, f"acc {v['conf']:.3f} (was {B[nm]['conf']:.3f})", f"iou {v.get('iou', -1):.3f}", f"{v['sec']:.1f}s {v['n_decodes']}")
            if k % 5 == 0: fout.write_text(json.dumps(res))
        fout.write_text(json.dumps(res))
    (OUT / f"{M}_pass3{a.tag}.done").write_text("ok"); RS.log("DONE")
