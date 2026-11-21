"""tmbox3: map-independent wide rectangle search for TrustMark minority images with boxsearch2 conf < 0.79 (CPU).
python t1/tmbox3/wide.py --list train_miss|train_neg|test --threads 4 --tag A
Coarse grid of small boxes (stride 20-28; larger boxes were covered by boxsearch/pass3) + top-K distinct starts refined edge by edge (steps 8,4,2,1).
Same decode as boxsearch2: crop = rect + 3 px, resized to 245 (bilinear, antialias), objective acc + 0.1*soft."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK / "t1/boxsearch2"))
import rectsearch as RS  # noqa
from PIL import Image
OUT = Path(os.environ.get("TMBOX3_OUT", str(TASK / "t1/out/tmbox3"))); OUT.mkdir(parents=True, exist_ok=True)
SHAPES = [(28, 28), (28, 44), (44, 28), (56, 56)]
STRIDE = {28: 20, 44: 24, 56: 28}


def coarse():
    out = []
    for h, w in SHAPES:
        st = max(STRIDE[h], STRIDE[w])
        ys = list(range(0, 512 - h + 1, st)); xs = list(range(0, 512 - w + 1, st))
        out += [(y, y + h, x, x + w) for y in ys for x in xs]
    return out


def refine(D, x, msg, r, obj):
    for step in (8, 4, 2, 1):
        for _ in range(4):
            c = []
            for e in range(4):
                for d in (-step, step):
                    q = list(r); q[e] += d; c.append(tuple(q))
            for (a, b) in ((0, 1), (2, 3)):
                for d in (-step, step):
                    q = list(r); q[a] += d; q[b] += d; c.append(tuple(q))
            c = [q for q in c if RS.ok(q, 2)]
            if not c: break
            o = D.score(x, c, msg)[0]; j = int(np.argmax(o))
            if o[j] <= obj: break
            r, obj = c[j], o[j]
    return r, obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--list"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="A"); ap.add_argument("--K", type=int, default=3); ap.add_argument("--part", default="0/1")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    G = coarse(); RS.log("coarse boxes", len(G))
    D = RS.Dec(RS.load_watermark("trustmark", device="cpu"), "cpu"); D.bs = 128
    F0 = set(json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"][0])
    names = []
    for S in (["train"] if a.list.startswith("train") else ["test"]):
        B = json.loads((TASK / f"t1/out/boxsearch2/trustmark_{S}_boxes.json").read_text())["boxes"]
        low = [n for n, v in B.items() if v["conf"] < 0.79]
        if a.list == "train_miss": low = [n for n in low if B[n]["edit_type"] == "outpainting"]
        if a.list == "train_neg": low = sorted([n for n in low if B[n]["edit_type"] != "outpainting"], key=lambda n: n not in F0)
        names = [(S, n) for n in low]
    pi, pn = map(int, a.part.split("/")); names = names[pi::pn]
    fout = OUT / f"wide_{a.list}_{a.tag}.json"; res = json.loads(fout.read_text()) if fout.exists() else {}
    meta = {m["image_name"]: m for S in ("train", "test") for m in json.loads((TASK / f"release/{S}/metadata.json").read_text())}
    RS.log(a.list, len(names), "images")
    with torch.inference_mode():
        for k, (S, nm) in enumerate(names):
            if nm in res: continue
            t0 = time.time(); n0 = D.n
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{S}/images/{nm}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
            o = D.score(x, G, msg)[0]; st = []
            for j in np.argsort(-o):
                if all(RS.iou(G[j], s) < 0.3 for s, _ in st): st.append((G[j], o[j]))
                if len(st) >= a.K: break
            best = None
            for s, o0 in st:
                r, ob = refine(D, x, msg, s, o0)
                if best is None or ob > best[1]: best = (r, ob)
            r = best[0]; _, acc, mar = D.score(x, [r], msg)
            v = {"rect": list(map(int, r)), "conf": float(acc[0]), "margin": float(mar[0]), "n_decodes": D.n - n0,
                 "sec": time.time() - t0, "coarse_top": [list(map(int, s)) for s, _ in st], "split": S, "fold0": nm in F0}
            if S == "train":
                idx = json.loads((TASK / "t1/common/train_index.json").read_text())[nm]
                v["edit_type"] = idx["edit_type"]; v["bbox512"] = idx["bbox512"]
                if idx["edit_type"] == "outpainting": v["iou"] = RS.iou(r, idx["bbox512"])
            res[nm] = v; RS.log(k, nm, v.get("edit_type"), r, f"acc {v['conf']:.3f}", f"iou {v.get('iou', -1):.3f}", f"{v['sec']:.0f}s {v['n_decodes']}")
            fout.write_text(json.dumps(res))
    (OUT / f"wide_{a.list}_{a.tag}.done").write_text("ok"); RS.log("DONE")
