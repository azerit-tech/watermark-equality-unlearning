"""tmbbox : départs = boîtes du détecteur officiel TrustMark Q (~/tmbbox/det_{tag}.json, toutes échelles, label != 0),
puis recherche locale identique à ~/tmseed/seedrefine.py (décodeur crop R+3, refine budget PER, top K, seuil 0.79,
bords tmedge 2 passes sans décalage tmedge2). usage (depuis TASK) : python ~/tmbbox/bboxrefine.py train|test [SUF det]
-> ~/tmbbox/bbseeds_{tag}.json"""
import sys, os, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path.home() / "tmseed"))
import seedrefine as SR  # noqa  (charge rectsearch, tmedge)
RS = SR.RS; TASK = SR.TASK; H = Path.home() / "tmbbox"; THR = SR.THR
PER, K = int(os.environ.get("PER", 60)), int(os.environ.get("K", 3))


def starts_of(d):
    S = []
    for s, bs in d.items():
        for b in bs:
            if b[5] == 0: continue
            q = (max(0, int(round(b[0]))), min(512, int(round(b[1]))), max(0, int(round(b[2]))), min(512, int(round(b[3]))))
            if q[1] - q[0] >= 8 and q[3] - q[2] >= 8 and all(max(abs(q[i] - t[i]) for i in range(4)) > 4 for t in S): S.append(q)
    return S


if __name__ == "__main__":
    tag = sys.argv[1]; split = "test" if tag == "test" else "train"; suf = sys.argv[2] if len(sys.argv) > 2 else ""
    torch.set_num_threads(int(os.environ.get("NT", 4)))
    D = RS.Dec(RS.load_watermark("trustmark", device="cpu"), "cpu"); D.bs = 128
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{split}/metadata.json").read_text())}
    idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    det = json.loads((H / f"det_{tag}{suf}.json").read_text())
    names = os.environ["NAMES"].split(",") if os.environ.get("NAMES") else list(det)
    fout = H / f"bbseeds_{tag}{suf}.json"; res = json.loads(fout.read_text()) if fout.exists() else {}
    with torch.inference_mode():
        for nm in names:
            if nm in res: continue
            t0 = time.time(); n0 = D.n
            msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
            x = torch.from_numpy(np.array(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
            S = starts_of(det[nm]); r_ = {"n_starts": len(S)}
            if tag == "train": r_["edit_type"] = idx[nm]["edit_type"]
            if S:
                o, acc, _ = D.score(x, S, msg); order = np.argsort(-o); best = None
                for j in order[:K]:
                    r, ob = SR.refine(D, x, msg, S[j], o[j], D.n + PER)
                    _, ac, _ = D.score(x, [r], msg)
                    if best is None or ac[0] > best[1]: best = (r, float(ac[0]), S[j])
                    if ac[0] >= 0.9: break
                r_.update(rect_seed=list(map(int, best[0])), conf=best[1], start=list(map(int, best[2])), start_acc_max=float(acc.max()))
                if best[1] >= THR:
                    r1 = SR.edge_pass(D, x, msg, best[0]); r2 = SR.edge_pass(D, x, msg, r1)
                    _, ac2, mar2 = D.score(x, [tuple(r2)], msg)
                    r_.update(rect_p1=r1, rect=r2, conf_edge=float(ac2[0]), margin_edge=float(mar2[0]))
                if tag == "train" and idx[nm].get("bbox512") and idx[nm]["edit_type"] == "outpainting":
                    r_["iou_seed"] = RS.iou(best[0], idx[nm]["bbox512"])
                    if "rect" in r_: r_["iou"] = RS.iou(r_["rect"], idx[nm]["bbox512"])
            r_.update(n_decodes=D.n - n0, sec=round(time.time() - t0, 1)); res[nm] = r_
            print(tag, nm, r_, flush=True); fout.write_text(json.dumps(res))
    (H / f"DONE_bb_{tag}{suf}").write_text("ok"); print("DONE", flush=True)
