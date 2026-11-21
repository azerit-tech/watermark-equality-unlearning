"""compare.py <new.pt> <sg_boxes.json> : compare à sub18p (test) ; masque = rectangle ? couverture, régime."""
import json, sys
from pathlib import Path
import numpy as np, torch
TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task"); H = Path.home()
sys.path[:0] = [str(TASK)]
from wmloc.utils import downsample_mask
new, old = torch.load(sys.argv[1], weights_only=True), torch.load(H/"t1w/sub18p/sub18p.pt", weights_only=True)
assert new["image_names"] == old["image_names"]; names = new["image_names"]
M, O = new["masks"].numpy(), old["masks"].numpy()
tm = {x["image_name"]: x["method"] for x in json.loads((TASK/"release/test/metadata.json").read_text())}
def L(p):
    d = json.loads(Path(p).read_text()); return d["boxes"] if "boxes" in d else d
B = {"trustmark": L(H/"tmseed/trustmark_test_boxes.json"), "stegastamp": L(sys.argv[2]), "pixelseal": L(H/"psseed/pixelseal_test_boxes.json")}
sys.path.insert(0, str(TASK/"t1/guard2")); from rule import THR
reg = {}
for m in THR:
    z = np.load(TASK/"t1/out/signal"/f"fullacc_{m}_test.npz"); reg.update({str(n): float(a) >= THR[m] for n, a in zip(z["names"], z["acc"])})
print("dtype", new["masks"].dtype, tuple(new["masks"].shape), "size MB", round(Path(sys.argv[1]).stat().st_size/1e6, 2))
cnt = {}
for i, n in enumerate(names):
    d = int((M[i] != O[i]).sum())
    if not d: continue
    m = tm[n]; cnt[m] = cnt.get(m, 0) + 1
    b = B[m].get(n, {}); r = b.get("rect") or b.get("box")
    full = np.zeros((512, 512), np.uint8); y0, y1, x0, x1 = r; full[y0:y1, x0:x1] = 1
    isrect = np.array_equal(downsample_mask(full).astype(bool), M[i].astype(bool))
    print(f"{m:10s} {n} cells_changed {d:5d} conf {b.get('conf')} rect[y0,y1,x0,x1] {r} ({y1-y0}x{x1-x0}px) cov {O[i].mean():.3f}->{M[i].mean():.3f} "
          f"mask==rect {isrect} cov<0.5 {M[i].mean()<0.5} decoded_majority {reg[n]} flip {(O[i].mean()>=0.5)!=(M[i].mean()>=0.5)}")
print("images changed per method", cnt, "total cells", int((M != O).sum()))
