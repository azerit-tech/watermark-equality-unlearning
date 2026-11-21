"""Intrinsic ceiling: re-embed PixelSeal on the 43 real outpainting geometries (R + k, k~U{2..5} per side, H6a recipe)
using a stegastamp train image as cover, then decode R+3 bilinear-aa. If thin rects also fail here, the ceiling is the embedder's."""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK))
from wmloc.watermarks import load_watermark  # noqa
torch.set_num_threads(int(sys.argv[2])); M = sys.argv[1]
wm = load_watermark(M, device="cpu")
RES, NET = (256, lambda z: wm.det(z)[:, 1:]) if M == "pixelseal" else (wm.resolution, lambda z: wm.net(z * 2 - 1))
idx = json.loads((TASK / "t1/common/train_index.json").read_text())
meta = json.loads((TASK / "release/train/metadata.json").read_text())
outp = [m for m in meta if m["method"] == M and idx[m["image_name"]]["edit_type"] == "outpainting"]
covers = [m for m in meta if m["method"] == "stegastamp"]
real = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 and Path(sys.argv[3]).exists() else {}
rng = np.random.default_rng(0); out = {}
for k, m in enumerate(outp):
    r = idx[m["image_name"]]["bbox512"]; ks = rng.integers(2, 6, 4)
    C = np.array(Image.open(TASK / f"release/train/images/{covers[k]['image_name']}.png").convert("RGB"))
    Y0, Y1, X0, X1 = max(r[0] - ks[0], 0), min(r[1] + ks[1], 512), max(r[2] - ks[2], 0), min(r[3] + ks[3], 512)
    X = C.copy(); X[Y0:Y1, X0:X1] = wm.embed(np.ascontiguousarray(C[Y0:Y1, X0:X1]), m["message"])
    x = torch.from_numpy(X).permute(2, 0, 1)[None].float().div(255); msg = np.array([int(c) for c in m["message"]], bool)
    with torch.no_grad():
        accs = []
        for (a, b, c, d) in [(Y0, Y1, X0, X1), (max(r[0] - 3, 0), min(r[1] + 3, 512), max(r[2] - 3, 0), min(r[3] + 3, 512))]:
            z = F.interpolate(x[:, :, a:b, c:d], size=(RES, RES), mode="bilinear", align_corners=False, antialias=True)
            accs.append(float(((NET(z).float().numpy()[0] > 0) == msg).mean()))
    rr = real.get(f"{m['image_name']}|outpainting", {}).get("bilinear-aa|m3")
    out[m["image_name"]] = {"rect": r, "minside": min(r[1] - r[0], r[3] - r[2]), "sim_exact": accs[0], "sim_m3": accs[1], "real_m3": rr}
    print(m["image_name"], r, "minside", out[m["image_name"]]["minside"], "sim exact %.3f m3 %.3f real %s" % (accs[0], accs[1], rr), flush=True)
Path(sys.argv[4]).write_text(json.dumps(out))
