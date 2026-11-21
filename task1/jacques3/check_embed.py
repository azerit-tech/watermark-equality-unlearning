"""Sanity check (CPU): batched Embedder.residual vs the official wm.embed (uint8) on 2 fold-0 images, plus the
correlation of r'(m) with the TRUE residual (Cw - C) on 2 synthetic covers. Run from TASK:
python t1/jacques3/check_embed.py <method> [ckpt_to_inspect]"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from informed import Embedder, luma  # noqa: E402
from shapes import meta, names_in_folds, read_img  # noqa: E402
from train_tm import bits  # noqa: E402

m = sys.argv[1]; torch.set_num_threads(4); emb = Embedder(m, "cpu")
for n in names_in_folds([0], m)[:2]:
    x = read_img(n); b = bits(meta()[n]["message"])
    off = emb.wm.embed(x, b).astype(np.float32) - x.astype(np.float32)
    X = torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255
    r = emb.residual(X, torch.from_numpy(b)[None])[0].permute(1, 2, 0).numpy() * 255
    c = np.corrcoef(off.ravel(), r.ravel())[0, 1]
    print(f"{n}: official vs batched residual corr={c:.4f} |off|={np.abs(off).mean():.3f} |ours|={np.abs(r).mean():.3f}", flush=True)
d = Path("t1/out/synth/v0") / m / "full" / "s0_0000"
C, Cw, msg = np.load(d / "C.npy", mmap_mode="r"), np.load(d / "Cw.npy", mmap_mode="r"), np.load(d / "msg.npy")
for i in range(2):
    c0, cw = np.asarray(C[i]), np.asarray(Cw[i]); true = luma(torch.from_numpy(cw.astype(np.float32) - c0.astype(np.float32)).permute(2, 0, 1)[None])
    X = torch.from_numpy(cw).permute(2, 0, 1)[None].float() / 255; mm = torch.from_numpy(msg[i].astype(np.uint8))[None]
    rm = luma(emb.residual(X, mm)) * 255; rc = luma(emb.residual(X, 1 - mm)) * 255
    cc = lambda u, v: np.corrcoef(u.numpy().ravel(), v.numpy().ravel())[0, 1]
    print(f"synth {i}: corr(r'(m), true)={cc(rm, true):.3f} corr(d, true)={cc(rm - rc, true):.3f}", flush=True)
if len(sys.argv) > 2:
    ck = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    w = ck["model"]["head.0.0.weight"]
    print(f"ckpt it={ck['it']} head new-input weights |w| max={w[:, -32:].abs().max():.2e} old mean={w[:, :-32].abs().mean():.2e}")
