"""Fold-0 evaluation of the PixelSeal majority-regime TTA. Score of every fold-0 PixelSeal image (34) with the map
binarized at 0.5 (raw) and after regime_guard.guard (guard); the TTA map replaces the tiles3 map ONLY for majority
images (fullacc >= 0.80). Paired bootstrap (10000) of the per-image delta vs tiles3, by edit type.
python t1/pstta/eval_pstta.py"""
import json, sys
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1" / "train", TASK / "t1" / "tmdense", TASK / "t1" / "common", TASK): sys.path.insert(0, str(p))
from regime_guard import load_regimes, guard  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402
O = TASK / "t1/out/pstta/pixelseal"
def ld(p):
    z = np.load(p); return {str(n): q.astype(np.float32) for n, q in zip(z["names"], z["probs"])}
base = ld(TASK / "t1/out/tiles3/pixelseal/fold0_probs128.npz"); names = list(base)
grid = ld(TASK / "t1/out/tta/pixelseal/fold0_probs128.npz")
V = {v: ld(O / f"views_fold0_{v}.npz") for v in ("id", "hf", "vf", "s2", "hv", "m2") if (O / f"views_fold0_{v}.npz").exists()}
z = np.load(TASK / "release/train/masks_128.npz"); gt = dict(zip([str(n) for n in z["names"]], z["masks"]))
ti = json.loads((TASK / "t1/common/train_index.json").read_text())
if isinstance(ti, list): ti = {e["image_name"]: e for e in ti}
reg = load_regimes("train"); maj = [n for n in names if reg[n]]
et = np.array([ti[n]["edit_type"] for n in names])
print("majority images:", len(maj), "types", {t: int(sum(ti[n]["edit_type"] == t for n in maj)) for t in ("hole", "transplant", "outpainting")})
def cand(f):  # map for each name: f(n) for majority images, base otherwise
    return {n: (f(n) if reg[n] else base[n]) for n in names}
C = {"base(tiles3 GPU)": base,
     "id(CPU)": cand(lambda n: V["id"][n]),
     "hf seul": cand(lambda n: V["hf"][n]), "vf seul": cand(lambda n: V["vf"][n]),
     "hv seul": cand(lambda n: V["hv"][n]), "s2 seul": cand(lambda n: V["s2"][n]),
     "grille 4x2 (t1/out/tta)": cand(lambda n: grid[n]),
     "flips 4 (id,hf,vf,hv)": cand(lambda n: np.mean([V[v][n] for v in ("id", "hf", "vf", "hv")], 0)),
     "id+s2": cand(lambda n: (V["id"][n] + V["s2"][n]) / 2),
     "grille+s2": cand(lambda n: (grid[n] + V["s2"][n]) / 2),
     **({"m2 seul": cand(lambda n: V["m2"][n]), "grille+m2": cand(lambda n: (grid[n] + V["m2"][n]) / 2),
         "grille+s2+m2": cand(lambda n: (grid[n] + V["s2"][n] + V["m2"][n]) / 3),
         "id+s2+m2": cand(lambda n: (V["id"][n] + V["s2"][n] + V["m2"][n]) / 3)} if "m2" in V else {}),
     "grille+flips": cand(lambda n: np.mean([grid[n]] + [V[v][n] for v in ("hf", "vf", "hv")], 0))}
rng = np.random.default_rng(0)
def sc(P, g):
    out = []
    for n in names:
        m = guard(P[n], reg[n])[0] if g else (P[n] >= .5).astype(np.uint8)
        out.append(miou_image(m, gt[n]))
    return np.array(out)
res = {}
for g in (False, True):
    lab = "guard" if g else "raw"; s0 = sc(base, g)
    print(f"--- {lab}  (n hole={int((et=='hole').sum())}, transplant={int((et=='transplant').sum())}, outpainting={int((et=='outpainting').sum())})")
    for k, P in C.items():
        s = sc(P, g); d = s - s0; bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        r = {"all": round(float(s.mean()), 4), **{t: round(float(s[et == t].mean()), 4) for t in ("hole", "transplant", "outpainting")},
             "delta_all": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bm, 2.5)), 4), round(float(np.percentile(bm, 97.5)), 4)],
             "delta_hole": round(float(d[et == "hole"].mean()), 4), "wins/losses": [int((d > 1e-9).sum()), int((d < -1e-9).sum())]}
        res[f"{k}/{lab}"] = r; print(f"{k:26s}", json.dumps(r))
(O / "eval_fold0.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
