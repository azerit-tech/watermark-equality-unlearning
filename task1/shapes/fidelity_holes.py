"""Fidélité des générateurs de trous : KS par descripteur (desc.KEYS), réel (255 trous du train) contre synthétique.
Depuis TASK (CPU, ~1 min) : python t1/shapes/fidelity_holes.py [--n 300]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(TASK / "t1/synth")]
from desc import mask_desc, fidelity, KEYS  # noqa: E402
from holes import sample_hole_mask  # noqa: E402
import shapes as S0  # noqa: E402  (générateur existant t1/synth/shapes.py)

OUT = TASK / "t1/out/shapes"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); a = ap.parse_args()
    real = [r for r in json.loads((OUT / "desc.json").read_text()) if r["kind"] == "hole"]
    rng = np.random.default_rng(1)
    lib = S0.Library()
    gens = {
        "new_holes": lambda: sample_hole_mask(rng),
        "synth_full": lambda: S0.sample_minority(lib, rng, "full"),
        "synth_blob_only": lambda: S0.shape_blob(lib.sample_frac(rng, "full"), rng, single=rng.random() >= 0.55),
    }
    res, ex = {}, {}
    for g, fn in gens.items():
        ms = [fn() for _ in range(a.n)]
        ex[g] = ms[:8]
        res[g] = fidelity(real, [mask_desc(m) for m in ms])
    print(f"{'desc':14s} " + " ".join(f"{g:>16s}" for g in gens))
    for k in list(KEYS) + ["mean_ks"]:
        print(f"{k:14s} " + " ".join(f"{res[g][k]:16.3f}" for g in gens))
    (OUT / "fidelity_holes.json").write_text(json.dumps(res, indent=1))
    rows = [np.concatenate([np.pad(m[::2, ::2] * 255, 2, constant_values=128) for m in ex[g]], 1) for g in gens]
    Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(OUT / "synth_holes.png")


if __name__ == "__main__":
    main()
