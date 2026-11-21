"""Merge runs/sisters/bank_sis_a.npz + bank_sis_b.npz -> runs/sisters/bank_sis.npz (bank_ext.npz format: key images,
plus src / strength / seed). Login node, CPU only.   ../.venv/bin/python sisters_merge.py"""
import numpy as np
parts = [np.load(f"runs/sisters/bank_sis_{n}.npz") for n in "ab"]
out = {k: np.concatenate([p[k] for p in parts]) for k in ("images", "src", "strength", "seed")}
np.savez("runs/sisters/bank_sis.npz", **out)
print("runs/sisters/bank_sis.npz", out["images"].shape, out["images"].dtype, "sources", len(np.unique(out["src"])))
