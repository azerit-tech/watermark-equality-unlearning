"""Weight-space combination (CPU, no evaluation): theta = theta_orig + sum_k c_k * (theta_k - theta_orig).
  ../.venv/bin/python sis_interp.py --out runs/sis/mix/x.pt runs/sis/sM/ckpt_ep08.pt:2.0 runs/oracle/E18.pt:0.5
"""
import argparse
import torch
from common import MODEL_FILE, check_submission
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("terms", nargs="+")
a = p.parse_args()
sd0 = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
out = {k: v.clone().float() if v.is_floating_point() else v.clone() for k, v in sd0.items()}
for t in a.terms:
    path, c = t.rsplit(":", 1); c = float(c)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k, v in sd0.items():
        if v.is_floating_point():
            out[k] += c * (sd[k].float() - v.float())
torch.save(out, a.out); check_submission(a.out); print("saved", a.out, a.terms)
