"""Full submission state dict from a noise_repair.py snapshot: original weights + the saved trainable tensors.
  ../.venv/bin/python noise_build.py runs/noise/x.part.pt runs/noise/x.pt
"""
import sys

import torch

from common import check_submission

part = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
sd = torch.load(part["base"], map_location="cpu", weights_only=True)
for k, v in part["tensors"].items():
    assert k in sd and sd[k].shape == v.shape, k
    sd[k] = v.float().contiguous()
sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
torch.save(sd, sys.argv[2])
check_submission(sys.argv[2])
print(f"{sys.argv[2]}: {len(part['tensors'])} tensors replaced, check_submission passed")
