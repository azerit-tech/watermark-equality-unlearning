"""Local evaluation on train with the OFFICIAL metric (wmloc.utils.miou_image / score).

    from evaluate import evaluate          # sys.path must contain t1/common and TASK
    res = evaluate({name: mask128, ...})   # res["all"], res["method"], res["regime"], res["method_regime"]

CLI (run from TASK):
    python t1/common/evaluate.py preds.pt            # {"image_names": [...], "masks": [N,128,128]} (submission format)
    python t1/common/evaluate.py preds.npz           # names + masks, or one array per image name
    python t1/common/evaluate.py preds.pt --fold 2   # restrict to fold 2 of splits_5fold.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
if str(TASK) not in sys.path:
    sys.path.insert(0, str(TASK))
from wmloc.utils import miou_image, score  # noqa: E402  (official metric, not reimplemented)

TRAIN = TASK / "release" / "train"
_CACHE: dict = {}


def load_train():
    """-> (meta {name: dict}, gt {name: (128,128) uint8})"""
    if not _CACHE:
        meta = json.loads((TRAIN / "metadata.json").read_text())
        z = np.load(TRAIN / "masks_128.npz")
        _CACHE["meta"] = {m["image_name"]: m for m in meta}
        _CACHE["gt"] = dict(zip(z["names"].tolist(), z["masks"]))
    return _CACHE["meta"], _CACHE["gt"]


def _as_mask(p):
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    p = np.squeeze(np.asarray(p))
    assert p.shape == (128, 128), p.shape
    if p.dtype == bool or np.issubdtype(p.dtype, np.integer):
        return (p != 0).astype(np.uint8)
    return (p >= 0.5).astype(np.uint8)          # float input: thresholded at 0.5


def regime(m):
    return "full" if m["full_watermark"] else "partial"


def evaluate(pred_dict, names=None, return_per_image=False):
    """pred_dict: {image_name: (128,128) 0/1}. names: subset to score (default: every train name in pred_dict).
    Returns {"all": (score, n), "method": {m: (score, n)}, "regime": {"full"/"partial": (score, n)},
             "method_regime": {"method/regime": (score, n)}} (+ "per_image" if asked)."""
    meta, gt = load_train()
    if names is None:
        names = [n for n in pred_dict if n in gt]
    missing = [n for n in names if n not in pred_dict]
    if missing:
        raise KeyError(f"{len(missing)} names without prediction, e.g. {missing[:3]}")
    preds = [_as_mask(pred_dict[n]) for n in names]
    gts = [gt[n] for n in names]
    per = {n: miou_image(p, g) for n, p, g in zip(names, preds, gts)}
    groups = defaultdict(list)
    for n in names:
        m = meta[n]
        groups[("method", m["method"])].append(per[n])
        groups[("regime", regime(m))].append(per[n])
        groups[("method_regime", f"{m['method']}/{regime(m)}")].append(per[n])
    res = {"all": (score(preds, gts), len(names)), "method": {}, "regime": {}, "method_regime": {}}
    for (kind, key), v in sorted(groups.items()):
        res[kind][key] = (float(np.mean(v)), len(v))
    if return_per_image:
        res["per_image"] = per
    return res


def format_result(res, title=""):
    lines = [f"== {title}" if title else "", f"ALL            {res['all'][0]:.4f}  (n={res['all'][1]})"]
    for kind in ("method", "regime", "method_regime"):
        for k, (s, n) in res[kind].items():
            lines.append(f"{k:<22} {s:.4f}  (n={n})")
    return "\n".join(l for l in lines if l)


def load_preds(path):
    path = Path(path)
    if path.suffix == ".pt":
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
    else:
        z = np.load(path, allow_pickle=False)
        obj = {k: z[k] for k in z.files}
    if "image_names" in obj or "names" in obj:
        names = obj.get("image_names", obj.get("names"))
        names = [str(n) for n in (names.tolist() if hasattr(names, "tolist") else names)]
        masks = obj["masks"]
        return {n: masks[i] for i, n in enumerate(names)}
    return dict(obj)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--fold", type=int, default=None, help="restrict to this fold of splits_5fold.json")
    a = ap.parse_args()
    pd = load_preds(a.preds)
    names = None
    if a.fold is not None:
        names = json.loads((Path(__file__).resolve().parent / "splits_5fold.json").read_text())["folds"][a.fold]
    print(format_result(evaluate(pd, names), f"{a.preds} fold={a.fold}"))
