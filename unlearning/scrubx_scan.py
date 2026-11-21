"""Weight-space extrapolation scan of SCRUB deltas + natural-image guard (70 % utility gate).

Level = forget "mean + std" of the per-image CE against the original's pseudo-labels, averaged over the clean forget
images and their fixed altered copies (metrics.ProxyEvaluator, seed 1234: the stats behind the public-score law
score ~ 0.034 x level; E12 3.1 -> 0.105, E13 3.6 -> 0.124, E14 -> 0.139).
Natural guard = the last --nat-n images of runs/ext/bank_ext.npz (COCO val2017 + Imagenette; never trained on by
ext_distill.py nor by ens_scrub.py --bank-holdout >= nat-n): argmax agreement with the original (all, original
confidence >= 0.5 / 0.9, original class among the forget classes or not) and KL(p_orig || p_model) at T=2.

  # scan: deltas saved by ens_scrub.py --delta-every; per delta, coarse alpha grid on 1,024 images, then a full eval
  # at the alpha interpolated for each target level
  python scrubx_scan.py scan --deltas runs/scrubx/X1/deltas.jsonl --levels 4 5 6 --out runs/scrubx/scan.jsonl
  # guard: full eval (level + natural) of existing submissions
  python scrubx_scan.py models runs/oracle/E14.pt runs/ens/B1x.pt --out runs/scrubx/guard.jsonl
  # build: theta0 + alpha * delta as a float32 submission (check_submission) + full eval
  # pick: per target level, the scanned (delta, alpha) with the best natural retain proxy -> float32 submission
  # <pick-dir>/<prefix>_L<level>.pt + note <queue-dir>/<prefix>_L<level>.txt
  python scrubx_scan.py pick runs/scrubx/scan_c.jsonl --levels 5 6 7 8 10 --prefix scrubC --out runs/scrubx/picks.jsonl
  python scrubx_scan.py build --delta runs/scrubx/X1/delta_s00110.pt --alpha 2.3 --save runs/scrubx/X1_a2.3.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import alter
from common import (FORGET_IMAGES_FILE, MODEL_FILE, build_model, check_submission, get_device, kl_T,
                    load_forget_images, preprocess, server_logits)

BS = 128


class Ctx:
    def __init__(self, nat_bank, nat_n, limit=None):
        self.dev = get_device()
        self.sd0 = {k: v.to(self.dev) for k, v in torch.load(MODEL_FILE, map_location="cpu", weights_only=True).items()}
        self.model = build_model().to(self.dev).eval()
        self.model.load_state_dict(self.sd0)
        self.images = load_forget_images(FORGET_IMAGES_FILE)[:limit].to(self.dev)
        with np.load(nat_bank, allow_pickle=False) as d:
            self.nat = torch.from_numpy(d["images"][-nat_n:].copy()).to(self.dev)
        z = self.forget_logits(len(self.images))
        self.labels = z["clean"].argmax(1)
        self.fclasses = torch.zeros(100, dtype=torch.bool, device=self.dev)
        self.fclasses[self.labels.unique()] = True
        self.nat_ref = self.nat_logits()
        p = self.nat_ref.softmax(1)
        self.nat_lab, conf = p.argmax(1), p.max(1).values
        self.nat_sel = {"c50": conf >= 0.5, "c90": conf >= 0.9, "fc": self.fclasses[self.nat_lab],
                        "oc": ~self.fclasses[self.nat_lab]}

    def forget_logits(self, n_img):
        """Clean + altered logits of the first n_img forget images; altered copies identical to ProxyEvaluator's."""
        g = torch.Generator(device=self.dev).manual_seed(1234 + 1)
        zc, za = [], []
        for i in range(0, min(n_img, len(self.images)), BS):
            x = preprocess(self.images[i:i + BS])
            zc.append(server_logits(self.model, x))
            za.append(server_logits(self.model, alter(x, g)))
        return {"clean": torch.cat(zc), "alt": torch.cat(za)}

    def nat_logits(self):
        return torch.cat([server_logits(self.model, preprocess(self.nat[i:i + BS])) for i in range(0, len(self.nat), BS)])

    def set_alpha(self, delta, a):
        self.model.load_state_dict({k: v0 + a * delta[k].float() if k in delta else v0 for k, v0 in self.sd0.items()})

    @torch.no_grad()
    def evaluate(self, n_img=None, nat=True):
        n_img = n_img or len(self.images)
        z, out = self.forget_logits(n_img), {}
        for k in ("clean", "alt"):
            ce = F.cross_entropy(z[k], self.labels[:len(z[k])], reduction="none")
            out[f"{k}_mean"], out[f"{k}_std"] = ce.mean().item(), ce.std().item()
            out[f"{k}_acc"] = (z[k].argmax(1) == self.labels[:len(z[k])]).float().mean().item()
        out["level"] = 0.5 * (out["clean_mean"] + out["clean_std"] + out["alt_mean"] + out["alt_std"])
        out["law"] = 0.034 * out["level"]
        if nat:
            zn = self.nat_logits()
            hit, kl = (zn.argmax(1) == self.nat_lab).float(), kl_T(self.nat_ref, zn)
            out["nat_agree"], out["nat_kl"] = hit.mean().item(), kl.mean().item()
            for s, m in self.nat_sel.items():
                out[f"nat_agree_{s}"], out[f"nat_kl_{s}"] = hit[m].mean().item(), kl[m].mean().item()
            out["ret_proxy"] = out["nat_agree_c50"] / (1 + out["nat_kl"])
        return out


def fmt(m):
    s = (f"level {m['level']:.2f} (law {m['law']:.4f}) clean {m['clean_mean']:.2f}/{m['clean_std']:.2f}/{m['clean_acc']:.3f} "
         f"alt {m['alt_mean']:.2f}/{m['alt_std']:.2f}/{m['alt_acc']:.3f}")
    if "nat_agree" in m:
        s += (f" | nat agree {m['nat_agree']:.3f} c50 {m['nat_agree_c50']:.3f} c90 {m['nat_agree_c90']:.3f} "
              f"fc {m['nat_agree_fc']:.3f} oc {m['nat_agree_oc']:.3f} kl {m['nat_kl']:.3f}")
    return s


def load_delta(path, dev):
    return {k: v.to(dev) for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["scan", "models", "build", "pick"])
    p.add_argument("paths", nargs="*", type=Path)
    p.add_argument("--deltas", nargs="*", type=Path, default=[], help="deltas.jsonl index files (scan)")
    p.add_argument("--levels", type=float, nargs="+", default=[4.0, 5.0, 6.0])
    p.add_argument("--grid", default="0.5,1,1.5,2,2.5,3,3.5,4,5,6,8")
    p.add_argument("--grid-n", type=int, default=1024)
    p.add_argument("--min-step", type=int, default=0, help="skip deltas saved before this optimizer step")
    p.add_argument("--delta", type=Path)
    p.add_argument("--alpha", type=float)
    p.add_argument("--save", type=Path)
    p.add_argument("--nat-bank", type=Path, default=Path("runs/ext/bank_ext.npz"))
    p.add_argument("--nat-n", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("runs/scrubx/scan.jsonl"))
    p.add_argument("--limit", type=int, default=None, help="first N forget images only (CPU smoke test)")
    p.add_argument("--prefix", default="scrubC")
    p.add_argument("--pick-dir", type=Path, default=Path("runs/scrubx"))
    p.add_argument("--queue-dir", type=Path, default=Path("runs/submit_queue"))
    p.add_argument("--recipe", default="SCRUB max-steps only (ens_scrub.py --retain-steps 0: no repair, no composites, "
                                        "no bank), fp16 deltas along the trajectory, theta0 + alpha x delta")
    args = p.parse_args()
    t0 = time.time()
    ctx = Ctx(args.nat_bank, args.nat_n, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.open("a")
    print(f"ready in {time.time() - t0:.0f}s | {len(ctx.fclasses.nonzero())} forget classes | nat {len(ctx.nat)} "
          f"(c50 {ctx.nat_sel['c50'].float().mean():.2f}, fc {ctx.nat_sel['fc'].float().mean():.2f})", flush=True)

    def emit(row):
        log.write(json.dumps(row) + "\n")
        log.flush()

    if args.mode == "models":
        for path in args.paths:
            ctx.model.load_state_dict({k: v.to(ctx.dev) for k, v in
                                       torch.load(path, map_location="cpu", weights_only=True).items()})
            m = ctx.evaluate()
            emit({"kind": "model", "file": str(path), **m})
            print(f"{path.name:28s} {fmt(m)}", flush=True)
        return

    if args.mode == "pick":
        rows = [json.loads(l) for f in args.paths for l in f.read_text().splitlines()]
        rows = [r for r in rows if r.get("kind") == "target"]
        for L in args.levels:
            cand = [r for r in rows if r["target"] == L and abs(r["level"] - L) < 0.5]
            if not cand:
                print(f"L{L:g}: no scanned point", flush=True)
                continue
            r = max(cand, key=lambda r: r["ret_proxy"])
            run, step = Path(r["file"]).parent.name, r["step"]
            save = args.pick_dir / f"{args.prefix}_L{L:g}.pt"
            ctx.set_alpha(load_delta(r["file"], ctx.dev), r["alpha"])
            m = ctx.evaluate()
            torch.save({k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                        for k, v in ctx.model.state_dict().items()}, save)
            check_submission(save)
            emit({"kind": "pick", "target": L, "run": run, "step": step, "alpha": r["alpha"], "file": str(save),
                  "n_cand": len(cand), **m})
            note = (f"{save.resolve()} | {args.recipe}: run {run} step {step} (epoch {r['epoch']}, {r['phase']}), "
                    f"alpha {r['alpha']:.2f}; best natural retain proxy of {len(cand)} scanned points at level {L:g} | "
                    f"{fmt(m)} | law 0.034 x level = {m['law']:.3f} | forget agreement with original clean "
                    f"{m['clean_acc']:.3f} alt {m['alt_acc']:.3f} | holdout 0")
            args.queue_dir.mkdir(parents=True, exist_ok=True)
            (args.queue_dir / f"{args.prefix}_L{L:g}.txt").write_text(note + "\n")
            print(f"L{L:g}: {run} step {step} alpha {r['alpha']:.2f} -> {save} | {fmt(m)}", flush=True)
        return

    if args.mode == "build":
        delta = load_delta(args.delta, ctx.dev)
        ctx.set_alpha(delta, args.alpha)
        m = ctx.evaluate()
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                    for k, v in ctx.model.state_dict().items()}, args.save)
        check_submission(args.save)
        emit({"kind": "build", "delta": str(args.delta), "alpha": args.alpha, "file": str(args.save), **m})
        print(f"saved + checked {args.save} | {fmt(m)}", flush=True)
        return

    grid = [float(a) for a in args.grid.split(",")]
    top = max(args.levels)
    for index in args.deltas:
        for line in index.read_text().splitlines():
            e = json.loads(line)
            if e["step"] < args.min_step:
                continue
            delta = load_delta(e["file"], ctx.dev)
            pts = []
            for a in grid:
                ctx.set_alpha(delta, a)
                lv = ctx.evaluate(args.grid_n, nat=False)["level"]
                pts.append((a, lv))
                if lv > top + 0.75:
                    break
            emit({"kind": "grid", **e, "grid": pts})
            print(f"{Path(e['file']).parent.name}/{Path(e['file']).name} ep{e['epoch']} {e['phase']:7s} grid "
                  + " ".join(f"{a:g}:{lv:.2f}" for a, lv in pts), flush=True)
            for L in args.levels:
                a = None
                for (a1, l1), (a2, l2) in zip(pts, pts[1:]):
                    if l1 <= L <= l2:
                        a = a1 + (L - l1) * (a2 - a1) / max(l2 - l1, 1e-6)
                        break
                if a is None:
                    continue
                ctx.set_alpha(delta, a)
                m = ctx.evaluate()
                emit({"kind": "target", **e, "target": L, "alpha": a, **m})
                print(f"    L{L:g} alpha {a:.2f} {fmt(m)}", flush=True)
            del delta
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
