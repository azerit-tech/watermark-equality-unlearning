"""Score checkpoints of the twin architecture with the EXACT server formula, against OUR gold models.

  python twin_score.py --regime tiny --self-check                 # validation: original -> 0, gold vs other golds
  python twin_score.py --regime tiny runs/twin/tiny/m/a/submission.pt runs/twin/tiny/m/b/submission.pt
  python twin_score.py --regime pb10q --interp runs/twin/pb10q/m/scrub_lr5e-6/ckpt_ep03.pt --alphas 0.25 0.5 1 1.5
  python twin_score.py --regime tiny --glob 'runs/twin/tiny/m/*/submission.pt' --json runs/twin/tiny/scores.json

Formula (twin_common.server_score), forward pass in eval mode under fp16 autocast like the server:
  forget = mean_std_score * prediction_score, each averaged over two groups: the F images and altered copies
    mean_std_score   = 1 - clip(d_sub / d_orig, 0, 1), d = |mean(L) - mean(L_gold)| + |std(L) - std(L_gold)|,
                       L = per-sample cross-entropy (T=1, true label); gold mean / std averaged over the gold models
    prediction_score = 1 - clip(mean KL(p_gold || p_sub) / mean KL(p_gold || p_orig), 0, 1), p = softmax(logits / 2),
                       p_gold = average of the gold models' distributions
  retain = clip(acc_sub / acc_orig, 0, 1) / (1 + mean KL(p_orig || p_sub)) at T=2 on the control set T
  final  = forget * retain
The server's alteration is unknown, so every score is given for two recipes of "altered copies":
  alter   augment.alter (mild crop + flip + small photometric change)
  unseen  each image gets one of augment.alter_unseen's kinds (blur, rotation, stronger crop, colour cast)
The 70% validation gate of the server is not applied (the twin-original is itself below it); val_acc is printed.
"""
import argparse
import glob
import json
from pathlib import Path

import torch

from twin_common import ALTER_RECIPES, TWIN_DIR, Bench, build, score_views

HEAD = (f"{'recipe':>6} | {'FINAL':>6} = {'forget':>6} x {'retain':>6} | {'ms':>5} ({'cln':>5} {'alt':>5}) {'pred':>5} ({'cln':>5} {'alt':>5}) | "
        f"{'ce cln':>6} {'std':>5} {'acc':>5} | {'ce alt':>6} {'std':>5} {'acc':>5} | {'gold ce':>7} {'std':>5} | "
        f"{'kl/kl_o cln':>13} {'alt':>13} | {'accT':>5} {'ratio':>5} {'retKL':>6}")


def fmt(recipe, s):
    return (f"{recipe:>6} | {s['final']:6.4f} = {s['forget']:6.4f} x {s['retain']:6.4f} | {s['mean_std_score']:5.3f} "
            f"({s['clean_mean_std_score']:5.3f} {s['alt_mean_std_score']:5.3f}) {s['prediction_score']:5.3f} "
            f"({s['clean_prediction_score']:5.3f} {s['alt_prediction_score']:5.3f}) | {s['clean_ce_mean']:6.3f} "
            f"{s['clean_ce_std']:5.3f} {s['clean_acc']:5.3f} | {s['alt_ce_mean']:6.3f} {s['alt_ce_std']:5.3f} {s['alt_acc']:5.3f} | "
            f"{s['clean_gold_ce_mean']:7.3f} {s['clean_gold_ce_std']:5.3f} | {s['clean_kl_sub']:6.3f}/{s['clean_kl_orig']:6.3f} "
            f"{s['alt_kl_sub']:6.3f}/{s['alt_kl_orig']:6.3f} | {s['val_acc']:5.3f} {s['retain_acc_ratio']:5.3f} {s['retain_kl']:6.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="*", type=Path)
    p.add_argument("--regime", required=True)
    p.add_argument("--glob", default=None)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--interp", type=Path, nargs="*", default=[], help="checkpoints to interpolate: theta_orig + alpha * (theta_ckpt - theta_orig)")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5])
    p.add_argument("--soup", type=Path, nargs="+", action="append", default=[], help="checkpoints whose weights are averaged (repeatable)")
    args = p.parse_args()
    bench = Bench(TWIN_DIR / args.regime, rebuild=args.rebuild_cache)
    K = len(bench.golds)
    print(f"regime {args.regime} ({bench.arch}): {K} gold models, F {len(bench.y_f)} images, T {len(bench.y_t)} images")
    results = {}

    if args.self_check:
        print("\n--- twin-original scored as a submission (must be 0) ---\n" + HEAD)
        s = bench.score(bench.orig)
        for r in ALTER_RECIPES:
            print(fmt(r, s[r]))
        results["orig"] = s
        print("\n--- each gold scored against the average of the OTHER golds: the ceiling of a perfect retraining ---\n" + HEAD)
        loo = []
        for k in range(K):
            s = bench.score(bench.golds[k], [g for j, g in enumerate(bench.golds) if j != k])
            loo.append(s)
            results[f"gold{k}_vs_others"] = s
            for r in ALTER_RECIPES:
                print(f"gold {bench.gold_files[k].stem[4:]} " + fmt(r, s[r]))
        for r in ALTER_RECIPES:
            for key in ("final", "forget", "retain", "mean_std_score", "prediction_score", "retain_kl"):
                v = torch.tensor([s[r][key] for s in loo])
                print(f"  ceiling [{r:>6}] {key:>16}: mean {v.mean():.4f}  min {v.min():.4f}  max {v.max():.4f}")
        print("\n--- each gold scored against ALL golds (itself included: optimistic) ---")
        for k in range(K):
            s = bench.score(bench.golds[k])
            results[f"gold{k}_vs_all"] = s
            print(f"gold {bench.gold_files[k].stem[4:]} " + " | ".join(f"{r}: final {s[r]['final']:.4f} forget {s[r]['forget']:.4f}" for r in ALTER_RECIPES))

    files = list(args.ckpts) + ([Path(f) for f in sorted(glob.glob(args.glob))] if args.glob else [])
    if files:
        print("\n" + f"{'checkpoint':<44} " + HEAD)
    for f in files:
        try:
            s = bench.score(bench.views(f))
        except Exception as e:  # a missing or malformed file must not hide the other rows
            print(f"{str(f):<44} FAILED: {e}")
            continue
        results[str(f)] = s
        name = str(f).replace(str(TWIN_DIR / args.regime) + "/", "").replace("/submission.pt", "")
        for r in ALTER_RECIPES:
            print(f"{name[-44:]:<44} " + fmt(r, s[r]))
    if args.interp:
        sd0 = torch.load(bench.dir / "orig.pt", map_location="cpu", weights_only=True)
        print("\nweight-space interpolation theta_orig + alpha * (theta_ckpt - theta_orig)\n" + f"{'checkpoint, alpha':<44} " + HEAD)
        for f in args.interp:
            sd1 = torch.load(f, map_location="cpu", weights_only=True)
            for a in args.alphas:
                model = build(bench.arch)
                model.load_state_dict({k: (v + a * (sd1[k] - v) if v.is_floating_point() else v) for k, v in sd0.items()})
                s = bench.score(score_views(model.to(bench.dev).eval(), bench.f_img, bench.t_img, bench.alt))
                results[f"{f}@{a}"] = s
                name = f"{str(f).replace(str(TWIN_DIR / args.regime) + '/', '')} a={a}"
                for r in ALTER_RECIPES:
                    print(f"{name[-44:]:<44} " + fmt(r, s[r]))
    for group in args.soup:
        sds = [torch.load(f, map_location="cpu", weights_only=True) for f in group]
        model = build(bench.arch)
        model.load_state_dict({k: (sum(sd[k] for sd in sds) / len(sds) if v.is_floating_point() else v) for k, v in sds[0].items()})
        s = bench.score(score_views(model.to(bench.dev).eval(), bench.f_img, bench.t_img, bench.alt))
        name = "soup(" + ",".join(Path(f).parent.name for f in group) + ")"
        results[name] = s
        print("\nweight average of " + ", ".join(str(f) for f in group) + "\n" + f"{'':<44} " + HEAD)
        for r in ALTER_RECIPES:
            print(f"{name[-44:]:<44} " + fmt(r, s[r]))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
