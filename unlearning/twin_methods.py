"""Method comparison on one twin regime, end to end. Every method sees only the twin-original and F.

  python twin_methods.py --regime tr --gpus 4            # on the card(s) given by CUDA_VISIBLE_DEVICES
  python twin_methods.py --regime test --gpus 1 --quick  # pipeline test on the login node

Stages (each writes its command list to jobs/twin_<regime>_<stage>.txt and runs it, several at a time per card;
  logs in logs/twin/<regime>/):
  1 targets   targets.py recipes on the twin-original (through twin_run.py, which only swaps the architecture):
                true3_*   the parametric family given gold's TRUE mean / std / error rate (it still has to guess
                          which image gets which loss and where the wrong-class mass goes)
                rank_*    the same, with gold's true per-image difficulty ranking
                lo_ / hi_ the family with the mean loss off by a factor 2 either way (cost of a wrong level)
                oracle_*  the true gold distributions (twin_analyze.py: oracle.pt): kl / opt / single / ens_T1
  2 sweep     unlearn.py on the oracle_kl targets over learning rates and trainable sets -> best setting
  3 methods   unlearn.py with the best setting on every target file, retention variants, and SCRUB
  4 score     twin_score.py on everything -> <regime>/methods.json and a table
Needs <regime>/analysis.json and oracle.pt (python twin_analyze.py --regime <regime>).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from twin_common import ALTER_RECIPES, TWIN_DIR

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--regime", required=True)
p.add_argument("--gpus", type=int, default=4)
p.add_argument("--parallel", type=int, default=None, help="runs at once (default 3 per card)")
p.add_argument("--quick", action="store_true", help="tiny version of every stage (pipeline test)")
p.add_argument("--slim", action="store_true", help="ViT-B regimes: the ~25 most informative runs (fits one 55-min single-card task)")
p.add_argument("--only", default=None, help="comma-separated target names: restrict stage 1 / 3 to these")
p.add_argument("--stages", default="1,2,3,4")
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--best", default=None, help="skip the sweep's choice: 'trainable,lr' e.g. all,1e-4")
args = p.parse_args()
R = TWIN_DIR / args.regime
P = f"twin_run.py --regime {args.regime}"
TG, M = R / "targets", R / "m"
an = json.loads((R / "analysis.json").read_text())
gm, gs, gacc = an["A"]["gold_F"]
err = 1 - gacc
stages = args.stages.split(",")
parallel = args.parallel or 3 * args.gpus
ep = 2 if args.quick else args.epochs


def run_batch(stage, lines):
    """Run the commands `parallel` at a time on the card(s) this process was given (CUDA_VISIBLE_DEVICES is
    inherited: the relay assigns it). With --gpus > 1 the runs are spread over the visible cards."""
    f = Path("jobs") / f"twin_{args.regime}_{stage}.txt"
    f.parent.mkdir(exist_ok=True)
    f.write_text("\n".join(lines) + "\n")
    log_dir = Path("logs/twin") / args.regime
    log_dir.mkdir(parents=True, exist_ok=True)
    cards = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(args.gpus))).split(",")
    print(f"\n##### stage {stage}: {len(lines)} commands ({f}), {parallel} at a time on card(s) {cards}", flush=True)
    pending, running, t0 = list(enumerate(lines)), [], time.time()
    while pending or running:
        while pending and len(running) < parallel:
            i, line = pending.pop(0)
            argv = shlex.split(line)
            name = Path(argv[argv.index("--out") + 1]).name if "--out" in argv else f"{stage}_{i}"
            log = log_dir / f"{name}.log"
            proc = subprocess.Popen([sys.executable, "-u", *argv], stdout=log.open("w"), stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": cards[i % len(cards)]})
            running.append((proc, name, log))
        for item in [r for r in running if r[0].poll() is not None]:
            running.remove(item)
            tail = item[2].read_text().strip().splitlines()[-1:] or [""]
            print(f"[{time.time() - t0:5.0f}s] exit {item[0].returncode} {item[1]}: {tail[0][:200]}", flush=True)
        time.sleep(1)


def score(glob, out):
    subprocess.run([sys.executable, "-u", "twin_score.py", "--regime", args.regime, "--glob", glob, "--json", str(out)], check=False)
    return json.loads(out.read_text()) if out.exists() else {}


def unlearn(name, target, trainable, lr, extra=""):
    return (f"{P} unlearn.py --twin --out {M / name} --targets {TG / (target + '.pt')} --holdout 0 --epochs {ep} --eval-every {ep} "
            f"--n-proxy 512 --trainable {trainable} --lr {lr} {extra}").strip()


# ---------------------------------------------------------------- 1 targets
true3 = f"--mean-loss {gm:.4f} --std-loss {gs:.4f} --err-rate {err:.4f}"
targets = {
    "true3_ln": true3,
    "true3_views": true3 + " --shape views",
    "true3_ent": true3 + " --shape views --difficulty entropy",
    "true3_ent_wc": true3 + " --shape views --difficulty entropy --rank-within-class",
    "true3_tau2": true3 + " --shape views --tau 2",
    "true3_tau4": true3 + " --shape views --tau 4",
    "true3_aug": true3 + " --shape views --others aug",
    "rank_ln": true3 + f" --ranking {R / 'analysis.pt'}:gold_L",
    "rank_views": true3 + f" --shape views --ranking {R / 'analysis.pt'}:gold_L",
    "lo_views": f"--mean-loss {gm / 2:.4f} --std-loss {gs * 0.75:.4f} --err-rate {err / 2:.4f} --shape views",
    "hi_views": f"--mean-loss {gm * 1.5:.4f} --std-loss {gs * 1.2:.4f} --err-rate {min(0.95, err * 1.4):.4f} --shape views",
    "oracle_kl": f"--probs {R / 'oracle.pt'}:kl",
    "oracle_opt": f"--probs {R / 'oracle.pt'}:opt",
    "oracle_single": f"--probs {R / 'oracle.pt'}:single",
    "oracle_ensT1": f"--probs {R / 'oracle.pt'}:ens_T1",
}
# the four margin-reduction levels used on the real model (jobs/sweep1.txt)
for name, (ml, sl, er) in {"L1": (0.10, 0.45, 0.02), "L2": (0.25, 0.90, 0.05), "L3": (0.50, 1.50, 0.12), "L4": (1.00, 2.20, 0.22)}.items():
    targets[name] = f"--mean-loss {ml} --std-loss {sl} --err-rate {er} --shape views --difficulty entropy"
if (R / "theory.pt").exists():   # the theorist's recipe rebuilt on the twin-original (twin_analyze.py)
    for key in ("th_A_mild", "th_B", "th_C_rec", "th_D", "th_E_strong", "th_x0.5", "th_x1.0", "th_x1.5", "th_x1.0_oracle"):
        targets[key.replace(".", "")] = f"--probs {R / 'theory.pt'}:{key}"
if (R / "recipes.pt").exists():  # hand-made recipes with the clean-entropy ranking (twin_recipes.py)
    for key in ("ent_true", "ent_goldshape", "ent_class05_goldshape"):
        targets[f"rc_{key}"] = f"--probs {R / 'recipes.pt'}:{key}"
if (R / "predict.pt").exists():  # learned predictor (twin_predict.py), cross-validated predictions
    for key in ("cv", "cv_cal", "cvclass_cal"):
        targets[f"pred_{key}"] = f"--probs {R / 'predict.pt'}:{key}"
extra_file = R / "extra_targets.json"   # {"name": "targets.py arguments"}: recipes added by hand (bridge numbers, theory)
if extra_file.exists():
    targets.update(json.loads(extra_file.read_text()))
if args.quick:
    targets = {k: targets[k] for k in ("true3_views", "oracle_kl")}
SLIM = ("L1", "L2", "L3", "L4", "th_A_mild", "th_B", "th_C_rec", "th_D", "th_E_strong", "th_x10", "th_x10_oracle", "true3_views", "true3_ln", "oracle_kl",
        "oracle_opt", "oracle_single", "pred_cv_cal", "rc_ent_true", "rc_ent_goldshape", "rc_ent_class05_goldshape")
if args.slim:
    targets = {k: v for k, v in targets.items() if k in SLIM}
if args.only:
    targets = {k: v for k, v in targets.items() if k in args.only.split(",")}
if "1" in stages:
    run_batch("targets", [f"{P} targets.py --twin --out {TG / (k + '.pt')} {v}" for k, v in targets.items()])
    subprocess.run([sys.executable, "twin_analyze.py", "--regime", args.regime, "--targets", *sorted(str(f) for f in TG.glob("*.pt"))], check=False)

# ---------------------------------------------------------------- 2 sweep
grid = [("all", lr) for lr in ("1e-5", "3e-5", "1e-4", "3e-4")] + [("last:4", lr) for lr in ("3e-5", "1e-4", "3e-4", "1e-3")] + \
       [("last:2", "3e-4"), ("head", "3e-3")]
if args.quick:
    grid = [("all", "1e-4"), ("last:4", "3e-4")]
if args.slim:   # around the settings used on the real model (last:4, lr 3e-5, 12 epochs)
    grid = [("all", "1e-5"), ("all", "3e-5"), ("last:4", "3e-5"), ("last:4", "1e-4"), ("last:2", "1e-4"), ("last:2", "3e-4")]
sweep_name = lambda tr, lr: f"sw_{tr.replace(':', '')}_lr{lr}"
if "2" in stages:
    run_batch("sweep", [unlearn(sweep_name(tr, lr), "oracle_kl", tr, lr) for tr, lr in grid])
best = tuple(args.best.split(",")) if args.best else None
if best is None:
    res = score(str(M / "sw_*" / "submission.pt"), R / "sweep.json")
    ranked = sorted(((v["mean_final"], k) for k, v in res.items()), reverse=True)
    for s, k in ranked:
        print(f"  sweep {k}: mean final {s:.4f}")
    top = Path(ranked[0][1]).parent.name
    best = next((tr, lr) for tr, lr in grid if sweep_name(tr, lr) == top)
print(f"best setting: --trainable {best[0]} --lr {best[1]}", flush=True)
(R / "best_setting.json").write_text(json.dumps(best))

# ---------------------------------------------------------------- 3 methods
if "3" in stages:
    tr, lr = best
    lines = [unlearn(f"td_{k}", k, tr, lr) for k in targets]
    ref = "th_C_rec" if "th_C_rec" in targets else "true3_views"
    var_targets = ("oracle_kl",) if args.quick else (ref,) if args.slim else ("oracle_kl", ref)
    for k in [v for v in var_targets if v in targets]:
        variants = [("comp1", "--w-comp 1"), ("strong1", "--w-strong 1"), ("noalt", "--alt-views 0"), ("T2", "--temps 2")]
        if not args.slim:
            variants += [("compT", "--w-comp 1 --comp-label targets"), ("l2sp", "--l2sp 100"), ("alt3", "--alt-views 3"), ("T1", "--temps 1")]
        lines += [unlearn(f"td_{k}_{n}", k, tr, lr, extra) for n, extra in variants]
        if not args.slim:
            lines.append(unlearn(f"td_{k}_ep{ep * 3}", k, tr, lr).replace(f"--epochs {ep} --eval-every {ep}", f"--epochs {ep * 3} --eval-every {ep * 3}"))
    if not (args.quick or args.slim):
        for tr2, lr2 in grid:   # the realistic recipe over the same grid: does the best setting depend on the targets?
            lines.append(unlearn(f"tdg_{ref}_{tr2.replace(':', '')}_lr{lr2}", ref, tr2, lr2))
    scrub_lrs = ("1e-4",) if args.quick else ("5e-6", "2e-5") if args.slim else ("1e-5", "3e-5", "1e-4", "3e-4")
    for slr in scrub_lrs:
        lines.append(f"{P} scrub.py --twin --out {M / ('scrub_lr' + slr)} --epochs {2 if args.quick else 6} --msteps {1 if args.quick else 3} "
                     f"--lr-max {slr} --lr-min {slr} --keep all")
    run_batch("methods", lines)

# ---------------------------------------------------------------- 4 score
if "4" in stages:
    res = score(str(M / "*" / "submission.pt"), R / "methods.json")
    res2 = score(str(M / "scrub*" / "ckpt_ep*.pt"), R / "methods_scrub_epochs.json")
    print("\n===== ranking (mean of the final score over the two alteration recipes) =====")
    for k, v in sorted({**res, **res2}.items(), key=lambda kv: -kv[1]["mean_final"]):
        a = v[ALTER_RECIPES[0]]
        print(f"  {v['mean_final']:.4f}  {k.replace(str(M) + '/', ''):<52} forget {a['forget']:.3f} (ms {a['mean_std_score']:.3f} pred {a['prediction_score']:.3f}) "
              f"retain {a['retain']:.3f} (acc ratio {a['retain_acc_ratio']:.3f}, KL {a['retain_kl']:.3f}) | F ce {a['clean_ce_mean']:.3f} std {a['clean_ce_std']:.3f} acc {a['clean_acc']:.3f}")
