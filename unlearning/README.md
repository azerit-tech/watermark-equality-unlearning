# Unlearning

Task: make the provided ViT-B/16 forget 3,500 training images, so that it behaves like a model retrained
without them, while staying close to the original everywhere else. Task sheet: `../EuroAIChampionshipParis2026-11-21.pdf`.

Method: SCRUB (Kurmanji et al., NeurIPS 2023, <https://arxiv.org/abs/2302.09880>), adapted because the
task gives no retain data, no labels and no validation set. The adaptations are listed at the top of `scrub.py`.

## Where things are

| What | Where |
|---|---|
| Code (this folder) | GitHub repo, edited locally |
| Code on JURECA | `/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/` |
| Data (`model.pt`, `forget_images.npz`) | one level up on JURECA: `.../Azerit/unlearning/` |
| Python environment | `.../Azerit/unlearning/.venv` (shared, set up by the team lead: torch 2.15, timm 1.0) |
| Checkpoints, metrics | `runs/<name>/` on JURECA (not in git) |
| Job logs | `logs/` on JURECA (not in git) |

## Files

| File | Purpose |
|---|---|
| `scrub.py` | First method (SCRUB). One run writes a checkpoint and a metrics row per epoch, then `submission.pt`. |
| `targets.py` | Second method, step 1: builds a target output for every forget image (our estimate of gold behaviour). |
| `unlearn.py` | Second method, step 2: trains the model to match those targets. Writes `submission.pt`. |
| `evaluate.py` | Format checks and local proxy metrics for any checkpoints. |
| `submit.py` | Uploads a checkpoint to the leaderboard and logs the score in `submissions.jsonl`. |
| `metrics.py` | The local proxy metrics (what each column means is documented at the top). |
| `augment.py` | GPU augmentations: light alterations of forget images, and composites used as retain data. |
| `common.py` | Model builder, preprocessing and prediction, identical to the organisers' `task_template.py`. |
| `run.sbatch` | Runs any of the scripts on one GPU. |
| `sync.sh` | Copies code to JURECA, and metrics and logs back. |

## Workflow

On your Mac (needs an open `ssh jureca` connection):

```bash
./sync.sh          # push the code to JURECA
./sync.sh pull     # fetch metrics and logs back
```

On JURECA:

```bash
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10

# 1. Train. Each epoch prints one row of metrics and saves runs/first/ckpt_epXX.pt
sbatch run.sbatch scrub.py --out runs/first --epochs 6 --msteps 3
tail -f logs/unlearn-<jobid>.out

# 2. Compare checkpoints from any runs
sbatch run.sbatch evaluate.py runs/first/ckpt_ep*.pt

# 3. Submit from the login node (compute nodes have no internet)
echo 'HACKATHON_API_KEY=<the team key>' > .env      # once
../.venv/bin/python submit.py runs/first/ckpt_ep03.pt --note "lr 2e-6, epoch 3"
```

## Reading the metrics

Each row compares a checkpoint with the original model. None of this is the leaderboard score, which needs
the gold models and hidden data.

- **clean / alt loss, std**: cross-entropy on the forget images and on lightly altered copies. The original
  model is near zero (it memorised them). The gold models should sit at the loss of unseen images, with a
  spread: the score compares both the mean and the std.
- **acc**: agreement with the original model's prediction. Gold models still classify most forget images
  correctly, so this should fall only a little, not collapse.
- **ret kl, agree, score**: drift from the original on synthetic composites, in the same form as the
  server's retain score. Lower `ret kl` and higher `agree` are better.

## What has to be tuned on the leaderboard

The loss a gold model has on the forget images is unknown: there is no held-out data to estimate it.
A run saves every epoch, so submit a few checkpoints with different `clean loss` values, see which scores
best, then rerun with `--target-loss <that value>` and a smaller learning rate to land on it precisely.

Main knobs: `--lr-max` (how fast it forgets), `--msteps` (how long), `--lr-min` and `--retain-steps`
(how hard it repairs), `--p-alter` (share of altered images in the forgetting steps), `--trainable last:N`
(touch only the last N blocks, which protects the retain score).

## Target distillation (`targets.py` + `unlearn.py`)

SCRUB pushes the model as far from the original as it can and then repairs it; on this task the repair
re-teaches the forget images. The second method instead states what a gold model should output on each
forget image and trains towards exactly that.

```bash
# 1. Targets: the loss on the forget images follows a prescribed distribution (mean, std, share of errors).
#    These three numbers are guesses of the gold models' behaviour and are what to tune on the leaderboard.
sbatch run.sbatch targets.py --out runs/targets/m06_s12_e12.pt --mean-loss 0.6 --std-loss 1.2 --err-rate 0.12

# 2. Unlearn. --holdout 500 keeps 500 forget images out of training to measure the drift on untouched images;
#    use --holdout 0 for the run that is submitted.
sbatch run.sbatch unlearn.py --out runs/td_first --targets runs/targets/m06_s12_e12.pt --holdout 500
```

Which image gets which loss follows a difficulty ranking (default: the original model's loss on strongly
augmented views). Another ranking (`--ranking file.pt[:key]`) or complete target probabilities
(`--probs file.pt[:key]`, shape [3500, 100]) can be plugged in. Retention options of `unlearn.py`:
`--w-comp` (distil on composites), `--l2sp` (pull towards the original weights), `--trainable`
(`all`, `last:N`, `head`, `lora:R[:N]`); details at the top of the file.

Reading a row of `unlearn.py` (full definitions at the top of `metrics.py`):

- **clean / alt / unseen ce, std, acc**: as above; should reach the targets' values printed above the table.
  `unseen` averages four alteration types that are never used in training (blur, rotation, strong crop,
  colour shift): it shows whether the forgetting transfers to alterations we did not think of.
- **tkl**: KL(target || model) at T=2, the local stand-in for the server's KL(gold || submission). Lower is better.
- **fsc, fsc_u**: the server's forget score if the targets were the gold models (trained / unseen alterations).
  It measures how well the targets are reached, not whether the targets are right.
- **ret, ret2, noise, held**: drift from the original (KL at T=2, argmax agreement) on training-recipe
  composites, other composites, noise images, and the held-out forget images. `held` is the one to trust most.
- **wdist**: relative L2 distance to the original weights.
