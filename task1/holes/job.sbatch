#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:58:00
# NOT SUBMITTED (coordinator decides). Longer fine-tune on faithful synthetic holes, 4 A100 in parallel (40 min each):
#   GPU0 PixelSeal  hole_src new=1,lib=1   from t1/out/tiles3/pixelseal/last.pt -> t1/out/holes/a100/pixelseal
#   GPU1 TrustMark  hole_src new=1,lib=1   from t1/out/tiles2/trustmark/last.pt -> t1/out/holes/a100/trustmark
#   GPU2 PixelSeal  hole_src old=1 (control: same recipe/duration, old hole sampler) -> .../pixelseal_ctrl
#   GPU3 TrustMark  hole_src old=1 (control)                                        -> .../trustmark_ctrl
# then eval_ft.py (LAST ckpt vs start ckpt, fold 0, paired bootstrap; test maps written only if gain >= 0.015 & CI > 0).
# Usage (from TASK): sbatch --job-name=holes-ft --output=t1/out/holes/a100/slurm-%j.out t1/holes/job.sbatch
set -uo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd $TASK && source .venv/bin/activate; O=t1/out/holes/a100; mkdir -p $O
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "job ${SLURM_JOB_ID:-?} $(hostname -s) start $(date +%H:%M:%S)"; nvidia-smi -L
COMMON="--p_real 0.25 --mix outpainting=1,transplant=1,hole=4 --tiles 256,320 --tile_inf 320 --p_focus 0.7 --minutes 40 --eval_min 1000 --workers 14"
PS="--method pixelseal --res 256 --ppl 0 --bs 24 --lr 1.5e-4 --lr_enc 1e-5 --init t1/out/tiles3/pixelseal/last.pt"
TM="--method trustmark --res 256 --ppl 1 --bs 16 --lr 3e-4 --lr_enc 1.5e-5 --init t1/out/tiles2/trustmark/last.pt"
run() {  # gpu name args...
  local g=$1 n=$2; shift 2; mkdir -p $O/$n
  CUDA_VISIBLE_DEVICES=$g python t1/holes/train_tiles.py "$@" $COMMON --out $O/$n > $O/$n/train.log 2>&1
  case $n in pixelseal*) m=pixelseal; b=t1/out/tiles3/pixelseal/last.pt; s=256,320;; *) m=trustmark; b=t1/out/tiles2/trustmark/last.pt; s=320;; esac
  CUDA_VISIBLE_DEVICES=$g python t1/holes/eval_ft.py $m $b $O/$n/last.pt $O/$n --scales $s > $O/$n/eval.log 2>&1
  echo "$n done rc=$? $(date +%H:%M:%S)"
}
run 0 pixelseal      $PS --hole_src new=1,lib=1 &
run 1 trustmark      $TM --hole_src new=1,lib=1 &
run 2 pixelseal_ctrl $PS --hole_src old=1 &
run 3 trustmark_ctrl $TM --hole_src old=1 &
wait
echo "end $(date +%H:%M:%S)" | tee $O/JOB_DONE
