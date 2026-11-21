#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:50:00
# devel OOF job: 4 folds of one method, one process per A100. Usage (from TASK):
#   [variant: 3rd arg "it" = iteration-matched, outputs t1/out/devel/<m>_it]
#   sbatch --job-name=oof-<m> --output=t1/out/devel/<m>/slurm-%j.out t1/devel/job_v2.sbatch <pixelseal|trustmark> "1 2 3 4"
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
M=$1; FOLDS=${2:-"1 2 3 4"}; export VARIANT=${3:-time}; D=t1/out/devel/$M; [ "$VARIANT" = it ] && D=t1/out/devel/${M}_it
cd $TASK; mkdir -p $D
echo "job ${SLURM_JOB_ID:-?} $(hostname -s) $M folds=$FOLDS start $(date +%H:%M:%S)"; nvidia-smi -L
declare -A PID; g=0
for k in $FOLDS; do
  mkdir -p $D/fold$k
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 WORKERS=13 bash t1/devel/run_fold_v2.sh $M $k > $D/fold$k/run.log 2>&1 &
  PID[$k]=$!; g=$((g + 1))
done
FAIL=0
for k in $FOLDS; do
  rc=0; wait ${PID[$k]} || rc=$?
  echo "fold $k rc=$rc $(date +%H:%M:%S)"; [ $rc -ne 0 ] && FAIL=$((FAIL + 1)) && echo "rc=$rc FAILED" >> $D/fold$k/FAILED || true
done
echo "failed=$FAIL end $(date +%H:%M:%S)" | tee $D/JOB_DONE
