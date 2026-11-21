#!/bin/bash
# GPUS=1
# MINUTES=27
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
# Dense StegaStamp rectangle grid scan (t1/stegagrid/gridscan.py), resumable: each run scans for 24 min and
# saves the cursor per image in t1/out/stegagrid/scan_<SET>.json ; redeposit the same script to continue.
# SET=test (default) or SET=train (folds 0,1 minority-regime images, for measurement/FP check).
# WARNING: step 4 % of side = ~2e8 crops per image (~20 000 GPU-min at 160 decodes/s): not worth depositing as is.
set -e
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export OMP_NUM_THREADS=8
SET=${SET:-test}
python t1/stegagrid/gridscan.py --set "$SET" --folds 0,1 --minutes 24 --step ${STEP:-0.04} --smin ${SMIN:-16} --device cuda:0
chmod -R g+rwX t1/out/stegagrid 2>/dev/null || true
