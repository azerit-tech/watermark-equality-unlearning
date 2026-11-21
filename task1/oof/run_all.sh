#!/bin/bash
# Sequential: wait for leak-free cache, then the 5 fold trainings one after the other, then the analysis.
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while [ ! -f t1/out/oof/cache/syn_done.txt ]; do sleep 5; done
for k in 0 1 2 3 4; do
  [ -f t1/out/oof/trustmark/fold$k/done ] || python t1/oof/dec_oof.py --fold $k > t1/out/oof/fold$k.log 2>&1
done
python t1/oof/merge.py > t1/out/oof/merge.log 2>&1
touch t1/out/oof/all_done
