#!/bin/bash
# hyb_E_sis: T20c10 full pipeline (sM step -> s1 wide tail -> s2 t2i-bank repair -> x2 extrapolation) with targets
# runs/targets/hyb_E_sis.pt = 2*log(0.7*softmax(sM/2) + 0.3*mean softmax(E16,E18,T05 / 2)). Usage: bash hybE.sh (inside the relay srun).
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10; cd $W || exit 1
PY=../.venv/bin/python; C=runs/hybE/code; export PYTHONPATH=$W; R=runs/hybE; B=runs/pg/bank_t2i.npz; TG=runs/targets/hyb_E_sis.pt
echo "$(date +%T) start sH"
$PY -u $C/sis_unlearn.py --out $R/sH --targets $TG --retain-bank runs/ext/bank_ext.npz --w-bank 0.3 --bank-val 1000 \
  --lr 5e-5 --epochs 8 --keep all --holdout 0 --w-hard 0.5 --bank-start-epoch 3 > $R/sH.log 2>&1 || { echo sH FAIL; exit 1; }
Q="--targets $TG --trainable last:4 --w-hard 0.5 --retain-bank runs/ext/bank_ext.npz --w-bank 0.3
   --bank-val 1000 --temps 1,2 --keep all --holdout 0 --w-tail 1 --tail-frac 0.2 --tail-thr 0.2 --tail-cap 10"
echo "$(date +%T) start s1"
$PY -u $C/sis_unlearn.py --out $R/s1 --init $R/sH/ckpt_ep08.pt $Q --lr 5e-5 --epochs 4 > $R/s1.log 2>&1 || { echo s1 FAIL; exit 1; }
echo "$(date +%T) start s2"
$PY -u $C/sis_unlearn.py --out $R/s2 --init $R/s1/ckpt_ep04.pt $Q --lr 3e-5 --epochs 3 --bank2 $B --bank2-split 0 \
    --bank2-conf 0.99 --w-bank2 2 --bank2-temps 1,2 > $R/s2.log 2>&1 || { echo s2 FAIL; exit 1; }
$PY sis_interp.py --out runs/repair/hyb_E_sis_x.pt $R/s2/ckpt_ep03.pt:2 $R/s1/ckpt_ep04.pt:-1 > $R/x.log 2>&1 || { echo X FAIL; exit 1; }
echo "$(date +%T) measures"
$PY -u $C/vast_measure2.py --bank $B --orig ../model.pt --forget ../forget_images.npz --out $R/m2.jsonl \
    runs/repair/vast_T20c10_x.pt runs/repair/hyb_E_sis_x.pt > $R/m2.log 2>&1
$PY -u $C/grp_ce.py runs/repair/vast_T20c10_x.pt runs/repair/hyb_E_sis_x.pt > $R/grp.log 2>&1
$PY -u sis_eval.py --out $R/eval.jsonl runs/repair/vast_T20c10_x.pt runs/repair/hyb_E_sis_x.pt > $R/eval.log 2>&1
CUDA_VISIBLE_DEVICES= $PY -c 'from common import check_submission as c; c("runs/repair/hyb_E_sis_x.pt"); print("CHECK PASS")' > $R/check.log 2>&1
chmod -R g+rwX $R runs/repair/hyb_E_sis_x.pt 2>/dev/null
echo "$(date +%T) done"; touch $R/DONE
