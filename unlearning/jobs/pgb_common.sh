# shared settings of the pgb_/hyb_ relay jobs (sourced)
umask 002
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
PY=/p/scratch/training2666/trentelivres1/Azerit/unlearning/.venv/bin/python
cd $W
BENCH="--orig ../model.pt --forget ../forget_images.npz --cache runs/pgb/cache"
# retain set of the bench until runs/pg/bank_t2i.npz exists: held-out tail of the kept-class sisters (as sis_eval.py)
BRET="--retain runs/sisters/sis_keep.npz --n-ret 1000"
SIS="--sis runs/sisters/sis_probs.pt"
HYB="--orig ../model.pt --forget ../forget_images.npz --bank runs/ext/bank_ext.npz --steps 600 --ckpt-every 150"
# score every checkpoint of a run dir + E16/E18 with the sisters bench
score_run() { $PY pgb_score.py $BENCH $BRET $SIS --cands runs/oracle/E18.pt runs/oracle/E16.pt $1/ckpt_s*.pt --out $1/bench_sis.jsonl > $1/bench_sis.log 2>&1; }
