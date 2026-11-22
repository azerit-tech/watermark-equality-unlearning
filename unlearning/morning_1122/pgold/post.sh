umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
cp $HOME/pgw/mk_targets.py $HOME/pgw/pgold.sh runs/pgold/code/
grep -h FINAL runs/pgold/train_s*.log | cut -c1-900
CUDA_VISIBLE_DEVICES= ../.venv/bin/python runs/pgold/code/mk_targets.py
