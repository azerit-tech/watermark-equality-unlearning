#!/bin/bash
# GPUS=1
# MINUTES=50
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Twin bench, regime pb6 (blocks 0..5 of the real model frozen): scorer self-check and ceiling, ground-truth analysis
# (gold on F clean / altered, bridge table), recipe scores, then the method ranking re-check: unlearn.py last:4,
# lr 1e-4, 12 epochs on the main target files, scored with this regime's gold models.
# Results: logs/twin/pb6_{selfcheck,analyze,recipes,predict,methods}.log, runs/twin/pb6/{analysis,methods}.json
umask 002
R=pb6; PY="../.venv/bin/python -u"
mkdir -p logs/twin
$PY twin_score.py --regime $R --self-check > logs/twin/${R}_selfcheck.log 2>&1
$PY twin_analyze.py --regime $R > logs/twin/${R}_analyze.log 2>&1
$PY twin_recipes.py --regime $R > logs/twin/${R}_recipes.log 2>&1
$PY twin_predict.py --regime $R > logs/twin/${R}_predict.log 2>&1
$PY twin_methods.py --regime $R --gpus 1 --parallel 4 --slim --skip-scrub --stages 1,3,4 --best last:4,1e-4 \
  --only L2,L3,L4,HT,aim_true,th_C_rec,th_x10,rc_ent_goldshape,rc_ent_class05_goldshape,pred_cv_cal,oracle_kl,oracle_opt > logs/twin/${R}_methods.log 2>&1
chmod -R g+rwX runs/twin logs/twin jobs 2>/dev/null
tail -n 30 logs/twin/${R}_selfcheck.log | cut -c1-200
grep -A25 "===== ranking" logs/twin/${R}_methods.log | cut -c1-220
