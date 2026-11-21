#!/bin/bash
# Strategist: sequential logit dumps on login GPU 1, each process capped at 290 s.
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
export CUDA_VISIBLE_DEVICES=${GPU:-1}
P=../.venv/bin/python
R=runs
timeout 290 $P -u strat_logits.py orig=orig A_ep5=$R/cal_lr1e-5/ckpt_ep05.pt B_ep2=$R/cal_lr1e-5/ckpt_ep02.pt D_a3=$R/oracle/l5e3_a3.pt
timeout 290 $P -u strat_logits.py E_surg=$R/surgery/surg_ep2_g_lateMLPhead.pt C_L3=$R/sweep1/L3/submission.pt F_G2=$R/login/real_G2/submission.pt G_G1=$R/login/G1/submission.pt
timeout 290 $P -u strat_logits.py --strong
timeout 290 $P -u strat_logits.py G0_bank=$R/login/G0_bank/submission.pt G2_bank=$R/login/real_G2_bank/submission.pt G1_bank=$R/login/G1_bank/submission.pt G0=$R/login/G0/submission.pt
timeout 290 $P -u strat_logits.py thD=$R/login/theory_D/submission.pt thC=$R/sweep2/theory_C/submission.pt G2max=$R/recipe_train/G2max/submission.pt G2lr15=$R/login/real_G2_lr15/submission.pt
echo ALLDONE
