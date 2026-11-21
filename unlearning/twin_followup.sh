#!/bin/bash
# Forward-only follow-ups on a twin regime, after twin_methods.py has scored its runs (login node, a few minutes):
#   1 alpha-scaling theta_orig + alpha (theta_run - theta_orig) of the best unlearn.py run, alpha 0.6 .. 1.4
#   2 soups (weight averages) of two runs with the same targets and different seeds
#   3 which local retain proxy predicts the true retain KL (twin_proxy.py)
# Usage: CUDA_VISIBLE_DEVICES=1 bash twin_followup.sh [regime]  > logs/twin/<regime>_followup.log
umask 002
R=${1:-pb10q}; M=runs/twin/$R/m; PY="../.venv/bin/python -u"
BEST=$($PY - "$R" <<'EOF'
import json, sys
from pathlib import Path
d = json.loads(Path(f"runs/twin/{sys.argv[1]}/methods.json").read_text())
runs = {k: v for k, v in d.items() if "/td_" in k and "oracle" not in k and "rank" not in k and "true3" not in k and "pred_" not in k}
print(max(runs, key=lambda k: runs[k]["mean_final"]))
EOF
)
echo "best non-oracle unlearn.py run: $BEST"
$PY twin_score.py --regime $R --interp $BEST $M/td_th_C_rec/submission.pt --alphas 0.6 0.8 0.9 1.0 1.1 1.2 1.4 --json runs/twin/$R/followup_alpha.json
SOUPS=""
for t in th_C_rec rc_ent_goldshape; do
  if [ -f $M/td_$t/submission.pt ] && [ -f $M/td_${t}_s1/submission.pt ]; then
    SOUPS="$SOUPS --soup $M/td_$t/submission.pt $M/td_${t}_s1/submission.pt"
  fi
done
[ -f $M/td_th_C_rec/submission.pt ] && [ -f $M/td_rc_ent_goldshape/submission.pt ] && SOUPS="$SOUPS --soup $M/td_th_C_rec/submission.pt $M/td_rc_ent_goldshape/submission.pt"
[ -n "$SOUPS" ] && $PY twin_score.py --regime $R $SOUPS $M/td_th_C_rec/submission.pt $M/td_th_C_rec_s1/submission.pt $M/td_rc_ent_goldshape/submission.pt \
  $M/td_rc_ent_goldshape_s1/submission.pt --json runs/twin/$R/followup_soup.json
$PY twin_proxy.py --regime $R
