#!/bin/bash
# ~/t1w/run_integ.sh (compte mougeot1) — intégration tâche 1 à partir de sub20 (HOME de Lumine dans ~/lumine, liens ~/t1w,
# ~/sgseed, ~/sgseed2, ~/psseed, ~/tmseed). À lancer depuis le nœud de connexion : chaque commande passe par une étape
# srun --overlap (CPU, < 20 min) sur le job relais courant. N'écrit que sous ~/t1w et ~/jacques4m, jamais sur le scratch.
#
#   ~/t1w/run_integ.sh integ [options integrate20]   intègre les sorties DONE de la nuit sur la base sub20 :
#        jacques5/* (folds 1-4) et t1/out/jacques5b/sg_long, jacques7/sg_informed (StegaStamp) : chacun essayé SEUL au fold 0,
#        adopté si gain >= +0,01 ET IC95 bas > 0 (bootstrap apparié 10 000) ; jacques6/*_all : membres de test.
#        -> ~/t1w/integ20/run_HHMMSS/sub20i.pt + integ20_report.json (code 0 = contrôles OK ; 2 = ne pas envoyer)
#        Options utiles : --sgboxes-test ~/jacques4m/stegastamp_test_boxes.json  --psboxes-test ~/jacques4m/pixelseal_test_boxes.json
#                         --tmboxes-test F  --sg "t1/out/jacques7/sg_informed"  --j5 DIR  --j6 DIR  --root DIR
#        (rectangles TRAIN changés = fold 0 changé : à ne faire qu'en connaissance de cause, le contrôle sub20 tombe alors)
#   ~/t1w/run_integ.sh merge                          fusionne t1/out/jacques4/{a_sg,b_ps}_*/res.json (même partiels)
#        -> ~/jacques4m/{stegastamp,pixelseal}_{train,test}_boxes.json (base SG test = ~/sgseed2 de sub20 ; règle conf >= 0,9
#        et base < 0,9 ; STOP si faux positif transplant au fold 0)
#   ~/t1w/run_integ.sh sub20                          reconstruit sub20 (contrôle, doit être identique à ~/t1w/sub20/sub20.pt)
#
# Exemple après a_sg_test_* et b_ps_* DONE :
#   ~/t1w/run_integ.sh merge && ~/t1w/run_integ.sh integ --sgboxes-test ~/jacques4m/stegastamp_test_boxes.json \
#        --psboxes-test ~/jacques4m/pixelseal_test_boxes.json
# Rapatrier sur le Mac puis contrôler (essai à blanc, rien n'est envoyé) :
#   ssh jureca 'cat ~/t1w/integ20/run_XXXX/sub20i.pt' > task1/subs/sub20i.pt && python3 task1/submit_t1.py task1/subs/sub20i.pt
# Si /p/scratch est monté sur le nœud courant (ex. jrlogin06 = ssh jureca2), tout tourne DIRECTEMENT ici, en nice,
# OMP_NUM_THREADS=8 ; sinon (jrlogin10) via srun --overlap sur le job relais.
# Options integrate20 : --rule lumine (défaut : gain >= 0,01 et IC95 bas > 0) | --rule member (membre de moyenne).
set -u
mode=${1:?integ|merge|sub20}; shift
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
if [ -d "$TASK/t1" ]; then J=""; else
  J=$(squeue -u "$USER" -n relais -h -t R -o %i | head -1)
  [ -z "$J" ] && { echo "scratch non monté ici et aucun job relais en cours : utiliser jureca2 (jrlogin06) ou attendre"; exit 3; }
fi
stamp=$(date +%H%M%S)
case $mode in
  integ) R=~/t1w/integ20/run_$stamp; mkdir -p "$R"; L=$R/run.log
         CMD="python ~/t1w/integ20/code/integrate20.py --root $R $*" ;;
  merge) mkdir -p ~/jacques4m; L=~/jacques4m/merge_$stamp.log
         CMD="python ~/t1w/integ20/merge20.py --out ~/jacques4m" ;;
  sub20) R=~/t1w/sub20_repro; mkdir -p "$R"; L=$R/build_$stamp.log
         CMD="python ~/t1w/final/build_final.py --config $R/config.json && python -c \"import torch; a=torch.load('$R/sub20.pt',weights_only=True); b=torch.load('$HOME/t1w/sub20/sub20.pt',weights_only=True); print('IDENTIQUE à sub20 :', a['image_names']==b['image_names'] and torch.equal(a['masks'],b['masks']))\"" ;;
  *) echo "mode inconnu $mode"; exit 1 ;;
esac
INNER="cd $TASK && source .venv/bin/activate
export TASK=$TASK CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
t0=\$(date +%s); { $CMD ; } > $L 2>&1; rc=\$?; echo \"rc=\$rc sec=\$((\$(date +%s)-t0))\" >> $L; exit \$rc"
if [ -z "$J" ]; then
  echo "exécution directe sur $(hostname) (scratch monté) ; journal $L"
  nice -n 10 bash -c "$INNER"
else
  echo "job relais $J ; journal $L"
  timeout 1200 srun --jobid="$J" --overlap -N1 -n1 -c8 --gres=gpu:0 bash -c "$INNER"
fi
rc=$?
grep -E "score seul|médiane|seul |vs sub20 :|vs ref|fold 0 .* vs sub20|couverture|base fold 0|ADOPTÉ|rejeté|adoptés|final |vs sub20|images différentes|STOP|new rects|ÉCHEC|ALERTE|FIN|IDENTIQUE|rc=" "$L" | tail -40
exit $rc
