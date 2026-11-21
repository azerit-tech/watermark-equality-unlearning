#!/bin/bash
# Creneau generique : le contenu du calcul est lu AU DEMARRAGE dans t1/mgr/payload.sh, qu'on peut donc modifier
# tant que le job attend. Regles d'equipe : un seul job en attente, 4 GPU utilises, <= 1 h.
# Depuis TASK :
#   sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 \
#     --cpus-per-task=64 --time=00:59:00 --job-name=t1_slot --output=t1/out/mgr/%j.out t1/mgr/slot.sbatch
set -euo pipefail
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd "$TASK"
P=t1/mgr/payload.sh
if [ ! -s "$P" ]; then echo "payload absent ou vide : rien a lancer"; exit 1; fi
bash -n "$P"
cp "$P" "t1/out/mgr/payload_${SLURM_JOB_ID}.sh"
echo "start $(date) $(hostname) payload=$(md5sum < "$P" | cut -c1-8)"; nvidia-smi -L
bash "t1/out/mgr/payload_${SLURM_JOB_ID}.sh"
