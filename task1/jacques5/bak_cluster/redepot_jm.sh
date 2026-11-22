#!/bin/bash
# Redépôt (Jacques) de f_longcomb_tm UNIQUEMENT (f_longcomb_ps retiré le 22/11 02:20 : remplacé par divcov) (jamais sg_long, qui tourne sur vast.ai).
# Lancé toutes les 5 min depuis le Mac de Jacques (srun --overlap dans le job relais de mougeot1).
# Redépose si le dernier journal de failed/ dit « code 3 », a plus de 3 min, et que la tâche n'est ni en file ni en cours.
R=/p/scratch/training2666/trentelivres1/Azerit/relais
T=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1
L=$T/jacques5/redepot_jm.log
for spec in "f_longcomb_tm:jacques5/f_longcomb_tm.sh:out/jacques5/longcomb_trustmark"; do
  IFS=: read name script out <<< "$spec"
  [ -f "$T/$out/DONE" ] && continue
  ls $R/todo $R/running 2>/dev/null | grep -q -- "-$name.sh" && continue
  last=$(ls -t $R/failed/*-$name.sh.log 2>/dev/null | head -1)
  [ -z "$last" ] && continue
  tail -2 "$last" | grep -q "code 3" || { echo "$(date +%H:%M) $name: dernier échec non 3, rien fait ($last)" >> $L; continue; }
  [ $(( $(date +%s) - $(stat -c %Y "$last") )) -lt 180 ] && continue
  n=$(ls $R/failed/*-$name.sh.log 2>/dev/null | wc -l)
  [ "$n" -ge 8 ] && continue
  echo "$(date +%H:%M) $name: code 3 (tranche $n, $last) -> redépôt" >> $L
  $R/relais-add $T/$script >> $L 2>&1
done
