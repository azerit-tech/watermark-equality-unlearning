#!/bin/bash
# Redépose nos tâches longues sorties en code 3 (budget de tranche atteint). Arrête dès que la tâche est finie (DONE) ou échoue avec un autre code.
R=/p/scratch/training2666/trentelivres1/Azerit/relais
T=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1
DRY=${DRY:-0}
for spec in "f_longcomb_tm:jacques5/f_longcomb_tm.sh:out/jacques5/longcomb_trustmark" "f_longcomb_ps:jacques5/f_longcomb_ps.sh:out/jacques5/longcomb_pixelseal" "sg_long:jacques5b/sg_long.sh:out/jacques5b/sg_long"; do
  IFS=: read name script out <<< "$spec"
  if [ -f "$T/$out/DONE" ]; then echo "$name: DONE"; continue; fi
  if ls $R/todo $R/running 2>/dev/null | grep -q -- "-$name.sh"; then echo "$name: en file ou en cours"; continue; fi
  last=$(ls -t $R/failed/*-$name.sh.log 2>/dev/null | head -1)
  if [ -z "$last" ]; then echo "$name: ni en file ni en échec (rien à faire)"; continue; fi
  n=$(ls $R/failed/*-$name.sh.log 2>/dev/null | wc -l)
  if tail -2 "$last" | grep -q "code 3"; then
    if [ "$n" -ge 8 ]; then echo "$name: déjà $n tranches, arrêt"; continue; fi
    echo "$name: code 3 (tranche $n) -> redépôt"; [ "$DRY" = 1 ] || $R/relais-add $T/$script 2>&1 | tail -1
  else echo "$name: échec non 3 : $(tail -1 "$last")"; fi
done
