#!/bin/bash
# État GPU de l'équipe sur JURECA (lecture seule : squeue, sinfo, scontrol, getfacl).
#   ./gpu.sh              membres = utilisateurs nommés dans les droits du dossier d'équipe
#   ./gpu.sh user1 user2  membres donnés à la main
# Variables facultatives : RES (réservation, défaut cispahack), TEAM_DIR.
RES="${RES:-cispahack}"
TEAM_DIR="${TEAM_DIR:-/p/scratch/training2666/trentelivres1/Azerit}"

if [ $# -gt 0 ]; then
  MEMBERS="$*"
else
  MEMBERS=$( (getfacl -p "$TEAM_DIR" 2>/dev/null | sed -n 's/^user:\([^:][^:]*\):.*/\1/p'
              stat -c %U "$TEAM_DIR" 2>/dev/null) | sort -u | tr '\n' ' ')
  [ -z "$MEMBERS" ] && MEMBERS="$USER"
fi
MEMBERS=$(echo $MEMBERS)
MEMBERS_CSV=$(echo $MEMBERS | tr ' ' ',')

RESINFO=$(scontrol show reservation "$RES" 2>/dev/null)
if [ -z "$RESINFO" ]; then
  echo "Réservation $RES introuvable (terminée ?)."
else
  NODES=$(echo "$RESINFO" | grep -o ' Nodes=[^ ]*' | cut -d= -f2)
  TOTAL=$(echo "$RESINFO" | grep -o 'NodeCnt=[0-9]*' | cut -d= -f2)
  END=$(echo "$RESINFO" | grep -o 'EndTime=[^ ]*' | cut -d= -f2 | sed 's/T/ /; s/:00$//')
  FREE=$(sinfo -h -N -n "$NODES" -o '%N %t' 2>/dev/null | sort -u | awk '$2=="idle"' | wc -l)
  RUN=$(squeue -h -R "$RES" -t R -o '%b' 2>/dev/null)
  NRUN=$(echo "$RUN" | grep -c .)
  NRUN1=$(echo "$RUN" | grep -c 'gpu:1')
  NPD=$(squeue -h -R "$RES" -t PD 2>/dev/null | wc -l)
  echo "GPU JURECA, $(date '+%a %d %b %H:%M')   (réservation $RES, fin $END)"
  echo "  Réservation : $FREE/$TOTAL nœuds libres. $NRUN jobs en cours (dont $NRUN1 sur 1 seule carte), $NPD en attente."
fi
DEV=$(sinfo -h -p dc-gpu-devel -o '%t %D' 2>/dev/null | awk '{t+=$2} $1=="idle"{f+=$2} END{printf "%d/%d", f, t}')
echo "  dc-gpu-devel (hors réservation, 2 h max) : $DEV nœuds libres."

# Rang de chaque job en attente dans la file de la réservation (ordre de priorité).
ORDER=$(squeue -h -R "$RES" -t PD --sort=-p,i -o '%i' 2>/dev/null | awk '{printf "%s:%d ", $1, NR}')

echo "Équipe ($MEMBERS) :"
squeue -h -u "$MEMBERS_CSV" -o '%i|%u|%j|%T|%l|%M|%S|%b|%r' 2>/dev/null |
awk -F'|' -v order="$ORDER" -v npd="${NPD:-?}" '
BEGIN { n = split(order, a, " "); for (i = 1; i <= n; i++) { split(a[i], b, ":"); rank[b[1]] = b[2] } }
{
  g = "?"; if (match($8, /gpu:[0-9]+/)) g = substr($8, RSTART + 4, RLENGTH - 4)
  st = substr($7, 12, 5)
  if ($4 == "RUNNING")      etat = sprintf("EN COURS    %s écoulé / %s demandé", $6, $5)
  else if ($4 == "PENDING") {
    etat = sprintf("EN ATTENTE  %s demandé, début estimé %s", $5, (st == "" ? "inconnu" : st " (pessimiste)"))
    if ($1 in rank) etat = etat sprintf(", rang %d/%d", rank[$1], npd)
    else if ($9 != "Priority" && $9 != "Resources") etat = etat " [" $9 "]"
  }
  else if ($4 == "CONFIGURING") etat = "DÉMARRE"
  else if ($4 == "COMPLETING")  etat = "SE TERMINE"
  else etat = $4
  printf "  %-9s %-14s %-12s %s GPU  %s\n", $1, $2, substr($3, 1, 12), g, etat
  k++
}
END { if (k == 0) print "  aucun job en cours ou en attente." }'
