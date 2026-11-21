# Bibliothèque commune du relais (sourcée par relais-job.sh et les commandes relais-*).
# Réglages par variables d'environnement (valeurs par défaut ci-dessous).
: "${RELAIS_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${RELAIS_NAME:=relais}"           # nom des jobs Slurm du relais
: "${RELAIS_ACCOUNT:=training2666}"
: "${RELAIS_PARTITION:=dc-gpu}"
: "${RELAIS_RES:=cispahack}"
: "${RELAIS_MINUTES:=30}"            # durée demandée pour chaque passage
: "${RELAIS_MAX_MINUTES:=60}"        # plafond (le guide demande de prévenir au-delà d'1 h)
: "${RELAIS_TASK_MINUTES:=20}"       # durée supposée d'une tâche sans « # MINUTES= »
: "${RELAIS_IDLE:=2}"                # minutes à vide avant de rendre le nœud
: "${RELAIS_BASE_MINUTES:=30}"       # durée des successeurs (toujours courts)
: "${RELAIS_TASK_MAX:=470}"          # durée maximale déclarable pour une tâche
: "${RELAIS_LONG:=0}"                # 1 = relais long (relais-start --long) : reste sur le nœud
: "${RELAIS_LONG_IDLE:=30}"          #     et le rend après 30 min sans tâche en cours ni en attente
: "${RELAIS_NGPUS:=4}"
: "${RELAIS_SIM:=0}"                 # 1 = simulation : ni sbatch ni scancel, temps accéléré
: "${RELAIS_UNIT:=60}"               # secondes par « minute » (simulation : 1 ou 2)
: "${RELAIS_POLL:=10}"               # secondes entre deux tours de boucle
export RELAIS_DIR RELAIS_MINUTES RELAIS_LONG RELAIS_SIM RELAIS_UNIT RELAIS_POLL

R_TODO="$RELAIS_DIR/todo"; R_RUN="$RELAIS_DIR/running"; R_DONE="$RELAIS_DIR/done"
R_FAIL="$RELAIS_DIR/failed"; R_LOGS="$RELAIS_DIR/logs"; R_STOP="$RELAIS_DIR/STOP"
R_EMPTY="$RELAIS_DIR/.vides.$USER"   # passages vides de suite de MA chaîne
TEAM_DIR=$(dirname "$RELAIS_DIR")

now()  { date +%s; }
hhmm() { date -d "@$1" +%H:%M; }
log()  { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
# STOP arrête toutes les chaînes ; STOP.<compte> seulement celle de ce compte.
stopped() { [ -e "$R_STOP" ] || [ -e "$R_STOP.$USER" ]; }

# Membres de l'équipe : utilisateurs nommés dans les ACL du dossier d'équipe, plus soi-même.
team_members() {
  { getfacl -p "$TEAM_DIR" 2>/dev/null | sed -n 's/^user:\([^:][^:]*\):.*/\1/p'; echo "$USER"; } |
    sort -u | paste -sd, -
}

# En-têtes d'une tâche -> T_GPUS T_MIN T_PRIO T_DIR (T_HASDIR=1 si « # DIR= » présent).
# Code de retour non nul si les valeurs sont invalides.
read_header() {
  local line n=0 g='' m='' p='' d='' hd=''
  while [ $n -lt 40 ] && IFS= read -r line; do
    n=$((n + 1)); line=${line%$'\r'}
    case $line in
      '# GPUS='*)    [ -z "$g" ] && g=${line#'# GPUS='} ;;
      '# MINUTES='*) [ -z "$m" ] && m=${line#'# MINUTES='} ;;
      '# PRIO='*)    [ -z "$p" ] && p=${line#'# PRIO='} ;;
      '# DIR='*)     [ -z "$hd" ] && { d=${line#'# DIR='}; hd=1; } ;;
    esac
  done < "$1"
  g=${g//[[:space:]]/}; m=${m//[[:space:]]/}; p=${p//[[:space:]]/}
  d=${d%"${d##*[![:space:]]}"}
  T_GPUS=${g:-1}; T_MIN=${m:-$RELAIS_TASK_MINUTES}; T_PRIO=${p:-2}; T_DIR=${d:-$RELAIS_DIR}; T_HASDIR=$hd
  [[ $T_GPUS =~ ^[1-9]$ ]] && [ "$T_GPUS" -le "$RELAIS_NGPUS" ] &&
    [[ $T_MIN =~ ^[1-9][0-9]*$ ]] && [[ $T_PRIO =~ ^[1-3]$ ]]
}

# Tâches de todo/ dans l'ordre de passage (PRIO puis nom), une par ligne : « prio<TAB>chemin ».
todo_ordered() {
  local f
  for f in "$R_TODO"/*; do
    [ -f "$f" ] || continue
    read_header "$f" 2>/dev/null
    printf '%s\t%s\n' "${T_PRIO:-2}" "$f"
  done | LC_ALL=C sort -t $'\t' -k1,1n -k2,2
}

# Fin de la réservation (epoch) ; échec si elle n'existe plus.
resv_end() {
  local info; info=$(scontrol show reservation "$RELAIS_RES" 2>/dev/null) || return 1
  [[ $info =~ EndTime=([^ ]+) ]] || return 1
  date -d "${BASH_REMATCH[1]}" +%s
}

# --- Accès à Slurm (remplacés par des fichiers .sim_* en simulation) ---
# Relais de MON compte en attente (ids).
my_pending_relais() {
  if [ "$RELAIS_SIM" = 1 ]; then cat "$RELAIS_DIR/.sim_pending_relais" 2>/dev/null; return 0; fi
  squeue -h -u "$USER" -n "$RELAIS_NAME" -t PD -o %i 2>/dev/null
}
# Relais de MON compte en cours, sauf celui donné en argument.
my_other_running_relais() {
  if [ "$RELAIS_SIM" = 1 ]; then cat "$RELAIS_DIR/.sim_running_relais" 2>/dev/null; return 0; fi
  squeue -h -u "$USER" -n "$RELAIS_NAME" -t R -o %i 2>/dev/null | grep -vx "${1:-x}"
}
# Tous les jobs de MON compte dans Slurm (« id nom état »), sauf celui donné en argument.
my_jobs() {
  if [ "$RELAIS_SIM" = 1 ]; then cat "$RELAIS_DIR/.sim_user_jobs" 2>/dev/null; return 0; fi
  squeue -h -u "$USER" -o '%i %j %T' 2>/dev/null | while read -r i j t; do
    [ "$i" = "${1:-x}" ] || echo "$i $j $t"
  done
}
# Le job (de n'importe quel membre) est-il encore vivant ? Dans le doute : oui (on ne range jamais
# en échec une tâche dont on ne peut pas prouver que le job est mort).
job_alive() {
  local id=$1 out h
  case $id in
    sim.*)                                   # simulation : sim.<hôte>.<pid>
      h=${id#sim.}; h=${h%.*}
      [ "$h" = "$(hostname -s)" ] || return 0          # autre hôte : invérifiable
      ps -p "${id##*.}" >/dev/null 2>&1; return ;;     # ps voit aussi les processus des autres comptes
    *[!0-9]*|'') return 0 ;;                 # id inconnu (ancien format de simulation…) : on n'y touche pas
  esac
  out=$(squeue -h -j "$id" -o %T 2>&1)       # squeue voit les jobs de tous les comptes
  if [ $? -ne 0 ]; then
    case $out in *"Invalid job id"*) return 1 ;; esac
    return 0                                 # squeue en panne : dans le doute, vivant
  fi
  case $out in PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING) return 0 ;; esac
  return 1
}
sbatch_clean() {   # sbatch sans les variables SLURM_* héritées quand on soumet depuis un job
  ( for v in $(compgen -e); do
      if [[ $v == SLURM_* || $v == SBATCH_* || $v == SALLOC_* ]]; then unset "$v"; fi
    done
    exec sbatch "$@" )
}
# Soumet un relais de $1 minutes ; affiche son numéro. Code 2 = impossible (réservation finie).
submit_relais() {
  local min=$1 end left args out
  end=$(resv_end) || { log "réservation $RELAIS_RES introuvable : pas de soumission"; return 2; }
  left=$(( (end - $(now)) / 60 - 1 ))
  [ "$left" -lt 10 ] && { log "moins de 10 min avant la fin de la réservation : pas de soumission"; return 2; }
  [ "$min" -gt "$left" ] && min=$left
  args=(--parsable --job-name="$RELAIS_NAME" --account="$RELAIS_ACCOUNT"
        --partition="$RELAIS_PARTITION" --reservation="$RELAIS_RES" --nodes=1
        --gres=gpu:"$RELAIS_NGPUS" --time="$min" --deadline="$(date -d "@$end" +%Y-%m-%dT%H:%M:%S)"
        --signal=B:TERM@60 --chdir="$RELAIS_DIR" --output="$RELAIS_DIR/logs/relais-%j.out"
        "$RELAIS_DIR/relais-job.sh")
  if [ "$RELAIS_SIM" = 1 ]; then
    log "SIMULATION : (RELAIS_LONG=$RELAIS_LONG RELAIS_MINUTES=$RELAIS_MINUTES) sbatch ${args[*]}"; echo "simSUIVANT"; return 0
  fi
  # Soumis depuis un job : on retire les variables SLURM_* héritées.
  out=$(sbatch_clean "${args[@]}") || { log "échec de sbatch (voir ci-dessus)"; return 1; }
  echo "${out%%;*}"
}
