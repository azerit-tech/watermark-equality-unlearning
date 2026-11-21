#!/bin/bash
# Job du relais (voir LISEZMOI.txt). Soumis par relais-start, puis par le relais précédent.
# 1. exécute les tâches de todo/ (PRIO puis ordre des noms), une par carte, en parallèle ;
# 2. rend le nœud dès qu'il n'a plus rien à faire (au plus RELAIS_IDLE minutes à vide) ;
# 3. en sortant (ou sur TERM), soumet son successeur si le compte n'a aucun autre job Slurm,
#    sauf STOP, réservation finie, ou deux passages de suite sans tâche (la chaîne s'arrête).
# Simulation sur le nœud de connexion : RELAIS_SIM=1 RELAIS_UNIT=1 RELAIS_POLL=1 RELAIS_DIR=... ./relais-job.sh
set -u
if [ -z "${RELAIS_DIR:-}" ]; then
  RELAIS_DIR=$(cd "$(dirname "$0")" && pwd)
fi
if [ -f "$RELAIS_DIR/relais-lib.sh" ]; then . "$RELAIS_DIR/relais-lib.sh"
else . "$(dirname "$(readlink -f "$0")")/relais-lib.sh"; fi || exit 1   # essai sur un dossier de test
mkdir -p "$R_TODO" "$R_RUN" "$R_DONE" "$R_FAIL" "$R_LOGS"

START=$(now)
if [ "$RELAIS_SIM" = 1 ]; then
  JOB="sim.$(hostname -s).$$"; END=$((START + RELAIS_MINUTES * RELAIS_UNIT))
else
  JOB=${SLURM_JOB_ID:?pas de SLURM_JOB_ID : lancer avec relais-start ou RELAIS_SIM=1}
  e=$(squeue -h -j "$JOB" -o %e 2>/dev/null)
  END=$(date -d "$e" +%s 2>/dev/null) || END=$((START + RELAIS_MINUTES * 60))
fi
IDLE=$RELAIS_IDLE; [ "$RELAIS_LONG" = 1 ] && IDLE=$RELAIS_LONG_IDLE   # mode long : 30 min à vide
MARGIN=$((2 * RELAIS_UNIT))          # marge avant la fin du job (le signal arrive 60 s avant)
log "relais $JOB ($USER) sur $(hostname -s), fin du job à $(hhmm "$END")$([ "$RELAIS_LONG" = 1 ] && echo ", mode long (rend le nœud après $IDLE min sans tâche)")"

declare -A P_NAME=() P_GPUS=() P_END=()       # par pid de tâche
GPU_PID=(); for ((i = 0; i < RELAIS_NGPUS; i++)); do GPU_PID[i]=''; done

free_gpus() { local i n=0; for ((i = 0; i < RELAIS_NGPUS; i++)); do [ -z "${GPU_PID[i]}" ] && n=$((n + 1)); done; echo $n; }

# Tâches restées dans running/ alors que leur job est mort (tuées à la limite de temps).
clean_stale() {
  local f b id name
  for f in "$R_RUN"/*; do
    [ -f "$f" ] || continue
    b=${f##*/}; [[ $b == *__* ]] || continue
    id=${b%%__*}; name=${b#*__}
    [ "$id" = "$JOB" ] && continue
    job_alive "$id" && continue
    mv "$f" "$R_FAIL/$name" 2>/dev/null || continue
    mv "$R_LOGS/$name.log" "$R_FAIL/$name.log" 2>/dev/null
    echo "== relais $JOB : tâche trouvée dans running/ sans job vivant ($id) : tuée (temps dépassé ?), rangée en échec" >> "$R_FAIL/$name.log"
    log "tâche orpheline $name (job $id mort) -> failed/"
  done
}

# Successeur, soumis UNE fois, à la sortie seulement (règle : un seul job par compte dans Slurm).
# Pas de successeur si STOP, deux passages vides de suite, ou si le compte a un autre job.
NLAUNCHED=0; SUCC_DONE=''
successor() {
  [ -n "$SUCC_DONE" ] && return 0; SUCC_DONE=1
  local p need f nv k
  RELAIS_LONG=0; RELAIS_MINUTES=$RELAIS_BASE_MINUTES; export RELAIS_LONG RELAIS_MINUTES   # successeur court
  need=$RELAIS_MINUTES
  if stopped; then log "STOP présent : pas de successeur, chaîne arrêtée"; return; fi
  # Passage vide = aucune tâche lancée et rien qui attende dans todo/.
  if [ "$NLAUNCHED" -gt 0 ] || [ -n "$(todo_ordered)" ]; then nv=0; else nv=$(( $(cat "$R_EMPTY" 2>/dev/null || echo 0) + 1 )); fi
  echo "$nv" > "$R_EMPTY"
  if [ "$nv" -ge 2 ]; then log "deux passages de suite sans tâche : chaîne arrêtée (relais-start la relance)"; return; fi
  p=$(my_jobs "$JOB")
  if [ -n "$p" ]; then
    log "pas de successeur : $USER a déjà un job dans Slurm ($(echo $p)) ; un seul job par compte. Chaîne arrêtée (relais-start quand il sera fini)"
    return
  fi
  # Durée : de quoi passer la plus longue tâche en attente.
  for f in "$R_TODO"/*; do
    [ -f "$f" ] && read_header "$f" || continue
    [ $((T_MIN + 3)) -gt "$need" ] && [ $((T_MIN + 3)) -le "$RELAIS_MAX_MINUTES" ] && need=$((T_MIN + 3))
  done
  [ "$need" -gt "$RELAIS_MAX_MINUTES" ] && need=$RELAIS_MAX_MINUTES
  for k in 1 2; do
    if p=$(submit_relais "$need"); then log "successeur soumis à la sortie : job $p ($need min)"; return; fi
    [ $? -eq 2 ] && return; sleep 5
  done
  log "échec de soumission : chaîne arrêtée (relais-start pour la relancer)"
}

reject() {   # $1 = fichier de todo/, $2 = raison
  local name=${1##*/}
  mv "$1" "$R_FAIL/$name" 2>/dev/null || return
  echo "== relais $JOB : refusée : $2" > "$R_FAIL/$name.log"
  log "tâche $name refusée ($2) -> failed/"
}

launch() {   # $1 = fichier pris dans running/, $2 = nom
  local dest=$1 name=$2 lg="$R_LOGS/$2.log" gl='' i n=0 pid
  for ((i = 0; i < RELAIS_NGPUS && n < T_GPUS; i++)); do
    [ -z "${GPU_PID[i]}" ] && { gl+="${gl:+,}$i"; n=$((n + 1)); }
  done
  echo "== relais : tâche $name, job $JOB ($USER), nœud $(hostname -s), GPU $gl, dossier $T_DIR, début $(date '+%F %T'), $T_MIN min prévues" > "$lg"
  ( cd "$T_DIR" || { echo "== dossier introuvable ou interdit : $T_DIR"; exit 97; }
    export CUDA_VISIBLE_DEVICES=$gl RELAIS_TASK=$name RELAIS_JOB=$JOB
    if [ -x "$dest" ]; then exec "$dest"; else exec bash "$dest"; fi
  ) >> "$lg" 2>&1 < /dev/null &
  pid=$!
  NLAUNCHED=$((NLAUNCHED + 1))
  P_NAME[$pid]=$name; P_GPUS[$pid]=$gl; P_END[$pid]=$(( $(now) + T_MIN * RELAIS_UNIT ))
  for i in ${gl//,/ }; do GPU_PID[i]=$pid; done
  log "lance $name sur GPU $gl ($T_MIN min prévues, PRIO $T_PRIO)"
}

finish() {   # $1 = pid, $2 = code de sortie, $3 = note facultative
  local pid=$1 rc=$2 name=${P_NAME[$1]} to i
  [ "$rc" -eq 0 ] && to=$R_DONE || to=$R_FAIL
  [ -n "${3:-}" ] && echo "== $3" >> "$R_LOGS/$name.log"
  echo "== fin : code $rc, $(date '+%F %T')" >> "$R_LOGS/$name.log"
  mv "$R_RUN/${JOB}__$name" "$to/$name" 2>/dev/null
  mv "$R_LOGS/$name.log" "$to/$name.log" 2>/dev/null
  for i in ${P_GPUS[$pid]//,/ }; do GPU_PID[i]=''; done
  unset "P_NAME[$pid]" "P_GPUS[$pid]" "P_END[$pid]"
  log "fin de $name : code $rc -> ${to##*/}/"
}

reap() {
  local pid rc
  for pid in "${!P_NAME[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"; rc=$?
    finish "$pid" "$rc"
  done
}

# Heure prévue où $1 cartes seront libres (d'après les durées déclarées des tâches en cours).
cards_free_at() {
  local want=$1 f t pid g
  f=$(free_gpus); t=$(now)
  [ "$f" -ge "$want" ] && { echo "$t"; return; }
  while read -r t pid; do
    g=${P_GPUS[$pid]//[^,]/}; f=$((f + ${#g} + 1))
    [ "$f" -ge "$want" ] && { echo "$t"; return; }
  done < <(for pid in "${!P_END[@]}"; do echo "${P_END[$pid]} $pid"; done | sort -n)
  echo "$END"
}

# Prend les tâches dans l'ordre. Si la première tâche bloquée faute de cartes peut tourner
# dans ce job, les suivantes ne passent devant que si elles finissent avant son départ prévu ;
# sinon, elles doivent seulement finir avant la fin du job (elle passera en tête au relais suivant).
take_tasks() {
  local prio f name dest t_now limit=''
  while IFS=$'\t' read -r prio f; do
    [ -f "$f" ] || continue
    name=${f##*/}
    if ! read_header "$f"; then reject "$f" "en-tête invalide (GPUS=$T_GPUS MINUTES=$T_MIN PRIO=$T_PRIO)"; continue; fi
    t_now=$(now)
    [ $((t_now + T_MIN * RELAIS_UNIT + MARGIN)) -gt "$END" ] && continue     # trop longue pour ce job
    if [ "$T_GPUS" -gt "$(free_gpus)" ]; then
      if [ -z "$limit" ]; then
        limit=$(cards_free_at "$T_GPUS")
        [ $((limit + T_MIN * RELAIS_UNIT + MARGIN)) -gt "$END" ] && limit=$((END - MARGIN))
      fi
      continue
    fi
    [ -n "$limit" ] && [ $((t_now + T_MIN * RELAIS_UNIT)) -gt "$limit" ] && continue
    dest="$R_RUN/${JOB}__$name"
    mv "$f" "$dest" 2>/dev/null || continue          # prise atomique : un autre relais l'a peut-être prise
    launch "$dest" "$name"
  done < <(todo_ordered)
}

on_term() {
  local pid
  log "signal reçu (fin du temps ou scancel) : arrêt des tâches en cours"
  for pid in "${!P_NAME[@]}"; do pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; done
  sleep 2
  for pid in "${!P_NAME[@]}"; do finish "$pid" 143 "tuée : fin du temps du job $JOB (durée prévue dépassée ?)"; done
  successor          # en dernier : le job se termine juste après
  exit 143
}
trap on_term TERM INT

clean_stale
if stopped; then log "STOP présent : le relais rend le nœud sans rien lancer, pas de successeur"; exit 0; fi

IDLE_SINCE=$(now)
while :; do
  reap
  stopped || take_tasks
  if [ ${#P_NAME[@]} -eq 0 ]; then
    if stopped; then log "STOP présent : fin du relais"; break; fi
    n=$(todo_ordered | grep -c .)
    if [ "$n" -gt 0 ]; then take_tasks; [ ${#P_NAME[@]} -gt 0 ] && continue; fi   # tâche arrivée à l'instant
    if [ "$n" -gt 0 ]; then log "$n tâche(s) ne tiennent pas dans le temps restant : laissées au relais suivant"; break; fi
    if [ $(( $(now) - IDLE_SINCE )) -ge $((IDLE * RELAIS_UNIT)) ]; then
      log "aucune tâche depuis $IDLE min : le relais rend le nœud"; break
    fi
  else
    IDLE_SINCE=$(now)
  fi
  sleep "$RELAIS_POLL" & wait $!
done
successor              # en dernier : le job se termine juste après
log "relais $JOB terminé"
exit 0
