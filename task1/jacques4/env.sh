# Sourced by every worker of t1/jacques4 (copy of t1/jacques2/env.sh, outputs under t1/out/jacques4).
# Relay tasks: CUDA_VISIBLE_DEVICES is set by the relay. CPU smoke: SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" (-> $OUTJ/smoke/...).
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd $TASK && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
umask 0007
J=$TASK/t1/jacques4
OUTJ=$TASK/t1/out/jacques4
DEV=${DEV:-cuda}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = 1 ] && OUTJ=$OUTJ/smoke
T_START=${T_START:-$(date +%s)}; export T_START
log() { echo "[$(date +%H:%M:%S) $(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-none}] $*"; }
openperm() { chmod -R g+rwX "$@" 2>/dev/null || true; }
