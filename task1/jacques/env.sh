# Sourced by every worker of t1/jacques. Relay tasks: CUDA_VISIBLE_DEVICES is set by the relay.
# CPU smoke test: SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" (outputs under $OUTJ/smoke/...).
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd $TASK && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
umask 0007
J=$TASK/t1/jacques
OUTJ=$TASK/t1/out/jacques
DEV=${DEV:-cuda}; SMOKE=${SMOKE:-0}; WORKERS=${WORKERS:-12}
[ "$SMOKE" = 1 ] && { OUTJ=$OUTJ/smoke; WORKERS=${WORKERS_SMOKE:-2}; }
T_START=${T_START:-$(date +%s)}; export T_START   # set once per relay task (shared by chained workers)
# BUDGET_MIN = wall-clock budget of THIS run (header MINUTES minus a margin), set by the caller
left_min() { awk -v t0=$T_START -v b=${BUDGET_MIN:-50} -v now=$(date +%s) 'BEGIN{x=(t0+b*60-now)/60; if(x<0)x=0; printf "%.2f", x}'; }
log() { echo "[$(date +%H:%M:%S) $(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-none}] $*"; }
openperm() { chmod -R g+rwX "$@" 2>/dev/null || true; }
# ckpt_elapsed FILE -> cumulative training seconds stored in a checkpoint (-1 if missing)
ckpt_elapsed() { [ -f "$1" ] || { echo -1; return; }; python -c "import torch,sys; print(float(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('elapsed',-1)))" "$1"; }
