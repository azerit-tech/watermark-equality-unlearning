# Sourced by every worker of t1/jacques2 (copy of t1/jacques/env.sh, outputs under t1/out/jacques2).
# Relay tasks: CUDA_VISIBLE_DEVICES is set by the relay. CPU smoke: SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" (-> $OUTJ/smoke/...).
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
cd $TASK && source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} HF_HOME=/p/scratch/training2666/trentelivres1/Azerit/.cache
umask 0007
J=$TASK/t1/jacques2
OUTJ=$TASK/t1/out/jacques2
DEV=${DEV:-cuda}; SMOKE=${SMOKE:-0}; WORKERS=${WORKERS:-12}
[ "$SMOKE" = 1 ] && { OUTJ=$OUTJ/smoke; WORKERS=${WORKERS_SMOKE:-2}; }
T_START=${T_START:-$(date +%s)}; export T_START
left_min() { awk -v t0=$T_START -v b=${BUDGET_MIN:-50} -v now=$(date +%s) 'BEGIN{x=(t0+b*60-now)/60; if(x<0)x=0; printf "%.2f", x}'; }
log() { echo "[$(date +%H:%M:%S) $(hostname -s) gpu=${CUDA_VISIBLE_DEVICES:-none}] $*"; }
openperm() { chmod -R g+rwX "$@" 2>/dev/null || true; }
ckpt_elapsed() { [ -f "$1" ] || { echo -1; return; }; python -c "import torch,sys; print(float(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('elapsed',-1)))" "$1"; }
