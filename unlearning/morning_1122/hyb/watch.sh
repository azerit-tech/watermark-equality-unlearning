#!/bin/bash
# Waits for the relay job (mougeot1, name relais) to RUN, picks a GPU with < 1500 MiB used, launches hybE.sh there via srun --overlap.
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10; cd $W; L=runs/hybE/watch.log
end=$(date -d 08:45 +%s)
while [ $(date +%s) -lt $end ]; do
  J=$(squeue -h -u mougeot1 -n relais -t RUNNING -o %i | head -1)
  if [ -n "$J" ]; then
    for k in $(seq 1 40); do
      S=$(timeout 60 srun --jobid=$J --overlap -N1 -n1 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1)
      GRES=""
      if ! echo "$S" | grep -qE '^[0-9]+, '; then
        S=$(timeout 60 srun --jobid=$J --overlap --gres=gpu:4 -N1 -n1 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1); GRES="--gres=gpu:4"
      fi
      echo "$(date +%T) job $J gres[$GRES]: $(echo $S | tr '\n' ' ')" >> $L
      G=$(echo "$S" | awk -F', ' '$1 ~ /^[0-9]+$/ && $2 < 1500 && $3 < 5 {print $1; exit}')
      if [ -n "$G" ]; then
        echo "$(date +%T) launching on job $J GPU $G" >> $L
        srun --jobid=$J --overlap $GRES -N1 -n1 bash -c "export CUDA_VISIBLE_DEVICES=$G; bash runs/hybE/hybE.sh" >> $L 2>&1
        echo "$(date +%T) srun exit $?" >> $L; exit 0
      fi
      sleep 15
    done
    echo "$(date +%T) no free GPU in relay $J after 10 min, giving up" >> $L; exit 1
  fi
  sleep 20
done
echo "$(date +%T) relay never started before 08:45" >> $L
