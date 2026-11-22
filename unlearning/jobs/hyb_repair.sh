#!/bin/bash
# GPUS=1
# MINUTES=35
# PRIO=1
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# "Repair E18": init = E18; forget loss = KL(E18 || model) at T=1 on the clean + 3 augmented views of the forget images
# (keeps the forgetting of E18: level and tail); retain loss = KL(orig || model) on bank_t2i split 0 (in-domain synthetic),
# weight 2; all weights, lr 2e-5, 1500 steps. Variant 2: the same from E20 (higher level, more drift).
# Then hyb_law.py: level, KL_ext (bank_ext held-out), KL_t2i (split 1), law 0.047*level/(1+KL_ext); law >= 0.20 -> note
# in runs/submit_queue.
. /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10/jobs/pgb_common.sh
O=runs/pgb/repair; mkdir -p $O
for b in E18 E20; do
  $PY hyb_train.py --out $O/$b --orig ../model.pt --forget ../forget_images.npz --bank runs/pg/bank_t2i.npz \
    --init runs/oracle/$b.pt --targets runs/oracle/$b.pt --temps 1 --w-ret 2 --trainable all --lr 2e-5 \
    --steps 1500 --ckpt-every 250 > $O/$b.log 2>&1
done
$PY hyb_law.py --orig ../model.pt --forget ../forget_images.npz --t2i runs/pg/bank_t2i.npz --ext runs/ext/bank_ext.npz \
  --cands runs/oracle/E18.pt runs/oracle/E20.pt runs/oracle/T05.pt $O/E18/ckpt_s*.pt $O/E20/ckpt_s*.pt \
  --out $O/law.jsonl --queue-min 0.20 --tag repair \
  --recipe "repair: init E18/E20 (see dir), KL(init||model) T=1 on forget views + 2*KL(orig||model) on bank_t2i split 0, all weights lr 2e-5" > $O/law.log 2>&1
