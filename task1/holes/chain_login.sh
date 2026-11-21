#!/bin/bash
# Fine-tune PixelSeal then TrustMark on the login node (one free card, <= 10 min per run) + fold-0 evaluation of the LAST ckpt.
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
R=t1/holes/run_gpu.sh; O=t1/out/holes; mkdir -p $O; rm -f $O/CHAIN.DONE
COMMON="--p_real 0.25 --mix outpainting=1,transplant=1,hole=4 --hole_src new=1,lib=1 --tiles 256,320 --tile_inf 320 --p_focus 0.7 --minutes 6 --eval_min 100 --workers 8"
bash $R ft_ps t1/holes/train_tiles.py --method pixelseal --res 256 --ppl 0 --bs 24 --lr 1.5e-4 --lr_enc 1e-5 $COMMON \
  --init t1/out/tiles3/pixelseal/last.pt --out $O/pixelseal
bash $R ev_ps t1/holes/eval_ft.py pixelseal t1/out/tiles3/pixelseal/last.pt $O/pixelseal/last.pt $O/pixelseal --scales 256,320
bash $R ft_tm t1/holes/train_tiles.py --method trustmark --res 256 --ppl 1 --bs 16 --lr 3e-4 --lr_enc 1.5e-5 $COMMON \
  --init t1/out/tiles2/trustmark/last.pt --out $O/trustmark
bash $R ev_tm t1/holes/eval_ft.py trustmark t1/out/tiles2/trustmark/last.pt $O/trustmark/last.pt $O/trustmark --scales 320
date > $O/CHAIN.DONE
