cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
for k in 1 2 3; do python t1/sgoof/flip_tta_ema.py --ckpt t1/out/train/v1/stegastamp_f$k/last.pt --split train --fold $k --views 0,1,2,3 --out t1/out/sgoof/stegastamp || exit 1; done
echo ok > t1/out/sgoof/INFER_DONE
