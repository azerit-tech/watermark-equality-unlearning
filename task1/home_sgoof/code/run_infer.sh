cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
for k in 1 2 3; do python /p/home/jusers/trentelivres1/jureca/sgoof/code/flip_tta_ema.py --ckpt t1/out/train/v1/stegastamp_f$k/last.pt --split train --fold $k --views 0,1,2,3 --out /p/home/jusers/trentelivres1/jureca/sgoof/stegastamp || exit 1; done
echo ok > /p/home/jusers/trentelivres1/jureca/sgoof/INFER_DONE
python /p/home/jusers/trentelivres1/jureca/sgoof/code/flip_tta_ema.py --ckpt t1/out/train/v1/stegastamp_f0/last.pt --split train --fold 0 --views 0,1,2,3 --out /p/home/jusers/trentelivres1/jureca/sgoof/stegastamp_f0ref && echo ok > /p/home/jusers/trentelivres1/jureca/sgoof/F0_DONE
