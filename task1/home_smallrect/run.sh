set -e
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
python t1/tmedge/sweep.py --method trustmark --split train --rects ~/smallrect/r0.json --out ~/smallrect/p1.json
python t1/tmedge/mkrects.py ~/smallrect/p1.json ~/smallrect/r1.json
python t1/tmedge/sweep.py --method trustmark --split train --rects ~/smallrect/r1.json --out ~/smallrect/p2.json
python t1/tmedge/mkrects.py ~/smallrect/p2.json ~/smallrect/r2.json
echo ok > ~/smallrect/DONE
