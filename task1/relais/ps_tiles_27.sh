#!/bin/bash
# GPUS=1
# MINUTES=27
# PRIO=3
# DIR=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
set -euo pipefail
cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
MIN=22 WORKERS=12 bash t1/tiles/relay_pixelseal.sh
