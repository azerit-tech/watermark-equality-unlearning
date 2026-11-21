#!/bin/bash
exec ssh -o BatchMode=yes -o ConnectTimeout=25 -o ControlMaster=auto -o ControlPath=/private/tmp/claude-501/cm-saliency-%C -o ControlPersist=20m jureca "cd /p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task; $*"
