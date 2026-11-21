#!/bin/bash
# GPUS=1
# MINUTES=20
# PRIO=2
# DIR=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
# Retain-bank comparison, arms 2-3: composites alone, composites + shuffle bank (same setup as cmp_none / cmp_shuf).
umask 002
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10 || exit 1
ARM_TIMEOUT=9m bash retain_cmp.sh comp compshuf
