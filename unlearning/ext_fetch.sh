#!/bin/bash
# LOGIN NODE ONLY (internet, no GPU): download + extract external unlabeled images into runs/ext/raw/,
# delete each archive after extraction. Generic client string, no personal data.
#   nohup bash ext_fetch.sh > runs/ext/fetch.log 2>&1 &
umask 002
set -u
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
RAW=runs/ext/raw
mkdir -p $RAW
df -h . | tail -1
UA="Mozilla/5.0 (X11; Linux x86_64)"
get() {  # url file dir
  echo "$(date +%T) download $1"
  curl -sS -L --retry 3 -A "$UA" -o "$RAW/$2" "$1" || { echo "FAILED $1"; return 1; }
  ls -la "$RAW/$2"
  echo "$(date +%T) extract $2"
  mkdir -p "$RAW/$3"
  case $2 in
    *.zip) unzip -q -o "$RAW/$2" -d "$RAW/$3" ;;
    *.tgz) tar -xzf "$RAW/$2" -C "$RAW/$3" ;;
  esac && rm -f "$RAW/$2"
  echo "$(date +%T) $3: $(find $RAW/$3 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l) images"
}
get http://images.cocodataset.org/zips/val2017.zip coco.zip coco
get https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz imagenette.tgz imagenette
df -h . | tail -1
echo "$(date +%T) FETCH DONE"
