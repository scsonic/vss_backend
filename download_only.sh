#!/bin/bash
# 只負責下載（網路 I/O，快），跟 embedding（CPU，慢）脫鉤，讓下載可以跑在前面。
set -uo pipefail
cd /home/toyota-004/Desktop/vss_test

URLS_FILE="$1"
LOG="download_only.log"
rm -f DOWNLOAD_DONE

total=$(grep -c . "$URLS_FILE")
i=0
while IFS= read -r url; do
  [ -z "$url" ] && continue
  i=$((i+1))
  id=$(echo "$url" | grep -oE 'v=[A-Za-z0-9_-]{11}' | sed 's/v=//')
  if [ -f "video/${id}.mp4" ]; then
    echo "[$i/$total] $id already downloaded, skip" | tee -a "$LOG"
    continue
  fi
  echo "=== [$i/$total] $id : download ===" | tee -a "$LOG"
  ./.venv/bin/yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b" \
    --merge-output-format mp4 \
    -o "video/%(id)s.%(ext)s" \
    "$url" >> "$LOG" 2>&1
  if [ $? -ne 0 ] || [ ! -f "video/${id}.mp4" ]; then
    echo "DOWNLOAD FAILED: $url" | tee -a "$LOG"
    echo "$url (download failed)" >> download_failed.txt
  fi
done < "$URLS_FILE"

echo "=== ALL DOWNLOADS DONE ($total processed) ===" | tee -a "$LOG"
touch DOWNLOAD_DONE
