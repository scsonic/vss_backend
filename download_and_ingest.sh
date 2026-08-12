#!/bin/bash
# 下載一批 YouTube 影片到 video/，逐支下載完就馬上 ingest（CLIP embedding）進資料庫。
# 不用 --reset，累加進現有 collection。失敗的影片記錄下來，不中斷整批流程。
set -uo pipefail
cd /home/toyota-004/Desktop/vss_test

URLS_FILE="$1"
LOG="video_ingest.log"
FAILED="ingest_failed.txt"
: > "$FAILED"

total=$(grep -c . "$URLS_FILE")
i=0
while IFS= read -r url; do
  [ -z "$url" ] && continue
  i=$((i+1))
  id=$(echo "$url" | grep -oE 'v=[A-Za-z0-9_-]{11}' | sed 's/v=//')
  echo "=== [$i/$total] $id : download ===" | tee -a "$LOG"
  if [ -f "video/${id}.mp4" ]; then
    echo "already downloaded, skip download" | tee -a "$LOG"
  else
    ./.venv/bin/yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b" \
      --merge-output-format mp4 \
      -o "video/%(id)s.%(ext)s" \
      "$url" >> "$LOG" 2>&1
    if [ $? -ne 0 ] || [ ! -f "video/${id}.mp4" ]; then
      echo "DOWNLOAD FAILED: $url" | tee -a "$LOG"
      echo "$url (download failed)" >> "$FAILED"
      continue
    fi
  fi
  echo "=== [$i/$total] $id : ingest ===" | tee -a "$LOG"
  ./.venv/bin/python ingest.py "video/${id}.mp4" >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "INGEST FAILED: $id" | tee -a "$LOG"
    echo "$url (ingest failed)" >> "$FAILED"
  fi
done < "$URLS_FILE"

echo "=== ALL DONE ($total videos processed) ===" | tee -a "$LOG"
if [ -s "$FAILED" ]; then
  echo "=== FAILURES ===" | tee -a "$LOG"
  cat "$FAILED" | tee -a "$LOG"
fi
