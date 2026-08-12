#!/bin/bash
# 持續掃 video/ 資料夾，把還沒進資料庫的 mp4 逐一做 embedding。
# 跟 download_only.sh 平行跑：邊下載邊 embed，下載完的檔案不用等全部下載完才開始。
# 每次外層迴圈只查詢+處理一支影片（重新查一次 pending list），
# 避免長時間存活的 pipe/while-read 在跑很久之後拿到不明原因被截斷的檔名。
# 直到 DOWNLOAD_DONE 出現且沒有待處理檔案了才結束。
set -uo pipefail
cd /home/toyota-004/Desktop/vss_test

LOG="video_ingest.log"
FAILED="ingest_failed.txt"

next_pending() {
  ./.venv/bin/python - <<'PY'
from pathlib import Path
from store import VectorStore
s = VectorStore()
r = s.collection.get(include=["metadatas"])
done = set(m["video"] for m in r["metadatas"])
for p in sorted(Path("video").glob("*.mp4")):
    if p.name not in done:
        print(p.name)
        break
PY
}

while true; do
  name=$(next_pending)
  if [ -z "$name" ]; then
    if [ -f DOWNLOAD_DONE ]; then
      echo "=== no more pending videos, downloads finished, exiting ===" | tee -a "$LOG"
      break
    fi
    sleep 10
    continue
  fi
  echo "=== ingest: $name ===" | tee -a "$LOG"
  ./.venv/bin/python ingest.py "video/${name}" >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "INGEST FAILED: $name" | tee -a "$LOG"
    echo "$name (ingest failed)" >> "$FAILED"
  fi
done

echo "=== ALL INGEST DONE ===" | tee -a "$LOG"
