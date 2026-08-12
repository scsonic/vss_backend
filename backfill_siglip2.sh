#!/bin/bash
# 把 video/ 底下所有已存在的 mp4，用 siglip2-giant 模型也 ingest 一次，
# 補進獨立的 video_frames_siglip2_giant collection（跟 dfn5b 那組資料完全分開）。
set -uo pipefail
cd /home/toyota-004/Desktop/vss_test

LOG="siglip2_backfill.log"
FAILED="siglip2_backfill_failed.txt"
: > "$FAILED"

next_pending() {
  ./.venv/bin/python - <<'PY'
from pathlib import Path
from store import VectorStore
import config
s = VectorStore(collection_name=config.EMBED_MODELS["siglip2-giant"]["collection"])
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
    echo "=== no more pending videos, backfill done ===" | tee -a "$LOG"
    break
  fi
  echo "=== siglip2-giant ingest: $name (GPU) ===" | tee -a "$LOG"
  ./.venv/bin/python ingest.py "video/${name}" --model siglip2-giant --device cuda --batch-size 8 >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "INGEST FAILED: $name" | tee -a "$LOG"
    echo "$name (ingest failed)" >> "$FAILED"
  fi
done

echo "=== ALL SIGLIP2 BACKFILL DONE ===" | tee -a "$LOG"

# backfill 跑完了，把 GPU 讓出來的 Cosmos VLM(llama-server) 重新啟動。
echo "=== restarting VLM (llama-server) ===" | tee -a "$LOG"
cd /home/toyota-004/Desktop/vss_test
nohup bash serve_vlm.sh > vlm.log 2>&1 &
disown
echo "=== VLM restart requested ===" | tee -a "$LOG"
