#!/bin/bash
# 把 video/ 底下所有已存在的 mp4，用 Cosmos-Embed1-448p 也 ingest 一次，
# 補進獨立的 video_frames_cosmos_embed1_448p collection（跟 dfn5b / siglip2-giant 完全分開）。
#
# 前提：cosmos_embed_server.py 這支 sidecar 要先跑起來（獨立 venv，見該檔案說明）；
# 這支腳本會自己確認/啟動它。GPU 現在沒有本機 VLM 佔用（已改用 OpenRouter），
# 全部讓給這裡的 embedding 用。
set -uo pipefail
cd /home/toyota-004/Desktop/vss_test

LOG="cosmos_embed_backfill.log"
FAILED="cosmos_embed_backfill_failed.txt"
SKIPFILE="cosmos_embed_backfill_skip.txt"
: > "$FAILED"
: > "$SKIPFILE"
declare -A fail_count

ensure_sidecar() {
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8090/health 2>/dev/null | grep -q 200; then
    return 0
  fi
  echo "=== starting cosmos_embed sidecar ===" | tee -a "$LOG"
  nohup bash serve_cosmos_embed.sh >> cosmos_embed.log 2>&1 &
  disown
  for i in $(seq 1 60); do
    curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8090/health 2>/dev/null | grep -q 200 && return 0
    sleep 3
  done
  echo "sidecar 啟動逾時" | tee -a "$LOG"
  return 1
}

next_pending() {
  ./.venv/bin/python - <<PY
from pathlib import Path
from store import VectorStore
import config
s = VectorStore(collection_name=config.EMBED_MODELS["cosmos-embed1-448p"]["collection"])
r = s.collection.get(include=["metadatas"])
done = set(m["video"] for m in r["metadatas"])
try:
    skip = set(open("$SKIPFILE").read().split())
except FileNotFoundError:
    skip = set()
for p in sorted(Path("video").glob("*.mp4")):
    if p.name not in done and p.name not in skip:
        print(p.name)
        break
PY
}

ensure_sidecar || exit 1

while true; do
  name=$(next_pending)
  if [ -z "$name" ]; then
    echo "=== no more pending videos, backfill done ===" | tee -a "$LOG"
    break
  fi
  ensure_sidecar || { echo "$name (sidecar 起不來)" >> "$FAILED"; break; }
  echo "=== cosmos-embed1 ingest: $name ===" | tee -a "$LOG"
  ./.venv/bin/python ingest_cosmos_embed.py "video/${name}" --batch-size 4 >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    n=${fail_count[$name]:-0}
    n=$((n + 1))
    fail_count[$name]=$n
    echo "INGEST FAILED（第 $n 次）: $name" | tee -a "$LOG"
    echo "$name (ingest failed, attempt $n)" >> "$FAILED"
    if [ "$n" -ge 2 ]; then
      echo "$name" >> "$SKIPFILE"
      echo "$name 已重試 $n 次仍失敗，跳過不再重試" | tee -a "$LOG"
    fi
  fi
done

echo "=== ALL COSMOS-EMBED1 BACKFILL DONE ===" | tee -a "$LOG"
