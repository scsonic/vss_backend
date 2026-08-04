#!/usr/bin/env bash
# 啟動 FastAPI 網頁(影片搜尋 + Cosmos 對話)。需先 bash serve_vlm.sh 開 Cosmos 服務。
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
exec uvicorn app:app --host 0.0.0.0 --port 8000
