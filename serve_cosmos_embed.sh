#!/bin/bash
# 啟動 Cosmos-Embed1-448p 的 HTTP sidecar（獨立 venv，見 cosmos_embed_server.py 說明）。
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv-cosmosembed/bin/python -m uvicorn cosmos_embed_server:app --host 127.0.0.1 --port 8090
