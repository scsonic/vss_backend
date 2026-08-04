#!/usr/bin/env bash
# 一次性同步(遠端已裝好 venv/模型/llama.cpp)：只傳 db + keyframe + mp4，再重啟服務。
set -euo pipefail
REMOTE="scsonic@192.168.31.135"
DIR="vss_test"
LOCAL="/Users/scsonic/vss_test"

echo "==> rsync db/"
rsync -az --delete "${LOCAL}/db/" "${REMOTE}:~/${DIR}/db/"

echo "==> rsync keyframes(video/_frames)"
rsync -az "${LOCAL}/video/_frames/" "${REMOTE}:~/${DIR}/video/_frames/"

echo "==> rsync mp4(video/*.mp4)"
rsync -az --include='*.mp4' --exclude='*' "${LOCAL}/video/" "${REMOTE}:~/${DIR}/video/"

echo "==> 重啟遠端服務"
ssh "${REMOTE}" 'bash -s' <<'REMOTE'
set -e
cd ~/vss_test
source .venv/bin/activate
fuser -k 8080/tcp 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
nohup bash serve_vlm.sh > vlm.log 2>&1 &
nohup bash serve_web.sh > web.log 2>&1 &
# tunnel(named, 固定網址)
pgrep -f 'cloudflared.*tunnel.*run' >/dev/null || nohup ~/bin/cloudflared tunnel --config ~/.cloudflared/vss-config.yml run > tunnel.log 2>&1 &
sleep 8
echo "--- 服務狀態 ---"
for p in 8080 8000; do (echo >/dev/tcp/127.0.0.1/$p) 2>/dev/null && echo "$p UP" || echo "$p down"; done
.venv/bin/python -c "import config,chromadb;c=chromadb.PersistentClient(path=str(config.DB_DIR));print('遠端 DB 筆數:',c.get_collection(config.COLLECTION_NAME).count())"
REMOTE
echo "==> 同步完成。"
