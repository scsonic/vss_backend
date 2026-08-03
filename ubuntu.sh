#!/usr/bin/env bash
# 部署 vss_test 到區網 Ubuntu 機(scsonic@192.168.31.135, 有 RTX 3090)。
# 在「本資料夾」執行： bash ubuntu.sh
#
# 這台已備妥：python3.12/venv、ffmpeg、cmake、git、cloudflared、CUDA(/usr/local/cuda)。
# 無需 sudo：全部裝在家目錄；llama.cpp 用系統 CUDA 編譯(把 nvcc 加進 PATH)。
# 頻寬：只上傳程式+db+keyframe(~460MB)；大模型由遠端自己抓。
set -euo pipefail

REMOTE="${REMOTE:-scsonic@192.168.31.135}"
REMOTE_DIR="${REMOTE_DIR:-vss_test}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> 1/2 rsync 專案到 ${REMOTE}:~/${REMOTE_DIR}"
rsync -avz --progress \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  --exclude 'models' --exclude '*.mp4' --exclude 'video/_frames_hires' \
  "${LOCAL_DIR}/" "${REMOTE}:~/${REMOTE_DIR}/"

echo "==> 2/2 遠端安裝與設定（免 sudo）"
ssh "${REMOTE}" "REMOTE_DIR='${REMOTE_DIR}' bash -s" <<'REMOTE_SETUP'
set -euo pipefail
cd ~/"${REMOTE_DIR}"
export PATH="/usr/local/cuda/bin:$PATH"     # 讓 nvcc 可用（CUDA 已裝但不在 PATH）

echo "--- Python venv + 套件 ---"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "--- 下載 Cosmos-Reason2-8B GGUF（遠端抓）---"
mkdir -p models/cosmos-r2-8b-gguf && cd models/cosmos-r2-8b-gguf
B="https://huggingface.co/mradermacher/Cosmos-Reason2-8B-GGUF/resolve/main"
[ -f Cosmos-Reason2-8B.Q4_K_M.gguf ]      || curl -L -C - --retry 10 --retry-all-errors -o Cosmos-Reason2-8B.Q4_K_M.gguf      "$B/Cosmos-Reason2-8B.Q4_K_M.gguf"
[ -f Cosmos-Reason2-8B.mmproj-Q8_0.gguf ] || curl -L -C - --retry 10 --retry-all-errors -o Cosmos-Reason2-8B.mmproj-Q8_0.gguf "$B/Cosmos-Reason2-8B.mmproj-Q8_0.gguf"
cd ~/"${REMOTE_DIR}"

echo "--- 建置 llama.cpp（CUDA，用系統 /usr/local/cuda）---"
if [ ! -x "$HOME/llama.cpp/build/bin/llama-server" ]; then
  [ -d "$HOME/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
  cd "$HOME/llama.cpp"
  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(nproc)" --target llama-server llama-mtmd-cli
  cd ~/"${REMOTE_DIR}"
fi
# 免 sudo：把 llama-server 連進 venv/bin（venv 啟用時即在 PATH）
ln -sf "$HOME/llama.cpp/build/bin/llama-server"   .venv/bin/llama-server
ln -sf "$HOME/llama.cpp/build/bin/llama-mtmd-cli" .venv/bin/llama-mtmd-cli 2>/dev/null || true

echo
echo "=========================================="
echo "遠端設定完成。啟動（在 ~/${REMOTE_DIR}，先 source .venv/bin/activate）："
echo "  1) Cosmos 服務： bash serve_vlm.sh        # GPU(3090) 加速，port 8080"
echo "  2) 網頁：        bash serve_web.sh        # port 8000"
echo "  3) 對外通道：    bash tunnel.sh           # cloudflared → 公開網址"
echo "=========================================="
REMOTE_SETUP

echo "==> 部署完成。"
