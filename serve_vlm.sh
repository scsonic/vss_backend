#!/bin/bash
# 啟動 Cosmos-Reason2-8B (GGUF Q4_K_M) 的 llama-server，供 vlm.py / app.py 呼叫。
# --jinja：啟用 function/tool calling。 -c 32768：context 32k。
# 有 NVIDIA GPU → 視覺編碼器也放 GPU（快）；Mac/Metal → --no-mmproj-offload 放 CPU 避免 OOM。
cd "$(dirname "$0")"

EXTRA=""
if ! command -v nvidia-smi >/dev/null 2>&1; then
  EXTRA="--no-mmproj-offload"   # 無 NVIDIA GPU（例如 Mac Metal）：視覺放 CPU
fi

# 找 llama-server：環境變數 > PATH > 本機 build 目錄
LLAMA_BIN="${LLAMA_SERVER:-$(command -v llama-server 2>/dev/null)}"
[ -z "$LLAMA_BIN" ] && LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"

exec "$LLAMA_BIN" \
  -m models/cosmos-r2-8b-gguf/Cosmos-Reason2-8B.Q4_K_M.gguf \
  --mmproj models/cosmos-r2-8b-gguf/Cosmos-Reason2-8B.mmproj-Q8_0.gguf \
  -ngl 99 $EXTRA -c 32768 \
  --host 127.0.0.1 --port 8080 --jinja "$@"
