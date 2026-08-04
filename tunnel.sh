#!/usr/bin/env bash
# 用 Cloudflare Tunnel 把本機服務對外公開（免費、給固定 https 網址）。
# 用法：  bash tunnel.sh [PORT]      # 預設 8000 = 網頁(影片搜尋站)
#
# 執行後會印出一個 https://xxxx.trycloudflare.com 網址，
# 別人打開那個網址就能搜尋影片、跟 Cosmos 問答。
set -euo pipefail
PORT="${1:-8000}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "安裝 cloudflared ..."
  if [ "$(uname)" = "Darwin" ]; then
    brew install cloudflared
  else
    # Ubuntu / Debian
    curl -L -o /tmp/cloudflared.deb \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i /tmp/cloudflared.deb
  fi
fi

# 優先用 ~/bin/cloudflared（較新版）；--config /dev/null 避免載到系統既有的 named-tunnel 設定
CF="$HOME/bin/cloudflared"; [ -x "$CF" ] || CF="cloudflared"
echo "開通道 → http://127.0.0.1:${PORT}"
echo "（下面會出現一個 https://<隨機>.trycloudflare.com 的公開網址）"
exec "$CF" tunnel --config /dev/null --no-autoupdate --url "http://127.0.0.1:${PORT}"
