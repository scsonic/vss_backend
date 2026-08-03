"""從『原始影片』在指定時間碼重新擷取高畫質影格。

embedding 用的是 2 秒取樣、JPEG 壓縮過的 keyframe（給 CLIP 378² 綽綽有餘）；
但要讓 Cosmos-Reason2 看清細節（看板小字、動作），改回原始 mp4 用時間碼
重新 decode 一張高畫質影格再送進 VLM。擷取結果會快取，避免重複 decode。
"""
import subprocess
from pathlib import Path

import config

HIRES_DIR = config.HIRES_DIR


def _find_video(video_name: str) -> Path | None:
    """依檔名找原始影片：先找 video/ 資料夾，再找專案根目錄。"""
    name = Path(video_name).name
    for cand in (config.VIDEO_DIR / name, config.ROOT / name):
        if cand.exists():
            return cand
    return None


def extract(video_name: str, t_sec: float) -> str | None:
    """從原始影片在 t_sec 擷取高畫質影格，回傳路徑；失敗回 None。"""
    video_path = _find_video(video_name)
    if video_path is None:
        return None
    stem = Path(video_name).stem
    d = HIRES_DIR / stem
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"t{t_sec:08.2f}.jpg"
    if out.exists():
        return str(out)
    # -ss 放在 -i 前：快速定位；-q:v 2：高畫質 JPEG（保留原始解析度）
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t_sec:.3f}", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", str(out)],
        capture_output=True,
    )
    return str(out) if out.exists() and r.returncode == 0 else None


def resolve(video_name: str, t_sec: float, fallback: str) -> str:
    """優先回傳高畫質擷取；失敗則退回原本的 keyframe。"""
    hi = extract(video_name, t_sec)
    return hi or fallback
