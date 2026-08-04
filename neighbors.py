"""給定某張影格，取得同一支影片在時間上「前後相鄰」的影格。

供 VLM 的 function call 使用：命中某個時間點後，可再把前後幾張影格拿來看，
判斷一個動作(例如「亂丟垃圾」)在時間上的前因後果，再下結論。

影格檔名為 frame_%05d.jpg（ffmpeg 1-indexed），時間 t = (num-1) * FRAME_INTERVAL_SEC。
"""
import re
from pathlib import Path

import config


def hhmmss(t: float) -> str:
    t = int(round(t))
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def neighbors(frame_path: str, before: int = 2, after: int = 2) -> list[dict]:
    """回傳 [{path, t_sec, timecode, is_center}]，含中心影格，依時間排序。"""
    p = Path(frame_path)
    d = p.parent
    m = re.search(r"frame_(\d+)\.jpg", p.name)
    if not m:
        return []
    idx = int(m.group(1))
    out = []
    for n in range(idx - before, idx + after + 1):
        fp = d / f"frame_{n:05d}.jpg"
        if fp.exists():
            t = (n - 1) * config.FRAME_INTERVAL_SEC
            out.append({
                "path": str(fp),
                "t_sec": round(t, 2),
                "timecode": hhmmss(t),
                "is_center": n == idx,
            })
    return out


def neighbors_by_time(video_stem: str, t_sec: float, before: int = 2, after: int = 2) -> list[dict]:
    """用影片名 + 秒數定位中心影格，再取前後鄰居。"""
    idx = round(t_sec / config.FRAME_INTERVAL_SEC) + 1  # 還原 1-indexed 檔號
    center = config.FRAMES_DIR / video_stem / f"frame_{idx:05d}.jpg"
    return neighbors(str(center), before, after)
