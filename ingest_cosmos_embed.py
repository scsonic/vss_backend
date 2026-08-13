"""幫已下載好的影片，用 Cosmos-Embed1-448p（NVIDIA 的 video-clip embedding，跟 CLIP 系列不同：
吃 8 張連續影格的短片段算一個 embedding，不是單張圖片）另外建一份獨立的向量資料庫。

跟 ingest.py 用同一套「t_sec 網格」（每 config.FRAME_INTERVAL_SEC 秒一個 keyframe 時間點），
thumbnail 直接沿用 FRAMES_DIR 底下（dfn5b/siglip2-giant ingest 時）已經抽好的同一批 keyframe
jpg，不重複存；只有真正拿去算 embedding 用的「密集短片段」是這支腳本自己另外抽、算完就丟。

實際 embedding 呼叫 cosmos_embed_server.py 這支 sidecar（獨立 venv，因為模型的自訂程式碼
只認 transformers==4.51.3，見該檔案開頭說明），跑之前要先：bash serve_cosmos_embed.sh

用法：
    python ingest_cosmos_embed.py video/xxx.mp4
    python ingest_cosmos_embed.py video/xxx.mp4 --reset   # 清空重建整個 collection
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import config
from store import VectorStore

MODEL_KEY = "cosmos-embed1-448p"
CLIP_FRAMES = 8       # Cosmos-Embed1-448p 官方建議的 clip 長度
DENSE_FPS = 5.0        # 密集抽格 fps：8 張間隔 0.2s、span 1.4s，遠小於 FRAME_INTERVAL_SEC(2.0s)，
                       # 相鄰 keyframe 的 clip 不會互相重疊


def get_duration(video: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def hhmmss(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_dense_frames(video: str, out_dir: Path, fps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "d_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vf", f"fps={fps}", "-q:v", "3", pattern],
        check=True, capture_output=True,
    )
    return sorted(out_dir.glob("d_*.jpg"))


def clip_window(dense: list[Path], center_idx: int, n: int) -> list[Path]:
    """回傳 n 張連續密集影格，盡量置中在 center_idx；超出邊界就往內夾，太短則重複最後一張湊滿。"""
    if not dense:
        return []
    half = n // 2
    start = max(0, center_idx - half)
    end = start + n
    if end > len(dense):
        end = len(dense)
        start = max(0, end - n)
    window = dense[start:end]
    while window and len(window) < n:
        window.append(window[-1])
    return window


def embed_clips(server_url: str, clip_paths: list[list[str]]) -> list[list[float]]:
    req = urllib.request.Request(
        server_url.rstrip("/") + "/embed_clips",
        data=json.dumps({"clips": clip_paths}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)["embeddings"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Cosmos-Embed1 sidecar 錯誤 {e.code}：{body[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--reset", action="store_true", help="清空並重建 collection")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="一次送幾個 clip 去 sidecar 做 embedding（實測 4 在跟其他 process 共用 GPU 時最穩）")
    ap.add_argument("--server-url", default=config.COSMOS_EMBED_SERVER_URL)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"找不到影片: {args.video}")
    video_name = video_path.name
    video_stem = video_path.stem

    duration = get_duration(str(video_path))
    if not duration:
        sys.exit(f"讀不到影片長度: {args.video}")
    n_keyframes = int(duration // config.FRAME_INTERVAL_SEC) + 1
    print(f"[ingest_cosmos_embed] {video_name} duration={duration:.1f}s, {n_keyframes} 個 keyframe 時間點")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"cosmos_embed_dense_{video_stem}_"))
    try:
        print(f"[ingest_cosmos_embed] ffmpeg 密集抽格 fps={DENSE_FPS} ...")
        dense = extract_dense_frames(str(video_path), tmp_dir, DENSE_FPS)
        print(f"[ingest_cosmos_embed] 密集影格共 {len(dense)} 張")
        if not dense:
            sys.exit("ffmpeg 密集抽格失敗，沒有輸出任何影格。")

        mcfg = config.EMBED_MODELS[MODEL_KEY]
        store = VectorStore(collection_name=mcfg["collection"])
        if args.reset:
            store.reset()

        ids, embs, metas = [], [], []
        pending_clips: list[list[str]] = []
        pending_meta: list[dict] = []

        def flush():
            if not pending_clips:
                return
            vecs = embed_clips(args.server_url, pending_clips)
            for meta, vec in zip(pending_meta, vecs):
                ids.append(f"{video_name}::{meta['t_sec']}")
                embs.append(vec)
                metas.append(meta)
            pending_clips.clear()
            pending_meta.clear()

        for k in range(n_keyframes):
            t_sec = round(k * config.FRAME_INTERVAL_SEC, 2)
            center_idx = round(t_sec * DENSE_FPS)
            window = clip_window(dense, center_idx, CLIP_FRAMES)
            if not window:
                continue
            # thumbnail 沿用 dfn5b/siglip2-giant ingest 時已經抽好的同一張 keyframe（同個 t_sec 網格），
            # 缺的話（例如這支影片還沒被其他模型 ingest 過）就用密集影格裡最靠近中心的那張補上。
            existing_path = config.FRAMES_DIR / video_stem / f"frame_{k+1:05d}.jpg"
            if not existing_path.exists():
                existing_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(window[len(window) // 2], existing_path)
            pending_clips.append([str(p) for p in window])
            pending_meta.append({
                "video": video_name, "path": str(existing_path),
                "t_sec": t_sec, "timecode": hhmmss(t_sec),
            })
            if len(pending_clips) >= args.batch_size:
                flush()
                print(f"[ingest_cosmos_embed] {len(ids)}/{n_keyframes}", end="\r")
        flush()
        print()

        if ids:
            store.add(ids, embs, metas)
        print(f"[ingest_cosmos_embed] 完成，資料庫現有 {store.count()} 筆影格向量。")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
