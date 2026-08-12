"""Ingestion pipeline：mp4 -> keyframes -> CLIP embedding -> ChromaDB。

對應 NVIDIA VSS 的 Stream Handler(切段) + Visual Encoder(嵌入) + Milvus(儲存)。
只處理前 config.MAX_DURATION_SEC 秒。

用法:
    python ingest.py path/to/video.mp4
    python ingest.py path/to/video.mp4 --reset   # 重建資料庫
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import config
from embedder import ClipEmbedder
from store import VectorStore


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


def extract_frames(video: str) -> list[dict]:
    """用 ffmpeg 每 FRAME_INTERVAL_SEC 秒抽一張影格，回傳 [{path, t_sec}]。

    每支影片的影格放在各自的子資料夾 frames/<影片檔名>/，避免多支影片互相覆蓋。
    """
    frame_dir = config.FRAMES_DIR / Path(video).stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    # 只清掉「這支影片」的舊影格
    for old in frame_dir.glob("frame_*.jpg"):
        old.unlink()

    duration = get_duration(video)
    fps = 1.0 / config.FRAME_INTERVAL_SEC
    cmd = ["ffmpeg", "-y", "-i", video]
    if config.MAX_DURATION_SEC:  # None = 整部影片
        limit = min(duration, config.MAX_DURATION_SEC) if duration else config.MAX_DURATION_SEC
        cmd += ["-t", str(limit)]
        print(f"[ingest] duration={duration:.1f}s, 只處理前 {limit:.1f}s, 每 {config.FRAME_INTERVAL_SEC}s 抽一張")
    else:
        print(f"[ingest] duration={duration:.1f}s, 處理整部影片, 每 {config.FRAME_INTERVAL_SEC}s 抽一張")

    pattern = str(frame_dir / "frame_%05d.jpg")
    subprocess.run(
        cmd + ["-vf", f"fps={fps}", "-q:v", "2", pattern],
        check=True, capture_output=True,
    )

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    out = []
    for i, f in enumerate(frames):
        out.append({"path": str(f), "t_sec": round(i * config.FRAME_INTERVAL_SEC, 2)})
    print(f"[ingest] 共抽出 {len(out)} 張 keyframe")
    return out


def hhmmss(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--reset", action="store_true", help="清空並重建 collection")
    ap.add_argument("--model", choices=list(config.EMBED_MODELS.keys()), default=config.DEFAULT_EMBED_MODEL,
                    help="用哪組 embedding 模型（各自存在獨立 collection，見 config.EMBED_MODELS）")
    args = ap.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"找不到影片: {args.video}")

    frames = extract_frames(args.video)
    if not frames:
        sys.exit("沒有抽到任何影格。")

    mcfg = config.EMBED_MODELS[args.model]
    embedder = ClipEmbedder(device="cpu", clip_model=mcfg["clip_model"],
                             clip_pretrained=mcfg["clip_pretrained"])  # GPU 留給常駐的 VLM(llama-server)，避免 OOM
    store = VectorStore(collection_name=mcfg["collection"])
    if args.reset:
        store.reset()

    video_name = Path(args.video).name
    embs = embedder.embed_images([fr["path"] for fr in frames])  # 批次嵌入(GPU)
    ids, metas = [], []
    for fr in frames:
        ids.append(f"{video_name}::{fr['t_sec']}")
        metas.append({
            "video": video_name,
            "path": fr["path"],
            "t_sec": fr["t_sec"],
            "timecode": hhmmss(fr["t_sec"]),
        })

    store.add(ids, embs, metas)
    print(f"[ingest] 完成，資料庫現有 {store.count()} 筆影格向量。")

    # 存一份 manifest 方便 debug
    (config.ROOT / "last_ingest.json").write_text(
        json.dumps({"video": video_name, "frames": len(frames), "model": args.model}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
