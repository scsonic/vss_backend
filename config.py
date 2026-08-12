"""全域設定。對應 nvidia_vss_flow.md 中的精簡版參數。"""
import os
from pathlib import Path

ROOT = Path(__file__).parent

# --- 影片切段 (chunking) ---
MAX_DURATION_SEC = None     # None = 處理整部影片（不截斷）；也可設數字（秒）只處理前段
FRAME_INTERVAL_SEC = 2.0    # 每幾秒抽一張 keyframe

# --- 路徑 ---
VIDEO_DIR = ROOT / "video"                 # 影片存放處
FRAMES_DIR = VIDEO_DIR / "_frames"         # keyframe 快取（放在 video 資料夾下）
HIRES_DIR = VIDEO_DIR / "_frames_hires"    # 送 VLM 的高畫質影格快取
DB_DIR = ROOT / "db"
COLLECTION_NAME = "video_frames"

# --- CLIP embedding 模型（取代 VSS 的 TensorRT visual encoder）---
# Apple DFN5B-CLIP-ViT-H/14 @ 378px（open_clip 名稱），1024 維、ViT-H 級，
# 為 Mac 上能跑、最接近 VSS NV-CLIP 的高品質 CLIP。
CLIP_MODEL = "ViT-H-14-378-quickgelu"
# 優先用本機權重（curl 續傳下載，避免 HF downloader 卡住）；否則用 open_clip tag 自動下載。
_LOCAL_CLIP = ROOT / "models" / "dfn5b" / "open_clip_pytorch_model.bin"
CLIP_PRETRAINED = str(_LOCAL_CLIP) if _LOCAL_CLIP.exists() else "dfn5b"

# 可選的多組 embedding 模型：各自存在獨立的 ChromaDB collection（同一個 DB_DIR 底下），
# 呼叫 /api/search、/api/agent_chat 的 search_video 工具時可用 model 參數指定要查哪一組，
# 不指定就用 DEFAULT_EMBED_MODEL。兩組資料庫互不相干，dim 也不同（1024 vs 1536），不能混用。
EMBED_MODELS = {
    "dfn5b": {
        "label": "DFN5B-CLIP-ViT-H/14 @ 378px",
        "clip_model": CLIP_MODEL,
        "clip_pretrained": CLIP_PRETRAINED,
        "collection": COLLECTION_NAME,
    },
    "siglip2-giant": {
        "label": "SigLIP2-Giant (ViT-gopt-16 @ 384px)",
        "clip_model": "ViT-gopt-16-SigLIP2-384",
        "clip_pretrained": "webli",
        "collection": "video_frames_siglip2_giant",
    },
}
DEFAULT_EMBED_MODEL = "dfn5b"

# --- VLM ---
# NVIDIA Cosmos-Reason2-8B（reasoning VLM，Qwen3-VL 底）GGUF Q4_K_M，
# 由 llama-server(Metal 加速) 提供服務；vlm.py 透過 HTTP API 呼叫。
# 啟動：bash serve_vlm.sh
VLM_SERVER_URL = "http://127.0.0.1:8080"
# 搜到影格後，會回原始 mp4 用時間碼重新擷取高畫質影格再送 VLM（hires.py）。
USE_HIRES_FRAMES = True
# 送進 VLM 前把影格縮到最長邊此像素；None = 送 mp4 原始解析度（最清楚，但 1080p 每張約 147s）。
# 實測折衷：1280 幾乎等同原始細節（招牌小字可讀）但每張僅約 39s；1024 約 19s；1080p(None) 約 147s。
VLM_IMG_MAX_PX = 1280
VLM_MAX_TOOL_HOPS = 3       # look_around 這類工具最多連續呼叫幾次，避免無限迴圈
# 舊版 MLX 2B（保留參考）：VLM_MODEL = "hzang/Cosmos-Reason2-2B-4bit"

# --- 嵌入批次 ---
EMBED_BATCH_SIZE = 32       # 一次丟幾張到 GPU forward（越大越快、越吃記憶體；OOM 就調小）

# --- 檢索 ---
TOP_K = 4                   # 最終交給 VLM 綜合的影格數（門檻過濾後）
SEARCH_CANDIDATES = 12      # RAG 先「多撈」幾張候選，再交給 Cosmos 過濾
SCORE_THRESHOLD = 0.20      # 相似度門檻（低於此的候選直接丟掉；DFN5B 街景約 0.2~0.35）
MERGE_FRAME_GAP = 10        # 同影片內，相鄰影格(frame 編號差 ≤ 此值)串成同一筆結果，避免結果擠在一起
LOOK_AROUND_BEFORE = 3      # look_around 預設往前看幾張
LOOK_AROUND_AFTER = 3       # look_around 預設往後看幾張

# --- Agent Search（/agent 頁面，用 OpenRouter LLM 自主決定要不要搜尋/解讀）---
try:
    from local_secrets import OPENROUTER_API_KEY  # 本機專用，不進 git
except ImportError:
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "deepseek/deepseek-v4-flash-0731"
AGENT_MAX_TOOL_HOPS = 6      # search_video/explain_clips 最多連續呼叫幾輪，避免無限迴圈
