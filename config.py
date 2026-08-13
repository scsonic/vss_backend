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
    # NVIDIA Cosmos-Embed1-448p：跟前兩個不一樣，是「video-clip」embedding（吃 8 張連續影格算一個
    # embedding，不是單張圖片），自訂 modeling 程式碼只認 transformers==4.51.3，跟本專案主 venv
    # 現在用的 transformers 版本不相容，所以獨立跑在 cosmos_embed_server.py 這支 sidecar
    # （見 COSMOS_EMBED_SERVER_URL），app.py/ingest 都是透過 HTTP 呼叫，不在主 process 內 load 模型。
    "cosmos-embed1-448p": {
        "label": "Cosmos-Embed1-448p（NVIDIA，video-clip embedding）",
        "family": "cosmos_embed1",
        "collection": "video_frames_cosmos_embed1_448p",
    },
}
DEFAULT_EMBED_MODEL = "siglip2-giant"
COSMOS_EMBED_SERVER_URL = "http://127.0.0.1:8090"

# --- VLM（/api/explain、/api/chat 用；已改用 OpenRouter，見下面「Agent Search」區塊的
# OPENROUTER_* 設定 —— 跟 agent_search.py 共用同一顆模型，純文字推理、不看畫面）---
# 舊版本機 Cosmos-Reason2-8B + llama-server 已停用（GPU 讓給 embedding 用），vlm.py 不再需要它。

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

# --- 語音轉文字（/api/transcribe，單純轉發到 OpenRouter 的 Whisper，key 藏在後端）---
OPENROUTER_TRANSCRIBE_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
OPENROUTER_TRANSCRIBE_MODEL = "openai/whisper-large-v3-turbo"
