"""Cosmos-Embed1-448p 的獨立 HTTP sidecar。

這顆 NVIDIA 的 video-clip embedding 模型（trust_remote_code 自訂 modeling 程式碼）只認
transformers==4.51.3，跟本專案主 venv 現在用的 transformers（給 SigLIP2 tokenizer 用，版本新很多）
不相容，所以獨立開一個 venv（.venv-cosmosembed）跑這支 server，主 app.py 透過 HTTP 呼叫，
跟以前 vlm.py 呼叫 llama-server 的模式一樣。

啟動：bash serve_cosmos_embed.sh   （對應 config.py 的 COSMOS_EMBED_SERVER_URL）

模型輸入是「8 張影格的短片段」而不是單張圖片（NVIDIA 原生設計就是 video-clip embedding），
所以 /embed_clip(s) 吃的是「本機檔案路徑列表」，由呼叫端（ingest_cosmos_embed.py）先用 ffmpeg
密集抽好影格再把路徑傳過來，這支 server 只負責讀圖 + forward。
"""
import gc

import numpy as np
import torch
import transformers.pytorch_utils as _pu
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# transformers 4.51.3 之後的版本移除了這個工具函式，但這個模型的自訂 Q-Former 程式碼還在用；
# 補回原本 HF transformers 裡的實作（標準 BERT head-pruning 邏輯，跟版本無關）。
if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index
    _pu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

from transformers import AutoModel, AutoProcessor  # noqa: E402  (要在補丁後才 import)

MODEL_ID = "nvidia/Cosmos-Embed1-448p"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"[cosmos_embed_server] loading {MODEL_ID} on {DEVICE} ({DTYPE}) ...")
_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE, dtype=DTYPE).eval()
_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print("[cosmos_embed_server] ready")

app = FastAPI(title="Cosmos-Embed1-448p sidecar")


class TextReq(BaseModel):
    texts: list[str]


class ClipReq(BaseModel):
    clips: list[list[str]]  # 每個 clip 是 8 張影格的本機絕對路徑


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}


def _run_no_grad(fn):
    """跑 fn() 並確保 GPU 記憶體一定被清乾淨，包括失敗的時候。

    重點是要有真正的 except（不能只有 finally）：Python 的例外物件會透過
    __traceback__ 一路拉住失敗當下呼叫鏈上每一層 frame 的區域變數（包括模型內部深處
    算到一半的巨大 attention tensor），只要例外還在往外傳、還沒被 except 接住，
    這些 tensor 就不會真的變成「沒人參照」，gc.collect()/empty_cache() 也清不掉。
    用 except 接住後，Python 會在 except 區塊結束時自動 `del` 例外變數斷開這條參照鏈，
    finally 才清得乾淨——不然一次 OOM 會讓 GPU 記憶體卡在高檔，後面小 batch 也連環失敗。
    """
    try:
        with torch.no_grad():
            return fn()
    except torch.cuda.OutOfMemoryError as e:
        msg = str(e)
        raise HTTPException(status_code=503, detail=f"GPU OOM：{msg[:300]}") from None
    finally:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


@app.post("/embed_text")
def embed_text(req: TextReq):
    def _do():
        inputs = _processor(text=req.texts).to(DEVICE, dtype=DTYPE)
        out = _model.get_text_embeddings(**inputs)
        return out.text_proj.float().cpu().tolist()
    return {"embeddings": _run_no_grad(_do)}


@app.post("/embed_clips")
def embed_clips(req: ClipReq):
    batch = []
    for paths in req.clips:
        frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in paths])  # (T,H,W,C)
        batch.append(np.transpose(frames, (0, 3, 1, 2)))  # (T,C,H,W)
    arr = np.stack(batch)  # (B,T,C,H,W)

    def _do():
        inputs = _processor(videos=arr).to(DEVICE, dtype=DTYPE)
        out = _model.get_video_embeddings(**inputs)
        return out.visual_proj.float().cpu().tolist()
    return {"embeddings": _run_no_grad(_do)}
