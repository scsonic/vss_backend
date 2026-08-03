# Video Search + VLM 範例（NVIDIA VSS 精簡版）

在 **MacBook Pro (Apple Silicon)** 上重現 NVIDIA
[Video Search and Summarization (VSS)](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
的核心精神：**把影片切段嵌入 → 存進向量資料庫 → 用文字語意搜尋 → 檢索到的片段交給 VLM 解釋**。

原始 VSS 需要 NVIDIA GPU / TensorRT / Milvus 叢集；本專案把每個角色換成能在 Apple Silicon
上跑的小模型。VSS 完整流程與對照表見 [nvidia_vss_flow.md](nvidia_vss_flow.md)。

## 架構對照

| 角色 | NVIDIA VSS | 本專案 |
|------|-----------|--------|
| 切段 / 解碼 | Stream Handler | `ffmpeg` 每 2 秒抽 1 張 keyframe |
| Embedding | NV-CLIP ViT-H/14 (1024維, 336px) | **DFN5B-CLIP-ViT-H/14 @ 378px**（open_clip，1024 維，MPS）|
| 向量資料庫 | Milvus | ChromaDB（嵌入式） |
| VLM | Cosmos Reason | **Cosmos-Reason2-8B**（GGUF Q4_K_M，llama-server + Metal）|
| 檢索合成 (CA-RAG) | graph-RAG + 檢索策略 | 門檻多撈候選 → Cosmos 過濾 → `look_around` 看前後張 → 綜合 |

## 資料流

```
[ingest] mp4 ─ffmpeg─► keyframes ─DFN5B(MPS,批次)─► 1024維 embedding ─► ChromaDB
                                                                          ▲
[search] 文字 query ─DFN5B text─► embedding ─► ChromaDB 多撈候選 + 門檻過濾 ┘
                                                     │
                                                     ▼
     候選影格 ─回原始 mp4 用時間碼重新擷取「高畫質」影格(hires.py)─┐
                                                     │            │
              高畫質影格 ─Cosmos-Reason2-8B 逐格 caption─◄─────────┘
                                                     │
                                       Cosmos 過濾掉不相關的候選
                                                     │
                        (可選) Cosmos 呼叫 look_around 取某時間點「前後張」影格
                                                     │
                                                     ▼
                              Cosmos-Reason2-8B 綜合 ─► 繁中解釋
```

> - **多撈 + 過濾**：RAG 先撈 `SEARCH_CANDIDATES` 張（門檻 `SCORE_THRESHOLD`），再讓 Cosmos 判斷哪些真的相關。
> - **look_around function call**：Cosmos 若要確認一個「動作」的前因後果（例如是否真的在丟垃圾），
>   可自行呼叫工具取得該時間點前後影格再下結論（見 [neighbors.py](neighbors.py)、[vlm.py](vlm.py)）。
> - **高畫質擷取**：embedding 用的是 2 秒取樣、壓縮過的 keyframe（給 CLIP 378² 夠用）；
>   但送進 Cosmos 前會**回原始 mp4 用時間碼重新 decode 一張高畫質影格**，讓它看得清小字/細節。
> - Cosmos-Reason2 是 reasoning VLM，會先輸出 `<think>…</think>`，程式會自動去除。
> - 送 VLM 前縮到 `VLM_IMG_MAX_PX`(預設 768) 平衡細節與速度；設 `None` = 原始解析度(最清楚但慢)。

## 安裝

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 另需 ffmpeg：brew install ffmpeg
```

## 使用

**1) 建立索引（ingest）** — 抽影格、嵌入、寫入向量資料庫：

```bash
python ingest.py NewYork.mp4 --reset   # 第一支用 --reset 重建
python ingest.py LosAngeles.mp4        # 之後的用 append
```

**2) 啟動 VLM 服務（Cosmos-Reason2-8B，需先跑，會常駐）**：

```bash
bash serve_vlm.sh          # llama-server，載入 8B GGUF（Metal 加速）
# 等到看到 server ready / 或 curl http://127.0.0.1:8080/health 回 ok
```

**3) 文字搜尋 + VLM 解釋 + 過濾 + look_around**：

```bash
python search.py "有沒有人在亂丟垃圾"
python search.py "bright billboards" --candidates 12 --threshold 0.2
python search.py "yellow taxi" --no-vlm     # 只做向量檢索、不跑 VLM（很快，不需 server）
python search.py "goal" --no-chat           # 不進入互動續問
```

- 結果後會**印出各階段耗時**（載入/嵌入/檢索/caption/filter/summarize/合計）。
- 總結完會進入**互動對話 session**：可對 Cosmos 續問（它記得剛剛的影格、也能再 look_around）；
  直接 Enter 或 `exit` 離開。用 `--no-chat` 可略過。

輸出：
1. **RAG 候選**：門檻過濾後的候選影格（含時間碼與相似度分數）。
2. **Dense Captions**：Cosmos 對每張候選的繁中逐格描述（會讀出看板文字）。
3. **Cosmos 過濾後保留**：判定為相關的影格。
4. **look_around 呼叫紀錄**：Cosmos 有無去看前後張、看了哪些時間點。
5. **綜合解釋**：一段繁中總結。

## 參數（config.py）

| 參數 | 預設 | 說明 |
|------|------|------|
| `MAX_DURATION_SEC` | None | None=整部影片；設數字則只處理前 N 秒 |
| `FRAME_INTERVAL_SEC` | 2.0 | 每幾秒抽一張影格 |
| `EMBED_BATCH_SIZE` | 32 | 嵌入時一次丟幾張到 GPU |
| `SEARCH_CANDIDATES` | 12 | RAG 先多撈幾張候選 |
| `SCORE_THRESHOLD` | 0.20 | 相似度門檻（低於直接丟） |
| `CLIP_MODEL` | ViT-H-14-378-quickgelu | embedding（DFN5B，1024 維）|
| `VLM_SERVER_URL` | 127.0.0.1:8080 | llama-server 位址 |
| `USE_HIRES_FRAMES` | True | 送 VLM 前回原片重新擷取高畫質影格 |
| `VLM_IMG_MAX_PX` | 1280 | 送 VLM 前縮到此邊長；None=1080p原始(~147s/張)、1280≈39s、1024≈19s |
| `VIDEO_DIR` | video/ | 影片存放處；frame 快取放在 `video/_frames`、`video/_frames_hires` |
| `VLM_MAX_TOOL_HOPS` | 3 | look_around 最多連續呼叫次數 |

## 效能參考（本機實測，M5 Pro 24GB）

- **嵌入**（DFN5B ViT-H/378，MPS 批次）：NewYork 170 張 ~48 秒；LosAngeles 1432 張 ~5 分 52 秒（比逐張快 ~4×）。
- **VLM**（Cosmos-Reason2-8B GGUF Q4_K_M，llama-server + Metal，視覺編碼器在 CPU 避開 Metal OOM）：
  - 每張 caption 依送入解析度：**1024px ≈ 19s、1280px ≈ 39s、1080p原始(None) ≈ 147s**。
  - 一次查詢 = 候選 caption + 過濾 + 數次 look_around(每次再抓前後 7 張) + 總結；
    以 1280px、4 候選約 8 分鐘（look_around 抓的張數最吃時間）。
- 權重：CLIP `models/dfn5b/`、VLM `models/cosmos-r2-8b-gguf/`；影格快取在 `video/_frames*`。

> 找不到結果就降 `--threshold`；瞬間動作（如一瞬間丟垃圾）建議把 `FRAME_INTERVAL_SEC` 調小、
> 讓 Cosmos 多用 look_around（見 nvidia_vss_flow.md 對「瞬間事件」的討論）。

## 檔案

| 檔案 | 用途 |
|------|------|
| `nvidia_vss_flow.md` | NVIDIA VSS 架構與流程解析（參考來源） |
| `config.py` | 全域參數 |
| `embedder.py` | DFN5B (open_clip) 影像/文字嵌入、批次 |
| `store.py` | ChromaDB 向量資料庫封裝 |
| `ingest.py` | 抽影格 → 嵌入 → 入庫 |
| `neighbors.py` | 前後相鄰影格查詢（look_around 用） |
| `hires.py` | 回原始 mp4 用時間碼重新擷取高畫質影格（送 VLM 用） |
| `vlm.py` | Cosmos-Reason2-8B（llama-server）封裝：caption / 過濾 / look_around 工具 |
| `serve_vlm.sh` | 啟動 llama-server（8B GGUF, Metal） |
| `search.py` | 檢索 + 門檻過濾 + Cosmos 解釋 |
