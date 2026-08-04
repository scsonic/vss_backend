# NVIDIA VSS (Video Search and Summarization) 流程解析

> 本文件整理 NVIDIA AI Blueprint「Video Search and Summarization」的架構與資料流，
> 作為我們在 MacBook Pro (Apple Silicon) 上打造精簡版 `video search + VLM` 範例的參考。
> 原始專案：<https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization>

---

## 1. VSS 是什麼

VSS 是一套 **GPU 加速的 video analytics agent 參考架構**，可以做到：

- **Long Video Summarization**：把很長的影片切段、逐段用 VLM 產生 dense caption，再彙整成一份摘要。
- **Video Search**：用自然語言查詢，在影片庫中找出相關片段。
- **Visual Q&A**：針對影片內容問答。
- **Alert Verification**：即時偵測異常，並用 VLM 降低誤報 (false positive)。

核心理念：**影片太長、太大，無法一次丟給模型**，所以要「切段 → 逐段理解 → 建立可檢索的語意索引 → 檢索後再用 LLM/VLM 合成答案」。

---

## 2. 三層架構

| 層級 | 職責 | 代表元件 |
|------|------|----------|
| **Real-Time Video Intelligence** | 從串流/檔案抽取視覺特徵與語意 embedding，發佈到 message broker | Stream Handler、TensorRT visual encoder |
| **Downstream Analytics** | 把偵測結果 enrich 成商業事件（軌跡、事件偵測、空間分析） | behavior-analytics、video-analytics-api |
| **Agent & Offline Processing** | 用 MCP (Model Context Protocol) 提供統一的工具介面給 agent 使用 | agent service、CA-RAG |

---

## 3. 核心資料流 (Pipeline)

```
影片輸入 (stream / 檔案)
      │
      ▼
[Stream Handler] 解碼 + 切成 chunk（小片段）
      │
      ├──────────────► [Visual Encoder (TensorRT)] 產生每個 chunk 的 embedding
      │                                                  │
      ▼                                                  ▼
[VLM] 針對每個 chunk 產生 dense caption / 回答           [Milvus Vector DB] 儲存 embedding + metadata
      │                                                  ▲
      ▼                                                  │
[CA-RAG] 從逐段 VLM 回應萃取資訊、動態建 graph ──────────┘
      │
      ▼
彙整成單一摘要 / 回答 / 檢索結果
```

### 各階段說明

1. **Ingestion（輸入）**：接受即時串流或已存檔的影片。
2. **Chunking（切段）**：Stream Handler 把長影片切成較小的 chunk（時間窗），方便逐段處理。
3. **Embedding Generation（產生嵌入）**：用 TensorRT-based visual encoder 把每個 chunk 轉成語意 embedding。
4. **Dense Captioning（密集描述）**：VLM 對每個 chunk 產生 frame-level 的文字描述，作為語意索引的基礎。
5. **Storage（儲存）**：embedding 與 metadata 存進 **Milvus** vector DB，成為可檢索的索引。
6. **Retrieval（檢索）**：使用者用自然語言查詢，與已嵌入的 caption / metadata 做語意比對，找出相關片段。
7. **Enhancement（增強）**：把檢索到的 context 塞進 LLM prompt 做合成 (augment)。
8. **Delivery（產出）**：agent 合成出摘要、警報或答案。

---

## 4. CA-RAG（Context-Aware RAG）

VSS 的靈魂模組。它從「逐段 VLM 回應」萃取有用資訊並彙整，支援多種檢索策略：

- **Vector Retrieval**：語意相似度檢索 + contextual compression。
- **Graph Retrieval**：知識圖譜檢索，處理實體之間的關係。
- **VLM Retrieval**：多模態視覺檢索。
- **Chain-of-Thought (CoT) Retrieval**：帶信心分數的迭代式檢索。
- **Advanced Retrieval**：模組化的 iterative planning & execution。

特點：**在 ingest chunk 的同時動態建立 graph**，讓「建圖」與「摘要」可以平行處理。

---

## 5. 用到的模型（原始 VSS）

| 角色 | 模型（範例） |
|------|--------------|
| LLM（推理/報告） | NVIDIA Nemotron-Nano-9B-v2 |
| VLM（視覺理解/Q&A） | Cosmos Reason / VILA 系列 VLM |
| Visual Encoder（embedding） | TensorRT-based encoder |
| Vector DB | Milvus |
| Reranker / Embedding | NeMo Retriever 系列 |
| 訊息中介 | Kafka / Redis pub-sub |

---

## 6. 對應到我們的精簡版（Mac / Apple Silicon）

因為只有 MacBook Pro（無 NVIDIA GPU、無 TensorRT/Milvus 叢集），我們把每個角色替換成「同類型但小一點、能在 Apple Silicon 跑」的方案：

| VSS 元件 | VSS 用的東西 | 我們的替代方案 | 原因 |
|----------|--------------|----------------|------|
| Chunking / 解碼 | Stream Handler | `ffmpeg` 每 N 秒抽 keyframe | 無需串流，檔案即可 |
| Visual Encoder (embedding) | TensorRT encoder | **DFN5B-CLIP-ViT-H/14 @ 378px**（1024 維，跑在 MPS） | ViT-H 級、最接近 VSS 的 NV-CLIP（同為 1024 維） |
| Vector DB | Milvus | **ChromaDB**（嵌入式、單機） | 免叢集、pip 即裝 |
| Dense Captioning / VLM | Cosmos / VILA | **Cosmos-Reason2-8B**（GGUF Q4_K_M，llama-server + Metal） | 正宗 NVIDIA Cosmos Reason2、Mac 可跑、支援 function call |
| CA-RAG | 複雜 graph-RAG | **簡化版 vector retrieval + VLM 合成** | 保留「檢索→VLM 解釋」的精神 |
| LLM 報告 | Nemotron-9B | 直接用同一顆 VLM 產生解釋 | 省資源 |

### 我們的資料流（簡化後）

```
mp4（只處理前 10 分鐘）
   │  ffmpeg 每 N 秒抽一張 keyframe
   ▼
keyframes ──► OpenCLIP image encoder ──► image embedding
   │                                          │
   │                                          ▼
   │                                   ChromaDB（存 embedding + 時間戳 metadata）

── 查詢時 ──
文字 query ──► OpenCLIP text encoder ──► text embedding
   │
   ▼
ChromaDB 語意檢索 top-k keyframes（含時間戳）
   │
   ▼
mlx-vlm (Cosmos-Reason2-2B) 對每張影格產生 dense caption（逐格、單張最穩）
   │
   ▼
把「query + 帶時間碼的 caption」再丟給同一顆模型做綜合(CA-RAG) ──► 繁中解釋
```

> 註：dense captioning 放在「查詢時只對 top-k 做」，讓 ingest 維持極快（僅嵌入）。
> 若要更貼近 VSS，可把 caption 移到 ingest 階段對每格產生並存入 DB（時間換品質）。

這樣就完整保留了 VSS 的三個關鍵精神：
1. **切段 + 逐格語意嵌入**（chunk + embedding）
2. **向量資料庫做語意檢索**（vector retrieval）
3. **檢索後交給 VLM 合成解釋**（CA-RAG 的核心動作）
