"""Agent Search：用 OpenRouter LLM（DeepSeek）當推理大腦，自主決定要不要對影片資料庫做
CLIP 語意搜尋、要不要往前後多看幾張影格的相似度分數，再把結果整理成回答。

這是一個「萬用影片搜尋 agent」，完全用 embedding 分數運作、不呼叫 Cosmos VLM（不做視覺解讀/caption），
所以速度快、不吃 GPU，但回答時只能依據「視覺特徵相似度」做推論，不是「真的看過畫面」。

跟 /api/search + /api/explain 的固定流程不同：這裡搜尋幾次、搜什麼、要不要往前後多看幾張，
全部由 LLM 依對話內容自己判斷，並用 function calling 呼叫下面兩個工具。

啟動：/agent 頁面 → POST /api/agent_chat，見 app.py。
"""
import json
import urllib.request
import urllib.error

import config

SYSTEM_PROMPT = """你是一個「萬用影片搜尋助理」，背後是影片畫面的向量資料庫（CLIP embedding + ChromaDB）。
你完全靠「語意向量相似度」運作，沒有視覺模型幫你看畫面內容、也看不到圖片本身 —— 你只看得到分數、時間碼、檔名。

你有兩個工具可以用：

1) search_video(query, top_n, model)
   用語意向量搜尋畫面，回傳最相關的候選片段（影片檔名、時間碼、相似度分數、合併張數）。
   query 請用「畫面看起來像什麼」來下，用簡短的名詞/場景描述，
   不要照抄使用者的完整問句。這裡用的 CLIP 模型是英文語料訓練的，中文查詢的效果明顯較差，
   所以 query 請儘量用英文下（就算使用者是用中文問你），例如使用者問「有沒有人闖紅燈」，
   你應該下 "person crossing street against red light" 之類的英文描述，而不是中文。
   必要時可以連續呼叫多次、用不同角度/關鍵字(仍然用英文)去找，增加找到的機會。
   model 是要用哪組 embedding 模型查，幾組是完全獨立的資料庫：
     - "siglip2-giant"（預設，不指定就是用這個）
     - "dfn5b"
     - "cosmos-embed1-448p"（NVIDIA 的 video-clip embedding，跟前兩個原理不同）
   不用主動切換，維持預設就好；只有使用者明確要求指定模型時才改用另一個。
   如果同一件事在預設模型都找不到，也可以換另一個模型再試一次。

2) look_around(index, before, after)
   針對「最近一次 search_video」結果裡的某一筆（index 是結果的 #編號，從 1 開始），
   往前、往後各多看幾張影格（每張間隔是固定的抽格秒數），回傳這些影格對「同一個查詢」的相似度分數。
   不會呼叫視覺模型、只是用已經算好的 embedding 比對分數，所以很快。
   用途：判斷一個事件的時間範圍（分數在前後哪裡開始升高/降低）、確認附近有沒有分數更高的畫面、
   或是單一命中點的分數是不是噪音（前後分數都很低，可能是誤判）。

工作原則：
- search_video / look_around 給的都只是「視覺特徵相似度分數」，不是確認過的事實 ——
  分數高不代表答案一定對，回答時用「畫面特徵符合」「相似度高」這類措辭，不要講得像親眼確認過。
- 可以先 search_video 找候選，覺得某個候選有可能但還想確認時間範圍/前後脈絡，再對它 look_around。
- 找不到相關內容、或分數普遍偏低時要老實說找不到/不確定，不要編造畫面內容。
- 最後一律用「英文」回覆使用者（即使使用者用中文或其他語言提問），整理成一段對使用者有幫助、
  清楚的回覆：找到了什麼（用詞保守）、在哪支影片的哪個時間點、相似度大概如何。
- 如果使用者的問題跟影片內容搜尋無關（例如閒聊、問你是誰），直接以一般助理身份用英文回答即可，不用勉強呼叫工具。
"""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_video",
        "description": "用語意向量搜尋影片畫面，回傳最相關的候選片段（影片、時間碼、相似度分數）。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要搜尋的畫面內容，簡短場景/物件描述，"
                                                             "請用英文（CLIP 模型是英文語料訓練，英文查詢效果明顯較好）"},
                "top_n": {"type": "integer", "description": "回傳幾筆結果，預設 10"},
                "model": {"type": "string", "enum": list(config.EMBED_MODELS.keys()),
                          "description": "要用哪組 embedding 模型查，預設 siglip2-giant，各組是各自獨立的資料庫"},
            },
            "required": ["query"],
        },
    },
}

LOOK_AROUND_TOOL = {
    "type": "function",
    "function": {
        "name": "look_around",
        "description": "針對最近一次 search_video 結果裡的某一筆，往前/往後多看幾張影格的相似度分數"
                       "（不呼叫視覺模型，只是比對既有 embedding，速度快），用來判斷事件時間範圍或確認命中是否可靠。",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "最近一次 search_video 結果的 #編號，從 1 開始"},
                "before": {"type": "integer", "description": "往前看幾張，預設 5"},
                "after": {"type": "integer", "description": "往後看幾張，預設 5"},
            },
            "required": ["index"],
        },
    },
}

TOOLS = [SEARCH_TOOL, LOOK_AROUND_TOOL]


class AgentError(RuntimeError):
    pass


def _call_openrouter(messages: list[dict], max_tokens: int = 1500, use_tools: bool = True) -> dict:
    if not config.OPENROUTER_API_KEY:
        raise AgentError("尚未設定 OPENROUTER_API_KEY（見 local_secrets.py 或環境變數）")
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if use_tools:
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"
    req = urllib.request.Request(
        config.OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://4070ti.scsonic.com",
            "X-Title": "VSS Agent Search",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise AgentError(f"OpenRouter API 錯誤 {e.code}：{body[:500]}")


def _run_search(store, embedder, base: str, query: str, top_n: int = 10) -> tuple[list[dict], list[dict], list[float]]:
    """回傳 (完整候選 list，含 path/t_sec，供 look_around 用, 給 LLM 看的精簡版, 這次查詢用的 embedding)。"""
    from app import _cluster_hits  # 沿用 app.py 既有的「相鄰影格合併」邏輯，避免重複實作

    q_emb = embedder.embed_text(query)
    pool = min(store.count(), max(int(top_n or 10) * 8, 60))
    hits = store.query(q_emb, pool)
    clustered = _cluster_hits(hits, config.MERGE_FRAME_GAP, base)
    cands = clustered[: max(1, int(top_n or 10))]
    # thumb/mp4 只給網頁 UI 呈現用，不算進 LLM 看到的 JSON（省 token，見 run_agent_turn 組 tool 訊息時濾掉）
    brief = [{"#": i + 1, "video": c["video"], "timecode": c["timecode"], "score": c["score"],
              "span": c["span"], "merged": c["merged"], "thumb": c["thumb"], "mp4": c["mp4"]}
             for i, c in enumerate(cands)]
    return cands, brief, q_emb


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _run_look_around(store, base: str, cands: list[dict], q_emb: list[float],
                      index: int, before: int = 5, after: int = 5) -> dict:
    if not cands or q_emb is None:
        return {"error": "還沒有 search_video 的結果可以往前後看，請先呼叫 search_video。"}
    i = int(index) - 1
    if i < 0 or i >= len(cands):
        return {"error": f"index 超出範圍，最近一次 search_video 只有 {len(cands)} 筆結果。"}
    cand = cands[i]
    video, t_sec = cand["video"], cand["t_sec"]
    center_idx = round(t_sec / config.FRAME_INTERVAL_SEC)
    before, after = int(before or 5), int(after or 5)
    offsets = list(range(-before, after + 1))
    id_by_offset = {}
    for off in offsets:
        n = center_idx + off
        if n < 0:
            continue
        t = round(n * config.FRAME_INTERVAL_SEC, 2)
        id_by_offset[off] = f"{video}::{t}"
    found = store.get_by_ids(list(id_by_offset.values()))
    frames = []
    for off, id_ in id_by_offset.items():
        rec = found.get(id_)
        if not rec:
            continue
        m = rec["meta"]
        frames.append({
            "offset": off, "timecode": m["timecode"], "score": round(_cosine(q_emb, rec["embedding"]), 3),
            "is_center": off == 0, "thumb": _frame_url(m["path"], base),
        })
    frames.sort(key=lambda f: f["offset"])
    return {"video": video, "center_timecode": cand["timecode"], "frames": frames}


def _frame_url(path: str, base: str) -> str:
    from app import frame_url
    return frame_url(path, base)


def run_agent_turn(session: dict, user_message: str, *, get_store_fn, get_embedder_fn, base: str) -> dict:
    """在既有 agent session 上跑一輪對話（含工具呼叫迴圈）。

    get_store_fn / get_embedder_fn：傳 model_key 進去，回傳對應模型的 VectorStore / ClipEmbedder
    單例（見 app.py 的 get_store/get_embedder）——因為每組 embedding 模型各自有獨立資料庫，
    要由每次 search_video 呼叫時帶的 model 參數決定要用哪一組。

    session 結構：{"messages": [...OpenRouter 對話格式...],
                   "last_cands": [...最近一次 search_video 的完整候選...],
                   "last_q_emb": [...最近一次 search_video 用的查詢向量...],
                   "last_model": "最近一次 search_video 用的 model_key（look_around 要用同一組資料庫）"}
    回傳 {"answer", "trace": [{"tool","args","result_brief"}], "usage": {...累計}}
    """
    messages = session.setdefault("messages", [])
    if not messages:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_message})

    trace = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for _ in range(config.AGENT_MAX_TOOL_HOPS):
        resp = _call_openrouter(messages)
        u = resp.get("usage") or {}
        usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += u.get("completion_tokens", 0)
        usage_total["total_tokens"] += u.get("total_tokens", 0)

        choice = resp["choices"][0]
        msg = choice["message"]
        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            answer = msg.get("content") or "(No reply content was returned.)"
            messages.append({"role": "assistant", "content": answer})
            return {"answer": answer, "trace": trace, "usage": usage_total}

        messages.append({"role": "assistant", "content": msg.get("content") or "",
                          "tool_calls": tool_calls})

        for tc in tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            if fn == "search_video":
                query = args.get("query", "")
                top_n = args.get("top_n", 10)
                model_key = args.get("model") or config.DEFAULT_EMBED_MODEL
                if model_key not in config.EMBED_MODELS:
                    model_key = config.DEFAULT_EMBED_MODEL
                store = get_store_fn(model_key)
                embedder = get_embedder_fn(model_key)
                cands, brief, q_emb = _run_search(store, embedder, base, query, top_n)
                session["last_cands"] = cands
                session["last_q_emb"] = q_emb
                session["last_model"] = model_key
                # LLM 是文字模型，thumb/mp4 URL 對它沒用、只會浪費 token；那兩欄只留給網頁 UI 顯示。
                llm_view = [{k: v for k, v in item.items() if k not in ("thumb", "mp4")} for item in brief]
                result_text = json.dumps({"model": model_key, "results": llm_view}, ensure_ascii=False)
                trace.append({"tool": fn, "args": args, "model": model_key, "result_brief": brief})
            elif fn == "look_around":
                model_key = session.get("last_model", config.DEFAULT_EMBED_MODEL)
                store = get_store_fn(model_key)
                out = _run_look_around(store, base, session.get("last_cands"), session.get("last_q_emb"),
                                        args.get("index"), args.get("before", 5), args.get("after", 5))
                llm_frames = [{k: v for k, v in f.items() if k != "thumb"} for f in out.get("frames", [])]
                llm_out = {**out, "frames": llm_frames} if "frames" in out else out
                result_text = json.dumps(llm_out, ensure_ascii=False)
                trace.append({"tool": fn, "args": args, "result_brief": out})
            else:
                result_text = json.dumps({"error": f"未知工具：{fn}"}, ensure_ascii=False)
                trace.append({"tool": fn, "args": args, "result_brief": {"error": "unknown tool"}})

            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                              "name": fn, "content": result_text})

    # 超過 hop 上限，強制不帶工具做最後總結（use_tools=False：不然模型還是可能選擇再呼叫一次工具
    # 而不是回答，導致 content 是空的、只能顯示下面這句 fallback 文字）
    resp = _call_openrouter(messages, use_tools=False)
    u = resp.get("usage") or {}
    usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
    usage_total["completion_tokens"] += u.get("completion_tokens", 0)
    usage_total["total_tokens"] += u.get("total_tokens", 0)
    msg = resp["choices"][0]["message"]
    # 推理模型有時把 max_tokens 都花在 reasoning 上，content 被截斷成空字串；
    # 這種情況退而求其次顯示 reasoning 摘要，總比空白回覆好。
    answer = msg.get("content") or msg.get("reasoning") or "(Reached the tool-call limit and no reply content was returned.)"
    messages.append({"role": "assistant", "content": answer})
    return {"answer": answer, "trace": trace, "usage": usage_total}
