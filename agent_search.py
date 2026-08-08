"""Agent Search：用 OpenRouter LLM（DeepSeek）當推理大腦，自主決定要不要對影片資料庫做
CLIP 語意搜尋、要不要進一步呼叫 Cosmos Reason 實際「看」畫面，再把結果整理成回答。

跟 /api/search + /api/explain 的固定流程不同：這裡搜尋幾次、搜什麼、要不要做視覺解讀，
全部由 LLM 依對話內容自己判斷、自己下查詢字串，並用 function calling 呼叫下面兩個工具。

啟動：/agent 頁面 → POST /api/agent_chat，見 app.py。
"""
import json
import urllib.request
import urllib.error
from pathlib import Path

import config

SYSTEM_PROMPT = """你是一個「影片監視器內容搜尋助理」，背後有一個影片畫面的向量資料庫（CLIP embedding + ChromaDB）。

你有兩個工具可以用：

1) search_video(query, top_n)
   用語意向量搜尋畫面，回傳最相關的候選片段（影片檔名、時間碼、相似度分數、合併張數）。
   這只是「視覺特徵相近」的分數，不代表你真的看到內容、也不保證正確 —— 分數高不等於答案就對。
   query 請用「畫面看起來像什麼」來下，用簡短的名詞/場景描述（中英文皆可），
   不要照抄使用者的完整問句。例如使用者問「有沒有人闖紅燈」，你可以下
   "person crossing street against red light"、"紅綠燈路口 行人" 等，
   必要時可以連續呼叫多次、用不同角度/關鍵字去找，增加找到的機會。

2) explain_clips(query, indices)
   針對「最近一次 search_video」回傳的候選，實際呼叫視覺模型（Cosmos Reason）看畫面內容，
   過濾掉不相關的、並產生詳細描述與總結。比 search_video 慢很多（要跑 VLM 推論），
   indices 是 search_video 回傳結果的編號（從 1 開始，例如 [1,3,5]），不填就是全部候選。

工作原則：
- search_video 的分數只能拿來「篩選候選」，如果使用者的問題需要確認「畫面裡到底發生了什麼」
  （動作、事件、有沒有某個東西），一定要接著呼叫 explain_clips 實際看過畫面再回答，
  不要只憑相似度分數就下結論或編造畫面內容。
- 如果只是要列出「大概在哪裡/哪些片段」而不需要確認細節，search_video 的結果就夠了。
- 找不到相關內容時要老實說找不到，不要編造。
- 最後一律用繁體中文，整理成一段對使用者有幫助、清楚的回覆：找到了什麼、
  在哪支影片的哪個時間點，如果經過 explain_clips 就以視覺解讀為準。
- 如果使用者的問題跟影片內容搜尋無關（例如閒聊、問你是誰），直接以一般助理身份回答即可，不用勉強呼叫工具。
"""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_video",
        "description": "用語意向量搜尋影片畫面，回傳最相關的候選片段（不代表已確認內容，只是視覺特徵相近的分數）。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要搜尋的畫面內容，簡短場景/物件描述，中英文皆可"},
                "top_n": {"type": "integer", "description": "回傳幾筆結果，預設 10"},
            },
            "required": ["query"],
        },
    },
}

EXPLAIN_TOOL = {
    "type": "function",
    "function": {
        "name": "explain_clips",
        "description": "針對最近一次 search_video 找到的候選影格，呼叫視覺模型(Cosmos Reason)實際看畫面內容，"
                       "過濾不相關的並產生詳細描述與總結。比 search_video 慢很多，只有需要確認畫面實際內容時才呼叫。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要確認/解讀的問題或內容"},
                "indices": {"type": "array", "items": {"type": "integer"},
                            "description": "要解讀的候選編號（對應最近一次 search_video 結果的 #編號，從 1 開始）；不填則用全部候選"},
            },
            "required": ["query"],
        },
    },
}

TOOLS = [SEARCH_TOOL, EXPLAIN_TOOL]


class AgentError(RuntimeError):
    pass


def _call_openrouter(messages: list[dict], max_tokens: int = 1500) -> dict:
    if not config.OPENROUTER_API_KEY:
        raise AgentError("尚未設定 OPENROUTER_API_KEY（見 local_secrets.py 或環境變數）")
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
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


def _run_search(store, embedder, base: str, query: str, top_n: int = 10) -> tuple[list[dict], list[dict]]:
    """回傳 (完整候選 list，含 path/t_sec，供 explain_clips 用, 給 LLM 看的精簡版)。"""
    from app import _cluster_hits  # 沿用 app.py 既有的「相鄰影格合併」邏輯，避免重複實作

    q_emb = embedder.embed_text(query)
    pool = min(store.count(), max(int(top_n or 10) * 8, 60))
    hits = store.query(q_emb, pool)
    clustered = _cluster_hits(hits, config.MERGE_FRAME_GAP, base)
    cands = clustered[: max(1, int(top_n or 10))]
    brief = [{"#": i + 1, "video": c["video"], "timecode": c["timecode"], "score": c["score"],
              "span": c["span"], "merged": c["merged"]} for i, c in enumerate(cands)]
    return cands, brief


def _run_explain(get_vlm_fn, query: str, candidates: list[dict], indices: list[int] | None) -> dict:
    vlm = get_vlm_fn()
    if indices:
        idxset = {int(i) - 1 for i in indices}
        picked = [c for i, c in enumerate(candidates) if i in idxset and 0 <= i < len(candidates)]
    else:
        picked = list(candidates)
    if not picked:
        return {"answer": "指定的編號超出範圍，沒有可解讀的候選。", "kept": []}
    # explain() 會就地在 candidate dict 上加 hires/caption 欄位，複製一份避免污染呼叫端的原始資料
    picked = [dict(c) for c in picked]
    result = vlm.explain(query, picked)
    kept = [{"video": c["video"], "timecode": c["timecode"], "caption": c.get("caption", "")}
            for c in result["kept"]]
    return {"answer": result["answer"], "kept": kept}


def run_agent_turn(session: dict, user_message: str, *, store, embedder, get_vlm_fn, base: str) -> dict:
    """在既有 agent session 上跑一輪對話（含工具呼叫迴圈）。

    session 結構：{"messages": [...OpenRouter 對話格式...], "last_cands": [...最近一次 search_video 的完整候選...]}
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
            answer = msg.get("content") or "（沒有取得回覆內容）"
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
                cands, brief = _run_search(store, embedder, base, query, top_n)
                session["last_cands"] = cands
                result_text = json.dumps({"results": brief}, ensure_ascii=False)
                trace.append({"tool": fn, "args": args, "result_brief": brief})
            elif fn == "explain_clips":
                cands = session.get("last_cands") or []
                if not cands:
                    result_text = json.dumps({"error": "還沒有 search_video 的結果可以解讀，請先呼叫 search_video。"},
                                              ensure_ascii=False)
                    trace.append({"tool": fn, "args": args, "result_brief": {"error": "no prior search"}})
                else:
                    try:
                        out = _run_explain(get_vlm_fn, args.get("query", user_message),
                                            cands, args.get("indices"))
                        result_text = json.dumps(out, ensure_ascii=False)
                        trace.append({"tool": fn, "args": args, "result_brief": out})
                    except Exception as e:
                        result_text = json.dumps({"error": f"Cosmos 服務錯誤：{e}"}, ensure_ascii=False)
                        trace.append({"tool": fn, "args": args, "result_brief": {"error": str(e)}})
            else:
                result_text = json.dumps({"error": f"未知工具：{fn}"}, ensure_ascii=False)
                trace.append({"tool": fn, "args": args, "result_brief": {"error": "unknown tool"}})

            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                              "name": fn, "content": result_text})

    # 超過 hop 上限，強制不帶工具做最後總結
    resp = _call_openrouter(messages)
    u = resp.get("usage") or {}
    usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
    usage_total["completion_tokens"] += u.get("completion_tokens", 0)
    usage_total["total_tokens"] += u.get("total_tokens", 0)
    answer = resp["choices"][0]["message"].get("content") or "（已達工具呼叫上限，且沒有取得回覆內容）"
    messages.append({"role": "assistant", "content": answer})
    return {"answer": answer, "trace": trace, "usage": usage_total}
