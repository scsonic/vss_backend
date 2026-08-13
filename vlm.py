"""VLM 封裝：改用 OpenRouter 的 DeepSeek（跟 agent_search.py 同一顆模型 config.OPENROUTER_MODEL），
不再依賴本機 llama-server + Cosmos-Reason2（該服務已停用，GPU 讓給 embedding 用）。

DeepSeek Flash 是純文字模型、看不到畫面，所以這裡不再對候選影格做視覺 caption/過濾，
改成跟 agent_search.py 一樣的原則：只用「候選片段的時間碼 + 相似度分數」給 LLM 做推理、
綜合出回答；用詞保守（分數高不代表確認過畫面內容）。

啟動：不需要另外啟動服務，只要 config.OPENROUTER_API_KEY 有設定即可。
"""
import json
import time
import urllib.request
import urllib.error

import config


class VlmError(RuntimeError):
    pass


class Vlm:
    def __init__(self):
        if not config.OPENROUTER_API_KEY:
            raise VlmError("尚未設定 OPENROUTER_API_KEY（見 local_secrets.py 或環境變數）")

    def _reset_usage(self):
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_sec": 0.0}

    def _usage_summary(self) -> dict:
        u = self._usage
        tps = (u["completion_tokens"] / u["elapsed_sec"]) if u["elapsed_sec"] > 0 else 0.0
        return {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                "total_tokens": u["total_tokens"], "tokens_per_sec": round(tps, 1)}

    def _chat(self, messages: list[dict], max_tokens: int = 700) -> str:
        payload = {"model": config.OPENROUTER_MODEL, "messages": messages,
                   "max_tokens": max_tokens, "temperature": 0.3}
        req = urllib.request.Request(
            config.OPENROUTER_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://4070ti.scsonic.com",
                "X-Title": "VSS Explain",
            },
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise VlmError(f"OpenRouter API 錯誤 {e.code}：{body[:500]}")
        elapsed = time.time() - t0
        if hasattr(self, "_usage"):
            usage = resp.get("usage") or {}
            self._usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self._usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self._usage["total_tokens"] += usage.get("total_tokens", 0)
            self._usage["elapsed_sec"] += elapsed
        msg = resp["choices"][0]["message"]
        # 推理模型有時把 max_tokens 都花在 reasoning 上，content 被截斷成空字串；
        # 這種情況退而求其次顯示 reasoning 摘要，總比空白回覆好。
        return msg.get("content") or msg.get("reasoning") or ""

    def explain(self, query: str, candidates: list[dict], image_size: int = 480, max_tokens: int = 1500) -> dict:
        """image_size 參數保留給 app.py 相容用（現在不再送圖片，故不使用）。

        直接把 /api/search 找到的候選（影片、時間碼、相似度分數）交給 DeepSeek 做綜合總結，
        不再逐張 caption/過濾——這些候選本來就已經是 /api/search 依分數排序、篩過 top_n 的結果。
        """
        self._reset_usage()
        timings = {}
        lines = []
        for i, c in enumerate(candidates):
            extra = f"（合併 {c['merged']} 張影格，時間範圍 {c['span']}）" if c.get("merged", 1) > 1 else ""
            lines.append(f"{i+1}. [{c['video']} {c['timecode']}] similarity score {c['score']}{extra}")
        system = (
            "你是影片搜尋助理。使用者對影片畫面資料庫做了一次語意向量搜尋，以下是找到的候選片段"
            "（影片檔名、時間碼、與查詢的相似度分數，已依分數排序）。"
            "你看不到實際畫面，只能依據分數與時間碼做推論——分數高不代表答案一定對，"
            "回答時請用「畫面特徵符合」「相似度高」這類保守措辭，不要講得像親眼確認過畫面內容。"
            "如果分數普遍偏低，要老實說不確定/找不到，不要編造畫面內容。"
            "最後請用「英文」（即使使用者是用中文或其他語言提問也一樣）寫『一段』通順的總結："
            "找到了什麼（用詞保守）、在哪支影片的哪些時間點、與查詢的關聯。"
        )
        user = f"使用者搜尋：「{query}」\n找到以下候選片段（依相似度排序）：\n" + "\n".join(lines)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t = time.time()
        answer = self._chat(messages, max_tokens=max_tokens)
        messages.append({"role": "assistant", "content": answer})
        timings["summarize"] = time.time() - t
        return {"candidates": candidates, "kept": candidates, "answer": answer,
                "trace": [], "messages": messages, "timings": timings, "usage": self._usage_summary()}

    def ask(self, messages: list[dict], question: str, max_tokens: int = 1500):
        """在既有對話上續問（沿用 explain() 留下的 messages）。回傳 (answer, trace, usage)。"""
        self._reset_usage()
        messages.append({"role": "user", "content": question})
        answer = self._chat(messages, max_tokens=max_tokens)
        messages.append({"role": "assistant", "content": answer})
        return answer, [], self._usage_summary()
