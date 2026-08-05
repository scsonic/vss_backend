"""VLM 封裝：NVIDIA Cosmos-Reason2-8B (GGUF Q4_K_M)，由 llama-server 提供服務。

流程（對應使用者需求）：
  1) RAG 先「多撈」候選影格（search.py 用門檻過濾）。
  2) caption 每張候選 → 用 Cosmos 過濾掉不相關的。
  3) 對留下的影格做綜合總結；過程中 Cosmos 可自行呼叫 look_around()
     取得某時間點「前後張」影格，判斷一個動作的前因後果後再總結。

啟動服務：bash serve_vlm.sh
"""
import base64
import io
import json
import re
import time
import urllib.request
from pathlib import Path

from PIL import Image

import config
import hires
from neighbors import neighbors_by_time

# Cosmos-Reason2 是 reasoning VLM，會先輸出 <think>...</think>，只取最後答案。
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    text = text or ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _parse_tc(tc: str) -> float:
    parts = [int(x) for x in str(tc).strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + s


# 給 Cosmos 的工具：看某時間點前後張影格
LOOK_AROUND_TOOL = {
    "type": "function",
    "function": {
        "name": "look_around",
        "description": "取得某支影片某時間點『前後相鄰』的影格影像，用來確認一個動作/事件的前因後果。"
                       "當單張影格看不出完整動作（例如是否真的在丟垃圾）時呼叫。",
        "parameters": {
            "type": "object",
            "properties": {
                "video": {"type": "string", "description": "影片檔名，例如 NewYork.mp4"},
                "timecode": {"type": "string", "description": "時間碼 HH:MM:SS"},
                "before": {"type": "integer", "description": "往前看幾張（預設 3）"},
                "after": {"type": "integer", "description": "往後看幾張（預設 3）"},
            },
            "required": ["video", "timecode"],
        },
    },
}


class Vlm:
    def __init__(self, url: str = config.VLM_SERVER_URL):
        self.url = url.rstrip("/")
        try:
            urllib.request.urlopen(self.url + "/health", timeout=5)
        except Exception:
            raise RuntimeError(
                f"連不到 llama-server ({self.url})。請先啟動：\n  bash serve_vlm.sh"
            )

    def _reset_usage(self):
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "predicted_ms": 0.0}

    def _usage_summary(self) -> dict:
        u = self._usage
        tps = (u["completion_tokens"] / (u["predicted_ms"] / 1000)) if u["predicted_ms"] > 0 else 0.0
        return {**u, "tokens_per_sec": round(tps, 1)}

    # ---------- 底層 ----------
    def _local_keyframe(self, path: str) -> str:
        """把 DB 存的(可能是別台機器的)絕對路徑，對應到本機 FRAMES_DIR/<影片>/<檔名>。"""
        p = Path(path)
        local = config.FRAMES_DIR / p.parent.name / p.name
        return str(local) if local.exists() else path

    def _resolve_img(self, video: str, t_sec: float, fallback: str) -> str:
        """搜到的影格 → 回原始 mp4 重新擷取高畫質影格（失敗則用本機 keyframe）。"""
        local = self._local_keyframe(fallback)
        if config.USE_HIRES_FRAMES and video and t_sec is not None:
            return hires.resolve(video, float(t_sec), local)
        return local

    def _img_content(self, path: str) -> dict:
        """讀圖 →（可選）縮到長邊 self._img_max_px → base64 data URL。

        用 PIL thumbnail()：只會往下縮、不會放大，並保持長寬比。
        """
        img = Image.open(path).convert("RGB")
        max_px = getattr(self, "_img_max_px", None) or config.VLM_IMG_MAX_PX
        if max_px:
            img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    def _chat(self, messages, tools=None, max_tokens=400) -> dict:
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": 0.2, "stream": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        usage = resp.get("usage") or {}
        timings = resp.get("timings") or {}
        if hasattr(self, "_usage"):
            self._usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self._usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self._usage["total_tokens"] += usage.get("total_tokens", 0)
            self._usage["predicted_ms"] += timings.get("predicted_ms", 0.0)
        return resp["choices"][0]

    # ---------- 步驟 ----------
    def caption(self, image_path: str, max_tokens: int = 200) -> str:
        msg = [{"role": "user", "content": [
            {"type": "text", "text": "用一句話詳細描述這張影片影格，包含畫面中的看板/招牌文字、車輛、人物與明顯動作。"},
            self._img_content(image_path),
        ]}]
        return _strip_think(self._chat(msg, max_tokens=max_tokens)["message"].get("content"))

    def filter_relevant(self, query: str, candidates: list[dict]) -> list[dict]:
        """candidates 需已含 'caption'。用一次 text-only 呼叫挑出相關的。"""
        lines = [f"{i+1}. [{c['video']} {c['timecode']}] {c['caption']}"
                 for i, c in enumerate(candidates)]
        prompt = (
            f"使用者要找的是：「{query}」。\n以下是候選影格的描述：\n" + "\n".join(lines) +
            "\n\n請只挑出『真的和使用者要找的內容相關』的編號，"
            "用 JSON 陣列回覆，例如 [1,3,4]；若全部都不相關就回 []。不要解釋。"
        )
        txt = _strip_think(self._chat([{"role": "user", "content": prompt}], max_tokens=150)["message"].get("content"))
        m = re.findall(r"\[([\d,\s]*)\]", txt)
        idxs = set()
        if m:
            idxs = {int(n) - 1 for n in re.findall(r"\d+", m[-1])}
        else:
            idxs = {int(n) - 1 for n in re.findall(r"\d+", txt)}
        kept = [c for i, c in enumerate(candidates) if i in idxs and 0 <= i < len(candidates)]
        return kept

    def _exec_tool(self, name: str, args: dict):
        """回傳 (文字結果, [影像 content])。"""
        if name == "look_around":
            video = args.get("video", "")
            stem = video[:-4] if video.lower().endswith(".mp4") else video
            t = _parse_tc(args.get("timecode", "00:00:00"))
            before = int(args.get("before", config.LOOK_AROUND_BEFORE))
            after = int(args.get("after", config.LOOK_AROUND_AFTER))
            frs = neighbors_by_time(stem, t, before, after)
            if not frs:
                return f"找不到 {video} {args.get('timecode')} 附近的影格。", []
            video_file = stem + ".mp4"
            desc = "、".join(f["timecode"] + ("(中心)" if f["is_center"] else "") for f in frs)
            # 前後張也回原始影片擷取高畫質
            imgs = [self._img_content(self._resolve_img(video_file, f["t_sec"], f["path"])) for f in frs]
            return f"已取得 {len(frs)} 張前後影格（原始畫質）：{desc}", imgs
        return f"未知工具：{name}", []

    def _tools_loop(self, messages, tools, max_tokens):
        """跑帶工具的對話迴圈；最終回答會 append 進 messages（供後續續問）。"""
        trace = []
        for _ in range(config.VLM_MAX_TOOL_HOPS):
            choice = self._chat(messages, tools=tools, max_tokens=max_tokens)
            msg = choice["message"]
            tcs = msg.get("tool_calls")
            if not tcs:
                answer = _strip_think(msg.get("content"))
                messages.append({"role": "assistant", "content": answer})
                return answer, trace
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result_text, images = self._exec_tool(fn, args)
                trace.append({"tool": fn, "args": args, "result": result_text})
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "name": fn, "content": result_text})
                if images:
                    messages.append({"role": "user",
                                     "content": [{"type": "text", "text": "這是你要求的前後影格："}] + images})
        # 用完 hop 次數，強制不帶工具做最後總結
        answer = _strip_think(self._chat(messages, tools=None, max_tokens=max_tokens)["message"].get("content"))
        messages.append({"role": "assistant", "content": answer})
        return answer, trace

    def ask(self, messages, question: str, max_tokens: int = 700):
        """在既有對話上續問（保留先前影格與工具能力）。回傳 (answer, trace, usage)。"""
        self._reset_usage()
        messages.append({"role": "user", "content": question})
        answer, trace = self._tools_loop(messages, [LOOK_AROUND_TOOL], max_tokens)
        return answer, trace, self._usage_summary()

    def explain(self, query: str, candidates: list[dict], image_size: int = 480) -> dict:
        """完整流程：回原片擷高畫質 → caption 候選 → 過濾 → (可 look_around) 綜合總結。

        image_size: 送進 LLM 前，圖片長邊縮到這個值以內（只往下縮、保持比例、不放大），
        用來加速推論；預設 480，<=0 則視為不限制（用原圖／config.VLM_IMG_MAX_PX）。
        回傳含 timings（各階段耗時）與 messages（對話 history，供 search.py 續問）。
        """
        self._reset_usage()
        self._img_max_px = image_size if image_size and image_size > 0 else None
        timings = {}
        t = time.time()
        for c in candidates:
            c["hires"] = self._resolve_img(c["video"], c.get("t_sec"), c["path"])
            c["caption"] = self.caption(c["hires"])
        timings["caption"] = time.time() - t

        t = time.time()
        kept = self.filter_relevant(query, candidates)
        timings["filter"] = time.time() - t
        if not kept:
            return {"candidates": candidates, "kept": [], "answer": "（Cosmos 判斷候選影格都與問題無關。）",
                    "trace": [], "messages": [], "timings": timings, "usage": self._usage_summary()}

        caps = "\n".join(f"- [{c['video']} {c['timecode']}] {c['caption']}" for c in kept)
        system = (
            "你是影片分析助理。以下是與使用者問題相關的影格（含影片名、時間碼、描述）。"
            "如果要確認某個『動作或事件』的前因後果（例如是否真的在丟垃圾、是否闖紅燈），"
            "可以呼叫 look_around 取得該時間點的前後影格再判斷。"
            "最後請用繁體中文寫『一段』通順的總結：發生了什麼、在哪支影片的哪些時間點、與問題的關聯。"
        )
        content = [{"type": "text", "text": f"使用者問題：「{query}」\n相關影格：\n{caps}"}]
        for c in kept[:4]:
            content.append(self._img_content(c.get("hires", c["path"])))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]
        t = time.time()
        answer, trace = self._tools_loop(messages, [LOOK_AROUND_TOOL], max_tokens=700)
        timings["summarize"] = time.time() - t
        return {"candidates": candidates, "kept": kept, "answer": answer,
                "trace": trace, "messages": messages, "timings": timings,
                "usage": self._usage_summary()}
