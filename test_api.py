"""搜尋 + Cosmos Reason 端對端小測試（不需 pytest，直接跑）。

用法：
    python test_api.py                       # 測本機 http://127.0.0.1:8000
    python test_api.py https://4070ti.scsonic.com
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
QUERY = sys.argv[2] if len(sys.argv) > 2 else "person walking on street"


def post(path: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


FAILED: list[str] = []

print(f"== 測試目標: {BASE}  query={QUERY!r} ==\n")

# 1) /api/search — 向量資料庫搜尋
d, dt = post("/api/search", {"query": QUERY, "top_n": 3})
check("search: 沒有 error", "error" not in d, d.get("error", ""))
check("search: 有 session_id", bool(d.get("session_id")))
check("search: results 非空", len(d.get("results", [])) > 0, f"{len(d.get('results', []))} 筆, {dt:.2f}s")
if d.get("results"):
    r0 = d["results"][0]
    check("search: 結果含 score/video/timecode", all(k in r0 for k in ("score", "video", "timecode")))
    print(f"       top1: {r0['video']} {r0['timecode']} score={r0['score']}")

sid = d.get("session_id")
if not sid:
    print("\n沒有 session_id，無法繼續測 /api/explain。")
    sys.exit(1)

# 2) /api/explain — Cosmos Reason 推理（caption/filter/look_around/summarize）
print()
e, dt = post("/api/explain", {"session_id": sid}, timeout=300)
check("explain: 沒有 error", "error" not in e, e.get("error", ""))
check("explain: answer 非空", bool(e.get("answer")), f"{dt:.2f}s")
u = e.get("usage") or {}
check("explain: usage 含 tokens_per_sec", "tokens_per_sec" in u)
if u:
    print(f"       usage: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} "
          f"total={u.get('total_tokens')} tok/s={u.get('tokens_per_sec')}")
if e.get("answer"):
    print(f"       answer[:120]: {e['answer'][:120]}")
t = e.get("timings") or {}
if t:
    print(f"       timings: {t}")

# 3) /api/chat — 續問（沿用同一 session，驗證 usage 也有回傳）
print()
c, dt = post("/api/chat", {"session_id": sid, "message": "這是哪一支影片？"}, timeout=120)
check("chat: 沒有 error", "error" not in c, c.get("error", ""))
check("chat: answer 非空", bool(c.get("answer")), f"{dt:.2f}s")
cu = c.get("usage") or {}
check("chat: usage 含 tokens_per_sec", "tokens_per_sec" in cu)
if cu:
    print(f"       usage: prompt={cu.get('prompt_tokens')} completion={cu.get('completion_tokens')} "
          f"total={cu.get('total_tokens')} tok/s={cu.get('tokens_per_sec')}")
if c.get("answer"):
    print(f"       answer[:120]: {c['answer'][:120]}")

print("\n" + ("=" * 40))
if FAILED:
    print(f"{len(FAILED)} 項失敗: {FAILED}")
    sys.exit(1)
print("全部通過。")
