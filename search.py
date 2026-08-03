"""Search pipeline：文字 query -> DFN5B text embedding -> ChromaDB 檢索
   -> (多撈候選 + 門檻過濾) -> Cosmos-Reason2-8B 過濾不相關 + 綜合總結
   -> 過程中 Cosmos 可 look_around 看前後張影格
   -> 最後進入互動 session，可對 Cosmos 續問。

各階段耗時會即時印出。

用法:
    python search.py "有人在亂丟垃圾嗎"
    python search.py "a red car" --candidates 15 --threshold 0.18
    python search.py "yellow taxi" --no-vlm     # 只做向量檢索
    python search.py "goal" --no-chat           # 不進入續問對話
"""
import argparse
import time

import config
from embedder import ClipEmbedder
from store import VectorStore


def _fmt(sec: float) -> str:
    return f"{sec:.1f}s" if sec < 60 else f"{int(sec // 60)}m{int(sec % 60)}s"


def search(query: str, candidates: int, threshold: float, use_vlm: bool, use_chat: bool):
    timings = {}
    store = VectorStore()
    if store.count() == 0:
        print("資料庫是空的，請先跑：python ingest.py <video.mp4> --reset")
        return

    t = time.time()
    embedder = ClipEmbedder()
    timings["載入embedding模型"] = time.time() - t

    t = time.time()
    q_emb = embedder.embed_text(query)
    timings["query嵌入"] = time.time() - t

    t = time.time()
    hits = store.query(q_emb, candidates)
    timings["向量檢索"] = time.time() - t

    cands = []
    for h in hits:
        if h["score"] < threshold:
            continue
        m = h["meta"]
        cands.append({
            "video": m["video"], "timecode": m["timecode"],
            "t_sec": m["t_sec"], "path": m["path"], "score": h["score"],
        })

    print(f"\n=== RAG 候選（query: {query!r}，門檻 {threshold}，取回 {len(cands)} 張）===")
    for i, c in enumerate(cands, 1):
        print(f"{i}. [{c['video']}] {c['timecode']}  score={c['score']:.3f}")
        print(f"     {c['path']}")

    if not cands:
        print("沒有超過門檻的候選（可降低 --threshold）。")
        _print_timings(timings)
        return
    if not use_vlm:
        _print_timings(timings)
        return

    from vlm import Vlm
    vlm = Vlm()
    result = vlm.explain(query, cands)
    timings.update(result["timings"])  # caption / filter / summarize

    print("\n=== Dense Captions（逐格描述）===")
    for c in result["candidates"]:
        print(f"[{c['video']} {c['timecode']}] {c.get('caption','')}")
        print(f"     {c['path']}")

    kept = result["kept"]
    print(f"\n=== Cosmos 過濾後保留 {len(kept)} 張（相關影格，原始畫質擷取）===")
    for c in kept:
        print(f"[{c['video']} {c['timecode']}]  {c.get('hires', c['path'])}")

    if result["trace"]:
        print("\n=== look_around 呼叫紀錄 ===")
        for tr in result["trace"]:
            print(f"  {tr['tool']}({tr['args']}) -> {tr['result']}")

    print("\n=== Cosmos-Reason2 綜合解釋 ===")
    print(result["answer"])

    _print_timings(timings)

    # ---------- 互動續問 session ----------
    if use_chat and result.get("messages"):
        _chat_loop(vlm, result["messages"])


def _print_timings(timings: dict):
    print("\n=== 各階段耗時 ===")
    total = 0.0
    for k, v in timings.items():
        total += v
        print(f"  {k:<16} {_fmt(v)}")
    print(f"  {'合計':<16} {_fmt(total)}")


def _chat_loop(vlm, messages):
    print("\n" + "=" * 50)
    print("進入對話模式：可對 Cosmos-Reason2 續問（它仍記得剛剛的影格，也能再 look_around）。")
    print("輸入問題後 Enter；直接 Enter 或輸入 exit/quit 離開。")
    print("=" * 50)
    while True:
        try:
            q = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(離開對話)")
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            print("(離開對話)")
            break
        t = time.time()
        answer, trace = vlm.ask(messages, q)
        dt = time.time() - t
        if trace:
            for tr in trace:
                print(f"  [look_around] {tr['args']} -> {tr['result']}")
        print(f"Cosmos > {answer}")
        print(f"  (耗時 {_fmt(dt)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--candidates", type=int, default=config.SEARCH_CANDIDATES,
                    help="RAG 先多撈幾張候選")
    ap.add_argument("--threshold", type=float, default=config.SCORE_THRESHOLD,
                    help="相似度門檻")
    ap.add_argument("--no-vlm", action="store_true", help="只做向量檢索，不跑 VLM")
    ap.add_argument("--no-chat", action="store_true", help="不進入互動續問對話")
    args = ap.parse_args()
    search(args.query, args.candidates, args.threshold,
           use_vlm=not args.no_vlm, use_chat=not args.no_chat)


if __name__ == "__main__":
    main()
