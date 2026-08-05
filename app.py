"""FastAPI 測試站：影片語意搜尋 + Cosmos-Reason2 對話。

頁面：
  /         搜尋頁：輸入文字 + top N(預設10) → 顯示 top N(縮圖/score/檔名)
            → 下方 Cosmos Reason「準備中」→ 完成後出現對話框，可續問。
            （同一次開啟的分頁保留對話；重新整理即開新對話。）
  /dbinfo   資料庫資訊頁：有哪些影片(封面)、向量資料庫、使用的 AI 模型。

啟動： bash serve_web.sh   （需先 bash serve_vlm.sh 開 Cosmos 服務）
"""
import json
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from embedder import ClipEmbedder
from store import VectorStore

app = FastAPI(title="Video Search + Cosmos-Reason2")

# 開放給所有來源呼叫（純 JSON API，不用 cookie，開 * 沒有 CSRF 疑慮）。
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- 影格靜態服務 ----
config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(config.FRAMES_DIR)), name="frames")

# ---- 原始影片靜態服務（給「播放影片」全螢幕 overlay 用） ----
config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=str(config.VIDEO_DIR)), name="videos")

# ---- 延遲載入的單例 ----
_embedder = None
_store = None
_vlm = None
# session_id -> {"cands":[...], "messages":[...], "explained":bool}
SESSIONS: dict[str, dict] = {}


def get_embedder():
    global _embedder
    if _embedder is None:
        # CPU：查詢只需單張 embedding（夠快），把 GPU 記憶體整個留給 Cosmos VLM。
        _embedder = ClipEmbedder(device="cpu")
    return _embedder


def get_store():
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def get_vlm():
    """可能因 llama-server 未啟動而丟例外。"""
    global _vlm
    if _vlm is None:
        from vlm import Vlm
        _vlm = Vlm()
    return _vlm


def public_base(request: Request) -> str:
    """求出對外可見的完整 origin（scheme+host），供組「完整 URL」用。

    本機直連時用 request 自己的 scheme/host；經 cloudflared 通道時，
    通道對外是 https 但對內轉給 uvicorn 是 http，所以優先看反代加的
    X-Forwarded-Proto / Cf-Visitor 標頭來還原真正的 https。
    """
    scheme = request.headers.get("x-forwarded-proto")
    if not scheme:
        cfv = request.headers.get("cf-visitor")
        if cfv:
            try:
                scheme = json.loads(cfv).get("scheme")
            except (json.JSONDecodeError, AttributeError):
                scheme = None
    scheme = scheme or request.url.scheme
    host = request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host}"


def frame_url(path: str, base: str = "") -> str:
    # DB 存的是絕對路徑（可能是別台機器的），只取「<影片資料夾>/<檔名>」對應到 /frames 掛載。
    p = Path(path)
    return f"{base}/frames/{p.parent.name}/{p.name}"


def video_url(video: str, base: str = "") -> str:
    return f"{base}/videos/{video}"


# ========== API ==========
class SearchReq(BaseModel):
    query: str
    top_n: int = 10


def _frame_idx(path: str, t_sec: float) -> int:
    m = re.search(r"frame_(\d+)", Path(path).name)
    if m:
        return int(m.group(1))
    return int(round(t_sec / config.FRAME_INTERVAL_SEC)) + 1


def _cluster_hits(hits: list[dict], gap: int, base: str = "") -> list[dict]:
    """同影片內、frame 編號相鄰(差<=gap)的命中串成一筆；每筆取分數最高者為代表。"""
    from collections import defaultdict
    by_video = defaultdict(list)
    for h in hits:
        m = h["meta"]
        by_video[m["video"]].append({
            "video": m["video"], "timecode": m["timecode"], "t_sec": m["t_sec"],
            "path": m["path"], "score": h["score"], "idx": _frame_idx(m["path"], m["t_sec"]),
        })
    clusters = []
    for items in by_video.values():
        items.sort(key=lambda x: x["idx"])
        cur = [items[0]]
        for it in items[1:]:
            if it["idx"] - cur[-1]["idx"] <= gap:   # 與前一張連續 → 同群
                cur.append(it)
            else:
                clusters.append(cur); cur = [it]
        clusters.append(cur)
    out = []
    for c in clusters:
        best = max(c, key=lambda x: x["score"])
        ts = sorted(c, key=lambda x: x["t_sec"])
        out.append({
            "video": best["video"], "timecode": best["timecode"], "t_sec": best["t_sec"],
            "path": best["path"], "score": round(best["score"], 3),
            "thumb": frame_url(best["path"], base), "filename": Path(best["path"]).name,
            "mp4": video_url(best["video"], base),
            "span": f"{ts[0]['timecode']}~{ts[-1]['timecode']}" if len(c) > 1 else ts[0]["timecode"],
            "merged": len(c),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


@app.post("/api/search")
def api_search(req: SearchReq, request: Request):
    store = get_store()
    if store.count() == 0:
        return JSONResponse({"error": "資料庫是空的"}, status_code=400)
    q_emb = get_embedder().embed_text(req.query)
    # 多撈一個 pool 再做「相鄰影格合併」，最後回 top_n 筆不同片段
    pool = min(store.count(), max(req.top_n * 8, 60))
    hits = store.query(q_emb, pool)
    clustered = _cluster_hits(hits, config.MERGE_FRAME_GAP, public_base(request))
    cands = clustered[:max(1, req.top_n)]
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {"query": req.query, "cands": cands, "messages": [], "explained": False}
    return {"session_id": sid, "results": cands}


class SidReq(BaseModel):
    session_id: str
    image_size: int = 640  # 送進 LLM 前圖片長邊縮到這個值以內（只往下縮、保持比例、不放大）


@app.post("/api/explain")
def api_explain(req: SidReq, request: Request):
    s = SESSIONS.get(req.session_id)
    if not s:
        return JSONResponse({"error": "session 不存在（可能已重新整理）"}, status_code=404)
    try:
        vlm = get_vlm()
    except Exception as e:
        return JSONResponse({"error": f"Cosmos 服務未啟動：{e}"}, status_code=503)
    result = vlm.explain(s["query"], s["cands"], image_size=req.image_size)
    s["messages"] = result["messages"]
    s["explained"] = True
    base = public_base(request)
    kept = [{"video": c["video"], "timecode": c["timecode"],
             "thumb": frame_url(c.get("hires", c["path"]), base), "mp4": video_url(c["video"], base)}
            for c in result["kept"]]
    caps = [{"video": c["video"], "timecode": c["timecode"], "caption": c.get("caption", "")}
            for c in result["candidates"]]
    return {"answer": result["answer"], "kept": kept, "captions": caps,
            "trace": result["trace"], "timings": result.get("timings", {}),
            "usage": result.get("usage", {})}


class ChatReq(BaseModel):
    session_id: str
    message: str


@app.post("/api/chat")
def api_chat(req: ChatReq):
    s = SESSIONS.get(req.session_id)
    if not s or not s.get("explained"):
        return JSONResponse({"error": "請先完成搜尋與 Cosmos 分析"}, status_code=400)
    try:
        vlm = get_vlm()
    except Exception as e:
        return JSONResponse({"error": f"Cosmos 服務未啟動：{e}"}, status_code=503)
    answer, trace, usage = vlm.ask(s["messages"], req.message)
    return {"answer": answer, "trace": trace, "usage": usage}


@app.get("/api/dbinfo")
def api_dbinfo(request: Request):
    store = get_store()
    r = store.collection.get(include=["metadatas"])
    metas = r["metadatas"]
    from collections import defaultdict
    per = defaultdict(list)
    for m in metas:
        per[m["video"]].append(m)
    base = public_base(request)
    videos = []
    for v, ms in sorted(per.items()):
        ms_sorted = sorted(ms, key=lambda x: x["t_sec"])
        videos.append({
            "video": v, "frames": len(ms),
            "duration": ms_sorted[-1]["timecode"],
            "cover": frame_url(ms_sorted[0]["path"], base),
            "mp4": video_url(v, base),
        })
    dim = len(store.collection.get(limit=1, include=["embeddings"])["embeddings"][0]) if metas else 0
    return {
        "videos": videos,
        "total_frames": len(metas),
        "vectordb": {"engine": "ChromaDB", "collection": config.COLLECTION_NAME,
                     "dim": dim, "metric": "cosine", "index": "HNSW"},
        "models": {
            "embedding": "Apple DFN5B-CLIP-ViT-H/14 @ 378px (open_clip, 1024維)",
            "vlm": "NVIDIA Cosmos-Reason2-8B (GGUF Q4_K_M, llama.cpp)",
        },
        "params": {"frame_interval_sec": config.FRAME_INTERVAL_SEC,
                   "vlm_img_max_px": config.VLM_IMG_MAX_PX},
    }


# ========== 頁面 ==========
@app.get("/", response_class=HTMLResponse)
def page_search():
    return SEARCH_HTML


@app.get("/dbinfo", response_class=HTMLResponse)
def page_dbinfo():
    return DBINFO_HTML


SEARCH_HTML = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>影片語意搜尋</title><style>
*{box-sizing:border-box} body{font-family:-apple-system,"PingFang TC",sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
a{color:#6cf} .wrap{max-width:1000px;margin:0 auto;padding:20px}
nav{display:flex;gap:16px;padding:12px 20px;background:#161922;border-bottom:1px solid #262b36}
h1{font-size:20px} .row{display:flex;gap:8px;margin:16px 0}
input,button{font-size:15px;padding:10px;border-radius:8px;border:1px solid #333;background:#1b1f29;color:#eee}
input[type=text]{flex:1} input[type=number]{width:90px} button{background:#2b6cff;border:0;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-top:12px}
.card{background:#161922;border:1px solid #262b36;border-radius:10px;overflow:hidden}
.card img{width:100%;height:120px;object-fit:cover;display:block;background:#000}
.card .meta{padding:8px;font-size:12px;line-height:1.5}
.card .actions{display:flex;gap:6px;padding:0 8px 8px}
.card .actions button{flex:1;padding:6px;font-size:12px;border-radius:6px}
.card .actions .btn-show{background:#2b6cff}
.card .actions .btn-video{background:#3a3f4d}
.score{color:#7fe08a;font-weight:600}
.section{margin-top:24px;background:#161922;border:1px solid #262b36;border-radius:12px;padding:16px}
.chat{display:flex;flex-direction:column;gap:10px;margin-top:10px}
.msg{padding:10px 12px;border-radius:12px;max-width:85%;white-space:pre-wrap;line-height:1.6}
.me{align-self:flex-end;background:#2b6cff}
.bot{align-self:flex-start;background:#232733;border:1px solid #313747}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #555;border-top-color:#6cf;border-radius:50%;animation:s 1s linear infinite;vertical-align:-2px}
@keyframes s{to{transform:rotate(360deg)}}
.muted{color:#8a93a3;font-size:13px} .tc{color:#cbd3e1}
.card img{cursor:zoom-in}
/* lightbox */
#lb{position:fixed;inset:0;background:rgba(0,0,0,.94);display:none;flex-direction:column;z-index:100}
#lb.on{display:flex}
#lb .top{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;color:#ddd;font-size:14px}
#lb .close{cursor:pointer;font-size:26px;line-height:1;padding:0 8px}
#lb .stage{flex:1;display:flex;align-items:center;justify-content:center;position:relative;min-height:0}
#lb .stage img{max-width:94vw;max-height:78vh;object-fit:contain}
#lb .arrow{position:absolute;top:50%;transform:translateY(-50%);font-size:44px;color:#fff;cursor:pointer;user-select:none;padding:10px 18px;opacity:.7}
#lb .arrow:hover{opacity:1} #lb .prev{left:6px} #lb .next{right:6px}
#lb .strip{display:flex;gap:6px;overflow-x:auto;padding:10px 12px;background:#0008}
#lb .strip img{height:56px;width:auto;border-radius:5px;cursor:pointer;opacity:.5;border:2px solid transparent}
#lb .strip img.sel{opacity:1;border-color:#2b6cff}
/* video overlay */
#vov{position:fixed;inset:0;background:rgba(0,0,0,.96);display:none;flex-direction:column;z-index:100}
#vov.on{display:flex}
#vov .top{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;color:#ddd;font-size:14px}
#vov .close{cursor:pointer;font-size:26px;line-height:1;padding:0 8px}
#vov .stage{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}
#vov .stage video{max-width:94vw;max-height:86vh}
</style></head><body>
<nav><b>🎬 影片搜尋</b><a href="/">搜尋</a><a href="/dbinfo">資料庫資訊</a></nav>
<div id="lb">
  <div class="top"><span id="lb-cap"></span><span class="close" onclick="closeLb()">✕</span></div>
  <div class="stage">
    <span class="arrow prev" onclick="navLb(-1)">‹</span>
    <img id="lb-img" src="">
    <span class="arrow next" onclick="navLb(1)">›</span>
  </div>
  <div class="strip" id="lb-strip"></div>
</div>
<div id="vov">
  <div class="top"><span id="vov-cap"></span><span class="close" onclick="closeVov()">✕</span></div>
  <div class="stage"><video id="vov-video" controls></video></div>
</div>
<div class="wrap">
<h1>輸入文字，語意搜尋影片畫面</h1>
<div class="row">
  <input id="q" type="text" placeholder="例如：守門員撲救 / 進球慶祝 / 亂丟垃圾" onkeydown="if(event.key==='Enter')doSearch()">
  <input id="topn" type="number" value="10" min="1" max="30" title="top N">
  <button id="btn" onclick="doSearch()">搜尋</button>
</div>
<div id="results"></div>
<div id="cosmos" class="section" style="display:none">
  <div id="cosmos-status"><span class="spin"></span> Cosmos Reason 準備中…（正在逐格解析、過濾、綜合，可能需數分鐘）</div>
  <div id="chat" class="chat"></div>
  <div class="row" id="chatbar" style="display:none">
    <input id="msg" type="text" placeholder="繼續問 Cosmos（例如：這是哪一隊？有進球嗎？）" onkeydown="if(event.key==='Enter')sendChat()">
    <button id="sendbtn" onclick="sendChat()">送出</button>
  </div>
</div>
</div>
<script>
let SID=null;  // 只存在記憶體，重新整理即消失 → 新對話
let RESULTS=[], LBI=0;
function openLb(i){LBI=i;var lb=document.getElementById('lb');lb.classList.add('on');renderLb()}
function closeLb(){document.getElementById('lb').classList.remove('on')}
function navLb(d){if(!RESULTS.length)return;LBI=(LBI+d+RESULTS.length)%RESULTS.length;renderLb()}
function renderLb(){var r=RESULTS[LBI];document.getElementById('lb-img').src=r.thumb;
  document.getElementById('lb-cap').textContent='#'+(LBI+1)+'  '+r.video+'  '+r.timecode+'  (score '+r.score+')';
  var s=RESULTS.map((x,i)=>'<img src="'+x.thumb+'" class="'+(i===LBI?'sel':'')+'" onclick="LBI='+i+';renderLb()">').join('');
  document.getElementById('lb-strip').innerHTML=s;
  var sel=document.querySelector('#lb-strip img.sel'); if(sel)sel.scrollIntoView({inline:'center',block:'nearest'});}
document.addEventListener('keydown',e=>{if(!document.getElementById('lb').classList.contains('on'))return;
  if(e.key==='Escape')closeLb();else if(e.key==='ArrowLeft')navLb(-1);else if(e.key==='ArrowRight')navLb(1)});
document.getElementById('lb').addEventListener('click',e=>{if(e.target.id==='lb')closeLb()});
function openVov(i){var r=RESULTS[i];var v=document.getElementById('vov-video');
  v.pause(); v.removeAttribute('src'); v.load();
  v.src='/videos/'+encodeURIComponent(r.video);
  v.onloadedmetadata=function(){v.currentTime=r.t_sec||0; v.play()};
  document.getElementById('vov-cap').textContent='#'+(i+1)+'  '+r.video+'  '+r.timecode;
  document.getElementById('vov').classList.add('on');}
function closeVov(){var v=document.getElementById('vov-video');v.pause();v.removeAttribute('src');v.load();
  document.getElementById('vov').classList.remove('on');}
document.addEventListener('keydown',e=>{if(!document.getElementById('vov').classList.contains('on'))return;
  if(e.key==='Escape')closeVov()});
document.getElementById('vov').addEventListener('click',e=>{if(e.target.id==='vov')closeVov()});
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json()}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function addMsg(cls,txt){const d=document.getElementById('chat');const el=document.createElement('div');el.className='msg '+cls;el.textContent=txt;d.appendChild(el);el.scrollIntoView();return el}
function fmtUsage(u){if(!u||!u.total_tokens)return '';
  return '🔢 '+u.total_tokens+' tokens（prompt '+u.prompt_tokens+' + completion '+u.completion_tokens+'） · ⚡ '+u.tokens_per_sec+' tok/s'}
function addUsage(u){const t=fmtUsage(u); if(!t)return;
  const d=document.getElementById('chat');const el=document.createElement('div');el.className='muted';el.style.alignSelf='flex-start';el.textContent=t;d.appendChild(el);el.scrollIntoView()}

async function doSearch(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const topn=parseInt(document.getElementById('topn').value||'10');
  document.getElementById('btn').disabled=true;
  document.getElementById('results').innerHTML='<p class="muted">搜尋中…</p>';
  document.getElementById('cosmos').style.display='none';
  document.getElementById('chat').innerHTML='';
  document.getElementById('chatbar').style.display='none';
  const d=await post('/api/search',{query:q,top_n:topn});
  document.getElementById('btn').disabled=false;
  if(d.error){document.getElementById('results').innerHTML='<p class="muted">'+esc(d.error)+'</p>';return}
  SID=d.session_id; RESULTS=d.results;
  let h='<h1>Top '+d.results.length+' 結果</h1><div class="grid">';
  d.results.forEach((r,i)=>{var merged=(r.merged&&r.merged>1)?(' <span style="color:#e0a34a">·合併'+r.merged+'張 '+r.span+'</span>'):'';
    h+='<div class="card"><img loading="lazy" onclick="openLb('+i+')" src="'+r.thumb+'">'
    +'<div class="meta"><div class="tc">#'+(i+1)+' '+esc(r.video)+' <b>'+r.timecode+'</b>'+merged+'</div>'
    +'<div class="score">score '+r.score+'</div><div class="muted">'+esc(r.filename)+'</div></div>'
    +'<div class="actions"><button class="btn-show" onclick="openLb('+i+')">顯示</button>'
    +'<button class="btn-video" onclick="openVov('+i+')">影片</button></div></div>'});
  h+='</div>';
  document.getElementById('results').innerHTML=h;
  // 啟動 Cosmos
  document.getElementById('cosmos').style.display='block';
  document.getElementById('cosmos-status').innerHTML='<span class="spin"></span> Cosmos Reason 準備中…（逐格解析／過濾／綜合，可能數分鐘）';
  const e=await post('/api/explain',{session_id:SID});
  if(e.error){document.getElementById('cosmos-status').innerHTML='⚠️ '+esc(e.error);return}
  let info='✅ Cosmos 分析完成';
  if(e.trace&&e.trace.length)info+='（期間呼叫 look_around '+e.trace.length+' 次看前後張）';
  if(e.usage&&e.usage.total_tokens)info+='<br><span class="muted">'+fmtUsage(e.usage)+'</span>';
  document.getElementById('cosmos-status').innerHTML=info;
  addMsg('bot',e.answer);
  addUsage(e.usage);
  document.getElementById('chatbar').style.display='flex';
}

async function sendChat(){
  const inp=document.getElementById('msg');const m=inp.value.trim();if(!m||!SID)return;
  inp.value='';addMsg('me',m);
  const wait=addMsg('bot','思考中…');
  document.getElementById('sendbtn').disabled=true;
  const d=await post('/api/chat',{session_id:SID,message:m});
  document.getElementById('sendbtn').disabled=false;
  wait.textContent = d.error? ('⚠️ '+d.error) : d.answer;
  if(!d.error)addUsage(d.usage);
}
</script></body></html>"""


DBINFO_HTML = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>資料庫資訊</title><style>
*{box-sizing:border-box} body{font-family:-apple-system,"PingFang TC",sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
a{color:#6cf} .wrap{max-width:1000px;margin:0 auto;padding:20px}
nav{display:flex;gap:16px;padding:12px 20px;background:#161922;border-bottom:1px solid #262b36}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px;margin-top:12px}
.card{background:#161922;border:1px solid #262b36;border-radius:10px;overflow:hidden}
.card img{width:100%;height:120px;object-fit:cover;display:block;background:#000}
.card .meta{padding:8px;font-size:13px;line-height:1.6}
.panel{background:#161922;border:1px solid #262b36;border-radius:12px;padding:16px;margin-top:16px}
.k{color:#8a93a3;display:inline-block;width:120px} b{color:#fff}
pre{background:#0d0f14;border:1px solid #262b36;border-radius:8px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.6}
code{color:#e0a34a} h3{margin-bottom:10px} .api h4{margin:16px 0 6px}
</style></head><body>
<nav><b>🎬 影片搜尋</b><a href="/">搜尋</a><a href="/dbinfo">資料庫資訊</a></nav>
<div class="wrap">
<h1>資料庫資訊</h1>
<div id="sys"></div>
<h2>影片清單</h2>
<div id="vids" class="grid"></div>
<div id="api" class="panel api"></div>
</div>
<script>
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function load(){
  const d=await (await fetch('/api/dbinfo')).json();
  document.getElementById('sys').innerHTML=
    '<div class="panel"><h3>向量資料庫</h3>'
    +'<div><span class="k">引擎</span><b>'+d.vectordb.engine+'</b></div>'
    +'<div><span class="k">Collection</span>'+d.vectordb.collection+'</div>'
    +'<div><span class="k">向量維度</span><b>'+d.vectordb.dim+'</b></div>'
    +'<div><span class="k">距離度量</span>'+d.vectordb.metric+' / '+d.vectordb.index+'</div>'
    +'<div><span class="k">總向量數</span><b>'+d.total_frames+'</b>（'+d.videos.length+' 支影片）</div></div>'
    +'<div class="panel"><h3>AI 模型</h3>'
    +'<div><span class="k">Embedding</span>'+esc(d.models.embedding)+'</div>'
    +'<div><span class="k">VLM</span>'+esc(d.models.vlm)+'</div>'
    +'<div><span class="k">抽格間隔</span>每 '+d.params.frame_interval_sec+' 秒；送 VLM 解析度上限 '+d.params.vlm_img_max_px+'px</div></div>';
  let h='';
  d.videos.forEach(v=>{h+='<div class="card"><img loading="lazy" src="'+v.cover+'">'
    +'<div class="meta"><b>'+esc(v.video)+'</b><br>影格 '+v.frames+' 張<br>長度 ~'+v.duration
    +'<br><a href="'+v.mp4+'" target="_blank">原始 mp4 ↗</a></div></div>'});
  document.getElementById('vids').innerHTML=h;

  const origin=location.origin;
  document.getElementById('api').innerHTML=
    '<h3>API 呼叫方式</h3>'
    +'<p class="k" style="width:auto">開放 CORS 給所有來源；只需這兩支 API 就能完成「搜尋 → Cosmos 解讀」全流程。'
    +'回傳的 thumb/mp4 欄位都是可直接打開的完整 URL。</p>'
    +'<div class="api">'
    +'<h4>1) POST /api/search — 語意搜尋影格</h4>'
    +'<pre>curl -X POST '+origin+'/api/search \\\n'
    +'  -H "Content-Type: application/json" \\\n'
    +'  -d \\'{"query": "有沒有人在亂丟垃圾", "top_n": 10}\\'</pre>'
    +'<p class="k" style="width:auto">回應：<code>{session_id, results: [{video, timecode, t_sec, score, thumb, mp4, filename, span, merged}]}</code>'
    +'　— <code>session_id</code> 要留著給下一步 /api/explain 用。</p>'
    +'<h4>2) POST /api/explain — 用 Cosmos Reason 解讀剛才的搜尋結果</h4>'
    +'<pre>curl -X POST '+origin+'/api/explain \\\n'
    +'  -H "Content-Type: application/json" \\\n'
    +'  -d \\'{"session_id": "上一步拿到的 session_id", "image_size": 640}\\'</pre>'
    +'<p class="k" style="width:auto"><code>image_size</code>（可選，預設 640）：送進 LLM 前，每張圖片長邊縮到這個'
    +'像素以內再送進去，加快推論；只會往下縮、保持長寬比、不會放大原圖。傳 0 或負數則不限制。</p>'
    +'<p class="k" style="width:auto">回應：<code>{answer, kept: [{video, timecode, thumb, mp4}], captions, trace, timings, '
    +'usage: {prompt_tokens, completion_tokens, total_tokens, tokens_per_sec}}</code>'
    +'　— <code>answer</code> 是繁中總結；<code>usage</code> 是這次呼叫 Cosmos 用掉的 token 數與生成速度。'
    +'此步驟需要 VLM 服務（serve_vlm.sh）已啟動，可能需數十秒到數分鐘。</p>'
    +'</div>';
}
load();
</script></body></html>"""
