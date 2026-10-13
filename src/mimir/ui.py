"""A local web UI over the same search the MCP server exposes.

Why a web UI rather than a native app: the engine is Python and MLX, so a Swift app would have
to bundle a Python runtime or reimplement the engine (DigUp reimplemented it, in ~15k lines).
A page served from this process calls the engine directly and needs no new dependency -
starlette and uvicorn already arrive with the MCP server.

Bound to 127.0.0.1 only, and it never reads a file's contents to serve a result: opening a hit
hands the path to macOS `open`. Search-as-you-type is the point, so the page debounces and
re-queries; instant search answers in ~11 ms, which is what makes that feel right.

    mimir ui            # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path

from .mcp_server import MimirSearch

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>mimir</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; --bg:#14161a; --panel:#1b1e24; --line:#2a2f38; --fg:#e6e8ec;
          --dim:#8b93a1; --accent:#7aa2f7; --mark:#4a3b1f; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
           padding:18px 24px 14px; }
  .row { display:flex; align-items:center; gap:12px; max-width:900px; margin:0 auto; }
  h1 { font-size:15px; font-weight:600; margin:0; letter-spacing:.02em; }
  h1 span { color:var(--dim); font-weight:400; }
  #q { flex:1; background:var(--panel); border:1px solid var(--line); border-radius:9px;
       color:var(--fg); font-size:15px; padding:11px 14px; outline:none; }
  #q:focus { border-color:var(--accent); }
  .modes { display:flex; gap:2px; background:var(--panel); border:1px solid var(--line);
           border-radius:9px; padding:2px; }
  .modes button { background:none; border:0; color:var(--dim); font:inherit; font-size:13px;
                  padding:6px 11px; border-radius:7px; cursor:pointer; }
  .modes button[aria-pressed="true"] { background:var(--accent); color:#10131a; font-weight:600; }
  .modes button:disabled { opacity:.4; cursor:default; }
  main { max-width:900px; margin:0 auto; padding:8px 24px 60px; }
  #status { color:var(--dim); font-size:12.5px; padding:12px 2px; }
  ul { list-style:none; margin:0; padding:0; }
  li { border:1px solid transparent; border-bottom-color:var(--line); padding:11px 12px;
       border-radius:8px; cursor:pointer; }
  li.sel { background:var(--panel); border-color:var(--line); }
  .top { display:flex; align-items:baseline; gap:9px; }
  .kind { font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim);
          border:1px solid var(--line); border-radius:5px; padding:1px 6px; flex:none; }
  .name { font-weight:600; }
  .loc { color:var(--accent); font-size:12.5px; flex:none; }
  .score { margin-left:auto; color:var(--dim); font-size:12px; flex:none; }
  .dir { color:var(--dim); font-size:12px; margin-top:2px; word-break:break-all; }
  .excerpt { color:#c7ccd6; font-size:13px; margin-top:5px; }
  mark { background:var(--mark); color:#ffd479; border-radius:3px; padding:0 2px; }
  .hint { color:var(--dim); font-size:12px; margin-top:10px; }
  kbd { background:var(--panel); border:1px solid var(--line); border-radius:4px;
        padding:0 4px; font-size:11px; }
</style></head>
<body>
<header><div class="row">
  <h1>mimir <span id="stats"></span></h1>
  <input id="q" placeholder="Describe what you're looking for, or paste an exact ID" autofocus
         autocomplete="off" spellcheck="false">
  <div class="modes">
    <button id="instant" aria-pressed="true">Instant</button>
    <button id="deep" aria-pressed="false" title="reranks candidates; slower, finds more">Deep</button>
  </div>
</div></header>
<main>
  <div id="status">Type to search. <span class="hint"><kbd>&uarr;</kbd><kbd>&darr;</kbd> move,
    <kbd>&crarr;</kbd> open, <kbd>&#8984;&crarr;</kbd> show in Finder</span></div>
  <ul id="results"></ul>
</main>
<script>
const $ = (id) => document.getElementById(id);
let deep = false, results = [], selected = -1, timer = null, seq = 0;

fetch('/api/info').then(r => r.json()).then(d => {
  $('stats').textContent = `${d.files} files, ${d.segments} segments`;
  if (!d.deep_available) { const b = $('deep'); b.disabled = true;
    b.title = 'run scripts/fetch_reranker.py to enable'; }
});

function esc(s) { return (s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function marked(s) { return esc(s).replace(/\\u00ab/g, '<mark>').replace(/\\u00bb/g, '</mark>'); }

function where(hit) {
  if (hit.loc === undefined) return '';
  const n = hit.loc;
  if (hit.kind === 'pdf') return `p.${Math.round(n)}`;
  const mm = Math.floor(n / 60), ss = String(Math.round(n % 60)).padStart(2, '0');
  return `${mm}:${ss}`;
}

function render() {
  const ul = $('results'); ul.innerHTML = '';
  results.forEach((hit, i) => {
    const li = document.createElement('li');
    if (i === selected) li.className = 'sel';
    const dir = hit.path.includes('/') ? hit.path.slice(0, hit.path.lastIndexOf('/')) : '';
    li.innerHTML =
      `<div class="top"><span class="kind">${esc(hit.kind)}</span>` +
      `<span class="name">${esc(hit.path.split('/').pop())}</span>` +
      (where(hit) ? `<span class="loc">${where(hit)}</span>` : '') +
      `<span class="score">${hit.score.toFixed(2)}</span></div>` +
      (dir ? `<div class="dir">${esc(dir)}</div>` : '') +
      (hit.excerpt ? `<div class="excerpt">${marked(hit.excerpt.trim().slice(0, 240))}</div>` : '');
    li.onclick = () => { selected = i; open_(false); };
    ul.appendChild(li);
  });
}

function open_(reveal) {
  if (selected < 0 || !results[selected]) return;
  const p = results[selected].absolute_path;
  fetch('/api/open?path=' + encodeURIComponent(p) + '&reveal=' + (reveal ? 1 : 0), {method:'POST'});
}

async function run() {
  const q = $('q').value.trim();
  if (!q) { results = []; selected = -1; render(); $('status').textContent = 'Type to search.'; return; }
  const mine = ++seq, t0 = performance.now();
  const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&deep=${deep ? 1 : 0}&limit=20`);
  const data = await r.json();
  if (mine !== seq) return;                       // a newer keystroke already answered
  const ms = Math.round(performance.now() - t0);
  results = data.results; selected = results.length ? 0 : -1;
  render();
  const mode = deep ? 'deep' : 'instant';
  $('status').textContent = results.length
    ? `${results.length} result${results.length === 1 ? '' : 's'} · ${mode} · ${ms} ms`
    : `nothing found · ${mode} · ${ms} ms`;
}

$('q').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(run, 120); });
function setMode(d) { deep = d; $('instant').setAttribute('aria-pressed', String(!d));
  $('deep').setAttribute('aria-pressed', String(d)); run(); }
$('instant').onclick = () => setMode(false);
$('deep').onclick = () => setMode(true);

document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown') { e.preventDefault();
    selected = Math.min(selected + 1, results.length - 1); render(); }
  else if (e.key === 'ArrowUp') { e.preventDefault();
    selected = Math.max(selected - 1, 0); render(); }
  else if (e.key === 'Enter' && document.activeElement === $('q')) { e.preventDefault();
    open_(e.metaKey); }
  else if (e.key === 'Escape') { $('q').value = ''; run(); $('q').focus(); }
});
</script></body></html>
"""


def create_app(search: MimirSearch):
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route

    # Handlers are plain `def` on purpose: Starlette runs them in a threadpool, so a blocking
    # model call cannot stall the event loop.
    def index(request):
        return HTMLResponse(PAGE)

    def info(request):
        return JSONResponse(search.info())

    def api_search(request):
        query = (request.query_params.get("q") or "").strip()
        deep = request.query_params.get("deep") in ("1", "true", "yes")
        limit = int(request.query_params.get("limit") or 20)
        if not query:
            return JSONResponse({"results": []})
        return JSONResponse({"results": search.search(query, limit=limit, deep=deep)})

    def api_open(request):
        ok = search.open_file(request.query_params.get("path") or "",
                              reveal=request.query_params.get("reveal") in ("1", "true"))
        return JSONResponse({"ok": ok})

    return Starlette(routes=[
        Route("/", index),
        Route("/api/info", info),
        Route("/api/search", api_search),
        Route("/api/open", api_open, methods=["POST"]),
    ])


def serve(index_dir: str | Path, port: int = 8765, open_browser: bool = True) -> int:
    import uvicorn

    search = MimirSearch(index_dir)
    url = f"http://127.0.0.1:{port}"
    print(f"mimir ui on {url}  (index: {search.index_dir})", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        # localhost only: this is not meant to be reachable from the network.
        uvicorn.run(create_app(search), host="127.0.0.1", port=port, log_level="warning")
    finally:
        search.close()
    return 0
