"""Small localhost-only screen for the additive Onkith assistant layer.

The authoritative transcript UI remains untouched.  This surface only mirrors
post-mask routing, the local conversation, TTS state, light state, and attached
device information.  It has no endpoint that accepts transcript text and never
writes a conversation to disk.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


HOST = "127.0.0.1"


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Onkith</title>
  <style>
    :root { color-scheme: dark; --ink:#ecf4ff; --muted:#8fa4bc;
      --line:#243244; --blue:#77b7ff; --green:#70e0ad; --amber:#ffd37a;
      --red:#ff8c99; --panel:rgba(14,24,37,.88); }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; color:var(--ink); font:16px/1.45
      system-ui,-apple-system,Segoe UI,sans-serif; background:
      radial-gradient(circle at 20% -10%,#163e68 0,transparent 42%),#07111d; }
    main { width:min(1050px,94vw); margin:auto; padding:34px 0 48px; }
    header { display:flex; align-items:end; justify-content:space-between;
      gap:20px; margin-bottom:22px; }
    h1 { margin:0; font-size:clamp(30px,5vw,56px); letter-spacing:-.04em; }
    .sub,.muted { color:var(--muted); }
    .grid { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(250px,.7fr);
      gap:18px; }
    .card { border:1px solid var(--line); background:var(--panel);
      box-shadow:0 18px 55px #0006; border-radius:20px; padding:20px; }
    .row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .pill { border:1px solid var(--line); border-radius:999px; padding:7px 11px;
      font:700 12px/1 ui-monospace,SFMono-Regular,monospace; letter-spacing:.06em;
      text-transform:uppercase; }
    .held { color:var(--green); }.sent { color:var(--amber); }.blocked { color:var(--red); }
    #turns { min-height:380px; max-height:62vh; overflow:auto; display:flex;
      flex-direction:column; gap:12px; padding-top:18px; }
    .turn { max-width:86%; border:1px solid var(--line); border-radius:16px;
      padding:12px 14px; white-space:pre-wrap; }
    .user { align-self:flex-end; background:#102a43; }
    .onkith { align-self:flex-start; background:#14261f; }
    .role { color:var(--muted); text-transform:uppercase; font:700 10px/1.2
      ui-monospace,SFMono-Regular,monospace; margin-bottom:5px; }
    h2 { margin:0 0 15px; font-size:15px; text-transform:uppercase;
      letter-spacing:.09em; color:var(--muted); }
    dl { margin:0; display:grid; grid-template-columns:1fr auto; gap:10px; }
    dt { color:var(--muted); } dd { margin:0; text-align:right; }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%;
      background:var(--green); margin-right:7px; box-shadow:0 0 14px var(--green); }
    #status { min-height:44px; margin-top:17px; color:var(--muted); }
    @media (max-width:760px) { .grid{grid-template-columns:1fr} header{align-items:start;
      flex-direction:column}.turn{max-width:94%} }
  </style>
</head>
<body><main>
  <header><div><h1>Onkith</h1><div class="sub">Whisper STT · local privacy · masked cloud reasoning</div></div>
    <div id="assistant" class="pill"><span class="dot"></span>idle</div></header>
  <div class="grid">
    <section class="card"><div class="row"><h2>Conversation</h2><div id="gate" class="pill held">held locally</div></div>
      <div id="turns"><div class="muted">Say “Onkith” followed by a question.</div></div></section>
    <aside id="deviceCard" class="card"><h2>Local device</h2><dl>
      <dt>STT</dt><dd>faster-whisper base.en</dd>
      <dt>Privacy</dt><dd>DeBERTa + structured PII</dd>
      <dt>LLM route</dt><dd>Groq, masked text only</dd>
      <dt>Speech</dt><dd id="speech">detecting…</dd>
      <dt>Microphone</dt><dd id="microphone">opening…</dd>
      <dt>Light</dt><dd id="light">not wired</dd>
    </dl><div id="status">Ordinary speech remains local.</div></aside>
  </div>
</main>
<script>
const q=s=>document.querySelector(s), esc=s=>String(s??'');
function render(s){
  const a=s.assistant||{}, g=s.privacy_gate||{}, d=s.devices||{}, l=s.light||{};
  q('#assistant').lastChild.textContent=' '+(a.state||'idle');
  q('#gate').textContent=g.label||'held locally'; q('#gate').className='pill '+(g.state||'held');
  q('#speech').textContent=d.speech||'none'; q('#microphone').textContent=d.microphone||'opening…';
  q('#deviceCard').style.display=d.open===false?'none':'block';
  q('#light').textContent=l.backend&&l.backend!=='none' ? ((l.on?'on ':'off ')+(l.brightness??0)+'%') : 'not wired';
  q('#status').textContent=(s.status&&s.status.message)||'Ordinary speech remains local.';
  const turns=((s.conversation||{}).turns)||[], box=q('#turns');
  if(!turns.length) return;
  box.replaceChildren(...turns.map(t=>{ const el=document.createElement('div');
    el.className='turn '+(t.role==='user'?'user':'onkith');
    const role=document.createElement('div'); role.className='role'; role.textContent=t.role==='user'?'you':'onkith';
    const body=document.createElement('div'); body.textContent=esc(t.text); el.append(role,body); return el; }));
  box.scrollTop=box.scrollHeight;
}
async function tick(){try{const r=await fetch('/state.json',{cache:'no-store'});if(r.ok)render(await r.json());}catch(e){}setTimeout(tick,300)}
tick();
</script></body></html>"""


class _SurfaceHandler(BaseHTTPRequestHandler):
    state_snapshot: Callable[[], dict[str, Any]]

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path.split("?", 1)[0] == "/state.json":
            body = json.dumps(
                self.state_snapshot(), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif self.path.split("?", 1)[0] in {"/", "/index.html"}:
            body = _PAGE.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class AssistantSurface:
    """Thread-safe event sink plus a read-only localhost web surface."""

    def __init__(
        self,
        *,
        port: int = 8765,
        open_window: bool = True,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.requested_port = int(port)
        self.open_window = bool(open_window)
        self.log = log
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "assistant": {"state": "idle"},
            "privacy_gate": {"state": "held", "label": "held locally"},
            "conversation": {"turns": []},
            "devices": {"open": True, "speech": "detecting…", "microphone": "opening…"},
            "light": {"backend": "none", "on": False, "brightness": 0},
            "panel": {"open": False},
            "status": {"message": "Ordinary speech remains local."},
            "revision": 0,
        }
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._browser: subprocess.Popen[Any] | None = None

    @property
    def url(self) -> str:
        port = self._server.server_port if self._server is not None else self.requested_port
        return f"http://{HOST}:{port}/"

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        safe_payload = copy.deepcopy(dict(payload or {}))
        with self._lock:
            current = self._state.get(topic)
            if isinstance(current, dict):
                current.update(safe_payload)
            else:
                self._state[topic] = safe_payload
            self._state["revision"] = int(self._state["revision"]) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def start(self) -> None:
        if self._server is not None:
            return
        handler = type("OnkithSurfaceHandler", (_SurfaceHandler,), {})
        handler.state_snapshot = staticmethod(self.snapshot)
        self._server = ThreadingHTTPServer((HOST, self.requested_port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="onkith-assistant-surface",
            daemon=False,
        )
        self._thread.start()
        self.log(f"Assistant UI: {self.url}")
        if self.open_window:
            self._open_browser()

    def _open_browser(self) -> None:
        requested = os.environ.get("ONKITH_UI_BROWSER", "").strip()
        executable = requested if requested and shutil.which(requested) else None
        if executable is None:
            executable = next(
                (
                    found
                    for name in ("chromium", "chromium-browser", "google-chrome")
                    if (found := shutil.which(name))
                ),
                None,
            )
        if executable is not None:
            try:
                self._browser = subprocess.Popen(
                    [executable, f"--app={self.url}", "--start-maximized"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except OSError as error:
                self.log(f"Could not open the assistant UI browser: {error}")
        try:
            webbrowser.open(self.url)
        except Exception as error:
            self.log(f"Open {self.url} manually ({error}).")

    def shutdown(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._browser is not None and self._browser.poll() is None:
            self._browser.terminate()
            try:
                self._browser.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._browser.kill()
        self._server = None
        self._thread = None
        self._browser = None
