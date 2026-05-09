#!/usr/bin/env python3
import html
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


RUNS_DIR = Path(os.environ.get("AIC_RUNS_DIR", "/workspace/aic_runs"))
RESULTS_DIR = Path(os.environ.get("AIC_RESULTS_DIR", "/workspace/aic_results"))
HOST = os.environ.get("AIC_WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("AIC_WEB_PORT", "8080"))

RUNS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROCESSES = {}
LOCK = threading.Lock()

SIMULATION_KINDS = {"eval_gui", "visual_only", "eval_headless", "warehouse_visual", "warehouse_policy"}
SINGLETON_KINDS = SIMULATION_KINDS | {"wave_policy"}


COMMANDS = {
    "eval_gui": [
        "/entrypoint.sh",
        "gazebo_gui:=true",
        "launch_rviz:=true",
        "ground_truth:=false",
        "start_aic_engine:=true",
        "shutdown_on_aic_engine_exit:=true",
        "model_discovery_timeout_seconds:=600",
    ],
    "visual_only": [
        "/entrypoint.sh",
        "gazebo_gui:=true",
        "launch_rviz:=true",
        "ground_truth:=false",
        "start_aic_engine:=false",
        "shutdown_on_aic_engine_exit:=false",
        "model_discovery_timeout_seconds:=600",
    ],
    "warehouse_visual": [
        "python3",
        "/opt/aic_web/warehouse_visual.py",
    ],
    "warehouse_policy": [
        "python3",
        "/opt/aic_web/warehouse_policy.py",
    ],
    "eval_headless": [
        "/entrypoint.sh",
        "gazebo_gui:=false",
        "launch_rviz:=false",
        "ground_truth:=false",
        "start_aic_engine:=true",
        "shutdown_on_aic_engine_exit:=true",
        "model_discovery_timeout_seconds:=600",
    ],
    "wave_policy": [
        "/bin/bash",
        "-lc",
        "source /ws_aic/install/setup.bash && "
        "ros2 run aic_model aic_model --ros-args "
        "-p use_sim_time:=true "
        "-p policy:=aic_example_policies.ros.WaveArm",
    ],
    "warehouse_demo": [
        "python3",
        "/opt/aic_web/warehouse_sim.py",
        "--sleep",
        "0.04",
    ],
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def meta_path(run_id):
    return RUNS_DIR / f"{run_id}.json"


def log_path(run_id):
    return RUNS_DIR / f"{run_id}.log"


def read_meta(run_id):
    path = meta_path(run_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_meta(meta):
    meta_path(meta["id"]).write_text(json.dumps(meta, indent=2, sort_keys=True))


def list_runs():
    runs = []
    for path in sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = json.loads(path.read_text())
        proc = PROCESSES.get(meta["id"])
        if proc and proc.poll() is None:
            meta["status"] = "running"
        elif proc:
            meta["status"] = "finished" if proc.returncode == 0 else "failed"
            meta["returncode"] = proc.returncode
            PROCESSES.pop(meta["id"], None)
            write_meta(meta)
        runs.append(meta)
    return runs


def append_log(run_id, line):
    with log_path(run_id).open("a", encoding="utf-8") as f:
        f.write(line)


def stop_process(run_id, reason):
    proc = PROCESSES.get(run_id)
    meta = read_meta(run_id)
    if not meta or not proc or proc.poll() is not None:
        return
    append_log(run_id, f"\n[web] stop requested: {reason}\n")
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    time.sleep(2)
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    meta["status"] = "stopping"
    write_meta(meta)


def stop_conflicting_runs(kind):
    if kind in SIMULATION_KINDS:
        for run_id, proc in list(PROCESSES.items()):
            if proc.poll() is None:
                meta = read_meta(run_id)
                if meta and meta.get("kind") in SINGLETON_KINDS:
                    stop_process(run_id, f"new {kind} run started")
    elif kind == "wave_policy":
        for run_id, proc in list(PROCESSES.items()):
            if proc.poll() is None:
                meta = read_meta(run_id)
                if meta and meta.get("kind") == "wave_policy":
                    stop_process(run_id, "new wave_policy run started")


def start_run(kind, command=None):
    if kind in COMMANDS:
        cmd = COMMANDS[kind]
    elif kind == "custom":
        if not command or not command.strip():
            raise ValueError("custom command is empty")
        cmd = ["/bin/bash", "-lc", command]
    else:
        raise ValueError(f"unknown run kind: {kind}")

    stop_conflicting_runs(kind)

    run_id = uuid.uuid4().hex[:12]
    meta = {
        "id": run_id,
        "kind": kind,
        "command": cmd if kind != "custom" else command,
        "created_at": now(),
        "status": "running",
        "returncode": None,
    }
    write_meta(meta)
    append_log(run_id, f"$ {cmd if kind != 'custom' else command}\n\n")

    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env.setdefault("AIC_RESULTS_DIR", str(RESULTS_DIR))
    env.setdefault("RMW_IMPLEMENTATION", "rmw_zenoh_cpp")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        preexec_fn=os.setsid,
    )
    PROCESSES[run_id] = proc

    def pump():
        assert proc.stdout is not None
        for line in proc.stdout:
            append_log(run_id, line)
        rc = proc.wait()
        with LOCK:
            current = read_meta(run_id) or meta
            current["status"] = "finished" if rc == 0 else "failed"
            current["returncode"] = rc
            current["finished_at"] = now()
            write_meta(current)

    threading.Thread(target=pump, daemon=True).start()
    return meta


def stop_run(run_id):
    proc = PROCESSES.get(run_id)
    meta = read_meta(run_id)
    if not meta:
        raise KeyError(run_id)
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        time.sleep(2)
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        meta["status"] = "stopping"
        write_meta(meta)
        append_log(run_id, "\n[web] stop requested\n")
    return meta


def read_log(run_id, offset):
    path = log_path(run_id)
    if not path.exists():
        raise KeyError(run_id)
    data = path.read_text(errors="replace")
    return {"offset": len(data), "chunk": data[offset:]}


def result_items():
    items = []
    if RESULTS_DIR.exists():
        for path in sorted(RESULTS_DIR.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
            if path.is_file():
                items.append(
                    {
                        "path": str(path.relative_to(RESULTS_DIR)),
                        "size": path.stat().st_size,
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                    }
                )
    return items


INDEX = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AIC Remote Runner</title>
  <style>
    :root { color-scheme: dark; --bg:#101214; --panel:#171b1f; --line:#2b3238; --text:#e8edf2; --muted:#98a3ad; --accent:#4fb38a; --bad:#e06464; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; border-bottom:1px solid var(--line); background:#0d0f11; }
    h1 { margin:0; font-size:18px; font-weight:650; }
    main { display:grid; grid-template-columns: 320px minmax(0, 1fr); min-height: calc(100vh - 55px); }
    aside { border-right:1px solid var(--line); padding:12px; overflow:auto; }
    section { padding:12px; min-width:0; }
    button, input, textarea, select { font: inherit; }
    button { border:1px solid var(--line); background:#22282e; color:var(--text); padding:9px 11px; border-radius:6px; cursor:pointer; }
    button:hover { border-color:#4c5964; }
    button.primary { background:#1e5b46; border-color:#257356; }
    button.danger { background:#5a2424; border-color:#743030; }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .panel { border:1px solid var(--line); border-radius:8px; padding:12px; background:var(--panel); margin-bottom:14px; }
    .panel h2 { margin:0 0 10px; font-size:14px; color:#d7dde3; }
    textarea { width:100%; min-height:88px; resize:vertical; border-radius:6px; border:1px solid var(--line); background:#0c0e10; color:var(--text); padding:10px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    td, th { padding:8px 6px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    tr.selected { background:#20272d; }
    .muted { color:var(--muted); }
    .status { display:inline-block; padding:2px 7px; border-radius:999px; background:#29313a; font-size:12px; }
    .status.running { background:#174b39; color:#aff2d5; }
    .status.failed { background:#502020; color:#ffc4c4; }
    .display-panel { padding:10px; }
    pre { margin:0; min-height:260px; max-height:34vh; overflow:auto; background:#070809; border:1px solid var(--line); border-radius:8px; padding:12px; white-space:pre-wrap; overflow-wrap:anywhere; font-size:12px; line-height:1.45; }
    .remote-placeholder { min-height:260px; border:1px solid var(--line); border-radius:8px; background:#050607; display:flex; align-items:center; justify-content:center; padding:20px; text-align:center; }
    .remote-frame { width:100%; height:72vh; min-height:620px; border:1px solid var(--line); border-radius:8px; background:#050607; }
    a { color:#8bd6ff; text-decoration:none; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } aside { border-right:0; border-bottom:1px solid var(--line); } }
  </style>
</head>
<body>
  <header>
    <h1>AIC Remote Runner</h1>
    <div class="row"><a href="/vnc/" target="_blank">Open noVNC</a><span class="muted">UI :8080, noVNC :6080</span></div>
  </header>
  <main>
    <aside>
      <div class="panel">
        <h2>Launch</h2>
        <div class="row">
          <button class="primary" onclick="startRun('eval_gui')">Start Gazebo/RViz + engine</button>
          <button onclick="startRun('visual_only')">Start Gazebo/RViz only</button>
          <button onclick="startRun('warehouse_visual')">Start visual warehouse robot</button>
          <button onclick="startRun('warehouse_policy')">Run warehouse policy</button>
          <button onclick="startRun('eval_headless')">Start headless</button>
          <button onclick="startRun('wave_policy')">Run WaveArm</button>
          <button onclick="startRun('warehouse_demo')">Run text warehouse demo</button>
        </div>
      </div>
      <div class="panel">
        <h2>Custom command</h2>
        <textarea id="custom">source /ws_aic/install/setup.bash && ros2 topic list</textarea>
        <div style="height:8px"></div>
        <button onclick="startCustom()">Run command</button>
      </div>
      <div class="panel">
        <h2>Runs</h2>
        <table>
          <thead><tr><th>Run</th><th>Status</th><th></th></tr></thead>
          <tbody id="runs"></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Results</h2>
        <table>
          <thead><tr><th>Path</th><th>Size</th></tr></thead>
          <tbody id="results"></tbody>
        </table>
      </div>
    </aside>
    <section>
      <div class="panel display-panel">
        <h2>Remote display</h2>
        <div id="remoteDisplay" class="remote-placeholder">
          <div>
            <div class="muted" style="margin-bottom:12px">For best performance open noVNC in a separate tab.</div>
            <div class="row" style="justify-content:center">
              <a href="http://localhost:6080/vnc.html?autoconnect=1&resize=scale" target="_blank">Open direct noVNC</a>
              <button onclick="loadEmbeddedVnc()">Load here</button>
            </div>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="row" style="justify-content:space-between;margin-bottom:10px">
          <h2 style="margin:0">Log</h2>
          <button class="danger" onclick="stopSelected()">Stop selected</button>
        </div>
        <pre id="log"></pre>
      </div>
    </section>
  </main>
  <script>
    let selected = null;
    let offset = 0;
    async function api(path, opts) {
      const r = await fetch(path, opts);
      if (!r.ok) throw new Error(await r.text());
      return await r.json();
    }
    async function startRun(kind) {
      const run = await api('/api/runs', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({kind})});
      selected = run.id; offset = 0; await refresh();
    }
    async function startCustom() {
      const command = document.getElementById('custom').value;
      const run = await api('/api/runs', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({kind:'custom', command})});
      selected = run.id; offset = 0; await refresh();
    }
    async function stopSelected() {
      if (!selected) return;
      await api('/api/runs/' + selected + '/stop', {method:'POST'});
      await refresh();
    }
    function loadEmbeddedVnc() {
      document.getElementById('remoteDisplay').outerHTML = '<iframe class="remote-frame" src="/vnc/vnc.html?autoconnect=1&resize=scale&password="></iframe>';
    }
    async function refresh() {
      const data = await api('/api/runs');
      const runs = document.getElementById('runs');
      runs.innerHTML = data.runs.map(r => `<tr class="${r.id===selected?'selected':''}" onclick="selectRun('${r.id}')"><td><strong>${r.kind}</strong><br><span class="muted">${r.id}<br>${r.created_at}</span></td><td><span class="status ${r.status}">${r.status}</span></td><td>${r.returncode ?? ''}</td></tr>`).join('');
      if (!selected && data.runs.length) { selected = data.runs[0].id; offset = 0; }
      const res = await api('/api/results');
      document.getElementById('results').innerHTML = res.items.map(i => `<tr><td>${escapeHtml(i.path)}<br><span class="muted">${i.updated}</span></td><td>${i.size}</td></tr>`).join('');
    }
    async function selectRun(id) {
      selected = id; offset = 0; document.getElementById('log').textContent = ''; await refresh(); await pollLog();
    }
    async function pollLog() {
      if (!selected) return;
      const data = await api('/api/runs/' + selected + '/log?offset=' + offset);
      offset = data.offset;
      const log = document.getElementById('log');
      if (data.chunk) {
        log.textContent += data.chunk;
        log.scrollTop = log.scrollHeight;
      }
    }
    function escapeHtml(s) {
      return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    setInterval(refresh, 3000);
    setInterval(pollLog, 1000);
    refresh().then(pollLog);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text, status=HTTPStatus.OK, content_type="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX, content_type="text/html; charset=utf-8")
        elif parsed.path == "/api/runs":
            with LOCK:
                self.send_json({"runs": list_runs()})
        elif parsed.path.startswith("/api/runs/") and parsed.path.endswith("/log"):
            parts = parsed.path.split("/")
            run_id = parts[3]
            offset = int(parse_qs(parsed.query).get("offset", ["0"])[0])
            try:
                self.send_json(read_log(run_id, offset))
            except KeyError:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        elif parsed.path == "/api/results":
            self.send_json({"items": result_items()})
        elif parsed.path == "/vnc/" or parsed.path.startswith("/vnc/"):
            target = "http://localhost:6080" + parsed.path.removeprefix("/vnc") + (("?" + parsed.query) if parsed.query else "")
            self.send_response(HTTPStatus.FOUND)
            self.send_header("location", target)
            self.end_headers()
        else:
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/runs":
                body = self.read_body()
                with LOCK:
                    run = start_run(body.get("kind", ""), body.get("command"))
                self.send_json(run, HTTPStatus.CREATED)
            elif parsed.path.startswith("/api/runs/") and parsed.path.endswith("/stop"):
                run_id = parsed.path.split("/")[3]
                with LOCK:
                    self.send_json(stop_run(run_id))
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt, *args):
        safe = html.escape(fmt % args)
        print(f"[web] {self.address_string()} {safe}", flush=True)


if __name__ == "__main__":
    print(f"AIC web runner listening on http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
