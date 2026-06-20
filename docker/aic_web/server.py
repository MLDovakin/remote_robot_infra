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
NAV2_STATE_FILE = RUNS_DIR / "nav2_state.json"
NAV2_TASK_FILE = RUNS_DIR / "nav2_task.json"
LIDAR_STATE_FILE = RUNS_DIR / "lidar_random_state.json"
LIDAR_TASK_FILE = RUNS_DIR / "lidar_random_task.json"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROCESSES = {}
LOCK = threading.Lock()

SIMULATION_KINDS = {"eval_gui", "visual_only", "eval_headless", "warehouse_visual", "warehouse_policy", "warehouse_nav2", "lidar_random", "vla_gazebo"}
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
    "warehouse_nav2": [
        "python3",
        "/opt/aic_web/warehouse_nav2_mode.py",
    ],
    "lidar_random": [
        "python3",
        "/opt/aic_web/warehouse_lidar_random_mode.py",
    ],
    "vla_gazebo": [
        "python3",
        "/opt/aic_web/vla_gazebo_bridge.py",
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
        try:
            meta = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "id" not in meta or "kind" not in meta:
            continue
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


def nav2_default_map():
    return {
        "origin": {"x": -11.0, "y": -8.0},
        "width_m": 22.0,
        "height_m": 16.0,
        "resolution": 0.1,
        "obstacles": [
            {"name": "front_wall", "x1": -11.0, "y1": -8.0, "x2": 11.0, "y2": -7.85},
            {"name": "back_wall", "x1": -11.0, "y1": 7.85, "x2": 11.0, "y2": 8.0},
            {"name": "left_wall", "x1": -11.0, "y1": -8.0, "x2": -10.85, "y2": 8.0},
            {"name": "right_wall", "x1": 10.85, "y1": -8.0, "x2": 11.0, "y2": 8.0},
            {"name": "storage_r", "x1": 4.9, "y1": -4.5, "x2": 7.1, "y2": -3.5},
            {"name": "storage_g", "x1": 4.9, "y1": -0.5, "x2": 7.1, "y2": 0.5},
            {"name": "storage_b", "x1": 4.9, "y1": 3.5, "x2": 7.1, "y2": 4.5},
            {"name": "rack_1", "x1": -3.5, "y1": -4.35, "x2": 1.5, "y2": -3.65},
            {"name": "rack_2", "x1": -3.5, "y1": -0.35, "x2": 1.5, "y2": 0.35},
            {"name": "rack_3", "x1": -3.5, "y1": 3.65, "x2": 1.5, "y2": 4.35},
        ],
        "dispatches": [
            {"name": "DispatchA", "x": -8.0, "y": -4.0, "w": 2.4, "h": 1.4},
            {"name": "DispatchB", "x": -8.0, "y": 4.0, "w": 2.4, "h": 1.4},
        ],
        "products": {
            "ProductR": {"storage": "StorageR", "slot": {"x": 6.55, "y": -4.2, "z": 1.02}, "pickup": {"x": 6.55, "y": -5.35, "yaw": 1.5708}},
            "ProductG": {"storage": "StorageG", "slot": {"x": 6.55, "y": -0.2, "z": 1.02}, "pickup": {"x": 6.55, "y": -1.35, "yaw": 1.5708}},
            "ProductB": {"storage": "StorageB", "slot": {"x": 6.55, "y": 3.8, "z": 1.02}, "pickup": {"x": 6.55, "y": 2.65, "yaw": 1.5708}},
        },
        "nav2_files": {
            "map": "/workspace/aic_results/nav2_warehouse_map/warehouse_map.yaml",
            "keepout": "/workspace/aic_results/nav2_warehouse_map/keepout_mask.yaml",
            "params": "/workspace/aic_results/nav2_warehouse_map/nav2_params.yaml",
            "behavior_tree": "/workspace/aic_results/nav2_warehouse_map/nav_to_pose_and_pause_near_goal_obstacle.xml",
        },
    }


def nav2_state():
    if NAV2_STATE_FILE.exists():
        try:
            return json.loads(NAV2_STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "status": "not_started",
        "message": "Start Nav2 map task mode first",
        "robot": {"x": -7.0, "y": -5.0, "yaw": 0.0},
        "task": None,
        "path": [],
        "cargo": None,
        "map": nav2_default_map(),
        "updated_at": None,
    }


def write_nav2_task(task):
    product = task.get("product", "ProductR")
    if product not in nav2_default_map()["products"]:
        raise ValueError(f"unknown product: {product}")
    drop = task.get("drop")
    if not isinstance(drop, dict):
        raise ValueError("drop point is required")
    x = float(drop["x"])
    y = float(drop["y"])
    keepouts = []
    for item in task.get("keepouts", []):
        keepouts.append({"x1": float(item["x1"]), "y1": float(item["y1"]), "x2": float(item["x2"]), "y2": float(item["y2"])})
    payload = {"product": product, "drop": {"x": x, "y": y, "yaw": float(drop.get("yaw", 0.0))}, "keepouts": keepouts, "created_at": now()}
    NAV2_TASK_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def lidar_random_state():
    if LIDAR_STATE_FILE.exists():
        try:
            return json.loads(LIDAR_STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "status": "not_started",
        "message": "Start Lidar Random Map first",
        "origin": {"x": -13.0, "y": -9.0},
        "width_m": 26.0,
        "height_m": 18.0,
        "resolution": 0.12,
        "grid_w": 216,
        "grid_h": 150,
        "known": [],
        "coverage": 0.0,
        "robot": {"x": -10.6, "y": -7.0, "yaw": 0.0},
        "true_obstacles": [],
        "products": {},
        "pickup_status": {"selected_product": None, "selected": None, "products": {}},
        "path": [],
        "task": None,
        "lidar": [],
        "updated_at": None,
    }


def write_lidar_random_task(task):
    state = lidar_random_state()
    products = state.get("products") or {"ProductR": {}, "ProductG": {}, "ProductB": {}}
    product = task.get("product") or next(iter(products))
    if product not in products:
        raise ValueError(f"unknown product: {product}")
    product_status = ((state.get("pickup_status") or {}).get("products") or {}).get(product, {})
    if state.get("status") not in {"mapped", "done"} and not product_status.get("discovered"):
        raise ValueError("TaskGoal is locked until the selected shelf is detected by lidar")
    drop = task.get("drop")
    if not isinstance(drop, dict):
        raise ValueError("drop point is required")
    keepouts = []
    for item in task.get("keepouts", []):
        keepouts.append({"x1": float(item["x1"]), "y1": float(item["y1"]), "x2": float(item["x2"]), "y2": float(item["y2"])})
    payload = {
        "product": product,
        "drop": {"x": float(drop["x"]), "y": float(drop["y"]), "yaw": float(drop.get("yaw", 0.0))},
        "keepouts": keepouts,
        "created_at": now(),
    }
    LIDAR_TASK_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


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
    <div class="row"><a href="/nav2" target="_blank">Open task map</a><a href="/lidar-random" target="_blank">Open lidar map</a><a href="/vnc/" target="_blank">Open noVNC</a><span class="muted">UI :8080, noVNC :6080</span></div>
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
          <button onclick="startRun('warehouse_nav2')">Start Nav2 map task mode</button>
          <button onclick="startRun('lidar_random')">Start Lidar Random Map</button>
          <button onclick="startRun('vla_gazebo')">Start VLA Gazebo bridge</button>
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


NAV2_INDEX = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Warehouse Nav2 Task Map</title>
  <style>
    :root { color-scheme: dark; --bg:#101214; --panel:#171b1f; --line:#2b3238; --text:#e8edf2; --muted:#98a3ad; --accent:#4fb38a; --bad:#e06464; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; border-bottom:1px solid var(--line); background:#0d0f11; }
    h1 { margin:0; font-size:18px; }
    main { display:grid; grid-template-columns:minmax(640px, 1fr) 340px; gap:12px; padding:12px; min-height:calc(100vh - 56px); }
    .panel { border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:12px; }
    .map-wrap { min-height:0; }
    canvas { width:100%; height:calc(100vh - 104px); min-height:640px; display:block; background:#d8d8d1; border:1px solid var(--line); border-radius:6px; }
    button, select { font:inherit; }
    button, select { border:1px solid var(--line); background:#22282e; color:var(--text); padding:9px 11px; border-radius:6px; }
    button { cursor:pointer; }
    button.primary { background:#1e5b46; border-color:#257356; }
    button.danger { background:#5a2424; border-color:#743030; }
    label { display:block; margin:10px 0 6px; color:#d7dde3; font-size:13px; }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .muted { color:var(--muted); }
    .stat { padding:8px 0; border-bottom:1px solid var(--line); font-size:13px; }
    pre { margin:10px 0 0; max-height:220px; overflow:auto; background:#070809; border:1px solid var(--line); border-radius:8px; padding:10px; white-space:pre-wrap; font-size:12px; }
    @media (max-width: 1100px) { main { grid-template-columns:1fr; } canvas { height:70vh; } }
  </style>
</head>
<body>
  <header>
    <h1>Warehouse Nav2 Task Map</h1>
    <div class="row"><a href="/" style="color:#8bd6ff">Runner</a><a href="/vnc/" target="_blank" style="color:#8bd6ff">noVNC</a></div>
  </header>
  <main>
    <div class="panel map-wrap"><canvas id="map" width="1100" height="800"></canvas></div>
    <aside class="panel">
      <div class="stat"><strong>Status:</strong> <span id="status">loading</span></div>
      <div class="stat"><strong>Robot:</strong> <span id="robot">-</span></div>
      <div class="stat"><strong>Drop:</strong> <span id="dropText">click map</span></div>
      <label for="product">Item</label>
      <select id="product">
        <option value="ProductR">ProductR / StorageR</option>
        <option value="ProductG">ProductG / StorageG</option>
        <option value="ProductB">ProductB / StorageB</option>
      </select>
      <label>Map edit mode</label>
      <div class="row">
        <button id="dropMode" class="primary" onclick="setMode('drop')">TaskGoal drop</button>
        <button id="keepoutMode" onclick="setMode('keepout')">Keepout zone</button>
      </div>
      <div style="height:10px"></div>
      <div class="row">
        <button class="primary" onclick="sendTask()">Send TaskGoal</button>
        <button onclick="clearKeepouts()">Clear zones</button>
        <button class="danger" onclick="clearTask()">Reset drop</button>
      </div>
      <pre id="details"></pre>
    </aside>
  </main>
  <script>
    let state = null;
    let mode = 'drop';
    let drop = null;
    let keepouts = [];
    let dragStart = null;
    const canvas = document.getElementById('map');
    const ctx = canvas.getContext('2d');

    function setMode(next) {
      mode = next;
      document.getElementById('dropMode').className = next === 'drop' ? 'primary' : '';
      document.getElementById('keepoutMode').className = next === 'keepout' ? 'primary' : '';
    }
    function clearKeepouts() { keepouts = []; draw(); }
    function clearTask() { drop = null; document.getElementById('dropText').textContent = 'click map'; draw(); }
    function worldToCanvas(x, y) {
      const m = (state && state.map) || fallbackMap();
      return {
        x: ((x - m.origin.x) / m.width_m) * canvas.width,
        y: canvas.height - ((y - m.origin.y) / m.height_m) * canvas.height,
      };
    }
    function canvasToWorld(px, py) {
      const m = (state && state.map) || fallbackMap();
      return {
        x: m.origin.x + (px / canvas.width) * m.width_m,
        y: m.origin.y + ((canvas.height - py) / canvas.height) * m.height_m,
      };
    }
    function fallbackMap() { return {origin:{x:-11,y:-8}, width_m:22, height_m:16, obstacles:[], products:{}, dispatches:[]}; }
    function rectCanvas(r) {
      const a = worldToCanvas(r.x1, r.y1);
      const b = worldToCanvas(r.x2, r.y2);
      return {x:Math.min(a.x,b.x), y:Math.min(a.y,b.y), w:Math.abs(a.x-b.x), h:Math.abs(a.y-b.y)};
    }
    function drawRectWorld(r, fill, stroke) {
      const c = rectCanvas(r);
      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2;
      ctx.fillRect(c.x, c.y, c.w, c.h);
      ctx.strokeRect(c.x, c.y, c.w, c.h);
    }
    function draw() {
      const m = (state && state.map) || fallbackMap();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#d6d6cf';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#b5b5ad';
      ctx.lineWidth = 1;
      for (let x = Math.ceil(m.origin.x); x <= m.origin.x + m.width_m; x++) {
        const p = worldToCanvas(x, m.origin.y);
        ctx.beginPath(); ctx.moveTo(p.x, 0); ctx.lineTo(p.x, canvas.height); ctx.stroke();
      }
      for (let y = Math.ceil(m.origin.y); y <= m.origin.y + m.height_m; y++) {
        const p = worldToCanvas(m.origin.x, y);
        ctx.beginPath(); ctx.moveTo(0, p.y); ctx.lineTo(canvas.width, p.y); ctx.stroke();
      }
      for (const obs of m.obstacles || []) drawRectWorld(obs, 'rgba(76,70,60,0.72)', 'rgba(48,44,38,0.9)');
      for (const d of m.dispatches || []) drawRectWorld({x1:d.x-d.w/2,y1:d.y-d.h/2,x2:d.x+d.w/2,y2:d.y+d.h/2}, 'rgba(47,150,95,0.35)', 'rgba(69,210,135,0.9)');
      for (const z of keepouts) drawRectWorld(z, 'rgba(220,70,70,0.35)', 'rgba(255,110,110,0.95)');
      if (state && state.path && state.path.length) {
        ctx.strokeStyle = '#1f7aff';
        ctx.lineWidth = 4;
        ctx.beginPath();
        state.path.forEach((p, i) => { const c = worldToCanvas(p.x, p.y); if (i) ctx.lineTo(c.x, c.y); else ctx.moveTo(c.x, c.y); });
        ctx.stroke();
      }
      for (const [name, item] of Object.entries(m.products || {})) {
        const p = worldToCanvas(item.slot.x, item.slot.y);
        ctx.fillStyle = name === 'ProductR' ? '#d04436' : name === 'ProductG' ? '#25a657' : '#2e5bd7';
        ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#111';
        ctx.font = '13px sans-serif';
        ctx.fillText(name, p.x + 10, p.y - 8);
      }
      if (drop) {
        const p = worldToCanvas(drop.x, drop.y);
        ctx.strokeStyle = '#111';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(p.x - 10, p.y); ctx.lineTo(p.x + 10, p.y); ctx.moveTo(p.x, p.y - 10); ctx.lineTo(p.x, p.y + 10); ctx.stroke();
      }
      if (state && state.robot) {
        const r = worldToCanvas(state.robot.x, state.robot.y);
        ctx.fillStyle = '#f4f5f0';
        ctx.strokeStyle = '#101214';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(r.x, r.y, 13, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        const yaw = state.robot.yaw || 0;
        ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(r.x + Math.cos(yaw) * 24, r.y - Math.sin(yaw) * 24); ctx.stroke();
      }
    }
    canvas.addEventListener('mousedown', ev => {
      const box = canvas.getBoundingClientRect();
      dragStart = canvasToWorld((ev.clientX - box.left) * canvas.width / box.width, (ev.clientY - box.top) * canvas.height / box.height);
    });
    canvas.addEventListener('mouseup', ev => {
      const box = canvas.getBoundingClientRect();
      const p = canvasToWorld((ev.clientX - box.left) * canvas.width / box.width, (ev.clientY - box.top) * canvas.height / box.height);
      if (mode === 'drop') {
        drop = p;
        document.getElementById('dropText').textContent = `${p.x.toFixed(2)}, ${p.y.toFixed(2)}`;
      } else if (dragStart) {
        keepouts.push({x1:dragStart.x, y1:dragStart.y, x2:p.x, y2:p.y});
      }
      dragStart = null;
      draw();
    });
    async function api(path, opts) {
      const r = await fetch(path, opts);
      if (!r.ok) throw new Error(await r.text());
      return await r.json();
    }
    async function sendTask() {
      if (!drop) return alert('Set drop point on the map first.');
      const payload = {product:document.getElementById('product').value, drop, keepouts};
      const result = await api('/api/nav2/task', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)});
      document.getElementById('details').textContent = JSON.stringify(result, null, 2);
    }
    async function poll() {
      state = await api('/api/nav2/state');
      document.getElementById('status').textContent = `${state.status}: ${state.message || ''}`;
      document.getElementById('robot').textContent = `${state.robot.x.toFixed(2)}, ${state.robot.y.toFixed(2)}, yaw ${state.robot.yaw.toFixed(2)}`;
      if (state.task) document.getElementById('details').textContent = JSON.stringify({task:state.task, nav2_files:state.map.nav2_files}, null, 2);
      draw();
    }
    setInterval(poll, 700);
    poll();
  </script>
</body>
</html>
"""


LIDAR_RANDOM_INDEX = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lidar Random Map</title>
  <style>
    :root { color-scheme: dark; --bg:#101214; --panel:#171b1f; --line:#2b3238; --text:#e8edf2; --muted:#98a3ad; --accent:#4fb38a; --bad:#e06464; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; border-bottom:1px solid var(--line); background:#0d0f11; }
    h1 { margin:0; font-size:18px; }
    main { display:grid; grid-template-columns:minmax(720px, 1fr) 360px; gap:12px; padding:12px; min-height:calc(100vh - 56px); }
    .panel { border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:12px; }
    canvas { width:100%; height:calc(100vh - 104px); min-height:680px; display:block; background:#f4f4f0; border:1px solid var(--line); border-radius:6px; }
    button, select { font:inherit; border:1px solid var(--line); background:#22282e; color:var(--text); padding:9px 11px; border-radius:6px; }
    button { cursor:pointer; }
    button.primary { background:#1e5b46; border-color:#257356; }
    button.danger { background:#5a2424; border-color:#743030; }
    label { display:block; margin:10px 0 6px; color:#d7dde3; font-size:13px; }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .muted { color:var(--muted); }
    .stat { padding:8px 0; border-bottom:1px solid var(--line); font-size:13px; }
    .pickup-card { margin:10px 0; padding:10px; border:1px solid var(--line); border-radius:8px; background:#111519; font-size:13px; }
    .pickup-card.ready { border-color:#2f8f68; background:#102019; }
    .pickup-card.waiting { border-color:#6b5b25; background:#201b10; }
    .pickup-card.picked { border-color:#3a7bc2; background:#101a25; }
    .pickup-title { display:flex; justify-content:space-between; gap:8px; margin-bottom:6px; }
    .badge { display:inline-block; padding:2px 7px; border-radius:999px; background:#29313a; color:#dce5ec; font-size:12px; white-space:nowrap; }
    .bar { height:8px; background:#0c0e10; border:1px solid var(--line); border-radius:999px; overflow:hidden; margin-top:8px; }
    .bar > div { height:100%; background:#4fb38a; width:0%; }
    pre { margin:10px 0 0; max-height:250px; overflow:auto; background:#070809; border:1px solid var(--line); border-radius:8px; padding:10px; white-space:pre-wrap; font-size:12px; }
    @media (max-width: 1120px) { main { grid-template-columns:1fr; } canvas { height:70vh; } }
  </style>
</head>
<body>
  <header>
    <h1>Lidar Random Map</h1>
    <div class="row"><a href="/" style="color:#8bd6ff">Runner</a><a href="/vnc/" target="_blank" style="color:#8bd6ff">noVNC</a></div>
  </header>
  <main>
    <div class="panel"><canvas id="map" width="1200" height="830"></canvas></div>
    <aside class="panel">
      <div class="stat"><strong>Status:</strong> <span id="status">loading</span></div>
      <div class="stat"><strong>Coverage:</strong> <span id="coverage">0%</span><div class="bar"><div id="coverageBar"></div></div></div>
      <div class="stat"><strong>Robot:</strong> <span id="robot">-</span></div>
      <div class="stat"><strong>Drop:</strong> <span id="dropText">click map after mapping</span></div>
      <div id="pickupCard" class="pickup-card waiting">
        <div class="pickup-title"><strong>Package pickup</strong><span id="pickupBadge" class="badge">waiting</span></div>
        <div id="pickupText" class="muted">Select an item; TaskGoal can start when its shelf is detected by lidar.</div>
      </div>
      <label for="product">Item</label>
      <select id="product" onchange="updatePickupCard(); draw();"></select>
      <label>Map edit mode</label>
      <div class="row">
        <button id="dropMode" class="primary" onclick="setMode('drop')">TaskGoal drop</button>
        <button id="keepoutMode" onclick="setMode('keepout')">Keepout zone</button>
      </div>
      <div style="height:10px"></div>
      <div class="row">
        <button class="primary" onclick="sendTask()">Send TaskGoal</button>
        <button onclick="clearKeepouts()">Clear zones</button>
        <button class="danger" onclick="clearTask()">Reset drop</button>
      </div>
      <pre id="details"></pre>
    </aside>
  </main>
  <script>
    let state = null, mode = 'drop', drop = null, keepouts = [], dragStart = null;
    const canvas = document.getElementById('map');
    const ctx = canvas.getContext('2d');
    function setMode(next) {
      mode = next;
      document.getElementById('dropMode').className = next === 'drop' ? 'primary' : '';
      document.getElementById('keepoutMode').className = next === 'keepout' ? 'primary' : '';
    }
    function clearKeepouts() { keepouts = []; draw(); }
    function clearTask() { drop = null; document.getElementById('dropText').textContent = 'click map after mapping'; draw(); }
    function mapInfo() { return state || {origin:{x:-13,y:-9}, width_m:26, height_m:18, grid_w:216, grid_h:150, resolution:0.12, known:[]}; }
    function worldToCanvas(x, y) {
      const m = mapInfo();
      return {x: ((x - m.origin.x) / m.width_m) * canvas.width, y: canvas.height - ((y - m.origin.y) / m.height_m) * canvas.height};
    }
    function canvasToWorld(px, py) {
      const m = mapInfo();
      return {x: m.origin.x + (px / canvas.width) * m.width_m, y: m.origin.y + ((canvas.height - py) / canvas.height) * m.height_m};
    }
    function drawRectWorld(r, fill, stroke) {
      const a = worldToCanvas(r.x1, r.y1), b = worldToCanvas(r.x2, r.y2);
      const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y), w = Math.abs(a.x - b.x), h = Math.abs(a.y - b.y);
      ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
    }
    function draw() {
      const m = mapInfo();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      const cellW = canvas.width / (m.grid_w || 216), cellH = canvas.height / (m.grid_h || 150);
      for (let gy = 0; gy < (m.known || []).length; gy++) {
        const row = m.known[gy];
        for (let gx = 0; gx < row.length; gx++) {
          const v = row[gx];
          if (v === '?') ctx.fillStyle = '#f6f6f2';
          else if (v === '#') ctx.fillStyle = '#0b1b24';
          else ctx.fillStyle = '#b5b0f5';
          ctx.fillRect(gx * cellW, canvas.height - (gy + 1) * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
      if (state && state.known && state.known.length) {
        ctx.globalAlpha = 0.52;
        for (const row of state.known) {}
        ctx.globalAlpha = 1;
      }
      for (const z of keepouts) drawRectWorld(z, 'rgba(220,70,70,0.28)', 'rgba(255,70,70,0.95)');
      if (state && state.path && state.path.length) {
        ctx.strokeStyle = '#e51b55'; ctx.lineWidth = 3; ctx.beginPath();
        state.path.forEach((p, i) => { const c = worldToCanvas(p.x, p.y); if (i) ctx.lineTo(c.x, c.y); else ctx.moveTo(c.x, c.y); });
        ctx.stroke();
      }
      if (state && state.lidar) {
        const r = worldToCanvas(state.robot.x, state.robot.y);
        ctx.strokeStyle = 'rgba(0,210,220,0.20)'; ctx.lineWidth = 1;
        for (const p of state.lidar) { const e = worldToCanvas(p.x, p.y); ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(e.x, e.y); ctx.stroke(); }
      }
      if (state && state.products) {
        for (const [name, item] of Object.entries(state.products)) {
          const p = worldToCanvas(item.slot.x, item.slot.y);
          const status = state.pickup_status && state.pickup_status.products ? state.pickup_status.products[name] : null;
          ctx.fillStyle = name.endsWith('R') ? '#d04436' : name.endsWith('G') ? '#25a657' : name.endsWith('B') ? '#2e5bd7' : '#c6a11c';
          ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.fill();
          if (status && status.discovered) {
            ctx.strokeStyle = status.picked ? '#2d7dd2' : '#23b26b';
            ctx.lineWidth = 3;
            ctx.beginPath(); ctx.arc(p.x, p.y, 13, 0, Math.PI * 2); ctx.stroke();
          }
          ctx.fillStyle = '#101214'; ctx.font = '12px sans-serif'; ctx.fillText(name, p.x + 9, p.y - 7);
        }
      }
      if (drop) { const p = worldToCanvas(drop.x, drop.y); ctx.strokeStyle = '#101214'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(p.x - 10, p.y); ctx.lineTo(p.x + 10, p.y); ctx.moveTo(p.x, p.y - 10); ctx.lineTo(p.x, p.y + 10); ctx.stroke(); }
      if (state && state.robot) {
        const r = worldToCanvas(state.robot.x, state.robot.y);
        ctx.fillStyle = '#f7f7ef'; ctx.strokeStyle = '#101214'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(r.x, r.y, 12, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(r.x + Math.cos(state.robot.yaw) * 24, r.y - Math.sin(state.robot.yaw) * 24); ctx.stroke();
      }
    }
    canvas.addEventListener('mousedown', ev => {
      const box = canvas.getBoundingClientRect();
      dragStart = canvasToWorld((ev.clientX - box.left) * canvas.width / box.width, (ev.clientY - box.top) * canvas.height / box.height);
    });
    canvas.addEventListener('mouseup', ev => {
      const box = canvas.getBoundingClientRect();
      const p = canvasToWorld((ev.clientX - box.left) * canvas.width / box.width, (ev.clientY - box.top) * canvas.height / box.height);
      if (mode === 'drop') { drop = p; document.getElementById('dropText').textContent = `${p.x.toFixed(2)}, ${p.y.toFixed(2)}`; }
      else if (dragStart) keepouts.push({x1:dragStart.x, y1:dragStart.y, x2:p.x, y2:p.y});
      dragStart = null; draw();
    });
    async function api(path, opts) { const r = await fetch(path, opts); if (!r.ok) throw new Error(await r.text()); return await r.json(); }
    async function sendTask() {
      if (!drop) return alert('Set drop point first.');
      const selectedProduct = document.getElementById('product').value;
      const productStatus = state && state.pickup_status && state.pickup_status.products ? state.pickup_status.products[selectedProduct] : null;
      const canStart = state && (['mapped','done'].includes(state.status) || (productStatus && productStatus.discovered));
      if (!canStart) return alert('TaskGoal is locked until the selected shelf is detected by lidar.');
      const payload = {product:selectedProduct, drop, keepouts};
      const result = await api('/api/lidar-random/task', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)});
      document.getElementById('details').textContent = JSON.stringify(result, null, 2);
    }
    function refreshProducts() {
      const select = document.getElementById('product');
      const current = select.value;
      const names = Object.keys((state && state.products) || {});
      if (!names.length) return;
      const statuses = state && state.pickup_status ? state.pickup_status.products || {} : {};
      select.innerHTML = names.map(n => {
        const s = statuses[n] || {};
        const suffix = s.picked ? 'picked' : s.discovered ? 'lidar found' : 'not found';
        return `<option value="${n}">${n} / ${state.products[n].storage} / ${suffix}</option>`;
      }).join('');
      const preferred = state && state.task && names.includes(state.task.product) ? state.task.product : current;
      if (names.includes(preferred)) select.value = preferred;
    }
    function updatePickupCard() {
      const selectedProduct = document.getElementById('product').value;
      const statuses = state && state.pickup_status ? state.pickup_status.products || {} : {};
      const s = statuses[selectedProduct] || null;
      const card = document.getElementById('pickupCard');
      const badge = document.getElementById('pickupBadge');
      const text = document.getElementById('pickupText');
      if (!s) {
        card.className = 'pickup-card waiting';
        badge.textContent = 'waiting';
        text.textContent = 'Waiting for lidar state.';
        return;
      }
      const picked = !!s.picked;
      card.className = picked ? 'pickup-card picked' : s.discovered ? 'pickup-card ready' : 'pickup-card waiting';
      badge.textContent = picked ? 'picked' : s.discovered ? 'lidar found' : 'not found';
      const seen = Math.round((s.known_ratio || 0) * 100);
      text.textContent = `${s.product} at ${s.storage}: ${s.message || s.phase}; shelf known ${seen}%, occupied hits ${s.occupied_hits || 0}.`;
    }
    async function poll() {
      state = await api('/api/lidar-random/state');
      refreshProducts();
      updatePickupCard();
      const cov = Math.round((state.coverage || 0) * 100);
      document.getElementById('status').textContent = `${state.status}: ${state.message || ''}`;
      document.getElementById('coverage').textContent = `${cov}%`;
      document.getElementById('coverageBar').style.width = `${cov}%`;
      document.getElementById('robot').textContent = `${state.robot.x.toFixed(2)}, ${state.robot.y.toFixed(2)}, yaw ${state.robot.yaw.toFixed(2)}`;
      document.getElementById('details').textContent = JSON.stringify({task:state.task, updated_at:state.updated_at}, null, 2);
      draw();
    }
    setInterval(poll, 500);
    poll();
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
        elif parsed.path == "/nav2":
            self.send_text(NAV2_INDEX, content_type="text/html; charset=utf-8")
        elif parsed.path == "/lidar-random":
            self.send_text(LIDAR_RANDOM_INDEX, content_type="text/html; charset=utf-8")
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
        elif parsed.path == "/api/nav2/state":
            self.send_json(nav2_state())
        elif parsed.path == "/api/nav2/map":
            self.send_json(nav2_state().get("map", nav2_default_map()))
        elif parsed.path == "/api/lidar-random/state":
            self.send_json(lidar_random_state())
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
            elif parsed.path == "/api/nav2/task":
                body = self.read_body()
                self.send_json(write_nav2_task(body), HTTPStatus.CREATED)
            elif parsed.path == "/api/lidar-random/task":
                body = self.read_body()
                self.send_json(write_lidar_random_task(body), HTTPStatus.CREATED)
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
