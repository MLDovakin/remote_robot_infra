#!/usr/bin/env python3
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


WORLD = Path("/opt/aic_web/warehouse_visual.sdf")
AIC_SETUP = Path("/ws_aic/install/setup.bash")
ROUTE = [
    (-7.0, -5.0, 0.0, "startup at dispatch lane"),
    (-3.5, -5.0, 0.0, "leaving dispatch area"),
    (3.0, -5.0, 0.0, "driving through lower aisle"),
    (6.55, -5.90, 0.35, "approaching StorageR front"),
    (6.55, -5.35, 1.5708, "pickup ProductR"),
    (6.55, -5.90, -1.5708, "backing out from StorageR"),
    (3.80, -5.90, 3.14159, "clearing StorageR side"),
    (3.80, -1.90, 1.5708, "driving around storage racks"),
    (6.55, -1.90, 0.0, "approaching StorageG front"),
    (6.55, -1.35, 1.5708, "pickup ProductG"),
    (6.55, -1.90, -1.5708, "backing out from StorageG"),
    (3.80, -1.90, 3.14159, "clearing StorageG side"),
    (3.80, 2.00, 1.5708, "driving around StorageG"),
    (2.0, 2.0, 2.65, "turning toward dispatch"),
    (-5.5, 4.2, 3.05, "driving to DispatchB"),
    (-8.0, 4.0, 3.14159, "dropoff at DispatchB"),
    (-7.0, -5.0, -1.5708, "returning to standby"),
]

HIDDEN_Z = -10.0
STEP_DELAY_SECONDS = 0.035
WAYPOINT_DELAY_SECONDS = 0.12
INTERPOLATION_STEPS = 16
POSE_COMMAND_TIMEOUT_SECONDS = 2.0
GZ_SERVICE_TIMEOUT_MS = 1500
IGNORED_GAZEBO_LOG_LINES = (
    "NodeShared::RecvSrvRequest() error sending response: Host unreachable",
)
PICKUP_EVENTS = {
    "pickup ProductR": {"product": "ProductR", "storage": "StorageR", "slot": (6.55, -4.2, 1.02)},
    "pickup ProductG": {"product": "ProductG", "storage": "StorageG", "slot": (6.55, -0.2, 1.02)},
}
DROPOFF_EVENTS = {
    "dropoff at DispatchB": {"dispatch": "DispatchB", "target": (-8.0, 4.0, 0.28)},
}


def run(cmd, timeout=None):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, exc.stdout or "")


def gz_shell(command):
    if AIC_SETUP.exists():
        return f"source {AIC_SETUP} && {command}"
    return command


def screen_geometry():
    screen = os.environ.get("AIC_SCREEN", "1600x900x24").split("x")
    try:
        return int(screen[0]), int(screen[1])
    except (IndexError, ValueError):
        return 1600, 900


def fit_gazebo_window():
    width, height = screen_geometry()
    for _ in range(30):
        wmctrl = run(["wmctrl", "-r", "Gazebo Sim", "-b", "add,maximized_vert,maximized_horz"])
        xdotool = run(
            [
                "xdotool",
                "search",
                "--name",
                "Gazebo Sim",
                "windowmove",
                "0",
                "0",
                "windowsize",
                str(width),
                str(height),
            ]
        )
        if wmctrl.returncode == 0 or xdotool.returncode == 0:
            return
        time.sleep(0.5)


def forward_gazebo_output(proc):
    assert proc.stdout is not None
    for line in proc.stdout:
        if any(noise in line for noise in IGNORED_GAZEBO_LOG_LINES):
            continue
        print(line, end="", flush=True)


def launch_gazebo(env):
    proc = subprocess.Popen(
        ["/bin/bash", "-lc", gz_shell(f"exec gz sim -v 3 {WORLD}")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    threading.Thread(target=forward_gazebo_output, args=(proc,), daemon=True).start()
    threading.Thread(target=fit_gazebo_window, daemon=True).start()
    return proc


def set_pose(x, y, yaw):
    req = (
        'name: "warehouse_robot", '
        f"position: {{x: {x:.3f}, y: {y:.3f}, z: 0.320}}, "
        f"orientation: {{z: {math.sin(yaw / 2):.6f}, w: {math.cos(yaw / 2):.6f}}}"
    )
    cmd = [
        "/bin/bash",
        "-lc",
        gz_shell(
            "gz service "
            "-s /world/warehouse_mobile/set_pose "
            "--reqtype gz.msgs.Pose "
            "--reptype gz.msgs.Boolean "
            f"--timeout {GZ_SERVICE_TIMEOUT_MS} "
            f"--req '{req}'"
        ),
    ]
    result = run(cmd, timeout=POSE_COMMAND_TIMEOUT_SECONDS)
    if result.returncode != 0:
        print(result.stdout, end="", flush=True)


def set_model_pose(model, x, y, z, yaw=0.0):
    req = (
        f'name: "{model}", '
        f"position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}}, "
        f"orientation: {{z: {math.sin(yaw / 2):.6f}, w: {math.cos(yaw / 2):.6f}}}"
    )
    cmd = [
        "/bin/bash",
        "-lc",
        gz_shell(
            "gz service "
            "-s /world/warehouse_mobile/set_pose "
            "--reqtype gz.msgs.Pose "
            "--reptype gz.msgs.Boolean "
            f"--timeout {GZ_SERVICE_TIMEOUT_MS} "
            f"--req '{req}'"
        ),
    ]
    result = run(cmd, timeout=POSE_COMMAND_TIMEOUT_SECONDS)
    if result.returncode != 0:
        print(result.stdout, end="", flush=True)


def set_cargo_visible(robot_x, robot_y, yaw):
    cargo_x = robot_x - math.cos(yaw) * 0.18
    cargo_y = robot_y - math.sin(yaw) * 0.18
    set_model_pose("cargo_item", cargo_x, cargo_y, 0.72, yaw)


def hide_cargo():
    set_model_pose("cargo_item", -7.0, -5.0, HIDDEN_Z)


def hide_delivered():
    set_model_pose("delivered_item", -8.0, 4.0, HIDDEN_Z)


def show_delivered(x, y):
    set_model_pose("delivered_item", x, y, 0.28)


def handle_waypoint_event(label, x, y, yaw, cargo):
    if label in PICKUP_EVENTS and not cargo:
        event = PICKUP_EVENTS[label]
        set_cargo_visible(x, y, yaw)
        print(f"pick_up product={event['product']} storage={event['storage']} slot={event['slot']}", flush=True)
        return event["product"]
    if label in DROPOFF_EVENTS and cargo:
        event = DROPOFF_EVENTS[label]
        show_delivered(event["target"][0], event["target"][1])
        hide_cargo()
        print(f"drop_off product={cargo} dispatch={event['dispatch']} target={event['target']}", flush=True)
        return None
    return cargo


def interpolate(a, b, steps):
    ax, ay, ayaw, _ = a
    bx, by, byaw, _ = b
    for i in range(1, steps + 1):
        t = i / steps
        yield (
            ax + (bx - ax) * t,
            ay + (by - ay) * t,
            ayaw + normalize_angle(byaw - ayaw) * t,
        )


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def main():
    if not WORLD.exists():
        raise FileNotFoundError(WORLD)

    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("GALLIUM_DRIVER", "llvmpipe")

    print("Starting visual warehouse Gazebo simulation")
    print("World contains a room, storage racks, dispatch areas, and a mobile warehouse robot")
    print("Gazebo camera starts above the robot/work area. Open noVNC to watch it.")

    gz = launch_gazebo(env)

    def shutdown(signum, frame):
        print("Stopping visual warehouse simulation", flush=True)
        try:
            os.killpg(os.getpgid(gz.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    time.sleep(8)
    set_pose(*ROUTE[0][:3])
    hide_cargo()
    hide_delivered()
    cargo = None
    lap = 1
    while gz.poll() is None:
        print(f"Route lap {lap}: executing warehouse pickup and dispatch path", flush=True)
        for start, end in zip(ROUTE, ROUTE[1:]):
            print(f"  {end[3]}", flush=True)
            for x, y, yaw in interpolate(start, end, INTERPOLATION_STEPS):
                if gz.poll() is not None:
                    return gz.returncode
                set_pose(x, y, yaw)
                if cargo:
                    set_cargo_visible(x, y, yaw)
                time.sleep(STEP_DELAY_SECONDS)
            cargo = handle_waypoint_event(end[3], end[0], end[1], end[2], cargo)
            time.sleep(WAYPOINT_DELAY_SECONDS)
        lap += 1
    return gz.returncode


if __name__ == "__main__":
    raise SystemExit(main())
