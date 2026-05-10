#!/usr/bin/env python3
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from warehouse_drive import GazeboDriveProjector


WORLD = Path("/opt/aic_web/warehouse_visual.sdf")
AIC_SETUP = Path("/ws_aic/install/setup.bash")
HIDDEN_Z = -10.0
CONTROL_DT_SECONDS = 0.04
SPEED_MULTIPLIER = 5.0
GOAL_PAUSE_SECONDS = 0.12
POSE_COMMAND_TIMEOUT_SECONDS = 2.0
GZ_SERVICE_TIMEOUT_MS = 1500
IGNORED_GAZEBO_LOG_LINES = (
    "NodeShared::RecvSrvRequest() error sending response: Host unreachable",
)


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class Goal:
    x: float
    y: float
    yaw: float
    label: str


class WarehousePolicy:
    def __init__(self):
        self.route = [
            Goal(-7.0, -5.0, 0.0, "dispatch start"),
            Goal(-3.5, -5.0, 0.0, "leave dispatch lane"),
            Goal(3.0, -5.0, 0.0, "drive lower aisle"),
            Goal(6.55, -5.90, 0.35, "approach StorageR front"),
            Goal(6.55, -5.35, math.pi / 2, "pickup at StorageR"),
            Goal(6.55, -5.90, -math.pi / 2, "back out from StorageR"),
            Goal(3.80, -5.90, math.pi, "clear StorageR side"),
            Goal(3.80, -1.90, math.pi / 2, "drive around storage racks"),
            Goal(6.55, -1.90, 0.0, "approach StorageG front"),
            Goal(6.55, -1.35, math.pi / 2, "pickup at StorageG"),
            Goal(6.55, -1.90, -math.pi / 2, "back out from StorageG"),
            Goal(3.80, -1.90, math.pi, "clear StorageG side"),
            Goal(3.80, 2.00, math.pi / 2, "drive around StorageG"),
            Goal(2.0, 2.0, 2.65, "turn to dispatch"),
            Goal(-8.0, 4.0, math.pi, "dropoff at DispatchB"),
            Goal(-7.0, -5.0, -math.pi / 2, "return standby"),
        ]
        self.index = 0
        self.completed_laps = 0
        self.cargo = None

    def observe(self, pose: Pose2D):
        goal = self.route[self.index]
        dx = goal.x - pose.x
        dy = goal.y - pose.y
        dist = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        yaw_error = normalize(goal.yaw - pose.yaw)
        heading_error = normalize(heading - pose.yaw)
        return {
            "goal": goal,
            "distance_to_goal": dist,
            "heading_error": heading_error,
            "yaw_error": yaw_error,
        }

    def act(self, observation):
        goal = observation["goal"]
        if observation["distance_to_goal"] < 0.28 and abs(observation["yaw_error"]) < 0.12:
            self.index += 1
            if self.index >= len(self.route):
                self.index = 0
                self.completed_laps += 1
            return {"advance_goal": True, "goal": goal, "linear": 0.0, "angular": 0.0}

        linear = min(0.16, observation["distance_to_goal"] * 0.45) * SPEED_MULTIPLIER
        angular = clip(observation["heading_error"] * 1.25 + observation["yaw_error"] * 0.25, -0.42, 0.42) * SPEED_MULTIPLIER
        if abs(observation["heading_error"]) > 0.65:
            linear *= 0.35
        return {"advance_goal": False, "goal": goal, "linear": linear, "angular": angular}

    def handle_goal_event(self, goal):
        if goal.label == "pickup at StorageR" and self.cargo is None:
            self.cargo = "ProductR"
            print("pick_up product=ProductR storage=StorageR slot=(6.55,-4.20,1.02)", flush=True)
            return
        if goal.label == "pickup at StorageG" and self.cargo is None:
            self.cargo = "ProductG"
            print("pick_up product=ProductG storage=StorageG slot=(6.55,-0.20,1.02)", flush=True)
            return
        if goal.label == "dropoff at DispatchB" and self.cargo:
            print(f"drop_off product={self.cargo} dispatch=DispatchB target=(-8.00,4.00,0.28)", flush=True)
            self.cargo = None


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


def clip(value, low, high):
    return max(low, min(high, value))


def normalize(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def set_pose(pose: Pose2D):
    req = (
        'name: "warehouse_robot", '
        f"position: {{x: {pose.x:.3f}, y: {pose.y:.3f}, z: 0.320}}, "
        f"orientation: {{z: {math.sin(pose.yaw / 2):.6f}, w: {math.cos(pose.yaw / 2):.6f}}}"
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


def set_cargo_visible(pose: Pose2D):
    cargo_x = pose.x - math.cos(pose.yaw) * 0.18
    cargo_y = pose.y - math.sin(pose.yaw) * 0.18
    set_model_pose("cargo_item", cargo_x, cargo_y, 0.72, pose.yaw)


def hide_cargo():
    set_model_pose("cargo_item", -7.0, -5.0, HIDDEN_Z)


def hide_delivered():
    set_model_pose("delivered_item", -8.0, 4.0, HIDDEN_Z)


def show_delivered():
    set_model_pose("delivered_item", -8.0, 4.0, 0.28)


def step_pose(pose: Pose2D, linear, angular, dt):
    next_yaw = normalize(pose.yaw + angular * dt)
    return Pose2D(
        x=pose.x + math.cos(next_yaw) * linear * dt,
        y=pose.y + math.sin(next_yaw) * linear * dt,
        yaw=next_yaw,
    )


def main():
    if not WORLD.exists():
        raise FileNotFoundError(WORLD)

    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("GALLIUM_DRIVER", "llvmpipe")

    print("Starting warehouse policy mode")
    print("This mode runs a separate observation-action control loop, not the scripted route replay.")
    print("Gazebo camera starts above the robot/work area. Open noVNC to watch it.")

    gz = launch_gazebo(env)

    def shutdown(signum, frame):
        print("Stopping warehouse policy mode", flush=True)
        try:
            os.killpg(os.getpgid(gz.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    policy = WarehousePolicy()
    pose = Pose2D(-7.0, -5.0, 0.0)
    dt = CONTROL_DT_SECONDS
    time.sleep(8)
    drive = GazeboDriveProjector("warehouse_mobile", map_origin=(-11.0, -8.0), map_resolution=0.05)
    drive.reset(pose.x, pose.y, pose.yaw)
    hide_cargo()
    hide_delivered()

    tick = 0
    while gz.poll() is None:
        observation = policy.observe(pose)
        action = policy.act(observation)
        goal = action["goal"]

        if action["advance_goal"]:
            previous_cargo = policy.cargo
            policy.handle_goal_event(goal)
            if policy.cargo:
                set_cargo_visible(pose)
            elif previous_cargo:
                hide_cargo()
                show_delivered()
            print(f"[policy] reached goal: {goal.label}; next_index={policy.index}; laps={policy.completed_laps}", flush=True)
            time.sleep(GOAL_PAUSE_SECONDS)
            continue

        odom_pose = drive.apply_cmd(action["linear"], action["angular"], dt, "policy_cmd_vel", log_every=8)
        pose = Pose2D(odom_pose.x, odom_pose.y, odom_pose.yaw)
        if policy.cargo:
            set_cargo_visible(pose)
        if tick % 8 == 0:
            print(
                "[policy] "
                f"goal={goal.label} "
                f"pos=({pose.x:.2f},{pose.y:.2f}) yaw={pose.yaw:.2f} "
                f"dist={observation['distance_to_goal']:.2f} "
                f"cmd=({action['linear']:.2f},{action['angular']:.2f})",
                flush=True,
            )
        tick += 1
        time.sleep(dt)
    return gz.returncode


if __name__ == "__main__":
    raise SystemExit(main())
