#!/usr/bin/env python3
"""
VLA Gazebo bridge — runs inside the container.

Launches Gazebo with the random warehouse world, runs the headless Python
simulation, and takes driving commands from the VLA model running on Mac
via shared files in aic_runs/.

Usage:
  # In one terminal on Mac:
  docker compose exec web bash -lc "python3 /opt/aic_web/vla_gazebo_bridge.py --seed 42"

  # In another terminal on Mac (after bridge prints "Ready"):
  conda activate vlm_env
  python3 warehouse_vla_gazebo.py --model smolvlm_lora --seed 42

  # Watch in browser: http://localhost:8080/lidar-random  (2D map)
  # Watch in 3D:      http://localhost:6080              (Gazebo noVNC)

Shared files (in aic_runs/, mounted as Docker volume):
  vla_sensor.json   — bridge writes: {lidar, pose, task, phase, t}
  vla_cmd.json      — VLA writes:    {linear, angular, t}
"""
import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from warehouse_lidar_random_mode import (
    DISPATCH_ZONES,
    GRID_H,
    GRID_W,
    HEIGHT_M,
    LIDAR_RANGE,
    LIDAR_RAYS,
    MAPPED_COVERAGE,
    ODOM_DT_SECONDS,
    ORIGIN_X,
    ORIGIN_Y,
    RESOLUTION,
    ROBOT_RADIUS,
    WIDTH_M,
    Pose2D,
    astar,
    coverage,
    frontier_goal,
    known_occupied,
    launch_gazebo,
    make_random_world,
    map_payload,
    reachable_free_cells,
    rect_occupancy,
    set_model_pose,
    simulate_lidar,
    simulate_lidar_ranges,
    write_world,
    world_to_grid,
    normalize_angle,
    STEP_MAX,
    YAW_MAX,
)
from warehouse_drive import GazeboDriveProjector

RUNS_DIR = Path(os.environ.get("AIC_RUNS_DIR", "/workspace/aic_runs"))
RESULTS_DIR = Path(os.environ.get("AIC_RESULTS_DIR", "/workspace/aic_results"))
SENSOR_FILE = RUNS_DIR / "vla_sensor.json"
CMD_FILE = RUNS_DIR / "vla_cmd.json"
STATE_FILE = RUNS_DIR / "lidar_random_state.json"

GOAL_RADIUS = 1.0    # metres to consider goal reached
POLL_HZ = 4          # how often bridge reads VLA command and updates Gazebo


def write_state_file(world, known, pose, reachable, status, message,
                     lidar_endpoints, task=None):
    data = map_payload(world, known, pose, status, message, task=task,
                       lidar=lidar_endpoints)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(STATE_FILE)


def write_sensor_file(lidar_ranges, pose, task_info):
    data = {
        "lidar": lidar_ranges,
        "pose": {"x": round(pose.x, 3), "y": round(pose.y, 3),
                 "yaw": round(pose.yaw, 4)},
        "task": task_info,
        "t": time.time(),
    }
    tmp = SENSOR_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    tmp.replace(SENSOR_FILE)


def read_cmd():
    """Read latest VLA command. Returns (linear, angular, timestamp) or None."""
    if not CMD_FILE.exists():
        return None
    try:
        d = json.loads(CMD_FILE.read_text())
        return d.get("linear", 0.0), d.get("angular", 0.0), d.get("t", 0.0)
    except Exception:
        return None


def apply_action(pose: Pose2D, linear: float, angular: float,
                 true_occupied: set):
    new_yaw = normalize_angle(pose.yaw + angular * YAW_MAX)
    step = STEP_MAX * max(0.0, min(1.0, linear))
    new_x = pose.x + math.cos(new_yaw) * step
    new_y = pose.y + math.sin(new_yaw) * step
    if world_to_grid(new_x, new_y) in true_occupied:
        return pose, True
    return Pose2D(new_x, new_y, new_yaw), False


def run_exploration(world, true_occupied, reachable):
    """Fast headless A* map build before VLA takes over."""
    known = [[-1] * GRID_W for _ in range(GRID_H)]
    pose = world["start"]
    simulate_lidar_ranges(pose, true_occupied, known)
    for _ in range(800):
        if coverage(known, reachable) >= MAPPED_COVERAGE:
            break
        planning_occ = true_occupied | known_occupied(known)
        goal, path = frontier_goal(pose, known, planning_occ, reachable)
        if not path:
            break
        current = pose
        for target in path[1:]:
            dx, dy = target.x - current.x, target.y - current.y
            yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.01 else current.yaw
            current = Pose2D(current.x + dx, current.y + dy, yaw)
            if world_to_grid(current.x, current.y) in true_occupied:
                break
            simulate_lidar_ranges(current, true_occupied, known)
        pose = current
        simulate_lidar_ranges(pose, true_occupied, known)
    print(f"  Exploration: coverage={coverage(known, reachable):.0%}  "
          f"pos=({pose.x:.1f},{pose.y:.1f})", flush=True)
    return pose, known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--goal-radius", type=float, default=GOAL_RADIUS)
    ap.add_argument("--no-gazebo", action="store_true",
                    help="Skip launching Gazebo (for testing bridge logic only)")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ── world ─────────────────────────────────────────────────────────────────
    print(f"Generating world seed={args.seed}…", flush=True)
    world = make_random_world(args.seed)
    write_world(world)
    true_occupied = rect_occupancy(world["rects"], ROBOT_RADIUS)
    reachable = reachable_free_cells(world["start"], true_occupied)

    # ── exploration (headless, fast) ──────────────────────────────────────────
    print("Running headless A* exploration…", flush=True)
    pose, known = run_exploration(world, true_occupied, reachable)

    # ── Gazebo ────────────────────────────────────────────────────────────────
    gz_proc = None
    drive = None
    if not args.no_gazebo:
        env = os.environ.copy()
        env.setdefault("DISPLAY", ":1")
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        env.setdefault("GALLIUM_DRIVER", "llvmpipe")
        print("Launching Gazebo…", flush=True)
        gz_proc = launch_gazebo(env)

        def shutdown(sig, frame):
            print("Shutting down…", flush=True)
            if gz_proc:
                try:
                    os.killpg(os.getpgid(gz_proc.pid), signal.SIGINT)
                except ProcessLookupError:
                    pass
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        print("Waiting 8s for Gazebo to start…", flush=True)
        time.sleep(8)
        drive = GazeboDriveProjector(
            "lidar_random_warehouse",
            map_origin=(ORIGIN_X, ORIGIN_Y),
            map_resolution=RESOLUTION,
        )
        drive.reset(pose.x, pose.y, pose.yaw)

    # ── task setup ────────────────────────────────────────────────────────────
    products = world["products"]
    if not products:
        print("No products in this world. Try a different seed.", flush=True)
        return

    product_name = next(iter(products))
    product = products[product_name]
    pickup = Pose2D(product["pickup"]["x"], product["pickup"]["y"],
                    product["pickup"]["yaw"])
    drop_zone = DISPATCH_ZONES[0]
    drop_pose = Pose2D(drop_zone["x"], drop_zone["y"], drop_zone["yaw"])

    tasks = [
        ("pickup", pickup,   product["storage"], product_name, None),
        ("dropoff", drop_pose, drop_zone["name"],  product_name, product_name),
    ]
    print(
        f"Task: {product_name} → {product['storage']} "
        f"({pickup.x:.1f},{pickup.y:.1f})  then → {drop_zone['name']} "
        f"({drop_pose.x:.1f},{drop_pose.y:.1f})",
        flush=True,
    )

    # ── clean stale command file ───────────────────────────────────────────────
    CMD_FILE.unlink(missing_ok=True)

    print("\n*** Ready. Start warehouse_vla_gazebo.py on Mac now. ***\n", flush=True)

    dt = 1.0 / POLL_HZ
    last_cmd_t = 0.0
    total_steps = 0
    goals_reached = 0

    for phase, goal_pose, goal_name, product_arg, cargo_arg in tasks:
        gx, gy = goal_pose.x, goal_pose.y
        task_info = {
            "phase": phase,
            "goal_name": goal_name,
            "goal": {"x": gx, "y": gy},
            "product": product_arg,
            "cargo": cargo_arg,
        }
        print(f"\n{'='*60}\n  PHASE: {phase.upper()}  goal={goal_name} ({gx:.1f},{gy:.1f})\n{'='*60}",
              flush=True)

        for step in range(args.steps):
            t0 = time.time()
            dist = math.hypot(gx - pose.x, gy - pose.y)

            if dist < args.goal_radius:
                print(f"  ✓ Goal reached in {step} VLA steps!", flush=True)
                goals_reached += 1
                break

            # simulate lidar
            lidar_ranges = simulate_lidar_ranges(pose, true_occupied, known)
            lidar_eps = simulate_lidar(pose, true_occupied, known)

            # write sensor state for VLA on Mac
            write_sensor_file(lidar_ranges, pose, task_info)

            # write 2D map state for web UI
            write_state_file(world, known, pose, reachable,
                             "vla_gazebo",
                             f"VLA {phase} step={total_steps} dist={dist:.1f}m  "
                             f"waiting for VLA command…",
                             lidar_eps, task={"product": product_name})

            # wait for a new VLA command (up to 30s)
            deadline = t0 + 30.0
            cmd = None
            while time.time() < deadline:
                c = read_cmd()
                if c and c[2] > last_cmd_t:
                    cmd = c
                    break
                time.sleep(dt)

            if cmd is None:
                print("  [warn] no VLA command received in 30s — stopping.", flush=True)
                break

            linear, angular, cmd_t = cmd
            last_cmd_t = cmd_t

            new_pose, collided = apply_action(pose, linear, angular, true_occupied)

            # update Gazebo 3D position
            if drive:
                drive.project_pose(new_pose.x, new_pose.y, new_pose.yaw,
                                   dt=time.time() - t0, prefix="vla", log_every=5)

            print(
                f"[{total_steps:>4}] {phase:<8}  "
                f"pos=({pose.x:+6.2f},{pose.y:+6.2f})  "
                f"dist={dist:5.2f}m  "
                f"cmd lin={linear:+.3f} ang={angular:+.3f}"
                f"{'  COLLISION' if collided else ''}",
                flush=True,
            )

            pose = new_pose
            total_steps += 1

        else:
            dist_final = math.hypot(gx - pose.x, gy - pose.y)
            print(f"  ✗ Max steps reached. Final dist={dist_final:.1f}m", flush=True)

    print(f"\nBridge done: {goals_reached}/{len(tasks)} goals reached, "
          f"{total_steps} steps total.", flush=True)

    if gz_proc:
        try:
            os.killpg(os.getpgid(gz_proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    main()
