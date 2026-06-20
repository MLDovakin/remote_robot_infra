#!/usr/bin/env python3
"""
VLA inference demo for the fine-tuned SmolVLM warehouse navigation model.

Runs on Mac (not in the Docker container). Uses the same headless simulation
as collect_dataset.py so no Gazebo is needed.

Live visualisation:
  Start the docker container first (`docker compose up -d`), then open
  http://localhost:8080 and click "Start Lidar Random Map" ONCE to load the
  map UI. Then interrupt it and run THIS script — it overwrites the same
  lidar_random_state.json, so the web UI refreshes automatically every second
  and shows the VLA robot moving on the map.

Trajectory log:
  Every step is written to aic_runs/vla_trajectory.jsonl.

Usage:
  conda activate vlm_env
  python3 warehouse_vla_mode.py --model smolvlm_lora --steps 400 --seed 42
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── import simulation helpers from collect_dataset.py ────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "docker" / "aic_web"))
from collect_dataset import (
    DISPATCH_ZONES,
    GRID_H,
    GRID_W,
    HEIGHT_M,
    LIDAR_RANGE,
    LIDAR_RAYS,
    ORIGIN_X,
    ORIGIN_Y,
    RESOLUTION,
    ROBOT_RADIUS,
    WIDTH_M,
    MAPPED_COVERAGE,
    Pose2D,
    astar,
    coverage,
    frontier_goal,
    grid_to_world,
    known_occupied,
    make_instruction,
    make_random_world,
    normalize_angle,
    reachable_free_cells,
    rect_occupancy,
    simulate_lidar_ranges,
    world_to_grid,
    STEP_MAX,
    YAW_MAX,
)

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
RUNS_DIR = Path("aic_runs")
RESULTS_DIR = Path("aic_results")
STATE_FILE = RUNS_DIR / "lidar_random_state.json"   # same file the web UI reads
TRAJ_FILE = RUNS_DIR / "vla_trajectory.jsonl"
IMG_SIZE = 128


# ── BEV renderer (identical to finetune_smolvlm.py) ──────────────────────────

def lidar_to_bev(lidar_ranges, pose_yaw, goal_x, goal_y, pose_x, pose_y):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    scale = IMG_SIZE / (LIDAR_RANGE * 2)
    for i, r in enumerate(lidar_ranges):
        angle = -math.pi + (2 * math.pi * i / LIDAR_RAYS)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for k in range(1, max(2, int(r / LIDAR_RANGE * 24))):
            frac = k / 24 * LIDAR_RANGE
            if frac > r:
                break
            px = int(cx - sin_a * frac * scale)
            py = int(cy - cos_a * frac * scale)
            if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                draw.point((px, py), fill=(38, 48, 56))
    for i, r in enumerate(lidar_ranges):
        if r >= LIDAR_RANGE - 0.05:
            continue
        angle = -math.pi + (2 * math.pi * i / LIDAR_RAYS)
        px = int(cx - math.sin(angle) * r * scale)
        py = int(cy - math.cos(angle) * r * scale)
        if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
            draw.ellipse([px - 1, py - 1, px + 2, py + 2], fill=(230, 210, 50))
    gdx, gdy = goal_x - pose_x, goal_y - pose_y
    grx = math.cos(-pose_yaw) * gdx - math.sin(-pose_yaw) * gdy
    gry = math.sin(-pose_yaw) * gdx + math.cos(-pose_yaw) * gdy
    d = math.hypot(grx, gry)
    if d > 0.1:
        f = min(1.0, LIDAR_RANGE * 0.9 / d)
        gpx = max(4, min(IMG_SIZE - 5, int(cx - gry * f * scale)))
        gpy = max(4, min(IMG_SIZE - 5, int(cy - grx * f * scale)))
        draw.line([(cx, cy), (gpx, gpy)], fill=(220, 60, 60))
        draw.ellipse([gpx - 4, gpy - 4, gpx + 4, gpy + 4], fill=(220, 60, 60))
    draw.polygon([(cx, cy - 6), (cx - 4, cy + 4), (cx + 4, cy + 4)], fill=(50, 200, 80))
    return img


# ── model inference ───────────────────────────────────────────────────────────

def load_model(model_path: str, device):
    print(f"Loading base model {MODEL_ID}…")
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)
    print(f"Loading LoRA adapter from {model_path}…")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def infer_action(model, processor, image: Image.Image, instruction: str, device):
    """Returns (linear, angular, raw_text, parse_ok)."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.3,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    # decode with special tokens to catch cases where action is wrapped
    raw_full = processor.decode(new_tokens, skip_special_tokens=False)
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
    # if clean decode is empty, try extracting from full decode
    if not raw:
        raw = re.sub(r"<[^>]+>", " ", raw_full).strip()
    linear, angular, ok = parse_action(raw)
    return linear, angular, raw, ok


def parse_action(text: str):
    """Parse 'linear=X angular=Y' from model output. Returns (linear, angular, success)."""
    m = re.search(r"linear=([+-]?\d*\.?\d+).*?angular=([+-]?\d*\.?\d+)", text)
    if m:
        lin = max(-1.0, min(1.0, float(m.group(1))))
        ang = max(-1.0, min(1.0, float(m.group(2))))
        return lin, ang, True
    return 0.0, 0.0, False


def astar_action(pose: Pose2D, gx: float, gy: float,
                 true_occupied: set, known) -> tuple[float, float]:
    """A* fallback: compute (linear, angular) toward goal."""
    planning_occ = true_occupied | known_occupied(known)
    goal = Pose2D(gx, gy, 0.0)
    path = astar(pose, goal, planning_occ)
    if len(path) < 2:
        # no path — try turning toward goal
        target_yaw = math.atan2(gy - pose.y, gx - pose.x)
        ang = max(-1.0, min(1.0, normalize_angle(target_yaw - pose.yaw) / YAW_MAX))
        return 0.0, ang
    target = path[1]
    dx, dy = target.x - pose.x, target.y - pose.y
    target_yaw = math.atan2(dy, dx)
    yaw_delta = normalize_angle(target_yaw - pose.yaw)
    linear = max(0.0, math.cos(yaw_delta))
    angular = max(-1.0, min(1.0, yaw_delta / YAW_MAX))
    return linear, angular


# ── motion ────────────────────────────────────────────────────────────────────

def apply_action(pose: Pose2D, linear: float, angular: float, true_occupied: set):
    """
    Apply (linear, angular) ∈ [-1,1] to current pose.
    angular encodes yaw_delta / pi  (same normalisation used in training).
    linear  encodes cos(heading_vs_movement) ≈ 1.0 for straight ahead.
    Returns (new_pose, collided).
    """
    new_yaw = normalize_angle(pose.yaw + angular * YAW_MAX)
    step = STEP_MAX * max(0.0, min(1.0, linear))   # only forward motion
    new_x = pose.x + math.cos(new_yaw) * step
    new_y = pose.y + math.sin(new_yaw) * step
    cell = world_to_grid(new_x, new_y)
    if cell in true_occupied:
        return pose, True   # blocked — stay in place
    return Pose2D(new_x, new_y, new_yaw), False


# ── state file (web UI) ───────────────────────────────────────────────────────

def lidar_endpoints(ranges, pose):
    """Convert range list to [{x,y}] endpoint list for the web UI."""
    eps = []
    for i, r in enumerate(ranges):
        if r >= LIDAR_RANGE - 0.05:
            continue
        angle = pose.yaw - math.pi + (2 * math.pi * i / LIDAR_RAYS)
        eps.append({
            "x": round(pose.x + math.cos(angle) * r, 3),
            "y": round(pose.y + math.sin(angle) * r, 3),
        })
    return eps


def write_state(world, known, pose, reachable, status, message,
                lidar_ranges, task=None, path=None):
    cov = coverage(known, reachable)
    data = {
        "status": status,
        "message": message,
        "origin": {"x": ORIGIN_X, "y": ORIGIN_Y},
        "width_m": WIDTH_M,
        "height_m": HEIGHT_M,
        "resolution": RESOLUTION,
        "grid_w": GRID_W,
        "grid_h": GRID_H,
        "known": [
            "".join("?" if v == -1 else "#" if v == 1 else "." for v in row)
            for row in known
        ],
        "coverage": round(cov, 4),
        "robot": {"x": round(pose.x, 3), "y": round(pose.y, 3), "yaw": round(pose.yaw, 4)},
        "true_obstacles": [
            {"name": r.name, "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2, "kind": r.kind}
            for r in world["rects"]
        ],
        "products": world["products"],
        "pickup_status": {
            "selected_product": (task or {}).get("product"),
            "selected": None,
            "products": {},
        },
        "path": [{"x": round(p.x, 3), "y": round(p.y, 3), "yaw": 0} for p in (path or [])],
        "task": task,
        "lidar": lidar_endpoints(lidar_ranges, pose),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ── headless A* exploration (builds the known map before VLA task) ────────────

def run_exploration(world, true_occupied, reachable):
    """Fast headless A* exploration. Returns final pose + fully updated known."""
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
    print(f"  Exploration done: coverage={coverage(known, reachable):.0%}  "
          f"pose=({pose.x:.1f},{pose.y:.1f})")
    return pose, known


# ── trajectory logger ─────────────────────────────────────────────────────────

class TrajectoryLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", encoding="utf-8")
        self._step = 0
        self._t0 = time.time()
        print(f"Trajectory log → {path}")

    def log(self, pose: Pose2D, goal_x, goal_y, goal_name,
            phase, instruction, model_raw, action, collided):
        dist = math.hypot(goal_x - pose.x, goal_y - pose.y)
        row = {
            "step": self._step,
            "t": round(time.time() - self._t0, 3),
            "pose": {"x": round(pose.x, 3), "y": round(pose.y, 3),
                     "yaw": round(pose.yaw, 4)},
            "goal": {"x": round(goal_x, 3), "y": round(goal_y, 3),
                     "name": goal_name},
            "phase": phase,
            "dist_to_goal": round(dist, 3),
            "instruction": instruction,
            "model_raw": model_raw,
            "action": {"linear": action[0], "angular": action[1]},
            "collision": collided,
        }
        self._f.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._f.flush()
        self._step += 1

    def close(self):
        self._f.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="VLA inference demo")
    ap.add_argument("--model", default="smolvlm_lora",
                    help="Path to LoRA adapter (default: smolvlm_lora/)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random world seed")
    ap.add_argument("--steps", type=int, default=400,
                    help="Max VLA inference steps")
    ap.add_argument("--goal-radius", type=float, default=1.0,
                    help="Distance (m) to consider goal reached")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── world ─────────────────────────────────────────────────────────────────
    print(f"Generating world seed={args.seed}…")
    world = make_random_world(args.seed)
    true_occupied = rect_occupancy(world["rects"], ROBOT_RADIUS)
    reachable = reachable_free_cells(world["start"], true_occupied)

    # ── exploration (headless, fast) ──────────────────────────────────────────
    print("Running headless A* exploration to build map…")
    pose, known = run_exploration(world, true_occupied, reachable)

    # ── task setup ────────────────────────────────────────────────────────────
    products = world["products"]
    if not products:
        print("No products in this world — try a different seed.")
        return
    product_name = next(iter(products))
    product = products[product_name]
    pickup = Pose2D(product["pickup"]["x"], product["pickup"]["y"],
                    product["pickup"]["yaw"])
    drop_zone = DISPATCH_ZONES[0]
    drop_pose = Pose2D(drop_zone["x"], drop_zone["y"], drop_zone["yaw"])

    tasks = [
        ("pickup", pickup, product["storage"], product_name, None),
        ("dropoff", drop_pose, drop_zone["name"], product_name, product_name),
    ]
    print(f"Task: pick up {product_name} from {product['storage']} "
          f"at ({pickup.x:.1f},{pickup.y:.1f}), "
          f"deliver to {drop_zone['name']} at ({drop_pose.x:.1f},{drop_pose.y:.1f})")

    # ── load model ────────────────────────────────────────────────────────────
    model, processor = load_model(args.model, device)

    # ── inference loop ────────────────────────────────────────────────────────
    logger = TrajectoryLogger(TRAJ_FILE)
    total_step = 0
    collisions = 0
    goal_reached_count = 0

    for phase, goal_pose, goal_name, product_arg, cargo_arg in tasks:
        gx, gy = goal_pose.x, goal_pose.y
        task_meta = {"product": product_name, "drop": {"x": gx, "y": gy}}
        print(f"\n{'='*70}")
        print(f"  PHASE: {phase.upper()}   goal={goal_name} ({gx:.1f}, {gy:.1f})")
        print(f"{'='*70}")

        for step in range(args.steps):
            dist = math.hypot(gx - pose.x, gy - pose.y)
            if dist < args.goal_radius:
                print(f"  ✓ Goal reached in {step} steps!")
                goal_reached_count += 1
                break

            # observe
            lidar = simulate_lidar_ranges(pose, true_occupied, known)
            instruction = make_instruction(
                phase, goal_name, gx, gy, pose,
                product=product_arg, cargo=cargo_arg,
            )

            # infer
            t_infer = time.time()
            bev = lidar_to_bev(lidar, pose.yaw, gx, gy, pose.x, pose.y)
            linear, angular, raw, parse_ok = infer_action(
                model, processor, bev, instruction, device
            )
            dt_ms = (time.time() - t_infer) * 1000

            # fallback to A* when model fails to produce a parseable action
            source = "VLA"
            if not parse_ok:
                linear, angular = astar_action(pose, gx, gy, true_occupied, known)
                source = "A*"

            # act
            new_pose, collided = apply_action(pose, linear, angular, true_occupied)
            if collided:
                collisions += 1

            # log
            logger.log(pose, gx, gy, goal_name, phase,
                       instruction, raw, (linear, angular), collided)

            # update state file for web UI
            write_state(world, known, new_pose, reachable,
                        "vla_inference",
                        f"[{source}] {phase} step={total_step} dist={dist:.1f}m → {raw}",
                        lidar, task_meta)

            # console
            blocked = "  *** COLLISION ***" if collided else ""
            src_tag = f"[{source}]" if source == "A*" else "     "
            print(
                f"[{total_step:>4}] {phase:<8}  "
                f"pos=({pose.x:+6.2f},{pose.y:+6.2f})  yaw={math.degrees(pose.yaw):+5.0f}°  "
                f"dist={dist:5.2f}m\n"
                f"         model → \"{raw}\"\n"
                f"         {src_tag} lin={linear:+.3f} ang={angular:+.3f}"
                f"  {dt_ms:.0f}ms{blocked}",
                flush=True,
            )

            pose = new_pose
            total_step += 1

        else:
            print(f"  ✗ Max steps reached without reaching goal. "
                  f"Final dist={math.hypot(gx-pose.x, gy-pose.y):.1f}m")

    logger.close()

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"VLA inference summary")
    print(f"  Total steps   : {total_step}")
    print(f"  Goals reached : {goal_reached_count}/{len(tasks)}")
    print(f"  Collisions    : {collisions}")
    print(f"  Trajectory    : {TRAJ_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
