#!/usr/bin/env python3
"""
VLA inference for the Gazebo 3D integration.

Reads sensor data written by vla_gazebo_bridge.py (in Docker container),
runs SmolVLM model inference on Mac (MPS), writes action back.

Usage:
  # Start bridge first (in container):
  docker compose exec web bash -lc "python3 /opt/aic_web/vla_gazebo_bridge.py --seed 42"
  # Wait for "*** Ready ***" message, then:
  conda activate vlm_env
  python3 warehouse_vla_gazebo.py --model smolvlm_lora --seed 42

Files used (in aic_runs/ — shared Docker volume):
  vla_sensor.json   ← bridge writes, we read
  vla_cmd.json      → we write, bridge reads
  vla_trajectory.jsonl → our full log
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

sys.path.insert(0, str(Path(__file__).parent / "docker" / "aic_web"))
from collect_dataset import (
    DISPATCH_ZONES,
    LIDAR_RANGE,
    LIDAR_RAYS,
    RESOLUTION,
    Pose2D,
    astar,
    known_occupied,
    make_instruction,
    normalize_angle,
    world_to_grid,
    rect_occupancy,
    STEP_MAX,
    YAW_MAX,
)

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
RUNS_DIR = Path("aic_runs")
SENSOR_FILE = RUNS_DIR / "vla_sensor.json"
CMD_FILE = RUNS_DIR / "vla_cmd.json"
TRAJ_FILE = RUNS_DIR / "vla_gazebo_trajectory.jsonl"
IMG_SIZE = 128


# ── BEV renderer ──────────────────────────────────────────────────────────────

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


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(model_path, device):
    print(f"Loading base model {MODEL_ID}…")
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)
    print(f"Loading LoRA adapter from {model_path}…")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def infer_action(model, processor, image, instruction, device):
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": instruction}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40,
                             do_sample=True, temperature=0.3)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw_full = processor.decode(new_tokens, skip_special_tokens=False)
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
    if not raw:
        raw = re.sub(r"<[^>]+>", " ", raw_full).strip()
    linear, angular, ok = parse_action(raw)
    return linear, angular, raw, ok


def parse_action(text):
    m = re.search(r"linear=([+-]?\d*\.?\d+).*?angular=([+-]?\d*\.?\d+)", text)
    if m:
        return (max(-1.0, min(1.0, float(m.group(1)))),
                max(-1.0, min(1.0, float(m.group(2)))),
                True)
    return 0.0, 0.0, False


# ── sensor file reading ───────────────────────────────────────────────────────

def read_sensor(last_t):
    """Return sensor dict if newer than last_t, else None."""
    if not SENSOR_FILE.exists():
        return None
    try:
        d = json.loads(SENSOR_FILE.read_text())
        if d.get("t", 0) > last_t:
            return d
    except Exception:
        pass
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvlm_lora")
    ap.add_argument("--poll", type=float, default=0.2,
                    help="Seconds between sensor file polls")
    ap.add_argument("--sensor-timeout", type=float, default=60.0,
                    help="Seconds to wait for first sensor data from bridge")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model, processor = load_model(args.model, device)

    # wait for bridge to start
    print(f"Waiting for bridge sensor data in {SENSOR_FILE}…")
    t_wait = time.time()
    while not SENSOR_FILE.exists():
        if time.time() - t_wait > args.sensor_timeout:
            print("ERROR: no sensor file after timeout. Is the bridge running?")
            return
        time.sleep(0.5)
    print("Sensor data found. Starting inference loop.")

    traj = TRAJ_FILE.open("w", encoding="utf-8")
    step = 0
    last_sensor_t = 0.0
    t0 = time.time()

    try:
        while True:
            # wait for fresh sensor reading
            sensor = None
            t_wait = time.time()
            while sensor is None:
                sensor = read_sensor(last_sensor_t)
                if sensor is None:
                    if time.time() - t_wait > 30.0:
                        print("No new sensor data for 30s — bridge finished?")
                        return
                    time.sleep(args.poll)

            last_sensor_t = sensor["t"]
            lidar = sensor["lidar"]
            p = sensor["pose"]
            pose = Pose2D(p["x"], p["y"], p["yaw"])
            task = sensor.get("task", {})
            phase = task.get("phase", "navigate")
            goal_name = task.get("goal_name", "goal")
            gx = task.get("goal", {}).get("x", 0.0)
            gy = task.get("goal", {}).get("y", 0.0)
            product_arg = task.get("product")
            cargo_arg = task.get("cargo")

            dist = math.hypot(gx - pose.x, gy - pose.y)
            instruction = make_instruction(
                phase, goal_name, gx, gy, pose,
                product=product_arg, cargo=cargo_arg,
            )

            # VLA inference
            t_inf = time.time()
            bev = lidar_to_bev(lidar, pose.yaw, gx, gy, pose.x, pose.y)
            linear, angular, raw, parse_ok = infer_action(
                model, processor, bev, instruction, device
            )
            dt_ms = (time.time() - t_inf) * 1000

            source = "VLA"
            if not parse_ok:
                # A* fallback (needs known grid — use empty grid, just aim toward goal)
                target_yaw = math.atan2(gy - pose.y, gx - pose.x)
                yaw_delta = normalize_angle(target_yaw - pose.yaw)
                linear = max(0.0, math.cos(yaw_delta))
                angular = max(-1.0, min(1.0, yaw_delta / YAW_MAX))
                source = "fallback"

            # write command for bridge
            CMD_FILE.write_text(json.dumps({
                "linear": round(linear, 4),
                "angular": round(angular, 4),
                "t": time.time(),
            }))

            # trajectory log
            traj.write(json.dumps({
                "step": step,
                "t": round(time.time() - t0, 3),
                "pose": {"x": round(pose.x, 3), "y": round(pose.y, 3),
                         "yaw": round(pose.yaw, 4)},
                "goal": {"x": round(gx, 3), "y": round(gy, 3), "name": goal_name},
                "phase": phase,
                "dist": round(dist, 3),
                "model_raw": raw,
                "source": source,
                "action": {"linear": round(linear, 4), "angular": round(angular, 4)},
            }, separators=(",", ":")) + "\n")
            traj.flush()

            src_tag = f"[{source}]" if source != "VLA" else "      "
            print(
                f"[{step:>4}] {phase:<8}  "
                f"pos=({pose.x:+6.2f},{pose.y:+6.2f})  "
                f"dist={dist:5.2f}m\n"
                f"         model → \"{raw}\"\n"
                f"         {src_tag} lin={linear:+.3f} ang={angular:+.3f}  "
                f"{dt_ms:.0f}ms",
                flush=True,
            )
            step += 1

    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps. Trajectory → {TRAJ_FILE}")
    finally:
        traj.close()


if __name__ == "__main__":
    main()
