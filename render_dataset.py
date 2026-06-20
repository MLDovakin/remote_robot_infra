#!/usr/bin/env python3
"""
Render BEV lidar images from collected dataset.jsonl and create
a HuggingFace-compatible dataset for SmolVLM fine-tuning.

Run on Mac after data collection:
  pip install pillow numpy datasets
  python3 render_dataset.py \
      --input aic_results/vla_dataset/dataset.jsonl \
      --output aic_results/vla_hf_dataset

Dataset format written:
  smolvlm_chat.jsonl  ← messages[] + image_b64 for each step
  hf_dataset/         ← HuggingFace Dataset (arrow format), optional via --hf

What the BEV image shows (egocentric, robot-centric):
  • Robot is at center, facing UP (forward = top of image)
  • Yellow dots = lidar obstacle hits
  • Dark grey rays = free space traced by lidar
  • Green triangle = robot heading
  • Red dot = goal direction (capped at image edge)
"""
import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError:
    print("Missing deps: pip install pillow numpy", file=sys.stderr)
    sys.exit(1)

LIDAR_RAYS = 144
LIDAR_RANGE = 5.2
IMG_SIZE = 128

# colours
BG = (18, 18, 22)
FREE = (38, 48, 56)
HIT = (230, 210, 50)
ROBOT_COL = (50, 200, 80)
GOAL_COL = (220, 60, 60)


# ── BEV renderer ─────────────────────────────────────────────────────────────

def lidar_to_bev(lidar_ranges, pose_yaw, goal_x, goal_y, pose_x, pose_y,
                 size=IMG_SIZE):
    """
    Egocentric BEV: robot always faces UP.
    goal_x/y are in world frame; converted to robot frame inside.

    Coordinate mapping (robot frame → pixel):
      forward (+x_robot) → UP   → row decreases
      left    (+y_robot) → LEFT → col decreases
      pixel_col = cx - robot_y * scale
      pixel_row = cy - robot_x * scale
    """
    img = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    scale = size / (LIDAR_RANGE * 2)   # px per metre

    # ── free-space gradient along each ray ───────────────────────────────────
    for i, r in enumerate(lidar_ranges):
        angle = -math.pi + (2 * math.pi * i / LIDAR_RAYS)  # robot frame
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        n_dots = max(2, int(r / LIDAR_RANGE * 24))
        for k in range(1, n_dots):
            frac = k / n_dots * r
            rx = cos_a * frac   # robot +x = forward
            ry = sin_a * frac   # robot +y = left
            px = int(cx - ry * scale)
            py = int(cy - rx * scale)
            if 0 <= px < size and 0 <= py < size:
                draw.point((px, py), fill=FREE)

    # ── obstacle hit dots ─────────────────────────────────────────────────────
    for i, r in enumerate(lidar_ranges):
        if r >= LIDAR_RANGE - 0.05:
            continue
        angle = -math.pi + (2 * math.pi * i / LIDAR_RAYS)
        rx = math.cos(angle) * r
        ry = math.sin(angle) * r
        px = int(cx - ry * scale)
        py = int(cy - rx * scale)
        if 0 <= px < size and 0 <= py < size:
            draw.ellipse([px - 1, py - 1, px + 2, py + 2], fill=HIT)

    # ── goal marker (world → robot frame) ────────────────────────────────────
    gdx = goal_x - pose_x
    gdy = goal_y - pose_y
    yaw = pose_yaw
    # rotate into robot frame: forward = x_rob, left = y_rob
    grx = math.cos(-yaw) * gdx - math.sin(-yaw) * gdy
    gry = math.sin(-yaw) * gdx + math.cos(-yaw) * gdy
    goal_dist = math.hypot(grx, gry)
    if goal_dist > 0.1:
        # clamp to 90 % of image radius
        cap = LIDAR_RANGE * 0.9
        factor = min(1.0, cap / goal_dist)
        gpx = int(cx - gry * factor * scale)
        gpy = int(cy - grx * factor * scale)
        gpx = max(4, min(size - 5, gpx))
        gpy = max(4, min(size - 5, gpy))
        draw.line([(cx, cy), (gpx, gpy)], fill=GOAL_COL, width=1)
        draw.ellipse([gpx - 4, gpy - 4, gpx + 4, gpy + 4],
                     fill=GOAL_COL, outline=(255, 120, 120))

    # ── robot triangle (forward = up) ─────────────────────────────────────────
    draw.polygon([(cx, cy - 6), (cx - 4, cy + 4), (cx + 4, cy + 4)],
                 fill=ROBOT_COL)

    return img


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


# ── dataset builder ───────────────────────────────────────────────────────────

def build_smolvlm_messages(instruction: str, action: str) -> list:
    """SmolVLM / Idefics3 chat format — returns a list of message dicts."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": action,
        },
    ]


def main():
    ap = argparse.ArgumentParser(description="Render BEV images → SmolVLM dataset")
    ap.add_argument("--input", required=True,
                    help="dataset.jsonl from collect_dataset.py")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Limit rows (useful for quick tests)")
    ap.add_argument("--phases", nargs="+",
                    default=["explore", "pickup", "dropoff"],
                    choices=["explore", "pickup", "dropoff"],
                    help="Which phases to include")
    ap.add_argument("--hf", action="store_true",
                    help="Also save as HuggingFace Dataset (needs `datasets`)")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_out = out_dir / "smolvlm_chat.jsonl"

    phase_set = set(args.phases)
    records = []

    with open(args.input, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if args.max_steps and n >= args.max_steps:
                break
            row = json.loads(line)
            if row["phase"] not in phase_set:
                continue

            lidar = row["lidar"]
            pose = row["pose"]    # [x, y, yaw]
            goal = row["goal"]    # [gx, gy]
            linear, angular = row["action"]
            action_text = f"linear={linear:.3f} angular={angular:.3f}"

            img = lidar_to_bev(
                lidar, pose[2], goal[0], goal[1], pose[0], pose[1],
                size=args.img_size,
            )

            records.append({
                "image_b64": img_to_b64(img),
                "instruction": row["instruction"],
                "action": action_text,
                "episode": row["episode"],
                "step": row["step"],
                "phase": row["phase"],
                "messages": build_smolvlm_messages(row["instruction"], action_text),
            })

            if len(records) % 2000 == 0 and records:
                print(f"  rendered {len(records)} steps...", flush=True)

    # ── save JSONL ────────────────────────────────────────────────────────────
    with jsonl_out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "image_b64": r["image_b64"],
                "instruction": r["instruction"],
                "action": r["action"],
                "episode": r["episode"],
                "step": r["step"],
                "phase": r["phase"],
                "messages": r["messages"],
            }, separators=(",", ":")) + "\n")
    print(f"Saved {len(records)} rows → {jsonl_out}")

    # ── optionally save HuggingFace Dataset ───────────────────────────────────
    if args.hf:
        try:
            from datasets import Dataset, Features, Value, Image as HFImage
        except ImportError:
            print("Install `datasets` for --hf: pip install datasets")
            return
        hf_images = [
            Image.open(io.BytesIO(base64.b64decode(r["image_b64"])))
            for r in records
        ]
        ds = Dataset.from_dict({
            "image": hf_images,
            "instruction": [r["instruction"] for r in records],
            "action": [r["action"] for r in records],
            "episode": [r["episode"] for r in records],
            "step": [r["step"] for r in records],
            "phase": [r["phase"] for r in records],
        })
        hf_path = out_dir / "hf_dataset"
        ds.save_to_disk(str(hf_path))
        print(f"HuggingFace Dataset saved → {hf_path}")
        print(f"  {ds}")


if __name__ == "__main__":
    main()
