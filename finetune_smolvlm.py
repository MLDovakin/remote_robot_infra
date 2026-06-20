#!/usr/bin/env python3
"""
Fine-tune SmolVLM-500M-Instruct on warehouse navigation data with LoRA.

Reads raw dataset.jsonl directly — render_dataset.py NOT needed.
BEV images are rendered on the fly (fast, no extra disk space).

Setup (once):
  conda create -n vla python=3.11 -y
  conda activate vla
  pip install torch torchvision transformers peft accelerate pillow numpy

Quick smoke test (500 samples, 1 epoch, ~5 min on MPS):
  python3 finetune_smolvlm.py --dataset aic_results/vla_dataset/dataset.jsonl \
      --output smolvlm_lora --max-samples 500 --epochs 1

Full run (50k samples, 3 epochs, ~6-10h on M2 Pro MPS):
  python3 finetune_smolvlm.py --dataset aic_results/vla_dataset/dataset.jsonl \
      --output smolvlm_lora --max-samples 50000 --epochs 3
"""
import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

# ── BEV renderer (egocentric, robot faces UP) ─────────────────────────────────
LIDAR_RAYS = 144
LIDAR_RANGE = 5.2
IMG_SIZE = 128


def lidar_to_bev(lidar_ranges, pose_yaw, goal_x, goal_y, pose_x, pose_y):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    scale = IMG_SIZE / (LIDAR_RANGE * 2)

    # free-space dots along each ray
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

    # obstacle hits
    for i, r in enumerate(lidar_ranges):
        if r >= LIDAR_RANGE - 0.05:
            continue
        angle = -math.pi + (2 * math.pi * i / LIDAR_RAYS)
        px = int(cx - math.sin(angle) * r * scale)
        py = int(cy - math.cos(angle) * r * scale)
        if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
            draw.ellipse([px - 1, py - 1, px + 2, py + 2], fill=(230, 210, 50))

    # goal marker (world → robot frame)
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

    # robot (triangle pointing up = forward)
    draw.polygon([(cx, cy - 6), (cx - 4, cy + 4), (cx + 4, cy + 4)],
                 fill=(50, 200, 80))
    return img


# ── Dataset ───────────────────────────────────────────────────────────────────

class NavDataset(Dataset):
    """
    Lazy-loads from raw dataset.jsonl using byte offsets.
    Each __getitem__ reads one line and renders BEV on the fly.
    Memory: O(n_offsets * 8 bytes) = ~2 MB for 292k samples.
    """

    def __init__(self, jsonl_path: str, max_samples: int | None = None,
                 shuffle_seed: int = 42):
        self.path = Path(jsonl_path)
        self.offsets: list[int] = []
        with self.path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(offset)
                if max_samples and len(self.offsets) >= max_samples:
                    break
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.offsets)
        print(f"NavDataset: {len(self.offsets)} samples from {self.path.name}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with self.path.open("rb") as f:
            f.seek(self.offsets[idx])
            row = json.loads(f.readline())
        pose, goal = row["pose"], row["goal"]
        linear, angular = row["action"]
        img = lidar_to_bev(row["lidar"], pose[2], goal[0], goal[1],
                           pose[0], pose[1])
        action_text = f"linear={linear:.3f} angular={angular:.3f}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": row["instruction"]},
                ],
            },
            {"role": "assistant", "content": action_text},
        ]
        return {"image": img, "messages": messages}


# ── Collator ──────────────────────────────────────────────────────────────────

class NavCollator:
    """
    Tokenises (image + text) pairs for SmolVLM.

    WHY no truncation: SmolVLM expands every image to ~1088 image tokens
    internally (resizes to 384×384 before the vision encoder). Truncating
    to a small max_length cuts those tokens mid-sequence and the processor
    raises a mismatch error. Total sequence length per sample is typically
    ~1100-1150 tokens which is fine for the 500M model.

    Label masking: prompt tokens get -100 so the loss is computed only on
    the short action string ("linear=X angular=Y", ~10 tokens).
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]):
        images = [item["image"] for item in batch]

        # full conversation: prompt + response
        full_texts = [
            self.processor.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            for item in batch
        ]
        # prompt only — used to measure where the response begins in the tokens
        prompt_texts = [
            self.processor.apply_chat_template(
                item["messages"][:1], tokenize=False, add_generation_prompt=True
            )
            for item in batch
        ]

        # No truncation — image tokens must not be cut off
        enc = self.processor(
            text=full_texts, images=images,
            return_tensors="pt", padding=True,
        )
        prompt_enc = self.processor(
            text=prompt_texts, images=images,
            return_tensors="pt", padding=True,
        )

        labels = enc["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        for i in range(len(batch)):
            labels[i][labels[i] == pad_id] = -100
            # everything up to end of prompt → -100 (no loss on prompt/image)
            prompt_len = int((prompt_enc["attention_mask"][i] == 1).sum())
            labels[i, :prompt_len] = -100

        enc["labels"] = labels
        return enc


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset.jsonl from collect_dataset.py")
    ap.add_argument("--output", default="smolvlm_warehouse_lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Per-device batch size (1 safe for 18 GB MPS; vision encoder is heavy)")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="Gradient accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=500,
                    help="Save checkpoint every N gradient updates")
    args = ap.parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32   # bfloat16 unstable on some MPS builds
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    # ── model + processor (load before DataLoader so collator gets processor) ──
    print(f"Loading {MODEL_ID}…")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype
    )

    # ── data ──────────────────────────────────────────────────────────────────
    ds = NavDataset(args.dataset, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=NavCollator(processor),
        num_workers=0,
        pin_memory=False,
    )
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Freeze the SigLIP vision encoder — we only fine-tune the LLM with LoRA.
    # WHY: (1) SigLIP forward on SmolVLM's internal 384×384 tiles uses ~8 GB on
    # MPS even at batch=1; freezing removes the backward pass through it entirely.
    # (2) The LLM already receives reasonable visual features; we need it to
    # learn navigation reasoning, not retrain the vision encoder on BEV images.
    frozen = 0
    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Frozen vision encoder: {frozen / 1e6:.1f}M params")

    # Gradient checkpointing: re-compute activations during backward instead of
    # storing them. Reduces peak memory ~40% at the cost of ~20% slower backward.
    model.enable_input_require_grads()   # required before gradient_checkpointing_enable with peft
    model.gradient_checkpointing_enable()

    model.print_trainable_parameters()
    model.to(device)
    model.train()

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_grad_steps = (len(loader) * args.epochs) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_grad_steps)
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── training ──────────────────────────────────────────────────────────────
    global_step = 0
    running_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for batch_i, raw_batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in raw_batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum

            if (batch_i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / (args.grad_accum * 10)
                    running_loss = 0.0
                    steps_left = total_grad_steps - global_step
                    eta_min = steps_left * (elapsed / max(global_step, 1)) / 60
                    print(
                        f"ep={epoch + 1}/{args.epochs}  "
                        f"step={global_step}/{total_grad_steps}  "
                        f"loss={avg_loss:.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}  "
                        f"ETA={eta_min:.0f}m"
                    )

                if global_step % args.save_steps == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt))
                    print(f"  → checkpoint saved: {ckpt}")

    # ── save final ────────────────────────────────────────────────────────────
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    elapsed_min = (time.time() - t0) / 60
    print(f"\nDone in {elapsed_min:.0f} min  →  {out_dir}/")
    print(f"\nInference example:")
    print(f"  from peft import PeftModel")
    print(f"  from transformers import AutoModelForImageTextToText, AutoProcessor")
    print(f"  base = AutoModelForImageTextToText.from_pretrained('{MODEL_ID}')")
    print(f"  model = PeftModel.from_pretrained(base, '{out_dir}')")


if __name__ == "__main__":
    main()
