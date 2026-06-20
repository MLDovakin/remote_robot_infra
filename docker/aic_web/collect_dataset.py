#!/usr/bin/env python3
"""
Headless dataset collector for VLA fine-tuning.

Runs the warehouse simulation WITHOUT Gazebo — pure Python, ~1000x faster.
Records (lidar_ranges, pose, goal, instruction, action) at every step.
Saves as JSONL compatible with render_dataset.py → SmolVLM fine-tuning.

Inside container:
  python3 /opt/aic_web/collect_dataset.py --episodes 200 --out /workspace/aic_results/vla_dataset

From Mac (mounts volume):
  docker compose exec web python3 /opt/aic_web/collect_dataset.py --episodes 200
"""
import argparse
import heapq
import json
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

# ── constants mirroring warehouse_lidar_random_mode.py ──────────────────────
ORIGIN_X = -13.0
ORIGIN_Y = -9.0
WIDTH_M = 26.0
HEIGHT_M = 18.0
RESOLUTION = 0.12
GRID_W = int(WIDTH_M / RESOLUTION)
GRID_H = int(HEIGHT_M / RESOLUTION)
ROBOT_RADIUS = 0.48
LIDAR_RANGE = 5.2
LIDAR_RAYS = 144
POSE_SPACING = 2.20
MAPPED_COVERAGE = 0.82
MAX_STEPS_PER_EPISODE = 4000

# ── action space (continuous, informative) ──────────────────────────────────
# linear  ∈ [0, 1]  = forward displacement this control step / STEP_MAX
# angular ∈ [-1, 1] = heading change this control step / YAW_MAX
# Both channels carry real signal (the old cos-angle linear was ~1.0 always).
STEP_MAX = 0.25          # metres travelled per control step at full speed
YAW_MAX = 0.60           # radians turned per control step at full lock

# ── pure-pursuit expert controller ──────────────────────────────────────────
LOOKAHEAD = 0.9          # metres ahead on the path to steer toward (smooths jagged A*)
K_HEADING = 0.7          # proportional gain: heading error → turn rate
MIN_SPEED_FRAC = 0.15    # floor on speed while moving (fraction of STEP_MAX)
NOISE_YAW = 0.04         # rad gaussian noise on steering (demonstration noise)
NOISE_SPEED = 0.10       # fractional gaussian noise on speed
GOAL_RADIUS = 0.45       # metres: a path goal is reached
COLLISION_INFLATE = 0.12 # hard obstacle inflation for recovery check; A* still
                         # plans on the ROBOT_RADIUS-inflated grid, so cutting
                         # into the safety margin is NOT treated as a collision
ALIGN_TOL = 0.08         # rad: in-place rotation tolerance at goals
EXPLORE_STEP_CAP = 900   # cap explore steps/episode so tasks aren't drowned
MAX_PRODUCTS_PER_EP = 3  # pick up to this many products per episode
PERTURB_PROB = 0.35      # chance to nudge pose before a task leg (recovery data)

# ── balancing (de-collapse the dominant 'go straight' action) ────────────────
STRAIGHT_ANG = 0.08      # |angular| below this (with high linear) = "straight"
STRAIGHT_LIN = 0.60
TARGET_STRAIGHT_FRAC = 0.40   # cap straight steps to this fraction of the set

DISPATCH_ZONES = [
    {"name": "DispatchA", "x": -8.4, "y": -6.5, "yaw": math.pi},
    {"name": "DispatchB", "x": -8.4, "y": 7.0, "yaw": math.pi},
]


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class Rect:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "obstacle"

    def normalized(self):
        return Rect(self.name, min(self.x1, self.x2), min(self.y1, self.y2),
                    max(self.x1, self.x2), max(self.y1, self.y2), self.kind)

    def inflated(self, amount):
        r = self.normalized()
        return Rect(r.name, r.x1 - amount, r.y1 - amount,
                    r.x2 + amount, r.y2 + amount, r.kind)

    def contains(self, x, y):
        r = self.normalized()
        return r.x1 <= x <= r.x2 and r.y1 <= y <= r.y2


# ── geometry helpers ─────────────────────────────────────────────────────────

def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def world_to_grid(x, y):
    gx = int((x - ORIGIN_X) / RESOLUTION)
    gy = int((y - ORIGIN_Y) / RESOLUTION)
    return max(0, min(GRID_W - 1, gx)), max(0, min(GRID_H - 1, gy))


def grid_to_world(gx, gy):
    return ORIGIN_X + (gx + 0.5) * RESOLUTION, ORIGIN_Y + (gy + 0.5) * RESOLUTION


def rect_occupancy(rects, inflate=0.0):
    occupied = set()
    for rect in rects:
        r = rect.inflated(inflate).normalized()
        gx1, gy1 = world_to_grid(r.x1, r.y1)
        gx2, gy2 = world_to_grid(r.x2, r.y2)
        for gy in range(min(gy1, gy2), max(gy1, gy2) + 1):
            for gx in range(min(gx1, gx2), max(gx1, gx2) + 1):
                occupied.add((gx, gy))
    return occupied


def reachable_free_cells(start, occupied):
    start_cell = world_to_grid(start.x, start.y)
    q = [start_cell]
    seen = {start_cell}
    for cell in q:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in seen:
                continue
            if not (0 <= nxt[0] < GRID_W and 0 <= nxt[1] < GRID_H and nxt not in occupied):
                continue
            seen.add(nxt)
            q.append(nxt)
    return seen


def coverage(known, reachable_free):
    if not reachable_free:
        return 0.0
    return sum(1 for gx, gy in reachable_free if known[gy][gx] != -1) / len(reachable_free)


def known_occupied(known):
    occ = set()
    for gy, row in enumerate(known):
        for gx, v in enumerate(row):
            if v == 1:
                occ.add((gx, gy))
    return occ


def cell_free(cell, occupied):
    return (0 <= cell[0] < GRID_W and 0 <= cell[1] < GRID_H
            and cell not in occupied)


# ── A* planner ───────────────────────────────────────────────────────────────

def astar(start_pose, goal_pose, occupied):
    start = world_to_grid(start_pose.x, start_pose.y)
    goal = world_to_grid(goal_pose.x, goal_pose.y)
    occ = set(occupied)
    occ.discard(start)
    occ.discard(goal)
    open_set = [(0.0, start)]
    came_from = {}
    gscore = {start: 0.0}
    neighbors = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                 (-1, -1, 1.41), (-1, 1, 1.41), (1, -1, 1.41), (1, 1, 1.41)]
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            return [Pose2D(*grid_to_world(x, y)) for x, y in cells]
        for dx, dy, cost in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not cell_free(nxt, occ):
                continue
            if dx and dy and ((current[0] + dx, current[1]) in occ
                              or (current[0], current[1] + dy) in occ):
                continue
            tentative = gscore[current] + cost
            if tentative >= gscore.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            gscore[nxt] = tentative
            heapq.heappush(open_set, (tentative + math.hypot(
                nxt[0] - goal[0], nxt[1] - goal[1]), nxt))
    return []


# ── lidar simulation ─────────────────────────────────────────────────────────

def simulate_lidar_ranges(pose, true_occupied, known):
    """Return 144 range values (m). Updates known occupancy as side effect."""
    ranges = []
    px, py = world_to_grid(pose.x, pose.y)
    max_steps = int(LIDAR_RANGE / RESOLUTION)
    for i in range(LIDAR_RAYS):
        angle = pose.yaw - math.pi + (2 * math.pi * i / LIDAR_RAYS)
        hit_range = LIDAR_RANGE
        for step in range(1, max_steps + 1):
            x = pose.x + math.cos(angle) * step * RESOLUTION
            y = pose.y + math.sin(angle) * step * RESOLUTION
            gx, gy = world_to_grid(x, y)
            if gx <= 0 or gy <= 0 or gx >= GRID_W - 1 or gy >= GRID_H - 1:
                hit_range = step * RESOLUTION
                break
            known[gy][gx] = 0
            if (gx, gy) in true_occupied:
                known[gy][gx] = 1
                hit_range = step * RESOLUTION
                break
        ranges.append(round(hit_range, 3))
    known[py][px] = 0
    return ranges


# ── frontier exploration ─────────────────────────────────────────────────────

def frontier_goal(pose, known, planning_occupied, reachable):
    candidates = []
    for gx, gy in reachable:
        if known[gy][gx] != 0:
            continue
        if any(0 <= gx + dx < GRID_W and 0 <= gy + dy < GRID_H
               and known[gy + dy][gx + dx] == -1
               for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            wx, wy = grid_to_world(gx, gy)
            dist = math.hypot(wx - pose.x, wy - pose.y)
            candidates.append((dist, gx, gy, Pose2D(wx, wy)))
    candidates.sort(key=lambda c: c[0], reverse=True)
    for _, _, _, candidate in candidates[:60]:
        path = astar(pose, candidate, planning_occupied)
        if len(path) > 2:
            return candidate, path
    return None, []


# ── world generator (identical to warehouse_lidar_random_mode.py) ─────────────

def _random_start(rng, occupied):
    """Pick a random free, varied start pose (random heading)."""
    candidates = [
        (-10.6, -7.0), (-9.6, -6.8), (10.4, -7.0), (-10.6, 7.0),
        (10.4, 7.0), (0.0, -7.2), (0.0, 7.2), (-10.6, 0.0), (10.4, 0.0),
    ]
    rng.shuffle(candidates)
    for x, y in candidates:
        if world_to_grid(x, y) not in occupied:
            return Pose2D(x, y, rng.uniform(-math.pi, math.pi))
    for _ in range(500):
        x = rng.uniform(ORIGIN_X + 1.0, ORIGIN_X + WIDTH_M - 1.0)
        y = rng.uniform(ORIGIN_Y + 1.0, ORIGIN_Y + HEIGHT_M - 1.0)
        if world_to_grid(x, y) not in occupied:
            return Pose2D(x, y, rng.uniform(-math.pi, math.pi))
    return Pose2D(-10.6, -7.0, 0.0)


def make_random_world(seed):
    """Diversified warehouse: randomized shelf rows/cols, sizes, jitter, clutter
    density, product set, and start pose. Map dimensions stay fixed so the grid
    constants hold; only the layout varies for scenario diversity."""
    rng = random.Random(seed)
    rects = [
        Rect("front_wall", ORIGIN_X, ORIGIN_Y, ORIGIN_X + WIDTH_M, ORIGIN_Y + 0.18, "wall"),
        Rect("back_wall", ORIGIN_X, ORIGIN_Y + HEIGHT_M - 0.18, ORIGIN_X + WIDTH_M, ORIGIN_Y + HEIGHT_M, "wall"),
        Rect("left_wall", ORIGIN_X, ORIGIN_Y, ORIGIN_X + 0.18, ORIGIN_Y + HEIGHT_M, "wall"),
        Rect("right_wall", ORIGIN_X + WIDTH_M - 0.18, ORIGIN_Y, ORIGIN_X + WIDTH_M, ORIGIN_Y + HEIGHT_M, "wall"),
    ]
    shelves = []
    products = {}
    product_names = ["ProductR", "ProductG", "ProductB", "ProductY", "ProductC"]
    all_rows = [-5.8, -3.2, -0.6, 2.0, 4.6, 6.6]
    all_cols = [-7.8, -4.6, -1.4, 1.8, 5.0, 8.2]
    rows = [y for y in all_rows if rng.random() > 0.15] or all_rows[:3]
    cols = [x for x in all_cols if rng.random() > 0.15] or all_cols[:3]
    skip_prob = rng.uniform(0.15, 0.45)
    shelf_index = 0
    for y in rows:
        for x in cols:
            if rng.random() < skip_prob:
                continue
            w = rng.uniform(1.2, 2.2)
            h = rng.uniform(0.55, 0.95)
            jx = rng.uniform(-0.25, 0.25)
            jy = rng.uniform(-0.25, 0.25)
            shelf_index += 1
            shelf = Rect(f"shelf_{shelf_index}", x + jx - w / 2, y + jy - h / 2,
                         x + jx + w / 2, y + jy + h / 2, "shelf")
            shelves.append(shelf)
            rects.append(shelf)
    for i in range(rng.randint(8, 28)):
        x = rng.uniform(-10.0, 10.0)
        y = rng.uniform(-7.0, 7.2)
        if abs(x + 10.5) < 1.6 and abs(y + 7.0) < 1.6:
            continue
        if any(r.contains(x, y) for r in rects):
            continue
        size = rng.uniform(0.40, 0.80)
        rects.append(Rect(f"pallet_stack_{i + 1}", x - size / 2, y - size / 2,
                          x + size / 2, y + size / 2, "crate"))
    rng.shuffle(shelves)
    for name, shelf in zip(product_names, shelves[:len(product_names)]):
        r = shelf.normalized()
        side_y = (r.y1 - 0.9 if abs(r.y1 - ORIGIN_Y) > abs(r.y2 - (ORIGIN_Y + HEIGHT_M))
                  else r.y2 + 0.9)
        slot_y = r.y1 + 0.18 if side_y < (r.y1 + r.y2) / 2 else r.y2 - 0.18
        products[name] = {
            "storage": shelf.name,
            "slot": {"x": (r.x1 + r.x2) / 2, "y": slot_y, "z": 1.38},
            "pickup": {
                "x": (r.x1 + r.x2) / 2,
                "y": side_y,
                "yaw": math.pi / 2 if side_y < (r.y1 + r.y2) / 2 else -math.pi / 2,
            },
        }
    occupied = rect_occupancy(rects, ROBOT_RADIUS)
    start = _random_start(rng, occupied)
    return {"seed": seed, "rects": rects, "products": products, "start": start}


# ── action + instruction helpers ─────────────────────────────────────────────

def compute_action(prev: Pose2D, curr: Pose2D):
    """
    Continuous normalized action recovered from one control step.
      linear  ∈ [0, 1]  : forward displacement / STEP_MAX
      angular ∈ [-1, 1] : heading change / YAW_MAX
    Inverse of the inference-time integrator (rotate by angular*YAW_MAX, then
    translate by linear*STEP_MAX along the new heading). Both channels vary:
    the expert slows for turns/near goals and rotates in place at targets.
    """
    dx = curr.x - prev.x
    dy = curr.y - prev.y
    dist = math.hypot(dx, dy)
    delta_yaw = normalize_angle(curr.yaw - prev.yaw)
    linear = round(max(0.0, min(1.0, dist / STEP_MAX)), 4)
    angular = round(max(-1.0, min(1.0, delta_yaw / YAW_MAX)), 4)
    return linear, angular


def _direction_hint(pose: Pose2D, gx, gy):
    rel = normalize_angle(math.atan2(gy - pose.y, gx - pose.x) - pose.yaw)
    if abs(rel) < 0.4:
        return "ahead"
    if abs(rel) > 2.7:
        return "behind"
    return "left" if rel > 0 else "right"


def make_instruction(phase, goal_name, gx, gy, pose: Pose2D,
                     product=None, cargo=None, rng=None):
    """Templated but diversified instruction. Pass an rng for reproducible
    paraphrase choice; defaults to the global random (fine at inference)."""
    r = rng or random
    dist = round(math.hypot(gx - pose.x, gy - pose.y), 1)
    hint = _direction_hint(pose, gx, gy)
    pos = f"x={pose.x:.1f} y={pose.y:.1f}"
    coord = f"({gx:.1f},{gy:.1f})"
    if phase == "explore":
        options = [
            f"Explore the warehouse. Current frontier: {coord}, {dist}m {hint}. Robot at {pos}.",
            f"Map the area. Move toward the unexplored frontier {coord}, {dist}m {hint}.",
            f"Keep exploring. Head to {coord}, about {dist}m to the {hint}.",
            f"Scan for shelves. Next frontier {coord} is {dist}m {hint}.",
            f"Expand the map: drive to frontier {coord}, {dist}m {hint}. Robot at {pos}.",
        ]
    elif phase == "pickup":
        options = [
            f"Navigate to pick up {product} from {goal_name}. Pickup: {coord}, {dist}m {hint}. Robot at {pos}.",
            f"Go to {goal_name} and pick up {product}. Target {coord}, {dist}m {hint}.",
            f"Retrieve {product} stored at {goal_name} {coord}, {dist}m {hint}.",
            f"Head to the {goal_name} shelf to grab {product}, {dist}m {hint}.",
            f"Pick up {product}: drive to {coord} ({goal_name}), {dist}m to your {hint}.",
        ]
    elif phase == "dropoff":
        options = [
            f"Deliver {cargo} to {goal_name}. Drop-off: {coord}, {dist}m {hint}. Robot at {pos}.",
            f"Bring {cargo} to {goal_name} at {coord}, {dist}m {hint}.",
            f"Drop off {cargo} at {goal_name} {coord}, {dist}m {hint}.",
            f"Transport {cargo} to the {goal_name} dispatch zone, {dist}m {hint}.",
            f"Take {cargo} to {coord} ({goal_name}), {dist}m {hint} of you.",
        ]
    else:
        options = [f"Navigate to {coord}.", f"Drive to {coord}, {dist}m {hint}."]
    return r.choice(options)


# ── path follower that collects dataset steps ─────────────────────────────────

def _step_record(phase, goal_name, gx, gy, cur, lidar, action, rng, product, cargo):
    linear, angular = action
    return {
        "phase": phase,
        "instruction": make_instruction(phase, goal_name, gx, gy, cur,
                                        product, cargo, rng=rng),
        "lidar": lidar,
        "pose": [round(cur.x, 3), round(cur.y, 3), round(cur.yaw, 4)],
        "goal": [round(gx, 3), round(gy, 3)],
        "action": [linear, angular],
    }


def drive_to(path, pose, known, true_occupied, collide_occupied,
             phase, goal_name, gx, gy, rng,
             product=None, cargo=None, max_steps=600):
    """Pure-pursuit follow of an A* path with continuous, noisy control.

    Each control step: steer toward a lookahead point (continuous turn rate),
    modulate speed by heading error + distance-to-goal, add demonstration
    noise, and recover when the next cell is blocked. Records (obs at current
    pose, action moving to the next pose). Produces smooth, varied actions
    instead of the old grid-quantized ones.
    """
    steps = []
    cur = pose
    if len(path) < 2:
        return steps, cur
    pts = path
    idx = 1
    goal_pt = pts[-1]
    stuck = 0
    for _ in range(max_steps):
        while idx < len(pts) - 1 and math.hypot(pts[idx].x - cur.x, pts[idx].y - cur.y) < LOOKAHEAD:
            idx += 1
        target = pts[idx]
        dist_goal = math.hypot(goal_pt.x - cur.x, goal_pt.y - cur.y)
        if idx >= len(pts) - 1 and dist_goal < GOAL_RADIUS:
            break
        desired = math.atan2(target.y - cur.y, target.x - cur.x)
        herr = normalize_angle(desired - cur.yaw)
        omega = max(-YAW_MAX, min(YAW_MAX, K_HEADING * herr + rng.gauss(0, NOISE_YAW)))
        turn_slow = max(MIN_SPEED_FRAC, math.cos(max(-math.pi / 2, min(math.pi / 2, herr))))
        goal_slow = min(1.0, dist_goal / (3 * STEP_MAX))
        speed = STEP_MAX * turn_slow * max(MIN_SPEED_FRAC, goal_slow) * (1.0 + rng.gauss(0, NOISE_SPEED))
        speed = max(0.0, min(STEP_MAX, speed))
        new_yaw = normalize_angle(cur.yaw + omega)
        nx = cur.x + math.cos(new_yaw) * speed
        ny = cur.y + math.sin(new_yaw) * speed
        if world_to_grid(nx, ny) in collide_occupied:
            # recovery: turn away and ease back instead of driving in
            new_yaw = normalize_angle(cur.yaw + YAW_MAX * rng.choice((-1.0, 1.0)))
            nx = cur.x - math.cos(cur.yaw) * STEP_MAX * 0.4
            ny = cur.y - math.sin(cur.yaw) * STEP_MAX * 0.4
            if world_to_grid(nx, ny) in collide_occupied:
                nx, ny = cur.x, cur.y
            stuck += 1
            if stuck > 12:
                break
        else:
            stuck = 0
        newpose = Pose2D(nx, ny, new_yaw)
        lidar = simulate_lidar_ranges(cur, true_occupied, known)
        action = compute_action(cur, newpose)
        steps.append(_step_record(phase, goal_name, gx, gy, cur, lidar, action,
                                  rng, product, cargo))
        cur = newpose
    return steps, cur


def align_to_yaw(target_yaw, pose, known, true_occupied, phase, goal_name, gx, gy,
                 rng, product=None, cargo=None, max_steps=24):
    """In-place rotation to a target heading — pure-rotation (linear≈0) samples
    that further diversify the action distribution at pickup/drop poses."""
    steps = []
    cur = pose
    for _ in range(max_steps):
        err = normalize_angle(target_yaw - cur.yaw)
        if abs(err) < ALIGN_TOL:
            break
        omega = max(-YAW_MAX, min(YAW_MAX, K_HEADING * err + rng.gauss(0, NOISE_YAW)))
        newpose = Pose2D(cur.x, cur.y, normalize_angle(cur.yaw + omega))
        lidar = simulate_lidar_ranges(cur, true_occupied, known)
        action = compute_action(cur, newpose)
        steps.append(_step_record(phase, goal_name, gx, gy, cur, lidar, action,
                                  rng, product, cargo))
        cur = newpose
    return steps, cur


# ── episode runner ────────────────────────────────────────────────────────────

def _maybe_perturb(pose, true_occupied, rng):
    """Occasionally nudge the pose off the ideal line so the controller has to
    recover — generates corrective actions the A* expert never showed."""
    if rng.random() > PERTURB_PROB:
        return pose
    for _ in range(8):
        dx = rng.uniform(-0.4, 0.4)
        dy = rng.uniform(-0.4, 0.4)
        nyaw = normalize_angle(pose.yaw + rng.uniform(-1.0, 1.0))
        if world_to_grid(pose.x + dx, pose.y + dy) not in true_occupied:
            return Pose2D(pose.x + dx, pose.y + dy, nyaw)
    return Pose2D(pose.x, pose.y, normalize_angle(pose.yaw + rng.uniform(-1.0, 1.0)))


def run_episode(seed):
    world = make_random_world(seed)
    true_occupied = rect_occupancy(world["rects"], ROBOT_RADIUS)
    hard_occupied = rect_occupancy(world["rects"], COLLISION_INFLATE)
    reachable = reachable_free_cells(world["start"], true_occupied)
    known = [[-1] * GRID_W for _ in range(GRID_H)]
    pose = world["start"]
    brng = random.Random((seed * 2654435761) & 0xFFFFFFFF)  # behavior/noise stream
    all_steps = []

    # Bootstrap: initial lidar scan so frontier_goal finds cells on the border
    # between known-free and unknown. Without this, known is all -1.
    simulate_lidar_ranges(pose, true_occupied, known)

    # ── Phase 1: frontier exploration (capped) ───────────────────────────────
    explore_steps = 0
    for _ in range(600):
        if coverage(known, reachable) >= MAPPED_COVERAGE:
            break
        if explore_steps >= EXPLORE_STEP_CAP or len(all_steps) >= MAX_STEPS_PER_EPISODE:
            break
        planning_occ = true_occupied | known_occupied(known)
        goal, path = frontier_goal(pose, known, planning_occ, reachable)
        if not path:
            break
        steps, pose = drive_to(path, pose, known, true_occupied, hard_occupied,
                               "explore", "frontier", goal.x, goal.y, brng)
        all_steps.extend(steps)
        explore_steps += len(steps)
        simulate_lidar_ranges(pose, true_occupied, known)

    # ── Phases 2/3: pick up + deliver a random subset of products ─────────────
    products = world["products"]
    if products:
        names = list(products)
        brng.shuffle(names)
        k = brng.randint(1, min(MAX_PRODUCTS_PER_EP, len(names)))
        for product_name in names[:k]:
            if len(all_steps) >= MAX_STEPS_PER_EPISODE:
                break
            product = products[product_name]
            pickup = Pose2D(product["pickup"]["x"], product["pickup"]["y"],
                            product["pickup"]["yaw"])
            pose = _maybe_perturb(pose, true_occupied, brng)
            occ = true_occupied | known_occupied(known)
            p1 = astar(pose, pickup, occ)
            if p1:
                steps, pose = drive_to(p1, pose, known, true_occupied, hard_occupied,
                                       "pickup", product["storage"], pickup.x, pickup.y,
                                       brng, product=product_name)
                all_steps.extend(steps)
                steps, pose = align_to_yaw(pickup.yaw, pose, known, true_occupied,
                                           "pickup", product["storage"],
                                           pickup.x, pickup.y, brng,
                                           product=product_name)
                all_steps.extend(steps)
            if len(all_steps) >= MAX_STEPS_PER_EPISODE:
                break
            drop = brng.choice(DISPATCH_ZONES)
            drop_pose = Pose2D(drop["x"], drop["y"], drop["yaw"])
            pose = _maybe_perturb(pose, true_occupied, brng)
            occ = true_occupied | known_occupied(known)
            p2 = astar(pose, drop_pose, occ)
            if p2:
                steps, pose = drive_to(p2, pose, known, true_occupied, hard_occupied,
                                       "dropoff", drop["name"], drop_pose.x, drop_pose.y,
                                       brng, product=product_name, cargo=product_name)
                all_steps.extend(steps)
                steps, pose = align_to_yaw(drop_pose.yaw, pose, known, true_occupied,
                                           "dropoff", drop["name"],
                                           drop_pose.x, drop_pose.y, brng,
                                           product=product_name, cargo=product_name)
                all_steps.extend(steps)

    return all_steps


# ── main ──────────────────────────────────────────────────────────────────────

def _run_one(seed):
    return run_episode(seed)


def _is_straight(action):
    lin, ang = action
    return abs(ang) < STRAIGHT_ANG and lin >= STRAIGHT_LIN


def main():
    ap = argparse.ArgumentParser(description="Headless VLA dataset collector")
    ap.add_argument("--episodes", type=int, default=400,
                    help="Number of episodes / random scenarios")
    ap.add_argument("--out", default="/workspace/aic_results/vla_dataset",
                    help="Output directory")
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel worker processes (0 = all CPUs)")
    ap.add_argument("--no-balance", action="store_true",
                    help="Skip downsampling the dominant 'straight' action")
    ap.add_argument("--target-straight", type=float, default=TARGET_STRAIGHT_FRAC,
                    help="Target fraction of 'straight' steps after balancing")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dataset.jsonl"
    raw_file = out_dir / "dataset.raw.jsonl"

    seeds = [args.seed_start + i for i in range(args.episodes)]
    workers = args.workers or (os.cpu_count() or 1)
    workers = max(1, min(workers, len(seeds)))
    print(f"Generating {args.episodes} scenarios with {workers} worker(s)…", flush=True)

    t0 = time.time()
    raw_total = straight = turn = 0
    phase_raw: dict = {}

    def consume(ep_id, steps):
        nonlocal raw_total, straight, turn
        for i, step in enumerate(steps):
            row = {"episode": ep_id, "step": i, **step}
            rf.write(json.dumps(row, separators=(",", ":")) + "\n")
            phase_raw[step["phase"]] = phase_raw.get(step["phase"], 0) + 1
            if _is_straight(step["action"]):
                straight += 1
            else:
                turn += 1
        raw_total += len(steps)

    with raw_file.open("w", encoding="utf-8") as rf:
        if workers > 1:
            with mp.Pool(workers) as pool:
                for ep_id, steps in enumerate(pool.imap_unordered(_run_one, seeds)):
                    consume(ep_id, steps)
                    if (ep_id + 1) % 10 == 0 or ep_id + 1 == args.episodes:
                        el = time.time() - t0
                        print(f"  {ep_id + 1:>4}/{args.episodes} eps  "
                              f"raw_steps={raw_total:>8}  {(ep_id + 1) / el:.1f} ep/s",
                              flush=True)
        else:
            for ep_id, seed in enumerate(seeds):
                consume(ep_id, _run_one(seed))
                if (ep_id + 1) % 10 == 0 or ep_id + 1 == args.episodes:
                    el = time.time() - t0
                    print(f"  {ep_id + 1:>4}/{args.episodes} eps  "
                          f"raw_steps={raw_total:>8}  {(ep_id + 1) / el:.1f} ep/s",
                          flush=True)

    # ── balancing pass: downsample the dominant 'straight' action ─────────────
    keep_p = 1.0
    if not args.no_balance and straight > 0 and turn > 0:
        f = args.target_straight
        desired_straight = (f / (1.0 - f)) * turn
        keep_p = min(1.0, desired_straight / straight)
    brng = random.Random(98765)
    final_total = final_straight = 0
    phase_final: dict = {}
    action_keys = set()
    with out_file.open("w", encoding="utf-8") as of, raw_file.open(encoding="utf-8") as rf:
        for line in rf:
            row = json.loads(line)
            act = row["action"]
            st = _is_straight(act)
            if st and brng.random() > keep_p:
                continue
            of.write(line)
            final_total += 1
            phase_final[row["phase"]] = phase_final.get(row["phase"], 0) + 1
            if st:
                final_straight += 1
            action_keys.add((round(act[0], 2), round(act[1], 2)))
    raw_file.unlink(missing_ok=True)

    meta = {
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "raw_total_steps": raw_total,
        "total_steps": final_total,
        "lidar_rays": LIDAR_RAYS,
        "lidar_range_m": LIDAR_RANGE,
        "phase_counts": phase_final,
        "balanced": not args.no_balance,
        "straight_fraction": round(final_straight / max(1, final_total), 3),
        "distinct_action_pairs_2dp": len(action_keys),
        "action_space": (f"linear ∈ [0,1] (forward step / STEP_MAX={STEP_MAX}m), "
                         f"angular ∈ [-1,1] (yaw delta / YAW_MAX={YAW_MAX}rad)"),
        "observation_space": (f"lidar: {LIDAR_RAYS} floats (meters), "
                              "pose: [x, y, yaw], goal: [x, y]"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print(f"  raw steps      : {raw_total}")
    print(f"  kept steps     : {final_total}  (straight kept p={keep_p:.3f})")
    print(f"  straight frac  : {meta['straight_fraction']}  (was ~0.71 in v1)")
    print(f"  distinct acts  : {len(action_keys)} pairs @2dp  (was 16 in v1)")
    print(f"  phases         : {phase_final}")
    print(f"  output         : {out_file}  ({out_file.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
