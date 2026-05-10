#!/usr/bin/env python3
import heapq
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from warehouse_drive import GazeboDriveProjector, wheel_models


AIC_SETUP = Path("/ws_aic/install/setup.bash")
RUNS_DIR = Path(os.environ.get("AIC_RUNS_DIR", "/workspace/aic_runs"))
RESULTS_DIR = Path(os.environ.get("AIC_RESULTS_DIR", "/workspace/aic_results"))
MAP_DIR = RESULTS_DIR / "lidar_random_map"
WORLD = MAP_DIR / "lidar_random_world.sdf"
STATE_FILE = RUNS_DIR / "lidar_random_state.json"
TASK_FILE = RUNS_DIR / "lidar_random_task.json"

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
LIDAR_VISUAL_RAYS = 48
EXPLORE_STEP_SECONDS = 0.004
TASK_STEP_SECONDS = 0.006
ODOM_DT_SECONDS = 0.350
EXPLORE_SPEED = 3.4
TASK_SPEED = 3.0
POSE_SPACING = 2.20
GZ_SERVICE_TIMEOUT_MS = 1500
POSE_COMMAND_TIMEOUT_SECONDS = 2.0
MAPPED_COVERAGE = 0.82
HIDDEN_Z = -10.0
DRIVE = None

IGNORED_GAZEBO_LOG_LINES = (
    "NodeShared::RecvSrvRequest() error sending response: Host unreachable",
    "[SignalHandler.cc:178] [GUI] 4 Interrupted system call",
)


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
        return Rect(self.name, min(self.x1, self.x2), min(self.y1, self.y2), max(self.x1, self.x2), max(self.y1, self.y2), self.kind)

    def inflated(self, amount):
        r = self.normalized()
        return Rect(r.name, r.x1 - amount, r.y1 - amount, r.x2 + amount, r.y2 + amount, r.kind)

    def contains(self, x, y):
        r = self.normalized()
        return r.x1 <= x <= r.x2 and r.y1 <= y <= r.y2

    def as_dict(self):
        r = self.normalized()
        return {"name": r.name, "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2, "kind": r.kind}


def run(cmd, timeout=None):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, exc.stdout or "")


def gz_shell(command):
    if AIC_SETUP.exists():
        return f"source {AIC_SETUP} && {command}"
    return command


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


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


def make_random_world(seed=None):
    rng = random.Random(seed or int(time.time()))
    rects = [
        Rect("front_wall", ORIGIN_X, ORIGIN_Y, ORIGIN_X + WIDTH_M, ORIGIN_Y + 0.18, "wall"),
        Rect("back_wall", ORIGIN_X, ORIGIN_Y + HEIGHT_M - 0.18, ORIGIN_X + WIDTH_M, ORIGIN_Y + HEIGHT_M, "wall"),
        Rect("left_wall", ORIGIN_X, ORIGIN_Y, ORIGIN_X + 0.18, ORIGIN_Y + HEIGHT_M, "wall"),
        Rect("right_wall", ORIGIN_X + WIDTH_M - 0.18, ORIGIN_Y, ORIGIN_X + WIDTH_M, ORIGIN_Y + HEIGHT_M, "wall"),
    ]
    shelves = []
    products = {}
    product_names = ["ProductR", "ProductG", "ProductB", "ProductY", "ProductC"]
    rows = [-5.8, -3.2, -0.6, 2.0, 4.6, 6.6]
    shelf_index = 0
    for y in rows:
        for x in [-7.8, -4.6, -1.4, 1.8, 5.0, 8.2]:
            if rng.random() < 0.24:
                continue
            w = rng.uniform(1.35, 2.05)
            h = rng.uniform(0.58, 0.88)
            shelf_index += 1
            shelf = Rect(f"shelf_{shelf_index}", x - w / 2, y - h / 2, x + w / 2, y + h / 2, "shelf")
            shelves.append(shelf)
            rects.append(shelf)
    for i in range(16):
        x = rng.uniform(-10.0, 10.0)
        y = rng.uniform(-7.0, 7.2)
        if abs(x + 10.5) < 1.5 and abs(y + 7.0) < 1.5:
            continue
        if any(r.contains(x, y) for r in rects):
            continue
        size = rng.uniform(0.42, 0.72)
        rects.append(Rect(f"pallet_stack_{i + 1}", x - size / 2, y - size / 2, x + size / 2, y + size / 2, "crate"))
    rng.shuffle(shelves)
    for name, shelf in zip(product_names, shelves[: len(product_names)]):
        r = shelf.normalized()
        side_y = r.y1 - 0.9 if abs(r.y1 - ORIGIN_Y) > abs(r.y2 - (ORIGIN_Y + HEIGHT_M)) else r.y2 + 0.9
        slot_y = r.y1 + 0.18 if side_y < (r.y1 + r.y2) / 2 else r.y2 - 0.18
        products[name] = {
            "storage": shelf.name,
            "slot": {"x": (r.x1 + r.x2) / 2, "y": slot_y, "z": 1.38},
            "pickup": {"x": (r.x1 + r.x2) / 2, "y": side_y, "yaw": math.pi / 2 if side_y < (r.y1 + r.y2) / 2 else -math.pi / 2},
        }
    start = Pose2D(-10.6, -7.0, 0.0)
    occupied = rect_occupancy(rects, ROBOT_RADIUS)
    if world_to_grid(start.x, start.y) in occupied:
        start = Pose2D(-9.6, -6.8, 0.0)
    return {"seed": seed, "rects": rects, "products": products, "start": start}


def material(kind):
    colors = {
        "wall": ("0.66 0.66 0.64 1", "0.76 0.76 0.74 1"),
        "floor": ("0.50 0.50 0.47 1", "0.68 0.68 0.63 1"),
        "shelf": ("0.70 0.52 0.18 1", "0.96 0.70 0.18 1"),
        "rack_frame": ("0.03 0.04 0.05 1", "0.05 0.06 0.07 1"),
        "crate": ("0.44 0.28 0.11 1", "0.62 0.42 0.18 1"),
        "box_dark": ("0.30 0.21 0.10 1", "0.44 0.32 0.14 1"),
        "label": ("0.86 0.86 0.76 1", "0.98 0.96 0.84 1"),
        "barcode": ("0.02 0.02 0.02 1", "0.02 0.02 0.02 1"),
        "marking": ("0.04 0.45 0.16 1", "0.04 0.55 0.20 1"),
        "hazard": ("0.90 0.75 0.03 1", "1.00 0.84 0.04 1"),
        "productr": ("0.72 0.05 0.04 1", "0.95 0.08 0.06 1"),
        "productg": ("0.02 0.45 0.14 1", "0.03 0.70 0.22 1"),
        "productb": ("0.04 0.17 0.64 1", "0.08 0.28 0.95 1"),
        "producty": ("0.75 0.58 0.03 1", "0.98 0.78 0.06 1"),
        "productc": ("0.02 0.58 0.62 1", "0.04 0.84 0.90 1"),
    }
    ambient, diffuse = colors.get(kind, colors["crate"])
    return f"<material><ambient>{ambient}</ambient><diffuse>{diffuse}</diffuse></material>"


def model_box(name, x, y, z, sx, sy, sz, kind, static=True):
    return f"""
    <model name="{name}">
      <static>{str(static).lower()}</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>{material(kind)}</visual>
      </link>
    </model>"""


def visual_box(name, x, y, z, sx, sy, sz, kind):
    return f"""
        <visual name="{name}">
          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          {material(kind)}
        </visual>"""


def visual_box_pose(name, x, y, z, sx, sy, sz, yaw, kind):
    return f"""
        <visual name="{name}">
          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          {material(kind)}
        </visual>"""


def rack_model(rect):
    r = rect.normalized()
    cx = (r.x1 + r.x2) / 2
    cy = (r.y1 + r.y2) / 2
    sx = r.x2 - r.x1
    sy = r.y2 - r.y1
    seed = sum(ord(ch) for ch in rect.name)
    rng = random.Random(seed)
    visuals = []
    post = 0.07
    height = 2.05
    for px in (-sx / 2 + post / 2, sx / 2 - post / 2):
        for py in (-sy / 2 + post / 2, sy / 2 - post / 2):
            visuals.append(visual_box(f"post_{len(visuals)}", px, py, height / 2, post, post, height, "rack_frame"))
    for level, z in enumerate((0.42, 1.04, 1.66), 1):
        visuals.append(visual_box(f"deck_{level}", 0, 0, z, sx + 0.14, sy + 0.18, 0.07, "shelf"))
        for bx in (-sx / 4, sx / 4):
            bw = min(0.48, sx * 0.36)
            bd = min(0.34, sy * 0.42)
            bh = rng.uniform(0.28, 0.48)
            by = rng.choice((-sy * 0.18, sy * 0.18))
            visuals.append(visual_box(f"box_{level}_{bx:.1f}", bx, by, z + 0.08 + bh / 2, bw, bd, bh, "crate"))
            visuals.append(visual_box(f"label_{level}_{bx:.1f}", bx, by - bd / 2 - 0.004, z + 0.10 + bh * 0.52, bw * 0.28, 0.012, bh * 0.38, "label"))
            visuals.append(visual_box(f"barcode_{level}_{bx:.1f}", bx + bw * 0.18, by - bd / 2 - 0.008, z + 0.10 + bh * 0.52, bw * 0.06, 0.014, bh * 0.34, "barcode"))
    return f"""
    <model name="{rect.name}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} 0 0 0 0</pose>
      <link name="link">
        <collision name="collision"><pose>0 0 {height / 2:.3f} 0 0 0</pose><geometry><box><size>{sx:.3f} {sy:.3f} {height:.3f}</size></box></geometry></collision>
        {''.join(visuals)}
      </link>
    </model>"""


def pallet_stack_model(rect):
    r = rect.normalized()
    cx = (r.x1 + r.x2) / 2
    cy = (r.y1 + r.y2) / 2
    sx = r.x2 - r.x1
    sy = r.y2 - r.y1
    seed = sum(ord(ch) for ch in rect.name)
    rng = random.Random(seed)
    visuals = [visual_box("pallet", 0, 0, 0.08, sx + 0.18, sy + 0.18, 0.16, "box_dark")]
    for layer in range(rng.randint(2, 4)):
        z = 0.20 + layer * 0.30
        visuals.append(visual_box(f"case_{layer}_a", -sx * 0.18, -sy * 0.12, z + 0.14, sx * 0.48, sy * 0.45, 0.28, "crate"))
        visuals.append(visual_box(f"case_{layer}_b", sx * 0.18, sy * 0.12, z + 0.14, sx * 0.48, sy * 0.45, 0.28, "crate"))
    return f"""
    <model name="{rect.name}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} 0 0 0 0</pose>
      <link name="link">
        <collision name="collision"><pose>0 0 0.55 0 0 0</pose><geometry><box><size>{sx + 0.18:.3f} {sy + 0.18:.3f} 1.10</size></box></geometry></collision>
        {''.join(visuals)}
      </link>
    </model>"""


def product_kind(name):
    return name.lower()


def product_model(name, slot):
    return model_box(f"item_{name}", slot["x"], slot["y"], slot["z"], 0.42, 0.34, 0.30, product_kind(name), static=False)


def floor_marking(name, x, y, sx, sy, kind="marking"):
    return model_box(name, x, y, 0.006, sx, sy, 0.012, kind)


def write_world(world):
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    models = [
        model_box("floor", 0, 0, -0.04, WIDTH_M, HEIGHT_M, 0.08, "floor"),
        floor_marking("dispatch_lane_a", -8.4, -7.1, 3.6, 0.08),
        floor_marking("dispatch_lane_b", -8.4, -6.0, 3.6, 0.08),
        floor_marking("main_aisle_left", -1.0, -7.65, 20.0, 0.07),
        floor_marking("main_aisle_right", -1.0, 7.65, 20.0, 0.07),
        floor_marking("pickup_hazard_front", -5.7, -7.55, 5.2, 0.09, "hazard"),
        floor_marking("pickup_hazard_back", -5.7, -5.15, 5.2, 0.09, "hazard"),
        floor_marking("pickup_hazard_left", -8.3, -6.35, 0.09, 2.4, "hazard"),
        floor_marking("pickup_hazard_right", -3.1, -6.35, 0.09, 2.4, "hazard"),
    ]
    for rect in world["rects"]:
        r = rect.normalized()
        if rect.kind == "shelf":
            models.append(rack_model(rect))
        elif rect.kind == "crate":
            models.append(pallet_stack_model(rect))
        else:
            height = 2.6
            models.append(model_box(rect.name, (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2, height / 2, r.x2 - r.x1, r.y2 - r.y1, height, rect.kind))
    for name, product in world["products"].items():
        models.append(product_model(name, product["slot"]))
    start = world["start"]
    lidar_visuals = [
        """
        <visual name="scan_disc">
          <pose>0 0 0.035 0 0 0</pose>
          <geometry><cylinder><radius>5.200</radius><length>0.010</length></cylinder></geometry>
          <material><ambient>0.05 0.10 1.00 0.18</ambient><diffuse>0.05 0.10 1.00 0.18</diffuse><transparency>0.72</transparency></material>
        </visual>"""
    ]
    for i in range(LIDAR_VISUAL_RAYS):
        angle = 2 * math.pi * i / LIDAR_VISUAL_RAYS
        length = LIDAR_RANGE * (0.55 + 0.45 * ((i % 4) / 3))
        lidar_visuals.append(visual_box_pose(f"scan_ray_{i}", math.cos(angle) * length / 2, math.sin(angle) * length / 2, 0.055, length, 0.018, 0.018, angle, "productb"))
    models.append(f"""
    <model name="warehouse_robot">
      <pose>{start.x:.3f} {start.y:.3f} 0.32 0 0 {start.yaw:.3f}</pose>
      <link name="base_link">
        <inertial><mass>20</mass></inertial>
        <collision name="base_collision"><geometry><box><size>1.0 0.72 0.38</size></box></geometry></collision>
        <visual name="base_visual"><geometry><box><size>1.0 0.72 0.38</size></box></geometry><material><ambient>0.86 0.86 0.82 1</ambient><diffuse>0.95 0.95 0.90 1</diffuse></material></visual>
        <visual name="left_wheel_visual"><pose>0 0.46 -0.16 1.5708 0 0</pose><geometry><cylinder><radius>0.17</radius><length>0.12</length></cylinder></geometry><material><ambient>0.03 0.04 0.05 1</ambient><diffuse>0.04 0.05 0.06 1</diffuse></material></visual>
        <visual name="right_wheel_visual"><pose>0 -0.46 -0.16 1.5708 0 0</pose><geometry><cylinder><radius>0.17</radius><length>0.12</length></cylinder></geometry><material><ambient>0.03 0.04 0.05 1</ambient><diffuse>0.04 0.05 0.06 1</diffuse></material></visual>
        <visual name="fork_left"><pose>0.63 0.18 -0.20 0 0 0</pose><geometry><box><size>0.75 0.07 0.06</size></box></geometry><material><ambient>0.08 0.08 0.08 1</ambient><diffuse>0.12 0.12 0.12 1</diffuse></material></visual>
        <visual name="fork_right"><pose>0.63 -0.18 -0.20 0 0 0</pose><geometry><box><size>0.75 0.07 0.06</size></box></geometry><material><ambient>0.08 0.08 0.08 1</ambient><diffuse>0.12 0.12 0.12 1</diffuse></material></visual>
        <visual name="lidar"><pose>0.30 0 0.40 0 0 0</pose><geometry><cylinder><radius>0.15</radius><length>0.14</length></cylinder></geometry><material><ambient>0.02 0.02 0.02 1</ambient><diffuse>0.02 0.02 0.02 1</diffuse></material></visual>
        <visual name="scan_marker"><pose>0 0 1.45 0 0 0</pose><geometry><sphere><radius>0.075</radius></sphere></geometry><material><ambient>0.0 0.95 1.0 1</ambient><diffuse>0.0 0.95 1.0 1</diffuse></material></visual>
        {''.join(lidar_visuals)}
      </link>
    </model>""")
    models.append(model_box("cargo_item", -10.6, -7.0, HIDDEN_Z, 0.42, 0.34, 0.30, "producty", static=False))
    models.append(model_box("pickup_marker", -10.6, -7.0, HIDDEN_Z, 0.70, 0.06, 0.06, "productg", static=False))
    models.append(model_box("delivered_item", -10.6, -7.0, HIDDEN_Z, 0.46, 0.38, 0.30, "productg", static=False))
    models.append(wheel_models())
    WORLD.write_text(
        f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="lidar_random_warehouse">
    <physics name="1ms" type="ignored"><max_step_size>0.01</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <scene><ambient>0.66 0.66 0.66 1</ambient><background>0.04 0.05 0.06 1</background></scene>
    <gui fullscreen="1"><camera name="lidar_random_overview"><pose>0 -8 18 0 1.05 1.5708</pose><view_controller>orbit</view_controller><projection_type>perspective</projection_type></camera></gui>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose><diffuse>0.9 0.9 0.85 1</diffuse><direction>-0.35 0.2 -0.9</direction></light>
    {''.join(models)}
  </world>
</sdf>
""",
        encoding="utf-8",
    )


def cell_free(cell, occupied):
    return 0 <= cell[0] < GRID_W and 0 <= cell[1] < GRID_H and cell not in occupied


def astar(start_pose, goal_pose, occupied):
    start = world_to_grid(start_pose.x, start_pose.y)
    goal = world_to_grid(goal_pose.x, goal_pose.y)
    occupied = set(occupied)
    occupied.discard(start)
    occupied.discard(goal)
    open_set = [(0.0, start)]
    came_from = {}
    gscore = {start: 0.0}
    neighbors = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.41), (-1, 1, 1.41), (1, -1, 1.41), (1, 1, 1.41)]
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
            if not cell_free(nxt, occupied):
                continue
            if dx and dy and ((current[0] + dx, current[1]) in occupied or (current[0], current[1] + dy) in occupied):
                continue
            tentative = gscore[current] + cost
            if tentative >= gscore.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            gscore[nxt] = tentative
            heapq.heappush(open_set, (tentative + math.hypot(nxt[0] - goal[0], nxt[1] - goal[1]), nxt))
    return []


def simulate_lidar(pose, true_occupied, known):
    endpoints = []
    px, py = world_to_grid(pose.x, pose.y)
    for i in range(LIDAR_RAYS):
        angle = pose.yaw - math.pi + (2 * math.pi * i / LIDAR_RAYS)
        hit = None
        max_steps = int(LIDAR_RANGE / RESOLUTION)
        for step in range(1, max_steps + 1):
            x = pose.x + math.cos(angle) * step * RESOLUTION
            y = pose.y + math.sin(angle) * step * RESOLUTION
            gx, gy = world_to_grid(x, y)
            if gx <= 0 or gy <= 0 or gx >= GRID_W - 1 or gy >= GRID_H - 1:
                hit = (gx, gy)
                break
            known[gy][gx] = 0
            if (gx, gy) in true_occupied:
                known[gy][gx] = 1
                hit = (gx, gy)
                break
        if hit:
            endpoints.append({"x": grid_to_world(hit[0], hit[1])[0], "y": grid_to_world(hit[0], hit[1])[1]})
    known[py][px] = 0
    return endpoints


def known_occupied(known):
    occ = set()
    for gy, row in enumerate(known):
        for gx, value in enumerate(row):
            if value == 1:
                occ.add((gx, gy))
    return occ


def coverage(known, reachable_free):
    if not reachable_free:
        return 0.0
    seen = sum(1 for gx, gy in reachable_free if known[gy][gx] != -1)
    return seen / len(reachable_free)


def reachable_free_cells(start, occupied):
    start_cell = world_to_grid(start.x, start.y)
    q = [start_cell]
    seen = {start_cell}
    for cell in q:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in seen or not cell_free(nxt, occupied):
                continue
            seen.add(nxt)
            q.append(nxt)
    return seen


def frontier_goal(pose, known, planning_occupied, reachable):
    candidates = []
    for gx, gy in reachable:
        if known[gy][gx] != 0:
            continue
        if any(0 <= gx + dx < GRID_W and 0 <= gy + dy < GRID_H and known[gy + dy][gx + dx] == -1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            wx, wy = grid_to_world(gx, gy)
            dist = math.hypot(wx - pose.x, wy - pose.y)
            candidates.append((dist, gx, gy, Pose2D(wx, wy)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, _, _, candidate in candidates[:60]:
        path = astar(pose, candidate, planning_occupied)
        if len(path) > 2:
            return candidate, path
    return None, []


def set_model_pose(model, x, y, z, yaw=0.0):
    req = f'name: "{model}", position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}}, orientation: {{z: {math.sin(yaw / 2):.6f}, w: {math.cos(yaw / 2):.6f}}}'
    cmd = ["/bin/bash", "-lc", gz_shell("gz service -s /world/lidar_random_warehouse/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean " f"--timeout {GZ_SERVICE_TIMEOUT_MS} --req '{req}'")]
    result = run(cmd, timeout=POSE_COMMAND_TIMEOUT_SECONDS)
    if result.returncode != 0:
        print(result.stdout, end="", flush=True)


def set_robot_pose(pose):
    if DRIVE:
        DRIVE.project_pose(pose.x, pose.y, pose.yaw, ODOM_DT_SECONDS, "lidar_random", log_every=10)
    else:
        set_model_pose("warehouse_robot", pose.x, pose.y, 0.320, pose.yaw)


def product_model_name(product_name):
    return f"item_{product_name}"


def set_product_item_pose(product_name, x, y, z, yaw=0.0):
    set_model_pose(product_model_name(product_name), x, y, z, yaw)


def hide_product_item(product_name):
    set_product_item_pose(product_name, -10.6, -7.0, HIDDEN_Z)


def set_pickup_marker(x, y, z, yaw=0.0):
    set_model_pose("pickup_marker", x, y, z, yaw)


def hide_pickup_marker():
    set_model_pose("pickup_marker", -10.6, -7.0, HIDDEN_Z)


def set_cargo_visible(pose, product_name=None):
    x = pose.x + math.cos(pose.yaw) * 0.18
    y = pose.y + math.sin(pose.yaw) * 0.18
    if product_name:
        set_product_item_pose(product_name, x, y, 0.62, pose.yaw)
    else:
        set_model_pose("cargo_item", x, y, 0.62, pose.yaw)


def hide_cargo():
    set_model_pose("cargo_item", -10.6, -7.0, HIDDEN_Z)


def hide_delivered():
    set_model_pose("delivered_item", -10.6, -7.0, HIDDEN_Z)


def show_delivered(x, y):
    set_model_pose("delivered_item", x, y, 0.28)


def animate_pickup(product_name, product, pose, world, known, reachable, task):
    slot = product["slot"]
    sx, sy, sz = float(slot["x"]), float(slot["y"]), float(slot["z"])
    tx = pose.x + math.cos(pose.yaw) * 0.18
    ty = pose.y + math.sin(pose.yaw) * 0.18
    tz = 0.62
    yaw = pose.yaw
    print(f"pickup animation product={product_name} from shelf slot=({sx:.2f},{sy:.2f},{sz:.2f})", flush=True)
    for i in range(1, 7):
        lift_z = sz + (1.95 - sz) * i / 6
        set_product_item_pose(product_name, sx, sy, lift_z, yaw)
        set_pickup_marker((sx + tx) / 2, (sy + ty) / 2, lift_z, yaw)
        write_state(world, known, pose, "executing", f"lifting {product_name} from {product['storage']}", reachable, task=task)
        time.sleep(0.08)
    for i in range(1, 11):
        t = i / 10
        x = sx + (tx - sx) * t
        y = sy + (ty - sy) * t
        z = 1.95 + (tz - 1.95) * t
        set_product_item_pose(product_name, x, y, z, yaw)
        set_pickup_marker((x + tx) / 2, (y + ty) / 2, z + 0.18, yaw)
        write_state(world, known, pose, "executing", f"transferring {product_name} onto robot forks", reachable, task=task)
        time.sleep(0.08)
    set_cargo_visible(pose, product_name)
    hide_pickup_marker()


def map_payload(world, known, pose, status, message, path=None, task=None, lidar=None):
    return {
        "status": status,
        "message": message,
        "origin": {"x": ORIGIN_X, "y": ORIGIN_Y},
        "width_m": WIDTH_M,
        "height_m": HEIGHT_M,
        "resolution": RESOLUTION,
        "grid_w": GRID_W,
        "grid_h": GRID_H,
        "known": ["".join("?" if v == -1 else "#" if v == 1 else "." for v in row) for row in known],
        "coverage": 0.0,
        "robot": {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
        "true_obstacles": [r.as_dict() for r in world["rects"]],
        "products": world["products"],
        "path": [{"x": p.x, "y": p.y, "yaw": p.yaw} for p in (path or [])],
        "task": task,
        "lidar": lidar or [],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_state(world, known, pose, status, message, reachable=None, path=None, task=None, lidar=None):
    data = map_payload(world, known, pose, status, message, path, task, lidar)
    if reachable is not None:
        data["coverage"] = coverage(known, reachable)
    write_json(STATE_FILE, data)


def move_path(path, pose, world, known, true_occupied, reachable, status, message, task=None, cargo=None, fast=True):
    current = pose
    for target in path[1:]:
        dx = target.x - current.x
        dy = target.y - current.y
        dist = math.hypot(dx, dy)
        steps = max(1, int(dist / POSE_SPACING))
        yaw = math.atan2(dy, dx) if dist > 0.01 else current.yaw
        for _ in range(steps):
            candidate = Pose2D(current.x + dx / steps, current.y + dy / steps, yaw)
            if world_to_grid(candidate.x, candidate.y) in true_occupied:
                lidar = simulate_lidar(current, true_occupied, known)
                write_state(world, known, current, status, "blocked by shelf; replanning", reachable, path, task, lidar)
                return current
            current = candidate
            set_robot_pose(current)
            if cargo:
                set_cargo_visible(current, cargo)
            lidar = simulate_lidar(current, true_occupied, known)
            write_state(world, known, current, status, message, reachable, path, task, lidar)
            time.sleep(EXPLORE_STEP_SECONDS if fast else TASK_STEP_SECONDS)
    return Pose2D(path[-1].x, path[-1].y, path[-1].yaw if path else current.yaw)


def execute_task(task, pose, world, known, true_occupied, reachable):
    if coverage(known, reachable) < MAPPED_COVERAGE:
        write_state(world, known, pose, "mapping", "TaskGoal locked until lidar exploration finishes", reachable, task=task)
        return pose
    products = world["products"]
    product_name = task.get("product", next(iter(products)))
    product = products.get(product_name)
    if not product:
        write_state(world, known, pose, "failed", f"unknown product {product_name}", reachable, task=task)
        return pose
    drop = task.get("drop") or {}
    keepouts = [Rect(f"keepout_{i}", float(k["x1"]), float(k["y1"]), float(k["x2"]), float(k["y2"]), "keepout") for i, k in enumerate(task.get("keepouts", []), 1)]
    occ = true_occupied | known_occupied(known) | rect_occupancy(keepouts, ROBOT_RADIUS)
    pickup = Pose2D(product["pickup"]["x"], product["pickup"]["y"], product["pickup"]["yaw"])
    drop_pose = Pose2D(float(drop["x"]), float(drop["y"]), float(drop.get("yaw", 0.0)))
    p1 = astar(pose, pickup, occ)
    p2 = astar(pickup, drop_pose, occ)
    if not p1 or not p2:
        write_state(world, known, pose, "failed", "no path on discovered map", reachable, task=task)
        return pose
    full = p1 + p2[1:]
    print(f"TaskGoal accepted product={product_name} pickup=({pickup.x:.2f},{pickup.y:.2f}) drop=({drop_pose.x:.2f},{drop_pose.y:.2f})", flush=True)
    pose = move_path(p1, pose, world, known, occ, reachable, "executing", "driving to pickup", task, None, fast=False)
    print(f"pick_up product={product_name} storage={product['storage']}", flush=True)
    animate_pickup(product_name, product, pose, world, known, reachable, task)
    pose = move_path(p2, pose, world, known, occ, reachable, "executing", "driving to drop", task, product_name, fast=False)
    set_product_item_pose(product_name, drop_pose.x, drop_pose.y, 0.38, drop_pose.yaw)
    show_delivered(drop_pose.x, drop_pose.y)
    print(f"drop_off product={product_name} target=({drop_pose.x:.2f},{drop_pose.y:.2f})", flush=True)
    write_state(world, known, pose, "done", "task completed", reachable, full, task)
    return pose


def forward_gazebo_output(proc):
    assert proc.stdout is not None
    for line in proc.stdout:
        if any(noise in line for noise in IGNORED_GAZEBO_LOG_LINES):
            continue
        print(line, end="", flush=True)


def launch_gazebo(env):
    proc = subprocess.Popen(["/bin/bash", "-lc", gz_shell(f"exec gz sim -v 3 {WORLD}")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, preexec_fn=os.setsid)
    threading.Thread(target=forward_gazebo_output, args=(proc,), daemon=True).start()
    return proc


def main():
    global DRIVE
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("LIDAR_RANDOM_SEED", "0")) or int(time.time())
    world = make_random_world(seed)
    write_world(world)
    true_occupied = rect_occupancy(world["rects"], ROBOT_RADIUS)
    reachable = reachable_free_cells(world["start"], true_occupied)
    known = [[-1 for _ in range(GRID_W)] for _ in range(GRID_H)]
    pose = world["start"]
    write_state(world, known, pose, "starting", f"generated random map seed={seed}", reachable)

    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("GALLIUM_DRIVER", "llvmpipe")

    print("Starting Lidar Random Map mode")
    print(f"Generated random world seed={seed}")
    print("Exploration must complete before TaskGoal is accepted.")
    gz = launch_gazebo(env)

    def shutdown(signum, frame):
        print("Stopping Lidar Random Map mode", flush=True)
        try:
            os.killpg(os.getpgid(gz.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    time.sleep(8)
    DRIVE = GazeboDriveProjector("lidar_random_warehouse", map_origin=(ORIGIN_X, ORIGIN_Y), map_resolution=RESOLUTION)
    DRIVE.reset(pose.x, pose.y, pose.yaw)
    hide_cargo()
    hide_delivered()
    hide_pickup_marker()

    last_task_version = TASK_FILE.stat().st_mtime_ns if TASK_FILE.exists() else None
    status = "mapping"
    while gz.poll() is None:
        lidar = simulate_lidar(pose, true_occupied, known)
        cov = coverage(known, reachable)
        if status == "mapping":
            write_state(world, known, pose, status, f"exploring with lidar {cov:.0%}", reachable, lidar=lidar)
            if cov >= MAPPED_COVERAGE:
                status = "mapped"
                write_state(world, known, pose, "mapped", "mapping complete; TaskGoal enabled", reachable, lidar=lidar)
            else:
                planning_occupied = true_occupied | known_occupied(known)
                _, path = frontier_goal(pose, known, planning_occupied, reachable)
                if path:
                    pose = move_path(path, pose, world, known, true_occupied, reachable, "mapping", "frontier exploration", None, None, fast=True)
                else:
                    status = "mapped"
                    write_state(world, known, pose, "mapped", "no more frontiers; TaskGoal enabled", reachable, lidar=lidar)
        else:
            if TASK_FILE.exists():
                version = TASK_FILE.stat().st_mtime_ns
                if version != last_task_version:
                    last_task_version = version
                    try:
                        pose = execute_task(read_json(TASK_FILE), pose, world, known, true_occupied, reachable)
                        status = "mapped"
                    except (json.JSONDecodeError, KeyError, ValueError) as exc:
                        print(f"Task rejected: {exc}", flush=True)
            write_state(world, known, pose, status, "mapping complete; waiting for TaskGoal", reachable, lidar=lidar)
            time.sleep(0.25)
    return gz.returncode


if __name__ == "__main__":
    raise SystemExit(main())
