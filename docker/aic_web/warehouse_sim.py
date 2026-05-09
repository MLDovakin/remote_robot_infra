#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from heapq import heappop, heappush
from pathlib import Path


DIRS = {
    "E": (1, 0),
    "N": (0, -1),
    "W": (-1, 0),
    "S": (0, 1),
}


@dataclass
class Pose:
    x: int
    y: int
    yaw: str = "E"


@dataclass
class Storage:
    name: str
    product: str
    pose: Pose
    inventory: int
    capacity: int

    def produce(self):
        if self.inventory < self.capacity:
            self.inventory += 1

    def request_product(self):
        if self.inventory <= 0:
            return None
        self.inventory -= 1
        return {"product": self.product, "source": self.name}


@dataclass
class Dispatch:
    name: str
    pose: Pose
    received: list = field(default_factory=list)

    def pick_product(self, item):
        self.received.append(item)


@dataclass
class Order:
    id: str
    dispatch: str
    items: dict
    status: str = "queued"


@dataclass
class Robot:
    name: str
    pose: Pose
    cargo_capacity: int = 8
    cargo: list = field(default_factory=list)
    state: str = "IDLE"
    battery: float = 100.0
    active_order: str | None = None


class Warehouse:
    def __init__(self):
        self.width = 18
        self.height = 12
        self.obstacles = self._build_obstacles()
        self.storages = {
            "StorageR": Storage("StorageR", "ProductR", Pose(14, 2), inventory=4, capacity=10),
            "StorageG": Storage("StorageG", "ProductG", Pose(14, 5), inventory=4, capacity=10),
            "StorageB": Storage("StorageB", "ProductB", Pose(14, 8), inventory=4, capacity=10),
        }
        self.dispatches = {
            "DispatchA": Dispatch("DispatchA", Pose(2, 3)),
            "DispatchB": Dispatch("DispatchB", Pose(2, 8)),
        }
        self.robot = Robot("amr_1", Pose(2, 10))
        self.orders = deque(
            [
                Order("ORD-001", "DispatchA", {"ProductR": 2, "ProductG": 1}),
                Order("ORD-002", "DispatchB", {"ProductB": 2, "ProductR": 1}),
                Order("ORD-003", "DispatchA", {"ProductG": 2, "ProductB": 1}),
            ]
        )
        self.completed = []
        self.trace = []
        self.step = 0

    def _build_obstacles(self):
        obstacles = set()
        for y in range(1, 11):
            if y not in (3, 7, 10):
                obstacles.add((6, y))
                obstacles.add((10, y))
        for x in range(4, 16):
            if x not in (6, 10, 14):
                obstacles.add((x, 0))
                obstacles.add((x, 11))
        return obstacles

    def walkable(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

    def nearest_storage(self, product):
        candidates = [s for s in self.storages.values() if s.product == product]
        return min(candidates, key=lambda s: manhattan((self.robot.pose.x, self.robot.pose.y), (s.pose.x, s.pose.y)))

    def snapshot(self, event):
        snap = {
            "step": self.step,
            "event": event,
            "robot": asdict(self.robot),
            "orders_waiting": [asdict(order) for order in self.orders],
            "completed_orders": [asdict(order) for order in self.completed],
            "storages": {name: asdict(storage) for name, storage in self.storages.items()},
            "dispatches": {name: asdict(dispatch) for name, dispatch in self.dispatches.items()},
            "map": self.render_ascii(),
        }
        self.trace.append(snap)
        print(f"[{self.step:03d}] {event}")
        print(snap["map"])
        print()

    def render_ascii(self):
        cells = [["." for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.obstacles:
            cells[y][x] = "#"
        for storage in self.storages.values():
            cells[storage.pose.y][storage.pose.x] = storage.product[-1]
        for dispatch in self.dispatches.values():
            cells[dispatch.pose.y][dispatch.pose.x] = "D"
        glyph = {"N": "^", "E": ">", "S": "v", "W": "<"}[self.robot.pose.yaw]
        cells[self.robot.pose.y][self.robot.pose.x] = glyph
        return "\n".join("".join(row) for row in cells)

    def move_to(self, target_pose, label, sleep_seconds):
        self.robot.state = label
        path = astar((self.robot.pose.x, self.robot.pose.y), (target_pose.x, target_pose.y), self)
        if not path:
            raise RuntimeError(f"no route from {self.robot.pose} to {target_pose}")
        for next_x, next_y in path[1:]:
            dx = next_x - self.robot.pose.x
            dy = next_y - self.robot.pose.y
            self.robot.pose.yaw = yaw_from_delta(dx, dy)
            self.robot.pose.x = next_x
            self.robot.pose.y = next_y
            self.robot.battery = max(0.0, self.robot.battery - 0.18)
            self.step += 1
            self.snapshot(f"{self.robot.name} {label} -> ({next_x},{next_y}) yaw={self.robot.pose.yaw}")
            if sleep_seconds:
                time.sleep(sleep_seconds)

    def run(self, sleep_seconds):
        self.snapshot("warehouse initialized")
        while self.orders:
            order = self.orders.popleft()
            order.status = "assigned"
            self.robot.active_order = order.id
            self.robot.state = "TO_PICKUP"
            self.snapshot(f"{order.id} assigned to {self.robot.name}: {order.items} -> {order.dispatch}")

            for product, quantity in order.items.items():
                storage = self.nearest_storage(product)
                self.move_to(storage.pose, f"TO_PICKUP:{storage.name}", sleep_seconds)
                self.robot.state = "LOADING"
                for _ in range(quantity):
                    item = storage.request_product()
                    if item is None:
                        storage.produce()
                        item = storage.request_product()
                    if len(self.robot.cargo) >= self.robot.cargo_capacity:
                        raise RuntimeError("cargo capacity exceeded")
                    self.robot.cargo.append(item)
                    self.step += 1
                    self.snapshot(f"loaded {product} from {storage.name}")
                    if sleep_seconds:
                        time.sleep(sleep_seconds)

            dispatch = self.dispatches[order.dispatch]
            self.move_to(dispatch.pose, f"TO_DROPOFF:{dispatch.name}", sleep_seconds)
            self.robot.state = "UNLOADING"
            while self.robot.cargo:
                item = self.robot.cargo.pop(0)
                dispatch.pick_product(item)
                self.step += 1
                self.snapshot(f"unloaded {item['product']} to {dispatch.name}")
                if sleep_seconds:
                    time.sleep(sleep_seconds)

            order.status = "completed"
            self.completed.append(order)
            self.robot.active_order = None
            self.robot.state = "IDLE"
            self.step += 1
            self.snapshot(f"{order.id} completed")

        self.snapshot("all orders completed")

    def write_results(self, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "robot": asdict(self.robot),
            "completed_orders": [asdict(order) for order in self.completed],
            "dispatches": {name: asdict(dispatch) for name, dispatch in self.dispatches.items()},
            "storages": {name: asdict(storage) for name, storage in self.storages.items()},
            "steps": self.step,
            "trace_file": "warehouse_trace.json",
        }
        (output_dir / "warehouse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "warehouse_trace.json").write_text(json.dumps(self.trace, indent=2), encoding="utf-8")
        (output_dir / "warehouse_final_map.txt").write_text(self.render_ascii() + "\n", encoding="utf-8")
        return summary


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def yaw_from_delta(dx, dy):
    for yaw, delta in DIRS.items():
        if delta == (dx, dy):
            return yaw
    raise ValueError(f"invalid delta {dx},{dy}")


def astar(start, goal, warehouse):
    frontier = []
    heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heappop(frontier)
        if current == goal:
            break
        for dx, dy in DIRS.values():
            nxt = (current[0] + dx, current[1] + dy)
            if not warehouse.walkable(*nxt):
                continue
            new_cost = cost_so_far[current] + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan(nxt, goal)
                heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return []

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def main():
    parser = argparse.ArgumentParser(description="Warehouse AMR picking simulation")
    parser.add_argument("--result-dir", default=os.environ.get("AIC_RESULTS_DIR", "/workspace/aic_results"))
    parser.add_argument("--sleep", type=float, default=0.04, help="Delay between simulation steps")
    args = parser.parse_args()

    run_dir = Path(args.result_dir) / f"warehouse_{time.strftime('%Y%m%d_%H%M%S')}"
    warehouse = Warehouse()
    warehouse.run(max(0.0, args.sleep))
    summary = warehouse.write_results(run_dir)

    print("Warehouse simulation completed")
    print(f"Results: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
