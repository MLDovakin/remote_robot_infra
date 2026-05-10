#!/usr/bin/env python3
import heapq
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


WORLD = Path("/opt/aic_web/warehouse_visual.sdf")
AIC_SETUP = Path("/ws_aic/install/setup.bash")
RUNS_DIR = Path(os.environ.get("AIC_RUNS_DIR", "/workspace/aic_runs"))
RESULTS_DIR = Path(os.environ.get("AIC_RESULTS_DIR", "/workspace/aic_results"))
STATE_FILE = RUNS_DIR / "nav2_state.json"
TASK_FILE = RUNS_DIR / "nav2_task.json"
MAP_DIR = RESULTS_DIR / "nav2_warehouse_map"

ORIGIN_X = -11.0
ORIGIN_Y = -8.0
WIDTH_M = 22.0
HEIGHT_M = 16.0
RESOLUTION = 0.10
GRID_W = int(WIDTH_M / RESOLUTION)
GRID_H = int(HEIGHT_M / RESOLUTION)
ROBOT_RADIUS = 0.65
WHEEL_RADIUS = 0.17
WHEEL_SEPARATION = 0.92
WHEEL_BASE = 0.72
WHEEL_Z = 0.30
MOTOR_LOG_INTERVAL_STEPS = 12

HIDDEN_Z = -10.0
STEP_SECONDS = 0.035
LINEAR_SPEED = 2.2
MAX_ANGULAR_SPEED = 2.5
POSE_COMMAND_TIMEOUT_SECONDS = 2.0
GZ_SERVICE_TIMEOUT_MS = 1500
TASK_POLL_SECONDS = 0.4

IGNORED_GAZEBO_LOG_LINES = (
    "NodeShared::RecvSrvRequest() error sending response: Host unreachable",
)


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass
class WheelCommand:
    linear: float
    angular: float
    left_velocity: float
    right_velocity: float
    left_angular_velocity: float
    right_angular_velocity: float
    left_angle: float
    right_angle: float
    steering_angle: float = 0.0


@dataclass(frozen=True)
class Rect:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self):
        return Rect(self.name, min(self.x1, self.x2), min(self.y1, self.y2), max(self.x1, self.x2), max(self.y1, self.y2))

    def inflated(self, amount):
        r = self.normalized()
        return Rect(r.name, r.x1 - amount, r.y1 - amount, r.x2 + amount, r.y2 + amount)

    def contains(self, x, y):
        r = self.normalized()
        return r.x1 <= x <= r.x2 and r.y1 <= y <= r.y2

    def as_dict(self):
        r = self.normalized()
        return {"name": self.name, "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2}


STATIC_OBSTACLES = [
    Rect("front_wall", -11.0, -8.0, 11.0, -7.85),
    Rect("back_wall", -11.0, 7.85, 11.0, 8.0),
    Rect("left_wall", -11.0, -8.0, -10.85, 8.0),
    Rect("right_wall", 10.85, -8.0, 11.0, 8.0),
    Rect("storage_r", 4.9, -4.5, 7.1, -3.5),
    Rect("storage_g", 4.9, -0.5, 7.1, 0.5),
    Rect("storage_b", 4.9, 3.5, 7.1, 4.5),
    Rect("rack_1", -3.5, -4.35, 1.5, -3.65),
    Rect("rack_2", -3.5, -0.35, 1.5, 0.35),
    Rect("rack_3", -3.5, 3.65, 1.5, 4.35),
]

PRODUCTS = {
    "ProductR": {
        "storage": "StorageR",
        "slot": {"x": 6.55, "y": -4.20, "z": 1.02},
        "pickup": Pose2D(6.55, -5.35, math.pi / 2),
    },
    "ProductG": {
        "storage": "StorageG",
        "slot": {"x": 6.55, "y": -0.20, "z": 1.02},
        "pickup": Pose2D(6.55, -1.35, math.pi / 2),
    },
    "ProductB": {
        "storage": "StorageB",
        "slot": {"x": 6.55, "y": 3.80, "z": 1.02},
        "pickup": Pose2D(6.55, 2.65, math.pi / 2),
    },
}

DISPATCH_AREAS = [
    {"name": "DispatchA", "x": -8.0, "y": -4.0, "w": 2.4, "h": 1.4},
    {"name": "DispatchB", "x": -8.0, "y": 4.0, "w": 2.4, "h": 1.4},
]


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


def world_to_grid(x, y):
    gx = int((x - ORIGIN_X) / RESOLUTION)
    gy = int((y - ORIGIN_Y) / RESOLUTION)
    return max(0, min(GRID_W - 1, gx)), max(0, min(GRID_H - 1, gy))


def grid_to_world(gx, gy):
    return ORIGIN_X + (gx + 0.5) * RESOLUTION, ORIGIN_Y + (gy + 0.5) * RESOLUTION


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return {
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
        "w": cr * cp * cy + sr * sp * sy,
    }


def transform_body_to_world(pose, local_x, local_y):
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    return pose.x + c * local_x - s * local_y, pose.y + s * local_x + c * local_y


def motor_command(linear, angular, wheel_angles, dt):
    left_velocity = linear - angular * WHEEL_SEPARATION * 0.5
    right_velocity = linear + angular * WHEEL_SEPARATION * 0.5
    left_angular_velocity = left_velocity / WHEEL_RADIUS
    right_angular_velocity = right_velocity / WHEEL_RADIUS
    wheel_angles["front_left"] += left_angular_velocity * dt
    wheel_angles["rear_left"] += left_angular_velocity * dt
    wheel_angles["front_right"] += right_angular_velocity * dt
    wheel_angles["rear_right"] += right_angular_velocity * dt
    return WheelCommand(
        linear=linear,
        angular=angular,
        left_velocity=left_velocity,
        right_velocity=right_velocity,
        left_angular_velocity=left_angular_velocity,
        right_angular_velocity=right_angular_velocity,
        left_angle=wheel_angles["front_left"],
        right_angle=wheel_angles["front_right"],
    )


def parse_keepouts(task):
    keepouts = []
    for index, item in enumerate(task.get("keepouts", []), start=1):
        try:
            keepouts.append(Rect(f"keepout_{index}", float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"])))
        except (KeyError, TypeError, ValueError):
            continue
    return keepouts


def point_in_rects(x, y, rects, inflate=0.0):
    return any(rect.inflated(inflate).contains(x, y) for rect in rects)


def product_blocked(product, keepouts):
    pickup = product["pickup"]
    slot = product["slot"]
    return point_in_rects(pickup.x, pickup.y, keepouts, ROBOT_RADIUS) or point_in_rects(slot["x"], slot["y"], keepouts, 0.0)


def build_occupancy(keepouts=None, inflate=True):
    keepouts = keepouts or []
    rects = STATIC_OBSTACLES + keepouts
    if inflate:
        rects = [r.inflated(ROBOT_RADIUS) for r in rects]
    return rect_occupancy(rects)


def rect_occupancy(rects):
    occupied = set()
    for gy in range(GRID_H):
        y = ORIGIN_Y + (gy + 0.5) * RESOLUTION
        for gx in range(GRID_W):
            x = ORIGIN_X + (gx + 0.5) * RESOLUTION
            if any(rect.contains(x, y) for rect in rects):
                occupied.add((gx, gy))
    return occupied


def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(path):
    if len(path) < 2:
        return 0.0
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:]))


def astar(start_pose, goal_pose, keepouts):
    start = world_to_grid(start_pose.x, start_pose.y)
    goal = world_to_grid(goal_pose.x, goal_pose.y)
    occupied = build_occupancy(keepouts, inflate=True)
    occupied.discard(start)
    if goal in occupied:
        return []
    open_set = [(0.0, start)]
    came_from = {}
    gscore = {start: 0.0}
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            return orient_path([Pose2D(*grid_to_world(x, y)) for x, y in cells], goal_pose.yaw)
        for dx, dy, cost in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt[0] < 0 or nxt[0] >= GRID_W or nxt[1] < 0 or nxt[1] >= GRID_H or nxt in occupied:
                continue
            if dx != 0 and dy != 0 and ((current[0] + dx, current[1]) in occupied or (current[0], current[1] + dy) in occupied):
                continue
            tentative = gscore[current] + cost
            if tentative >= gscore.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            gscore[nxt] = tentative
            heapq.heappush(open_set, (tentative + heuristic(nxt, goal), nxt))
    return []


def orient_path(points, final_yaw):
    for index, point in enumerate(points[:-1]):
        nxt = points[index + 1]
        point.yaw = math.atan2(nxt.y - point.y, nxt.x - point.x)
    if points:
        points[-1].yaw = final_yaw
    return points


def write_pgm(path, occupied, keepouts=None):
    raw = bytearray()
    for image_y in range(GRID_H - 1, -1, -1):
        for x in range(GRID_W):
            cell = (x, image_y)
            if cell in occupied:
                raw.append(0)
            else:
                raw.append(254)
    path.write_bytes(f"P5\n{GRID_W} {GRID_H}\n255\n".encode("ascii") + bytes(raw))


def write_map_yaml(path, image_name):
    path.write_text(
        "\n".join(
            [
                f"image: {image_name}",
                "mode: trinary",
                f"resolution: {RESOLUTION}",
                f"origin: [{ORIGIN_X}, {ORIGIN_Y}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_nav2_params():
    (MAP_DIR / "nav2_params.yaml").write_text(
        """amcl:
  ros__parameters:
    use_sim_time: true

map_server:
  ros__parameters:
    use_sim_time: true
    yaml_filename: /workspace/aic_results/nav2_warehouse_map/warehouse_map.yaml
    topic_name: map
    frame_id: map

keepout_filter_mask_server:
  ros__parameters:
    use_sim_time: true
    topic_name: keepout_filter_mask
    yaml_filename: /workspace/aic_results/nav2_warehouse_map/keepout_mask.yaml

keepout_costmap_filter_info_server:
  ros__parameters:
    use_sim_time: true
    type: 0
    filter_info_topic: keepout_costmap_filter_info
    mask_topic: keepout_filter_mask
    base: 0.0
    multiplier: 1.0

bt_navigator:
  ros__parameters:
    use_sim_time: true
    default_nav_to_pose_bt_xml: /workspace/aic_results/nav2_warehouse_map/nav_to_pose_and_pause_near_goal_obstacle.xml

planner_server:
  ros__parameters:
    use_sim_time: true
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: true
      allow_unknown: false

controller_server:
  ros__parameters:
    use_sim_time: true
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.6
      lookahead_dist: 0.7
      use_collision_detection: true

global_costmap:
  global_costmap:
    ros__parameters:
      use_sim_time: true
      global_frame: map
      robot_base_frame: base_link
      plugins: ["static_layer", "inflation_layer"]
      filters: ["keepout_filter"]
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        inflation_radius: 0.7
      keepout_filter:
        plugin: "nav2_costmap_2d::KeepoutFilter"
        enabled: true
        filter_info_topic: keepout_costmap_filter_info

local_costmap:
  local_costmap:
    ros__parameters:
      use_sim_time: true
      global_frame: map
      robot_base_frame: base_link
      rolling_window: true
      width: 5
      height: 5
      resolution: 0.05
      plugins: ["inflation_layer"]
      filters: ["keepout_filter"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        inflation_radius: 0.7
      keepout_filter:
        plugin: "nav2_costmap_2d::KeepoutFilter"
        enabled: true
        filter_info_topic: keepout_costmap_filter_info
""",
        encoding="utf-8",
    )
    (MAP_DIR / "nav_to_pose_and_pause_near_goal_obstacle.xml").write_text(
        """<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithReplanning">
        <ControllerSelector selected_controller="{selected_controller}" default_controller="FollowPath" topic_name="controller_selector"/>
        <PlannerSelector selected_planner="{selected_planner}" default_planner="GridBased" topic_name="planner_selector"/>
        <RateController hz="1.0">
          <RecoveryNode number_of_retries="1" name="ComputePathToPose">
            <ComputePathToPose goal="{goal}" path="{path}" planner_id="{selected_planner}" error_code_id="{compute_path_error_code}" error_msg="{compute_path_error_msg}"/>
            <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
          </RecoveryNode>
        </RateController>
        <ReactiveSequence name="MonitorAndFollowPath">
          <PathLongerOnApproach path="{path}" prox_len="3.0" length_factor="2.0">
            <RetryUntilSuccessful num_attempts="1">
              <SequenceWithMemory name="CancelingControlAndWait">
                <CancelControl name="ControlCancel"/>
                <Wait wait_duration="5.0"/>
              </SequenceWithMemory>
            </RetryUntilSuccessful>
          </PathLongerOnApproach>
          <RecoveryNode number_of_retries="1" name="FollowPath">
            <FollowPath path="{path}" controller_id="{selected_controller}" error_code_id="{follow_path_error_code}" error_msg="{follow_path_error_msg}"/>
            <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
          </RecoveryNode>
        </ReactiveSequence>
      </PipelineSequence>
      <ReactiveFallback name="RecoveryFallback">
        <GoalUpdated/>
        <RoundRobin name="RecoveryActions">
          <Sequence name="ClearingActions">
            <ClearEntireCostmap name="ClearLocalCostmap-Subtree" service_name="local_costmap/clear_entirely_local_costmap"/>
            <ClearEntireCostmap name="ClearGlobalCostmap-Subtree" service_name="global_costmap/clear_entirely_global_costmap"/>
          </Sequence>
          <Spin spin_dist="1.57" error_code_id="{spin_error_code}" error_msg="{spin_error_msg}"/>
          <Wait wait_duration="5.0"/>
          <BackUp backup_dist="0.30" backup_speed="0.05" error_code_id="{backup_error_code}" error_msg="{backup_error_msg}"/>
        </RoundRobin>
      </ReactiveFallback>
    </RecoveryNode>
  </BehaviorTree>
</root>
""",
        encoding="utf-8",
    )


def generate_nav2_files(keepouts=None):
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    static_occupied = rect_occupancy(STATIC_OBSTACLES)
    keepout_occupied = rect_occupancy(keepouts or [])
    write_pgm(MAP_DIR / "warehouse_map.pgm", static_occupied)
    write_map_yaml(MAP_DIR / "warehouse_map.yaml", "warehouse_map.pgm")
    write_pgm(MAP_DIR / "keepout_mask.pgm", keepout_occupied)
    write_map_yaml(MAP_DIR / "keepout_mask.yaml", "keepout_mask.pgm")
    write_nav2_params()


class RosTelemetry:
    def __init__(self):
        self.available = False
        self.node = None
        try:
            import rclpy
            from geometry_msgs.msg import TransformStamped, Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import JointState
            from tf2_msgs.msg import TFMessage
        except ImportError as exc:
            print(f"ROS telemetry disabled: {exc}", flush=True)
            return

        self.rclpy = rclpy
        self.TransformStamped = TransformStamped
        self.Twist = Twist
        self.Odometry = Odometry
        self.JointState = JointState
        self.TFMessage = TFMessage
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("warehouse_fallback_odometry")
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
        self.tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)
        self.joint_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.available = True
        print("ROS telemetry enabled: publishing /cmd_vel, /odom, /tf, /joint_states", flush=True)

    def shutdown(self):
        if self.available:
            self.node.destroy_node()
            self.rclpy.shutdown()

    def publish(self, pose, command):
        if not self.available:
            return
        stamp = self.node.get_clock().now().to_msg()
        q = quaternion_from_euler(0.0, 0.0, pose.yaw)

        twist = self.Twist()
        twist.linear.x = command.linear
        twist.angular.z = command.angular
        self.cmd_pub.publish(twist)

        odom = self.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = pose.x
        odom.pose.pose.position.y = pose.y
        odom.pose.pose.position.z = 0.32
        odom.pose.pose.orientation.x = q["x"]
        odom.pose.pose.orientation.y = q["y"]
        odom.pose.pose.orientation.z = q["z"]
        odom.pose.pose.orientation.w = q["w"]
        odom.twist.twist.linear.x = command.linear
        odom.twist.twist.angular.z = command.angular
        self.odom_pub.publish(odom)

        tf = self.TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = pose.x
        tf.transform.translation.y = pose.y
        tf.transform.translation.z = 0.32
        tf.transform.rotation.x = q["x"]
        tf.transform.rotation.y = q["y"]
        tf.transform.rotation.z = q["z"]
        tf.transform.rotation.w = q["w"]

        laser_tf = self.TransformStamped()
        laser_tf.header.stamp = stamp
        laser_tf.header.frame_id = "base_link"
        laser_tf.child_frame_id = "laser"
        laser_tf.transform.translation.x = 0.36
        laser_tf.transform.translation.y = 0.0
        laser_tf.transform.translation.z = 0.34
        laser_tf.transform.rotation.w = 1.0
        self.tf_pub.publish(self.TFMessage(transforms=[tf, laser_tf]))

        joints = self.JointState()
        joints.header.stamp = stamp
        joints.name = ["left_wheel_joint", "right_wheel_joint"]
        joints.position = [command.left_angle, command.right_angle]
        joints.velocity = [command.left_angular_velocity, command.right_angular_velocity]
        self.joint_pub.publish(joints)


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


def set_model_pose_rpy(model, x, y, z, roll, pitch, yaw):
    q = quaternion_from_euler(roll, pitch, yaw)
    req = (
        f'name: "{model}", '
        f"position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}}, "
        "orientation: "
        f"{{x: {q['x']:.6f}, y: {q['y']:.6f}, z: {q['z']:.6f}, w: {q['w']:.6f}}}"
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


def set_robot_pose(pose):
    set_model_pose("warehouse_robot", pose.x, pose.y, 0.320, pose.yaw)


def set_wheel_poses(pose, command):
    return


def set_cargo_visible(pose):
    cargo_x = pose.x - math.cos(pose.yaw) * 0.18
    cargo_y = pose.y - math.sin(pose.yaw) * 0.18
    set_model_pose("cargo_item", cargo_x, cargo_y, 0.72, pose.yaw)


def hide_cargo():
    set_model_pose("cargo_item", -7.0, -5.0, HIDDEN_Z)


def show_delivered(x, y):
    set_model_pose("delivered_item", x, y, 0.28)


def hide_delivered():
    set_model_pose("delivered_item", -8.0, 4.0, HIDDEN_Z)


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
    return proc


def command_payload(command):
    if not command:
        return None
    return {
        "linear_x": command.linear,
        "angular_z": command.angular,
        "left_wheel_linear": command.left_velocity,
        "right_wheel_linear": command.right_velocity,
        "left_wheel_rad_s": command.left_angular_velocity,
        "right_wheel_rad_s": command.right_angular_velocity,
        "left_angle": command.left_angle,
        "right_angle": command.right_angle,
        "steering_angle": command.steering_angle,
    }


def state_payload(status, pose, task=None, path=None, cargo=None, message="", command=None):
    return {
        "status": status,
        "message": message,
        "robot": {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
        "odom": {"frame_id": "odom", "child_frame_id": "base_link", "x": pose.x, "y": pose.y, "yaw": pose.yaw},
        "motors": command_payload(command),
        "task": task,
        "path": [{"x": p.x, "y": p.y, "yaw": p.yaw} for p in (path or [])],
        "cargo": cargo,
        "map": map_payload(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def map_payload():
    return {
        "origin": {"x": ORIGIN_X, "y": ORIGIN_Y},
        "width_m": WIDTH_M,
        "height_m": HEIGHT_M,
        "resolution": RESOLUTION,
        "obstacles": [r.as_dict() for r in STATIC_OBSTACLES],
        "dispatches": DISPATCH_AREAS,
        "products": {
            name: {
                "storage": item["storage"],
                "slot": item["slot"],
                "pickup": {"x": item["pickup"].x, "y": item["pickup"].y, "yaw": item["pickup"].yaw},
            }
            for name, item in PRODUCTS.items()
        },
        "nav2_files": {
            "map": str(MAP_DIR / "warehouse_map.yaml"),
            "keepout": str(MAP_DIR / "keepout_mask.yaml"),
            "params": str(MAP_DIR / "nav2_params.yaml"),
            "behavior_tree": str(MAP_DIR / "nav_to_pose_and_pause_near_goal_obstacle.xml"),
        },
    }


def move_along(path, pose, task, cargo, telemetry, wheel_angles):
    if not path:
        return pose
    current = pose
    tick = 0
    for target in path[1:]:
        segment_start = current
        dx = target.x - segment_start.x
        dy = target.y - segment_start.y
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx) if dist > 0.01 else target.yaw
        yaw_delta = normalize_angle(target_yaw - segment_start.yaw)
        linear_steps = math.ceil(dist / (LINEAR_SPEED * STEP_SECONDS))
        angular_steps = math.ceil(abs(yaw_delta) / (MAX_ANGULAR_SPEED * STEP_SECONDS))
        steps = max(1, linear_steps, angular_steps)
        for step in range(1, steps + 1):
            ratio = step / steps
            previous = current
            next_yaw = normalize_angle(segment_start.yaw + yaw_delta * ratio)
            current = Pose2D(segment_start.x + dx * ratio, segment_start.y + dy * ratio, next_yaw)
            linear = math.hypot(current.x - previous.x, current.y - previous.y) / STEP_SECONDS
            angular = normalize_angle(current.yaw - previous.yaw) / STEP_SECONDS
            command = motor_command(linear, angular, wheel_angles, STEP_SECONDS)
            set_robot_pose(current)
            set_wheel_poses(current, command)
            telemetry.publish(current, command)
            if cargo:
                set_cargo_visible(current)
            if tick % MOTOR_LOG_INTERVAL_STEPS == 0:
                print(
                        "[motors] "
                        f"odom=({current.x:.2f},{current.y:.2f},{current.yaw:.2f}) "
                        f"cmd_vel=(linear.x={command.linear:.2f}, angular.z={command.angular:.2f}) "
                        f"left=(v={command.left_velocity:.2f}m/s, omega={command.left_angular_velocity:.2f}rad/s, "
                        f"angle={command.left_angle:.2f}) "
                        f"right=(v={command.right_velocity:.2f}m/s, omega={command.right_angular_velocity:.2f}rad/s, "
                        f"angle={command.right_angle:.2f}) "
                    f"steering={command.steering_angle:.2f}",
                    flush=True,
                )
            write_json(STATE_FILE, state_payload("executing", current, task, path, cargo, command=command))
            tick += 1
            time.sleep(STEP_SECONDS)
    current = Pose2D(path[-1].x, path[-1].y, path[-1].yaw)
    set_robot_pose(current)
    return current


def plan_product_candidate(name, product, pose, drop_pose, keepouts):
    if product_blocked(product, keepouts):
        return None
    pickup_pose = product["pickup"]
    pickup_path = astar(pose, pickup_pose, keepouts)
    if not pickup_path:
        return None
    drop_path = astar(pickup_pose, drop_pose, keepouts)
    if not drop_path:
        return None
    score = path_length(pickup_path) + path_length(drop_path)
    return {
        "name": name,
        "product": product,
        "pickup_pose": pickup_pose,
        "pickup_path": pickup_path,
        "drop_path": drop_path,
        "score": score,
    }


def select_product_plan(requested_name, pose, drop_pose, keepouts):
    requested = PRODUCTS.get(requested_name)
    requested_plan = plan_product_candidate(requested_name, requested, pose, drop_pose, keepouts) if requested else None
    if requested_plan:
        return requested_plan, None

    candidates = []
    for name, product in PRODUCTS.items():
        if name == requested_name:
            continue
        candidate = plan_product_candidate(name, product, pose, drop_pose, keepouts)
        if candidate:
            candidates.append(candidate)
    if not candidates:
        return None, f"no reachable shelf for requested product={requested_name}"
    candidates.sort(key=lambda item: item["score"])
    reason = f"requested product={requested_name} is blocked or unreachable; switched to nearest reachable shelf={candidates[0]['name']}"
    return candidates[0], reason


def execute_task(task, pose, telemetry, wheel_angles):
    product_name = task.get("product", "ProductR")
    try:
        drop = task["drop"]
        drop_pose = Pose2D(float(drop["x"]), float(drop["y"]), float(drop.get("yaw", 0.0)))
    except (KeyError, TypeError, ValueError):
        print("Task rejected: drop pose is missing or invalid", flush=True)
        return pose

    keepouts = parse_keepouts(task)
    generate_nav2_files(keepouts)
    if point_in_rects(drop_pose.x, drop_pose.y, keepouts, ROBOT_RADIUS):
        print("Task rejected: drop point is inside a keepout zone", flush=True)
        write_json(STATE_FILE, state_payload("failed", pose, task, [], None, "drop point is inside keepout"))
        return pose

    plan, switch_reason = select_product_plan(product_name, pose, drop_pose, keepouts)
    if not plan:
        print(f"Task rejected: {switch_reason}", flush=True)
        write_json(STATE_FILE, state_payload("failed", pose, task, [], None, switch_reason or "no reachable shelf"))
        return pose

    product_name = plan["name"]
    product = plan["product"]
    pickup_pose = plan["pickup_pose"]
    pickup_path = plan["pickup_path"]
    drop_path = plan["drop_path"]
    full_path = pickup_path + drop_path[1:] if pickup_path and drop_path else []
    effective_task = dict(task)
    effective_task["requested_product"] = task.get("product", "ProductR")
    effective_task["product"] = product_name
    if switch_reason:
        effective_task["switch_reason"] = switch_reason
    write_json(STATE_FILE, state_payload("planned", pose, task, full_path, None, "path planned"))
    if not full_path:
        print("Task rejected: no route through current map and keepout zones", flush=True)
        write_json(STATE_FILE, state_payload("failed", pose, task, [], None, "no path"))
        return pose

    if switch_reason:
        print(f"TaskGoal rerouted: {switch_reason}", flush=True)
    print(
        "TaskGoal accepted "
        f"product={product_name} pickup=({pickup_pose.x:.2f},{pickup_pose.y:.2f}) "
        f"drop=({drop_pose.x:.2f},{drop_pose.y:.2f}) keepouts={len(keepouts)}",
        flush=True,
    )
    write_json(STATE_FILE, state_payload("planned", pose, effective_task, full_path, None, "path planned"))
    pose = move_along(pickup_path, pose, effective_task, None, telemetry, wheel_angles)
    print(
        f"pick_up product={product_name} storage={product['storage']} "
        f"slot=({product['slot']['x']:.2f},{product['slot']['y']:.2f},{product['slot']['z']:.2f})",
        flush=True,
    )
    set_cargo_visible(pose)
    time.sleep(0.3)
    pose = move_along(drop_path, pose, effective_task, product_name, telemetry, wheel_angles)
    hide_cargo()
    show_delivered(drop_pose.x, drop_pose.y)
    print(f"drop_off product={product_name} target=({drop_pose.x:.2f},{drop_pose.y:.2f})", flush=True)
    write_json(STATE_FILE, state_payload("done", pose, effective_task, full_path, None, "task completed"))
    return pose


def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_nav2_files()
    pose = Pose2D(-7.0, -5.0, 0.0)
    wheel_angles = {"front_left": 0.0, "rear_left": 0.0, "front_right": 0.0, "rear_right": 0.0}
    telemetry = RosTelemetry()
    write_json(STATE_FILE, state_payload("starting", pose, None, [], None, "starting gazebo"))

    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("GALLIUM_DRIVER", "llvmpipe")

    print("Starting warehouse Nav2 task mode")
    print("Generated Nav2 map/keepout files in /workspace/aic_results/nav2_warehouse_map")
    print("Use the Map Task window to set keepout zones and send TaskGoal.")

    gz = launch_gazebo(env)

    def shutdown(signum, frame):
        print("Stopping warehouse Nav2 task mode", flush=True)
        try:
            os.killpg(os.getpgid(gz.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        telemetry.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    time.sleep(8)
    set_robot_pose(pose)
    set_wheel_poses(pose, motor_command(0.0, 0.0, wheel_angles, STEP_SECONDS))
    hide_cargo()
    hide_delivered()
    write_json(STATE_FILE, state_payload("idle", pose, None, [], None, "waiting for TaskGoal"))
    last_task_version = TASK_FILE.stat().st_mtime_ns if TASK_FILE.exists() else None

    while gz.poll() is None:
        if TASK_FILE.exists():
            version = TASK_FILE.stat().st_mtime_ns
            if version != last_task_version:
                last_task_version = version
                try:
                    task = read_json(TASK_FILE)
                    pose = execute_task(task, pose, telemetry, wheel_angles)
                except json.JSONDecodeError as exc:
                    print(f"Task rejected: invalid JSON: {exc}", flush=True)
        else:
            write_json(STATE_FILE, state_payload("idle", pose, None, [], None, "waiting for TaskGoal"))
        time.sleep(TASK_POLL_SECONDS)
    telemetry.shutdown()
    return gz.returncode


if __name__ == "__main__":
    raise SystemExit(main())
