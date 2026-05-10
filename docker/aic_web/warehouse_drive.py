#!/usr/bin/env python3
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


AIC_SETUP = Path("/ws_aic/install/setup.bash")
WHEEL_RADIUS_M = 0.17
WHEEL_SEPARATION_M = 0.92
BASE_Z_M = 0.32
WHEEL_Z_M = 0.18
WHEEL_Y_M = 0.46
POSE_COMMAND_TIMEOUT_SECONDS = 2.0
GZ_SERVICE_TIMEOUT_MS = 1500


@dataclass
class OdomPose:
    x: float
    y: float
    yaw: float


@dataclass
class MotorCommand:
    linear_x: float
    angular_z: float
    left_mps: float
    right_mps: float
    left_rad_s: float
    right_rad_s: float
    left_angle: float
    right_angle: float


def normalize(angle):
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
        "w": cr * cp * cy + sr * sp * sy,
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
    }


def gz_shell(command):
    if AIC_SETUP.exists():
        return f"source {AIC_SETUP} && {command}"
    return command


def run(cmd, timeout=None):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, exc.stdout or "")


def world_offset(pose, forward, left):
    return (
        pose.x + math.cos(pose.yaw) * forward - math.sin(pose.yaw) * left,
        pose.y + math.sin(pose.yaw) * forward + math.cos(pose.yaw) * left,
    )


class DiffDriveOdometry:
    def __init__(self, x, y, yaw, wheel_radius=WHEEL_RADIUS_M, wheel_separation=WHEEL_SEPARATION_M):
        self.pose = OdomPose(x, y, yaw)
        self.wheel_radius = wheel_radius
        self.wheel_separation = wheel_separation
        self.left_angle = 0.0
        self.right_angle = 0.0

    def command_from_twist(self, linear_x, angular_z, dt):
        left_mps = linear_x - angular_z * self.wheel_separation / 2.0
        right_mps = linear_x + angular_z * self.wheel_separation / 2.0
        left_rad_s = left_mps / self.wheel_radius
        right_rad_s = right_mps / self.wheel_radius
        self.left_angle = normalize(self.left_angle + left_rad_s * dt)
        self.right_angle = normalize(self.right_angle + right_rad_s * dt)
        return MotorCommand(
            linear_x=linear_x,
            angular_z=angular_z,
            left_mps=left_mps,
            right_mps=right_mps,
            left_rad_s=left_rad_s,
            right_rad_s=right_rad_s,
            left_angle=self.left_angle,
            right_angle=self.right_angle,
        )

    def integrate_cmd(self, linear_x, angular_z, dt):
        cmd = self.command_from_twist(linear_x, angular_z, dt)
        mid_yaw = self.pose.yaw + angular_z * dt * 0.5
        self.pose = OdomPose(
            self.pose.x + math.cos(mid_yaw) * linear_x * dt,
            self.pose.y + math.sin(mid_yaw) * linear_x * dt,
            normalize(self.pose.yaw + angular_z * dt),
        )
        return self.pose, cmd

    def project_pose(self, x, y, yaw, dt):
        dt = max(dt, 1e-3)
        dx = x - self.pose.x
        dy = y - self.pose.y
        heading = self.pose.yaw
        linear_x = (math.cos(heading) * dx + math.sin(heading) * dy) / dt
        angular_z = normalize(yaw - self.pose.yaw) / dt
        cmd = self.command_from_twist(linear_x, angular_z, dt)
        self.pose = OdomPose(x, y, normalize(yaw))
        return self.pose, cmd


class RosDriveTelemetry:
    def __init__(self, node_name="warehouse_drive_projector"):
        self.available = False
        try:
            import rclpy
            from geometry_msgs.msg import TransformStamped, Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import JointState
            from tf2_msgs.msg import TFMessage
        except ImportError as exc:
            print(f"ROS drive telemetry disabled: {exc}", flush=True)
            return
        self.rclpy = rclpy
        self.TransformStamped = TransformStamped
        self.Twist = Twist
        self.Odometry = Odometry
        self.JointState = JointState
        self.TFMessage = TFMessage
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node(node_name)
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
        self.tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)
        self.joint_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.available = True
        print("ROS drive telemetry enabled: publishing /cmd_vel, /odom, /tf, /joint_states", flush=True)

    def publish(self, pose, cmd):
        if not self.available:
            return
        stamp = self.node.get_clock().now().to_msg()
        q = quaternion_from_euler(0.0, 0.0, pose.yaw)

        twist = self.Twist()
        twist.linear.x = cmd.linear_x
        twist.angular.z = cmd.angular_z
        self.cmd_pub.publish(twist)

        odom = self.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = pose.x
        odom.pose.pose.position.y = pose.y
        odom.pose.pose.position.z = BASE_Z_M
        odom.pose.pose.orientation.x = q["x"]
        odom.pose.pose.orientation.y = q["y"]
        odom.pose.pose.orientation.z = q["z"]
        odom.pose.pose.orientation.w = q["w"]
        odom.twist.twist.linear.x = cmd.linear_x
        odom.twist.twist.angular.z = cmd.angular_z
        self.odom_pub.publish(odom)

        base_tf = self.TransformStamped()
        base_tf.header.stamp = stamp
        base_tf.header.frame_id = "odom"
        base_tf.child_frame_id = "base_link"
        base_tf.transform.translation.x = pose.x
        base_tf.transform.translation.y = pose.y
        base_tf.transform.translation.z = BASE_Z_M
        base_tf.transform.rotation.x = q["x"]
        base_tf.transform.rotation.y = q["y"]
        base_tf.transform.rotation.z = q["z"]
        base_tf.transform.rotation.w = q["w"]

        laser_tf = self.TransformStamped()
        laser_tf.header.stamp = stamp
        laser_tf.header.frame_id = "base_link"
        laser_tf.child_frame_id = "laser"
        laser_tf.transform.translation.x = 0.36
        laser_tf.transform.translation.y = 0.0
        laser_tf.transform.translation.z = 0.34
        laser_tf.transform.rotation.w = 1.0
        self.tf_pub.publish(self.TFMessage(transforms=[base_tf, laser_tf]))

        joints = self.JointState()
        joints.header.stamp = stamp
        joints.name = ["left_wheel_joint", "right_wheel_joint"]
        joints.position = [cmd.left_angle, cmd.right_angle]
        joints.velocity = [cmd.left_rad_s, cmd.right_rad_s]
        self.joint_pub.publish(joints)


class GazeboDriveProjector:
    def __init__(self, world_name, robot_model="warehouse_robot", map_origin=None, map_resolution=None):
        self.world_name = world_name
        self.robot_model = robot_model
        self.map_origin = map_origin
        self.map_resolution = map_resolution
        self.odom = None
        self.tick = 0
        self.telemetry = RosDriveTelemetry(f"{world_name}_drive_projector")

    def reset(self, x, y, yaw):
        self.odom = DiffDriveOdometry(x, y, yaw)
        pose = OdomPose(x, y, yaw)
        self.set_robot_pose(pose)
        self.set_wheel_poses(pose)
        return pose

    def set_model_pose(self, model, x, y, z, yaw=0.0):
        q = quaternion_from_euler(0.0, 0.0, yaw)
        self.set_model_pose_quat(model, x, y, z, q)

    def set_model_pose_quat(self, model, x, y, z, q):
        req = (
            f'name: "{model}", '
            f"position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}}, "
            f"orientation: {{x: {q['x']:.6f}, y: {q['y']:.6f}, z: {q['z']:.6f}, w: {q['w']:.6f}}}"
        )
        cmd = [
            "/bin/bash",
            "-lc",
            gz_shell(
                f"gz service -s /world/{self.world_name}/set_pose "
                "--reqtype gz.msgs.Pose "
                "--reptype gz.msgs.Boolean "
                f"--timeout {GZ_SERVICE_TIMEOUT_MS} "
                f"--req '{req}'"
            ),
        ]
        result = run(cmd, timeout=POSE_COMMAND_TIMEOUT_SECONDS)
        if result.returncode != 0:
            print(result.stdout, end="", flush=True)

    def set_robot_pose(self, pose):
        self.set_model_pose(self.robot_model, pose.x, pose.y, BASE_Z_M, pose.yaw)

    def set_wheel_poses(self, pose):
        return

    def map_cell(self, pose):
        if not self.map_origin or not self.map_resolution:
            return None
        ox, oy = self.map_origin
        return int((pose.x - ox) / self.map_resolution), int((pose.y - oy) / self.map_resolution)

    def log_command(self, prefix, pose, cmd, log_every):
        if log_every <= 0 or self.tick % log_every != 0:
            return
        cell = self.map_cell(pose)
        cell_text = f" map_cell=({cell[0]},{cell[1]})" if cell else ""
        print(
            f"[drive:{prefix}] "
            f"cmd_vel=(linear.x={cmd.linear_x:.3f}, angular.z={cmd.angular_z:.3f}) "
            f"odom=({pose.x:.3f},{pose.y:.3f},{pose.yaw:.3f}){cell_text} "
            f"motors: left(v={cmd.left_mps:.3f}m/s phi_dot={cmd.left_rad_s:.3f}rad/s angle={cmd.left_angle:.3f}) "
            f"right(v={cmd.right_mps:.3f}m/s phi_dot={cmd.right_rad_s:.3f}rad/s angle={cmd.right_angle:.3f}) "
            f"wheel_xy: L={world_offset(pose, 0.0, WHEEL_Y_M)} R={world_offset(pose, 0.0, -WHEEL_Y_M)}",
            flush=True,
        )

    def apply_cmd(self, linear_x, angular_z, dt, prefix="cmd_vel", log_every=8):
        if self.odom is None:
            self.reset(0.0, 0.0, 0.0)
        pose, cmd = self.odom.integrate_cmd(linear_x, angular_z, dt)
        self.set_robot_pose(pose)
        self.set_wheel_poses(pose)
        self.log_command(prefix, pose, cmd, log_every)
        self.telemetry.publish(pose, cmd)
        self.tick += 1
        return pose

    def project_pose(self, x, y, yaw, dt, prefix="pose", log_every=8):
        if self.odom is None:
            self.reset(x, y, yaw)
        pose, cmd = self.odom.project_pose(x, y, yaw, dt)
        self.set_robot_pose(pose)
        self.set_wheel_poses(pose)
        self.log_command(prefix, pose, cmd, log_every)
        self.telemetry.publish(pose, cmd)
        self.tick += 1
        return pose


def wheel_model(name):
    return f"""
    <model name="{name}">
      <static>false</static>
      <pose>0 0 -10 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>{WHEEL_RADIUS_M:.3f}</radius><length>0.120</length></cylinder></geometry>
          <material><ambient>0.02 0.03 0.04 1</ambient><diffuse>0.03 0.04 0.05 1</diffuse></material>
        </visual>
        <visual name="stripe">
          <pose>{WHEEL_RADIUS_M * 0.52:.3f} 0 0 0 0 0</pose>
          <geometry><box><size>{WHEEL_RADIUS_M:.3f} 0.130 0.026</size></box></geometry>
          <material><ambient>0.90 0.92 0.95 1</ambient><diffuse>0.90 0.92 0.95 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def wheel_models():
    return ""
