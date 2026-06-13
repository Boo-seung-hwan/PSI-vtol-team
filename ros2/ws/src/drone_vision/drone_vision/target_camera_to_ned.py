import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleLocalPosition


class TargetCameraToNED(Node):
    def __init__(self):
        super().__init__("target_camera_to_ned")

        self.declare_parameter("camera_point_topic", "/vision/target_camera_xyz")
        self.declare_parameter("output_topic", "/target/ned")

        self.declare_parameter("camera_forward_offset_m", 0.35)
        self.declare_parameter("camera_right_offset_m", 0.0)
        self.declare_parameter("camera_down_offset_m", 0.08)

        self.camera_point_topic = self.get_parameter("camera_point_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.camera_forward_offset = float(
            self.get_parameter("camera_forward_offset_m").value
        )
        self.camera_right_offset = float(
            self.get_parameter("camera_right_offset_m").value
        )
        self.camera_down_offset = float(
            self.get_parameter("camera_down_offset_m").value
        )

        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.current_pos = None
        self.current_yaw = 0.0

        self.local_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.local_position_callback,
            px4_qos,
        )

        self.camera_sub = self.create_subscription(
            PointStamped,
            self.camera_point_topic,
            self.camera_point_callback,
            10,
        )

        self.target_pub = self.create_publisher(
            Float32MultiArray,
            self.output_topic,
            10,
        )

        self.get_logger().info("target_camera_to_ned started")
        self.get_logger().info(f"camera_point_topic: {self.camera_point_topic}")
        self.get_logger().info(f"output_topic: {self.output_topic}")

    def local_position_callback(self, msg: VehicleLocalPosition):
        self.current_pos = np.array([msg.x, msg.y, msg.z], dtype=float)

        try:
            self.current_yaw = float(msg.heading)
        except AttributeError:
            self.current_yaw = 0.0

    def camera_point_callback(self, msg: PointStamped):
        if self.current_pos is None:
            return

        x_cam = float(msg.point.x)
        y_cam = float(msg.point.y)
        z_cam = float(msg.point.z)

        if not all(math.isfinite(v) for v in [x_cam, y_cam, z_cam]):
            return

        if z_cam <= 0.05:
            return

        forward_body = self.camera_forward_offset - y_cam
        right_body = self.camera_right_offset + x_cam
        down_body = self.camera_down_offset + z_cam

        yaw = self.current_yaw

        north_offset = math.cos(yaw) * forward_body - math.sin(yaw) * right_body
        east_offset = math.sin(yaw) * forward_body + math.cos(yaw) * right_body
        down_offset = down_body

        target_x = self.current_pos[0] + north_offset
        target_y = self.current_pos[1] + east_offset
        target_z = self.current_pos[2] + down_offset

        out = Float32MultiArray()
        out.data = [
            float(target_x),
            float(target_y),
            float(target_z),
            float(yaw),
            1.0,
        ]

        self.target_pub.publish(out)

        self.get_logger().info(
            f"target_ned=({target_x:.2f}, {target_y:.2f}, {target_z:.2f}), "
            f"camera_xyz=({x_cam:.2f}, {y_cam:.2f}, {z_cam:.2f}), "
            f"vehicle=({self.current_pos[0]:.2f}, {self.current_pos[1]:.2f}, {self.current_pos[2]:.2f}), "
            f"yaw={math.degrees(yaw):.1f} deg"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TargetCameraToNED()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
