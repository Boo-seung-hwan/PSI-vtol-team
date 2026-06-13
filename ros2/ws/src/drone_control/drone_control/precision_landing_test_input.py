import rclpy
import math
import random

from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy

from std_msgs.msg import String, Float32MultiArray
from px4_msgs.msg import VehicleLocalPosition


class PrecisionLandingTestInput(Node):
    def __init__(self):
        super().__init__("precision_landing_test_input")

        self.current_pos = None
        self.current_heading = 0.0
        
        self.target_pos = None
        self.target_yaw = 0.0
        self.last_target_time = self.get_clock().now()
        self.target_interval = 5.0  
        self.target_radius = 5.0

        self.mission_pub = self.create_publisher(String, "/mission/state", 10)
        self.target_pub = self.create_publisher(Float32MultiArray, "/target/ned", 10)

        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.local_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.local_callback,
            px4_qos,
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

    def local_callback(self, msg):
        self.current_pos = (float(msg.x), float(msg.y), float(msg.z))
        self.current_heading = float(msg.heading)

    def timer_callback(self):
        mission_msg = String()
        mission_msg.data = "VERTIPORT_LANDING"
        self.mission_pub.publish(mission_msg)

        if self.current_pos is None:
            self.get_logger().warn("Waiting for /fmu/out/vehicle_local_position...")
            return

        now = self.get_clock().now()

        if self.target_pos is None:
            self.generate_random_target_on_circle()

        elapsed = (now - self.last_target_time).nanoseconds / 1e9

        if elapsed >= self.target_interval:
            self.generate_random_target_on_circle()
            self.last_target_time = now

        target_x, target_y, target_z = self.target_pos

        target_msg = Float32MultiArray()
        target_msg.data = [
            target_x,
            target_y,
            target_z,
            self.target_yaw,
            1.0,
        ]
        self.target_pub.publish(target_msg)

        self.get_logger().info(
            f"Published target: x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}, yaw={self.target_yaw:.2f}"
        )    


    def generate_random_target_on_circle(self):
        if self.current_pos is None:
            return

        x, y, z = self.current_pos

        theta = random.uniform(0.0, 2.0 * math.pi)

        target_x = x + self.target_radius * math.cos(theta)
        target_y = y + self.target_radius * math.sin(theta)
        target_z = z

        self.target_pos = (target_x, target_y, target_z)
        self.target_yaw = self.current_heading

        self.get_logger().info(
            f"New random target: x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}, yaw={self.target_yaw:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionLandingTestInput()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()