import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


class TargetDepthEstimator(Node):
    def __init__(self):
        super().__init__("target_depth_estimator")

        self.declare_parameter("bbox_topic", "/vision/yolo_bbox")
        self.declare_parameter("depth_topic", "/vtol/depth")
        self.declare_parameter("output_topic", "/vision/target_camera_xyz")
        self.declare_parameter("horizontal_fov", 1.3962634)
        self.declare_parameter("patch_radius", 3)

        self.bbox_topic = self.get_parameter("bbox_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.hfov = float(self.get_parameter("horizontal_fov").value)
        self.patch_radius = int(self.get_parameter("patch_radius").value)

        self.bridge = CvBridge()
        self.latest_bbox = None

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bbox_sub = self.create_subscription(
            Float32MultiArray,
            self.bbox_topic,
            self.bbox_callback,
            10,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            image_qos,
        )

        self.point_pub = self.create_publisher(
            PointStamped,
            self.output_topic,
            10,
        )

        self.get_logger().info("target_depth_estimator started")
        self.get_logger().info(f"bbox_topic: {self.bbox_topic}")
        self.get_logger().info(f"depth_topic: {self.depth_topic}")
        self.get_logger().info(f"output_topic: {self.output_topic}")

    def bbox_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 9:
            return

        valid = msg.data[0]
        if valid < 0.5:
            self.latest_bbox = None
            return

        self.latest_bbox = {
            "class_id": int(msg.data[1]),
            "conf": float(msg.data[2]),
            "cx": float(msg.data[3]),
            "cy": float(msg.data[4]),
            "bbox_w": float(msg.data[5]),
            "bbox_h": float(msg.data[6]),
            "image_w": float(msg.data[7]),
            "image_h": float(msg.data[8]),
        }

    def depth_callback(self, msg: Image):
        if self.latest_bbox is None:
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth = np.asarray(depth, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"depth conversion failed: {e}")
            return

        h, w = depth.shape[:2]

        u = int(round(self.latest_bbox["cx"]))
        v = int(round(self.latest_bbox["cy"]))

        if u < 0 or u >= w or v < 0 or v >= h:
            return

        r = self.patch_radius
        u1 = max(0, u - r)
        u2 = min(w, u + r + 1)
        v1 = max(0, v - r)
        v2 = min(h, v + r + 1)

        patch = depth[v1:v2, u1:u2]
        valid_depth = patch[np.isfinite(patch)]
        valid_depth = valid_depth[(valid_depth > 0.05) & (valid_depth < 100.0)]

        if valid_depth.size == 0:
            self.get_logger().warn("no valid depth around bbox center")
            return

        Z = float(np.median(valid_depth))

        image_w = float(w)
        image_h = float(h)

        fx = image_w / (2.0 * math.tan(self.hfov / 2.0))
        vfov = 2.0 * math.atan(math.tan(self.hfov / 2.0) * image_h / image_w)
        fy = image_h / (2.0 * math.tan(vfov / 2.0))

        cx0 = image_w / 2.0
        cy0 = image_h / 2.0

        X = (u - cx0) * Z / fx
        Y = (v - cy0) * Z / fy

        out = PointStamped()
        out.header = msg.header
        out.header.frame_id = "vtol_camera_link"
        out.point.x = float(X)
        out.point.y = float(Y)
        out.point.z = float(Z)

        self.point_pub.publish(out)

        self.get_logger().info(
            f"target_camera_xyz=({X:.2f}, {Y:.2f}, {Z:.2f}) "
            f"pixel=({u},{v}) depth={Z:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TargetDepthEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
