import time
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        self.declare_parameter("image_topic", "/vtol/camera")
        self.declare_parameter("model_path", "/workspace/ros2/ws/src/drone_vision/models/best.pt")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("target_class_id", 0)

        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.conf_th = float(self.get_parameter("confidence_threshold").value)
        self.target_class_id = int(self.get_parameter("target_class_id").value)

        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install it inside ros2-vision container."
            )

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos,
        )

        self.bbox_pub = self.create_publisher(
            Float32MultiArray,
            "/vision/yolo_bbox",
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            "/vision/yolo_status",
            10,
        )

        self.annotated_pub = self.create_publisher(
            Image,
            "/vision/yolo_annotated_image",
            10
        )

        self.last_log_time = 0.0

        self.get_logger().info("drone_vision: yolo_detector started")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"model_path: {self.model_path}")
        self.get_logger().info(f"confidence_threshold: {self.conf_th}")
        self.get_logger().info(f"target_class_id: {self.target_class_id}")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return

        annotated = frame.copy()

        results = self.model.predict(
            source=frame,
            conf=self.conf_th,
            verbose=False,
        )

        best = None

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id != self.target_class_id:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x1_i = int(x1)
                y1_i = int(y1)
                x2_i = int(x2)
                y2_i = int(y2)

                label = f"class={cls_id} conf={conf:.2f}"

                cv2.rectangle(
                    annotated,
                    (x1_i, y1_i),
                    (x2_i, y2_i),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1_i, max(20, y1_i - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                if best is None or conf > best["conf"]:
                    best = {
                        "cls_id": cls_id,
                        "conf": conf,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }

        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_msg.header = msg.header
        self.annotated_pub.publish(annotated_msg)

        bbox_msg = Float32MultiArray()

        # /vision/yolo_bbox format:
        # [valid, class_id, confidence, cx, cy, width, height, image_width, image_height]
        if best is None:
            bbox_msg.data = [
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                float(msg.width),
                float(msg.height),
            ]
            self.bbox_pub.publish(bbox_msg)
            return

        cx = 0.5 * (best["x1"] + best["x2"])
        cy = 0.5 * (best["y1"] + best["y2"])
        w = best["x2"] - best["x1"]
        h = best["y2"] - best["y1"]

        bbox_msg.data = [
            1.0,
            float(best["cls_id"]),
            float(best["conf"]),
            float(cx),
            float(cy),
            float(w),
            float(h),
            float(msg.width),
            float(msg.height),
        ]
        self.bbox_pub.publish(bbox_msg)

        now = time.monotonic()
        if now - self.last_log_time > 1.0:
            self.last_log_time = now

            status = String()
            status.data = (
                f"DETECTED class={best['cls_id']} "
                f"conf={best['conf']:.2f} "
                f"center=({cx:.1f},{cy:.1f}) "
                f"size=({w:.1f},{h:.1f}) "
                f"image=({msg.width},{msg.height})"
            )
            self.status_pub.publish(status)
            self.get_logger().info(status.data)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
