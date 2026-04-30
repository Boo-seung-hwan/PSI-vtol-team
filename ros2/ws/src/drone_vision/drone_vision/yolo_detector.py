import rclpy
from rclpy.node import Node


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.get_logger().info('drone_vision: yolo_detector started')


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
