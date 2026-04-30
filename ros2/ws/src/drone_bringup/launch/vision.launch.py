from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    vision_config = PathJoinSubstitution([
        FindPackageShare('drone_bringup'),
        'config',
        'vision.yaml'
    ])

    return LaunchDescription([
        Node(
            package='drone_vision',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen',
            parameters=[vision_config]
        )
    ])
