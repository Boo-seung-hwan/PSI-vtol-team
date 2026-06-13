from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    common_env = {
        "PYTHONOPTIMIZE": "1",
    }

    return LaunchDescription([
        Node(
            package="drone_control",
            executable="setpoint_mux",
            name="setpoint_mux",
            output="screen",
            emulate_tty=True,
            additional_env=common_env,
        ),

        Node(
            package="drone_control",
            executable="mission_manager",
            name="mission_manager",
            output="screen",
            emulate_tty=True,
            additional_env=common_env,
        ),

        Node(
            package="drone_control",
            executable="generator_tracking",
            name="generator_tracking",
            output="screen",
            emulate_tty=True,
            additional_env=common_env,
        ),

        Node(
            package="drone_control",
            executable="vertiport_tracking",
            name="vertiport_tracking",
            output="screen",
            emulate_tty=True,
            additional_env=common_env,
        ),
    
        Node(
            package="drone_vision",
            executable="yolo_detector",
            name="yolo_detector",
            output="screen",
            emulate_tty=True,
        ),

        Node(
            package="drone_vision",
            executable="target_depth_estimator",
            name="target_depth_estimator",
            output="screen",
            emulate_tty=True,
        ),

        Node(
            package="drone_vision",
            executable="target_camera_to_ned",
            name="target_camera_to_ned",
            output="screen",
            emulate_tty=True,
        ),
    ])
    