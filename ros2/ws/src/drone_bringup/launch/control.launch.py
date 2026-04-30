from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    control_config = PathJoinSubstitution([
        FindPackageShare('drone_bringup'),
        'config',
        'control.yaml'
    ])

    return LaunchDescription([
        Node(
            package='drone_control',
            executable='offboard_controller',
            name='offboard_controller',
            output='screen',
            parameters=[control_config]
        )
    ])
