from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_share = FindPackageShare('drone_bringup')

    sim_launch = PathJoinSubstitution([pkg_share, 'launch', 'sim.launch.py'])
    vision_launch = PathJoinSubstitution([pkg_share, 'launch', 'vision.launch.py'])
    control_launch = PathJoinSubstitution([pkg_share, 'launch', 'control.launch.py'])

    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(sim_launch)),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(vision_launch)),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(control_launch)),
    ])
