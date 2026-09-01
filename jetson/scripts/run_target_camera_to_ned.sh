#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /workspace/ros2/ws/install/setup.bash

ros2 run drone_vision target_camera_to_ned
