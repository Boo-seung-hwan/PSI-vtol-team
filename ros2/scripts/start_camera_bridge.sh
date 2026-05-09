#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /workspace/ros2/ws/install/setup.bash ]; then
  source /workspace/ros2/ws/install/setup.bash
fi

echo "[INFO] Starting camera bridge: /vtol/camera"

ros2 run ros_gz_bridge parameter_bridge \
  "/vtol/camera@sensor_msgs/msg/Image[gz.msgs.Image" \
  "/vtol/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
