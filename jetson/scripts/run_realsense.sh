#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /workspace/ros2/ws/install/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true
