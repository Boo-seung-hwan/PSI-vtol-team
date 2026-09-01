#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /workspace/ros2/ws/install/setup.bash

ros2 run drone_vision target_depth_estimator --ros-args \
  -p bbox_topic:=/vision/yolo_bbox \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/aligned_depth_to_color/camera_info \
  -p output_topic:=/vision/target_camera_xyz \
  -p max_bbox_depth_dt_s:=1.0 \
  -p patch_radius:=30 \
  -p depth_scale_m:=0.001
