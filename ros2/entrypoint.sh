#!/bin/bash
set -e


source /opt/ros/humble/setup.bash


cd /workspace/ros2

if [ -f /workspace/ros2/ws/install/setup.bash ]; then
    source /workspace/ros2/ws/install/setup.bash
fi

exec "$@"
