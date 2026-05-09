#!/bin/bash
set -e

echo "[CHECK] Docker containers"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "px4-gazebo|ros2-vision|microxrce-agent" || true

echo
echo "[CHECK] PX4 Gazebo model file"
docker exec px4-gazebo bash -lc '
wc -l /workspace/px4/PX4-Autopilot/Tools/simulation/gz/models/standard_vtol/model.sdf
grep -n "vtol_camera_link\|/vtol/camera\|sensor name=\"camera\"" /workspace/px4/PX4-Autopilot/Tools/simulation/gz/models/standard_vtol/model.sdf || true
'

echo
echo "[CHECK] Gazebo Sensors plugin"
docker exec px4-gazebo bash -lc '
grep -n "gz-sim-sensors-system\|render_engine\|ogre2" /workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf || true
'

echo
echo "[CHECK] Gazebo models"
docker exec px4-gazebo bash -lc 'gz model --list || true'

echo
echo "[CHECK] Gazebo camera topic"
docker exec px4-gazebo bash -lc '
gz topic -l | grep camera || true
gz topic -i -t /vtol/camera || true
'
