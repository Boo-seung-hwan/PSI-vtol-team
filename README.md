# PSI VTOL Team Drone Stack

PX4 SITL, Gazebo, ROS2, vision, mission control, and MuJoCo RL experiments for the PSI VTOL project.

This repository is intended to be reproducible on a new desktop after cloning the `orange-fix` branch.

---

## 1. Repository Structure

```text
drone_stack/
├── compose.yaml
├── start.sh
├── px4/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── PX4-Autopilot/              # external dependency, not tracked by this repo
├── px4_assets/
│   └── worlds/
│       └── psi_vtol_world.sdf      # custom Gazebo world files
├── models/
│   ├── mono_cam/
│   └── standard_vtol/
├── ros2/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── ws/
│       └── src/
│           ├── drone_control/
│           ├── drone_vision/
│           ├── drone_interfaces/
│           ├── drone_bringup/
│           ├── px4_msgs/
│           └── my_first_pkg/
└── mujoco_rl/

Important rule:

PX4-Autopilot is treated as an external dependency.
Do not store project-specific files only inside px4/PX4-Autopilot.
Project-specific Gazebo worlds, models, params, or patches should be stored in this repository.
2. First-Time Setup on a New Desktop

Clone the repository:

git clone -b orange-fix https://github.com/Boo-seung-hwan/PSI-vtol-team.git drone_stack
cd drone_stack

Build Docker images:

docker compose build

Start the full stack:

./start.sh

start.sh automatically sets QGC_HOST_IP.
Use ./start.sh instead of directly running docker compose up.

3. Containers

The main containers are:

microxrce-agent   PX4 ↔ ROS2 DDS bridge
px4-gazebo        PX4 SITL + Gazebo
ros2-vision       ROS2 mission, control, and vision nodes

Check container status:

docker compose ps -a

Expected status:

microxrce-agent   Up
px4-gazebo        Up
ros2-vision       Up
4. ROS2 Build

Enter the ROS2 container:

docker exec -it ros2-vision bash

Build the ROS2 workspace:

cd /workspace/ros2/ws

source /opt/ros/humble/setup.bash

touch src/my_first_pkg/COLCON_IGNORE

rm -rf build install log

colcon build

source install/setup.bash

Do not use --symlink-install in the current container environment.
In this project, the following command may fail:

colcon build --symlink-install

Use this instead:

colcon build

Check that the packages are detected:

ros2 pkg list | grep drone

Expected packages include:

drone_bringup
drone_control
drone_interfaces
drone_vision

Check executable registration:

ros2 pkg executables drone_control
ros2 pkg executables drone_vision

Expected examples:

drone_control mission_manager
drone_control setpoint_mux
drone_control generator_tracking
drone_control vertiport_tracking

drone_vision yolo_detector
drone_vision target_depth_estimator
drone_vision target_camera_to_ned
5. Launch Mission System

Inside the ros2-vision container:

cd /workspace/ros2/ws

source /opt/ros/humble/setup.bash
source install/setup.bash

export PYTHONOPTIMIZE=1

ros2 launch drone_control mission_system.launch.py

This launches:

setpoint_mux
mission_manager
generator_tracking
vertiport_tracking
yolo_detector
target_depth_estimator
target_camera_to_ned
6. PX4 / Gazebo Custom World Management

Custom Gazebo world files should be stored in:

px4_assets/worlds/

For example:

px4_assets/worlds/psi_vtol_world.sdf

These files are copied into PX4-Autopilot inside the px4-gazebo container before Gazebo starts.

Expected destination inside the container:

/workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds/

Do not manually add custom world files only to:

px4/PX4-Autopilot/Tools/simulation/gz/worlds/

because px4/PX4-Autopilot/ is an external dependency and is not tracked by this repository.

7. Starting PX4 + Gazebo

Start from the host WSL terminal:

cd ~/drone_stack
./start.sh

Check logs:

docker logs px4-gazebo --tail=200
docker logs ros2-vision --tail=200
docker logs microxrce-agent --tail=200

If px4-gazebo exits immediately, inspect:

docker compose ps -a
docker logs px4-gazebo --tail=200

A common world-file error looks like:

FileNotFoundError:
.../Tools/simulation/gz/worlds/psi_vtol_world.sdf

This means the custom world file was not copied or is missing from px4_assets/worlds/.

8. QGroundControl

start.sh sets:

QGC_HOST_IP

Do not run docker compose up directly unless QGC_HOST_IP is manually exported.

If needed:

export QGC_HOST_IP=$(ip route | grep default | awk '{print $3}')
docker compose up

Recommended:

./start.sh
9. Mission Control Architecture

The mission system uses the following high-level flow:

mission_manager
      ↓
/mission/state
      ↓
generator_tracking / vertiport_tracking
      ↓
/generator_tracking/trajectory_setpoint
/vertiport_tracking/trajectory_setpoint
      ↓
setpoint_mux
      ↓
/fmu/in/trajectory_setpoint

Important topics:

/mission/state
/mission/status
/setpoint_mux/source
/setpoint_mux/status

/generator_tracking/trajectory_setpoint
/generator_tracking/status

/vertiport_tracking/trajectory_setpoint
/vertiport_tracking/status

/fmu/in/trajectory_setpoint
/fmu/in/offboard_control_mode

/fmu/out/vehicle_local_position
/fmu/out/vehicle_status

/vision/yolo_bbox
/vision/target_camera_xyz
/target/ned
10. Vision Pipeline

The vision pipeline is:

/vtol/camera
      ↓
yolo_detector
      ↓
/vision/yolo_bbox
      ↓
target_depth_estimator
      ↓
/vision/target_camera_xyz
      ↓
target_camera_to_ned
      ↓
/target/ned

Run YOLO detector manually:

ros2 run drone_vision yolo_detector --ros-args \
  -p model_path:=/workspace/ros2/ws/src/drone_vision/models/aruco_best.pt \
  -p confidence_threshold:=0.25 \
  -p target_class_id:=0
11. Common Troubleshooting
Package not found

Example:

Package 'drone_vision' not found
Package 'drone_control' not found

Fix:

cd /workspace/ros2/ws

source /opt/ros/humble/setup.bash

touch src/my_first_pkg/COLCON_IGNORE

rm -rf build install log

colcon build

source install/setup.bash

Then check:

ros2 pkg list | grep drone
mission_system.launch.py not found

Example:

file 'mission_system.launch.py' was not found in the share directory of package 'drone_control'

Fix:

cd /workspace/ros2/ws

source /opt/ros/humble/setup.bash

rm -rf build install log

colcon build

source install/setup.bash

ls install/drone_control/share/drone_control/launch

Expected:

mission_system.launch.py
error: option --editable not recognized

If this happens during:

colcon build --symlink-install

use:

colcon build

instead.

my_first_pkg build failure

my_first_pkg is not required for the current drone stack.

Ignore it:

touch /workspace/ros2/ws/src/my_first_pkg/COLCON_IGNORE

This file should be committed to Git.

PX4 Gazebo world missing

If PX4 exits with:

FileNotFoundError:
.../Tools/simulation/gz/worlds/psi_vtol_world.sdf

Check that the file exists in the repository:

ls px4_assets/worlds/

Expected:

psi_vtol_world.sdf

Then restart:

docker compose down --remove-orphans
./start.sh
12. Clean Restart

From host WSL:

cd ~/drone_stack

docker compose down --remove-orphans

./start.sh

If Docker or WSL seems stuck:

wsl --shutdown

Then restart Docker Desktop and run:

cd ~/drone_stack
./start.sh
13. Git Workflow

Check changes:

git status

Stage only intended files:

git add README.md
git add ros2/ws/src/my_first_pkg/COLCON_IGNORE
git add ros2/ws/src/drone_control/setup.py
git add ros2/ws/src/drone_vision/setup.py
git add px4/entrypoint.sh
git add compose.yaml
git add px4_assets/worlds/psi_vtol_world.sdf

Commit:

git commit -m "Fix reproducible PX4 and ROS2 setup"

Push:

git push origin orange-fix

Do not commit generated folders:

ros2/ws/build/
ros2/ws/install/
ros2/ws/log/
px4/PX4-Autopilot/
14. MuJoCo RL

The MuJoCo RL environment is stored in:

mujoco_rl/

Typical workflow:

cd ~/drone_stack/mujoco_rl
docker run --rm -it \
  -p 6006:6006 \
  -v "$PWD":/workspace/mujoco_rl \
  mujoco-rl:landing

TensorBoard:

tensorboard --logdir ./runs --host 0.0.0.0 --port 6006
15. Current Known Notes
Use colcon build, not colcon build --symlink-install, in the current ROS2 container.
my_first_pkg should be ignored using COLCON_IGNORE.
PX4 custom worlds should be stored in px4_assets/worlds/, not only inside px4/PX4-Autopilot/.
Use ./start.sh instead of docker compose up to ensure QGC_HOST_IP is set.

---

그리고 지금 `git status`를 보면 README 수정은 아직 없고, MuJoCo 파일 세 개만 수정된 상태입니다.

```text
modified: mujoco_rl/envs/landing_env.py
modified: mujoco_rl/envs/landing_env_real2sim.py
modified: mujoco_rl/eval_compare_v2.py
