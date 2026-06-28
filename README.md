# PSI VTOL Team Drone Stack

PX4 SITL, Gazebo, ROS2 mission control, vision pipeline, and MuJoCo RL workspace for the PSI VTOL project.

Current working branch:

```bash
orange-fix
```

---

## 1. Repository Structure

```text
drone_stack/
├── compose.yaml
├── start.sh
├── px4/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── PX4-Autopilot/        # external dependency, not tracked by this repo
├── px4_assets/
│   └── worlds/               # custom Gazebo world files
├── models/                   # custom Gazebo models
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
```

Important rule:

PX4-Autopilot is treated as an external dependency.

Do not store project-specific files only inside:

```text
px4/PX4-Autopilot/
```

Project-specific Gazebo worlds, models, params, and patches should be stored in this repository.

Examples:

```text
px4_assets/worlds/
models/
px4/patches/
```

---

## 2. First-Time Setup on a New Desktop

Clone the repository:

```bash
git clone -b orange-fix https://github.com/Boo-seung-hwan/PSI-vtol-team.git drone_stack
cd drone_stack
```

Build Docker images:

```bash
docker compose build
```

Start the full stack:

```bash
./start.sh
```

Use `./start.sh` instead of `docker compose up`.

`start.sh` automatically sets `QGC_HOST_IP`, which is needed for QGroundControl communication.

---

## 3. Main Containers

Check container status:

```bash
docker compose ps -a
```

Expected containers:

```text
microxrce-agent   Up
px4-gazebo        Up
ros2-vision       Up
```

Container roles:

```text
microxrce-agent   PX4 ↔ ROS2 DDS bridge
px4-gazebo        PX4 SITL + Gazebo
ros2-vision       ROS2 mission, control, and vision nodes
```

---

## 4. ROS2 Workspace Build

Enter the ROS2 container:

```bash
docker exec -it ros2-vision bash
```

Build the ROS2 workspace:

```bash
cd /workspace/ros2/ws

source /opt/ros/humble/setup.bash

touch src/my_first_pkg/COLCON_IGNORE

rm -rf build install log

colcon build

source install/setup.bash
```

Do not use:

```bash
colcon build --symlink-install
```

In the current container environment, `--symlink-install` may fail with:

```text
error: option --editable not recognized
```

Use this instead:

```bash
colcon build
```

Check package detection:

```bash
ros2 pkg list | grep drone
```

Expected packages include:

```text
drone_bringup
drone_control
drone_interfaces
drone_vision
```

Check executable registration:

```bash
ros2 pkg executables drone_control
ros2 pkg executables drone_vision
```

Expected examples:

```text
drone_control mission_manager
drone_control setpoint_mux
drone_control generator_tracking
drone_control vertiport_tracking

drone_vision yolo_detector
drone_vision target_depth_estimator
drone_vision target_camera_to_ned
```

---

## 5. Launch Mission System

Inside the `ros2-vision` container:

```bash
cd /workspace/ros2/ws

source /opt/ros/humble/setup.bash
source install/setup.bash

export PYTHONOPTIMIZE=1

ros2 launch drone_control mission_system.launch.py
```

This launches:

```text
setpoint_mux
mission_manager
generator_tracking
vertiport_tracking
yolo_detector
target_depth_estimator
target_camera_to_ned
```

---

## 6. PX4 / Gazebo Custom Worlds

Custom Gazebo world files should be stored in:

```text
px4_assets/worlds/
```

Example:

```text
px4_assets/worlds/psi_vtol_world.sdf
```

These files should be copied into PX4-Autopilot inside the `px4-gazebo` container before Gazebo starts.

Expected destination inside the container:

```text
/workspace/px4/PX4-Autopilot/Tools/simulation/gz/worlds/
```

Do not manually add custom world files only to:

```text
px4/PX4-Autopilot/Tools/simulation/gz/worlds/
```

because `px4/PX4-Autopilot/` is not tracked by this repository.

If a custom world is used, check that `compose.yaml` sets the correct world name:

```yaml
PX4_GZ_WORLD=psi_vtol_world
```

If the default world is used:

```yaml
PX4_GZ_WORLD=default
```

---

## 7. PX4 + Gazebo Startup

Start from the host WSL terminal:

```bash
cd ~/drone_stack
./start.sh
```

Check logs:

```bash
docker logs px4-gazebo --tail=200
docker logs ros2-vision --tail=200
docker logs microxrce-agent --tail=200
```

If `px4-gazebo` exits immediately, inspect:

```bash
docker compose ps -a
docker logs px4-gazebo --tail=200
```

Common world-file error:

```text
FileNotFoundError:
.../Tools/simulation/gz/worlds/psi_vtol_world.sdf
```

This means the custom world file was not copied or is missing from:

```text
px4_assets/worlds/
```

Check:

```bash
ls px4_assets/worlds/
```

Expected example:

```text
psi_vtol_world.sdf
```

Then restart:

```bash
docker compose down --remove-orphans
./start.sh
```

---

## 8. QGroundControl

`start.sh` sets:

```text
QGC_HOST_IP
```

Do not run `docker compose up` directly unless `QGC_HOST_IP` is manually exported.

Recommended:

```bash
./start.sh
```

Manual alternative:

```bash
export QGC_HOST_IP=$(ip route | grep default | awk '{print $3}')
docker compose up
```

---

## 9. Mission System Architecture

High-level flow:

```text
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
```

Important topics:

```text
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
```

---

## 10. Vision Pipeline

Pipeline:

```text
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
```

Run YOLO detector manually:

```bash
ros2 run drone_vision yolo_detector --ros-args \
  -p model_path:=/workspace/ros2/ws/src/drone_vision/models/aruco_best.pt \
  -p confidence_threshold:=0.25 \
  -p target_class_id:=0
```


### Vision model weight files

YOLO / PyTorch model weight files are not tracked by this GitHub repository.

Do not commit files such as:

```text
*.pt
*.pth
*.onnx
*.engine
```

Required model files should be downloaded separately from:

```text
Microsoft Teams > 미션팀 자료
```

After downloading the files, place them in the following directory on the host WSL side:

```text
~/drone_stack/ros2/ws/src/drone_vision/models/
```

The same directory is visible inside the `ros2-vision` container as:

```text
/workspace/ros2/ws/src/drone_vision/models/
```

Required files:

```text
ros2/ws/src/drone_vision/models/best.pt
ros2/ws/src/drone_vision/models/aruco_best.pt
```

Optional file:

```text
ros2/ws/src/drone_vision/models/yolov8n.pt
```

`best.pt` is the default model path used by `yolo_detector`.

`aruco_best.pt` is used when running the detector manually with the ArUco / marker detection model.

Example check from the host WSL terminal:

```bash
cd ~/drone_stack

ls -lh ros2/ws/src/drone_vision/models/
```

Expected example:

```text
best.pt
aruco_best.pt
```

Example check from inside the `ros2-vision` container:

```bash
docker exec -it ros2-vision bash

ls -lh /workspace/ros2/ws/src/drone_vision/models/
```

If the `.pt` files are missing, the vision launch may fail when `yolo_detector` tries to load the model.

Do not use `git add .` to stage model weights accidentally.

Check whether model files are ignored by Git:

```bash
git status --ignored -s
git check-ignore -v ros2/ws/src/drone_vision/models/*.pt
```

---

## 11. Common Troubleshooting

### Package not found

Example:

```text
Package 'drone_vision' not found
Package 'drone_control' not found
```

Fix:

```bash
cd /workspace/ros2/ws

source /opt/ros/humble/setup.bash

touch src/my_first_pkg/COLCON_IGNORE

rm -rf build install log

colcon build

source install/setup.bash
```

Then check:

```bash
ros2 pkg list | grep drone
```

---

### `mission_system.launch.py` not found

Example:

```text
file 'mission_system.launch.py' was not found in the share directory of package 'drone_control'
```

Fix:

```bash
cd /workspace/ros2/ws

source /opt/ros/humble/setup.bash

rm -rf build install log

colcon build

source install/setup.bash

ls install/drone_control/share/drone_control/launch
```

Expected:

```text
mission_system.launch.py
```

---

### `error: option --editable not recognized`

If this happens during:

```bash
colcon build --symlink-install
```

use:

```bash
colcon build
```

instead.

---

### `my_first_pkg` build failure

`my_first_pkg` is not required for the current drone stack.

Ignore it:

```bash
touch /workspace/ros2/ws/src/my_first_pkg/COLCON_IGNORE
```

This file should be committed to Git.

---

### PX4 Gazebo world missing

If PX4 exits with:

```text
FileNotFoundError:
.../Tools/simulation/gz/worlds/psi_vtol_world.sdf
```

check:

```bash
ls px4_assets/worlds/
```

Expected example:

```text
psi_vtol_world.sdf
```

Then restart:

```bash
docker compose down --remove-orphans
./start.sh
```

---

### Docker namespace error

Example:

```text
OCI runtime create failed
namespace path: lstat /proc/.../ns/net: no such file or directory
```

This usually happens when `ros2-vision` tries to attach to the network namespace of `px4-gazebo`, but `px4-gazebo` has already exited.

Check:

```bash
docker compose ps -a
docker logs px4-gazebo --tail=200
```

Clean restart:

```bash
docker compose down --remove-orphans
./start.sh
```

If Docker or WSL is stuck, run this in Windows PowerShell:

```powershell
wsl --shutdown
```

Then restart Docker Desktop and run:

```bash
cd ~/drone_stack
./start.sh
```

---

## 12. Clean Restart

From host WSL:

```bash
cd ~/drone_stack

docker compose down --remove-orphans

./start.sh
```

Check status:

```bash
docker compose ps -a
```

---

## 13. Git Workflow

Check changes:

```bash
git status
```

Stage only intended files:

```bash
git add README.md
git add ros2/ws/src/my_first_pkg/COLCON_IGNORE
git add ros2/ws/src/drone_control/setup.py
git add ros2/ws/src/drone_vision/setup.py
git add px4/entrypoint.sh
git add compose.yaml
git add px4_assets/worlds/psi_vtol_world.sdf
```

Commit:

```bash
git commit -m "Fix reproducible PX4 and ROS2 setup"
```

Push:

```bash
git push origin orange-fix
```

Do not commit generated folders:

```text
ros2/ws/build/
ros2/ws/install/
ros2/ws/log/
px4/PX4-Autopilot/
```

Also avoid staging unrelated files with:

```bash
git add .
```

when only specific files should be committed.

---

## 14. MuJoCo RL

The MuJoCo RL workspace is stored in:

```text
mujoco_rl/
```

Typical workflow:

```bash
cd ~/drone_stack/mujoco_rl

docker run --rm -it \
  -p 6006:6006 \
  -v "$PWD":/workspace/mujoco_rl \
  mujoco-rl:landing
```

TensorBoard:

```bash
tensorboard --logdir ./runs --host 0.0.0.0 --port 6006
```

---

## 15. Current Notes

- Use `colcon build`, not `colcon build --symlink-install`.
- Ignore `my_first_pkg` using `COLCON_IGNORE`.
- Store custom PX4 worlds in `px4_assets/worlds/`.
- Use `./start.sh` instead of `docker compose up`.
- Treat `px4/PX4-Autopilot/` as an external dependency.