# PSI VTOL Team Drone Stack

PX4 SITL, Gazebo, ROS2 mission control, vision pipeline, and MuJoCo RL workspace for the PSI VTOL project.

This repository uses the `orange-fix` branch as the current working branch.

---

## 1. Repository Structure

```text
drone_stack/
├── compose.yaml
├── start.sh
├── px4/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── PX4-Autopilot/        # external dependency, not tracked
├── px4_assets/
│   └── worlds/               # custom Gazebo world files
├── models/                   # custom Gazebo models
├── ros2/
│   └── ws/
│       └── src/
│           ├── drone_control/
│           ├── drone_vision/
│           ├── drone_interfaces/
│           ├── drone_bringup/
│           ├── px4_msgs/
│           └── my_first_pkg/
└── mujoco_rl/
