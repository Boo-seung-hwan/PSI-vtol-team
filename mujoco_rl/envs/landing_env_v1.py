import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class LandingConfig:
    dt: float = 0.1
    max_steps: int = 120

    # PX4 NED convention:
    # x: North, y: East, z: Down
    # ground/target z = 0
    # vehicle above ground => z is negative
    init_xy_range_m: float = 3.0
    init_altitude_min_m: float = 2.0
    init_altitude_max_m: float = 5.0
    init_vel_range_mps: float = 0.15

    # Baseline PID velocity controller
    kp_xy: float = 0.45
    kd_xy: float = 0.18
    kp_z: float = 0.35
    kd_z: float = 0.12

    max_pid_xy_mps: float = 0.7
    max_pid_z_mps: float = 0.35

    # RL residual velocity limit
    residual_xy_mps: float = 0.25
    residual_z_mps: float = 0.15

    # Velocity response model
    # Larger alpha means vehicle velocity follows command faster.
    vel_response_alpha: float = 0.35

    # Safety limits
    max_xy_error_m: float = 8.0
    max_altitude_m: float = 8.0
    max_speed_mps: float = 2.5

    # Success condition
    success_xy_m: float = 0.20
    success_altitude_m: float = 0.15
    success_vxy_mps: float = 0.25
    success_vz_mps: float = 0.25


class LandingEnv(gym.Env):
    """
    Simple state-based precision landing environment.

    This environment intentionally matches the future PX4/Gazebo interface:
      observation = [dx, dy, dz, vx, vy, vz, prev_ax, prev_ay, prev_az]
      action      = RL residual velocity command normalized to [-1, 1]

    NED convention:
      target = [0, 0, 0]
      vehicle above ground has z < 0
      positive vz means descending.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: LandingConfig | None = None):
        super().__init__()
        self.cfg = config or LandingConfig()

        # normalized residual action: [-1, 1]^3
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # [dx, dy, dz, vx, vy, vz, prev_action(3)]
        high = np.array(
            [20.0, 20.0, 20.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            shape=(9,),
            dtype=np.float32,
        )

        self.target = np.zeros(3, dtype=np.float64)
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.prev_action = np.zeros(3, dtype=np.float64)
        self.step_count = 0

    def _get_obs(self):
        error = self.target - self.pos
        obs = np.concatenate([error, self.vel, self.prev_action]).astype(np.float32)
        return obs

    def _pid_velocity(self):
        error = self.target - self.pos

        vx = self.cfg.kp_xy * error[0] - self.cfg.kd_xy * self.vel[0]
        vy = self.cfg.kp_xy * error[1] - self.cfg.kd_xy * self.vel[1]
        vz = self.cfg.kp_z * error[2] - self.cfg.kd_z * self.vel[2]

        vxy = np.array([vx, vy], dtype=np.float64)
        norm_xy = np.linalg.norm(vxy)
        if norm_xy > self.cfg.max_pid_xy_mps:
            vxy *= self.cfg.max_pid_xy_mps / (norm_xy + 1e-9)

        vz = float(np.clip(vz, -self.cfg.max_pid_z_mps, self.cfg.max_pid_z_mps))

        return np.array([vxy[0], vxy[1], vz], dtype=np.float64)

    def _scale_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        residual = np.array(
            [
                action[0] * self.cfg.residual_xy_mps,
                action[1] * self.cfg.residual_xy_mps,
                action[2] * self.cfg.residual_z_mps,
            ],
            dtype=np.float64,
        )
        return residual

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        x = self.np_random.uniform(-self.cfg.init_xy_range_m, self.cfg.init_xy_range_m)
        y = self.np_random.uniform(-self.cfg.init_xy_range_m, self.cfg.init_xy_range_m)

        altitude = self.np_random.uniform(
            self.cfg.init_altitude_min_m,
            self.cfg.init_altitude_max_m,
        )

        # NED: above ground means z is negative.
        z = -altitude

        self.pos = np.array([x, y, z], dtype=np.float64)

        self.vel = self.np_random.uniform(
            -self.cfg.init_vel_range_mps,
            self.cfg.init_vel_range_mps,
            size=3,
        ).astype(np.float64)

        self.prev_action = np.zeros(3, dtype=np.float64)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        self.prev_action = action.copy()

        v_pid = self._pid_velocity()
        v_residual = self._scale_action(action)
        v_cmd = v_pid + v_residual

        # Total velocity command limit
        vxy_norm = np.linalg.norm(v_cmd[:2])
        if vxy_norm > 0.9:
            v_cmd[:2] *= 0.9 / (vxy_norm + 1e-9)
        v_cmd[2] = float(np.clip(v_cmd[2], -0.45, 0.45))

        # First-order velocity response, approximating PX4 velocity loop lag.
        alpha = self.cfg.vel_response_alpha
        self.vel = self.vel + alpha * (v_cmd - self.vel)

        # Light process noise for robustness.
        self.vel += self.np_random.normal(0.0, 0.005, size=3)

        self.pos = self.pos + self.vel * self.cfg.dt

        error = self.target - self.pos
        xy_error = float(np.linalg.norm(error[:2]))
        z_error = abs(float(error[2]))
        vxy = float(np.linalg.norm(self.vel[:2]))
        vz_abs = abs(float(self.vel[2]))
        speed = float(np.linalg.norm(self.vel))

        # Reward shaping
        reward = 0.0
        reward += -1.00 * xy_error
        reward += -0.35 * z_error
        reward += -0.08 * speed
        reward += -0.03 * float(np.linalg.norm(action))

        # Encourage progress toward target compared to previous step approximately
        # by giving small reward near the landing zone.
        if xy_error < 0.5:
            reward += 0.5
        if z_error < 0.5:
            reward += 0.5

        success = (
            xy_error < self.cfg.success_xy_m
            and z_error < self.cfg.success_altitude_m
            and vxy < self.cfg.success_vxy_mps
            and vz_abs < self.cfg.success_vz_mps
        )

        failed = (
            xy_error > self.cfg.max_xy_error_m
            or abs(float(self.pos[2])) > self.cfg.max_altitude_m
            or speed > self.cfg.max_speed_mps
            or self.pos[2] > 0.30  # below ground too much in NED
        )

        terminated = False
        if success:
            reward += 100.0
            terminated = True
        elif failed:
            reward -= 100.0
            terminated = True

        truncated = self.step_count >= self.cfg.max_steps

        info = {
            "xy_error": xy_error,
            "z_error": z_error,
            "vxy": vxy,
            "vz_abs": vz_abs,
            "success": success,
            "failed": failed,
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
        }

        return self._get_obs(), float(reward), terminated, truncated, info
