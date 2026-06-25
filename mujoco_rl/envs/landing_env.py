import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class LandingConfig:
    dt: float = 0.05
    max_steps: int = 320

    # NED convention:
    # x: North, y: East, z: Down
    # target ground z = 0
    # vehicle above ground => z < 0
    init_xy_range_m: float = 5.0
    init_altitude_min_m: float = 2.0
    init_altitude_max_m: float = 6.0
    init_vel_range_mps: float = 0.25

    # Baseline PID velocity controller
    kp_xy: float = 0.45
    kd_xy: float = 0.18
    kp_z: float = 0.35
    kd_z: float = 0.12

    max_pid_xy_mps: float = 0.70    
    max_pid_z_mps: float = 0.35

    # RL residual velocity limit
    residual_xy_mps: float = 0.25
    residual_z_mps: float = 0.10

    # Total velocity command limit
    max_cmd_xy_mps: float = 0.90
    max_cmd_z_mps: float = 0.38

    # Episode-wise randomized PX4 velocity-loop response
    vel_response_alpha_min: float = 0.20
    vel_response_alpha_max: float = 0.45

    # Action delay in control steps
    action_delay_steps: int = 2

    # Disturbances
    wind_accel_xy_max_mps2: float = 0.08
    wind_accel_z_max_mps2: float = 0.02
    process_noise_vel_std_mps: float = 0.006

    # Target measurement noise
    target_noise_xy_std_m: float = 0.04
    target_noise_z_std_m: float = 0.03
    target_noise_max_m: float = 0.15

    target_dropout_prob: float = 0.03
    target_outlier_prob: float = 0.01
    target_outlier_xy_m: float = 1.5
    target_stale_prob: float = 0.05

    # Observation noise for vehicle velocity
    velocity_obs_noise_std_mps: float = 0.015

    # Safety limits
    max_xy_error_m: float = 10.0
    max_altitude_m: float = 9.0
    max_speed_mps: float = 2.5

    # Success condition
    success_xy_m: float = 0.20
    success_altitude_m: float = 0.05
    success_vxy_mps: float = 0.20
    success_vz_mps: float = 0.15

    # Ground safety
    below_ground_limit_m: float = 0.20

    # Reward coefficients
    w_xy: float = 1.00
    w_z: float = 0.35
    w_speed: float = 0.08
    w_action: float = 0.03
    w_saturation: float = 0.7
    w_near_ground_vz: float = 1.50

    dt_nominal: float = 0.05
    dt_jitter_std: float = 0.008
    dt_min: float = 0.03
    dt_max: float = 0.08

    obs_delay_steps_min: int = 1
    obs_delay_steps_max: int = 4

    action_delay_steps_min: int = 1
    action_delay_steps_max: int = 4


class LandingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, config: LandingConfig | None = None):
        super().__init__()
        self.cfg = config or LandingConfig()

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        high = np.array(
            [30.0, 30.0, 30.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            shape=(9,),
            dtype=np.float32,
        )

        self.target_true = np.zeros(3, dtype=np.float64)
        self.target_measured = np.zeros(3, dtype=np.float64)

        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)

        self.prev_action = np.zeros(3, dtype=np.float64)
        self.action_queue: list[np.ndarray] = []

        self.step_count = 0

        # randomized per episode
        self.vel_response_alpha = 0.35
        self.wind_accel = np.zeros(3, dtype=np.float64)

    def _sample_target_measurement(self):
        if self.np_random.random() < self.cfg.target_dropout_prob:
            self.target_valid = False
            return

        self.target_valid = True

        noise = np.array(
            [
                self.np_random.normal(0.0, self.cfg.target_noise_xy_std_m),
                self.np_random.normal(0.0, self.cfg.target_noise_xy_std_m),
                self.np_random.normal(0.0, self.cfg.target_noise_z_std_m),
            ],
            dtype=np.float64,
        )

        norm = float(np.linalg.norm(noise))
        if norm > self.cfg.target_noise_max_m:
            noise *= self.cfg.target_noise_max_m / (norm + 1e-9)

        self.target_measured = self.target_true + noise

    def _get_obs(self):
        self._sample_target_measurement()

        error_obs = self.target_measured - self.pos

        vel_obs = self.vel + self.np_random.normal(
            0.0,
            self.cfg.velocity_obs_noise_std_mps,
            size=3,
        )

        obs = np.concatenate([error_obs, vel_obs, self.prev_action]).astype(np.float32)
        return obs

    def _pid_velocity(self):
        # PID uses noisy measured target, as actual vision pipeline would.
        error = self.target_measured - self.pos

        vx = self.cfg.kp_xy * error[0] - self.cfg.kd_xy * self.vel[0]
        vy = self.cfg.kp_xy * error[1] - self.cfg.kd_xy * self.vel[1]
        vz = self.cfg.kp_z * error[2] - self.cfg.kd_z * self.vel[2]

        vxy = np.array([vx, vy], dtype=np.float64)
        norm_xy = float(np.linalg.norm(vxy))
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
        self.target_queue = []
        self.obs_delay_steps = self.np_random.integers(
            self.cfg.obs_delay_steps_min,
            self.cfg.obs_delay_steps_max + 1,
        )

        self.action_delay_steps = int(
            self.np_random.integers(
                self.cfg.action_delay_steps_min,
                self.cfg.action_delay_steps_max + 1,
            )
        )

        self.action_queue = [
            np.zeros(3, dtype=np.float64)
            for _ in range(self.action_delay_steps)
        ]

        x = self.np_random.uniform(-self.cfg.init_xy_range_m, self.cfg.init_xy_range_m)
        y = self.np_random.uniform(-self.cfg.init_xy_range_m, self.cfg.init_xy_range_m)

        altitude = self.np_random.uniform(
            self.cfg.init_altitude_min_m,
            self.cfg.init_altitude_max_m,
        )

        z = -altitude

        self.pos = np.array([x, y, z], dtype=np.float64)

        self.vel = self.np_random.uniform(
            -self.cfg.init_vel_range_mps,
            self.cfg.init_vel_range_mps,
            size=3,
        ).astype(np.float64)

        height = -self.pos[2]  # NED에서 z<0이면 고도 양수
        if height < self.cfg.ground_effect_height_m:
            ground_factor = 1.0 - self.cfg.ground_effect_strength * (
            1.0 - height / self.cfg.ground_effect_height_m
            )
        self.vel[2] *= ground_factor

        self.prev_action = np.zeros(3, dtype=np.float64)

        self.action_queue = [
            np.zeros(3, dtype=np.float64)
            for _ in range(max(0, self.cfg.action_delay_steps))
        ]

        self.vel_response_alpha = float(
            self.np_random.uniform(
                self.cfg.vel_response_alpha_min,
                self.cfg.vel_response_alpha_max,
            )
        )

        self.wind_accel = np.array(
            [
                self.np_random.uniform(
                    -self.cfg.wind_accel_xy_max_mps2,
                    self.cfg.wind_accel_xy_max_mps2,
                ),
                self.np_random.uniform(
                    -self.cfg.wind_accel_xy_max_mps2,
                    self.cfg.wind_accel_xy_max_mps2,
                ),
                self.np_random.uniform(
                    -self.cfg.wind_accel_z_max_mps2,
                    self.cfg.wind_accel_z_max_mps2,
                ),
            ],
            dtype=np.float64,
        )

        self._sample_target_measurement()
        self.target_queue.append(self.target_measured.copy())

        if len(self.target_queue) > self.obs_delay_steps:
            delayed_target = self.target_queue.pop(0)
        else:
            delayed_target = self.target_queue[0]

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        raw_action = np.asarray(action, dtype=np.float64)
        raw_action = np.clip(raw_action, -1.0, 1.0)
        self.prev_action = raw_action.copy()

        # Apply action delay.
        if self.cfg.action_delay_steps > 0:
            self.action_queue.append(raw_action.copy())
            applied_action = self.action_queue.pop(0)
        else:
            applied_action = raw_action.copy()

        v_pid = self._pid_velocity()
        v_residual = self._scale_action(applied_action)
        v_cmd = v_pid + v_residual

        # Total velocity command limit.
        vxy_norm = float(np.linalg.norm(v_cmd[:2]))
        if vxy_norm > self.cfg.max_cmd_xy_mps:
            v_cmd[:2] *= self.cfg.max_cmd_xy_mps / (vxy_norm + 1e-9)

        v_cmd[2] = float(np.clip(v_cmd[2], -self.cfg.max_cmd_z_mps, self.cfg.max_cmd_z_mps))

        # First-order velocity response with episode-randomized alpha.
        alpha = self.vel_response_alpha
        self.vel = self.vel + alpha * (v_cmd - self.vel)

        # Wind disturbance is modeled as low-frequency acceleration.
        dt = self.cfg.dt + self.np_random.normal(0.0, self.cfg.dt_jitter_std)
        dt = float(np.clip(dt, self.cfg.dt_min, self.cfg.dt_max))
        self.vel = self.vel + self.wind_accel * self.cfg.dt

        # Small process noise.
        self.vel += self.np_random.normal(0.0, self.cfg.process_noise_vel_std_mps, size=3)

        self.pos = self.pos + self.vel * self.cfg.dt

        true_error = self.target_true - self.pos

        xy_error = float(np.linalg.norm(true_error[:2]))
        z_error = abs(float(true_error[2]))
        vxy = float(np.linalg.norm(self.vel[:2]))
        vz_abs = abs(float(self.vel[2]))
        speed = float(np.linalg.norm(self.vel))

        reward = 0.0
        reward += -self.cfg.w_xy * xy_error
        reward += -self.cfg.w_z * z_error
        reward += -self.cfg.w_speed * speed
        reward += -self.cfg.w_action * float(np.linalg.norm(raw_action))

        # Penalize action saturation.
        saturation = np.maximum(np.abs(raw_action) - 0.80, 0.0)
        reward += -self.cfg.w_saturation * float(np.sum(saturation ** 2))

        # Encourage being near target.
        if xy_error < 0.5:
            reward += 0.5
        if z_error < 0.5:
            reward += 0.5

        # Near the ground, fast vertical motion is dangerous.
        if z_error < 0.8:
            reward += -self.cfg.w_near_ground_vz * vz_abs

        success = (
            xy_error < self.cfg.success_xy_m
            and z_error < self.cfg.success_altitude_m
            and vxy < self.cfg.success_vxy_mps
            and vz_abs < self.cfg.success_vz_mps
        )

        failure_reason = "none"

        if xy_error > self.cfg.max_xy_error_m:
            failure_reason = "xy_error_limit"
        elif abs(float(self.pos[2])) > self.cfg.max_altitude_m:
            failure_reason = "altitude_limit"
        elif speed > self.cfg.max_speed_mps:
            failure_reason = "speed_limit"
        elif self.pos[2] > self.cfg.below_ground_limit_m:
            failure_reason = "below_ground_limit"

        failed = failure_reason != "none"

        terminated = False
        if success:
            reward += 120.0
            terminated = True
        elif failed:
            reward -= 120.0
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
            "target_measured": self.target_measured.copy(),
            "vel_response_alpha": self.vel_response_alpha,
            "wind_accel": self.wind_accel.copy(),
            "applied_action": applied_action.copy(),
            "raw_action": raw_action.copy(),
            "failure_reason": failure_reason,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
