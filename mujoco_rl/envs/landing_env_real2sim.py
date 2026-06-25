import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class LandingConfig:

    # ------------------------------------------------------------------
    # Simulation timing
    # ------------------------------------------------------------------
    dt: float = 0.05                 # nominal control period [s], 20 Hz
    max_steps: int = 320             # 16 s episode at 20 Hz
    dt_jitter_std: float = 0.008     # loop timing jitter [s]
    dt_min: float = 0.03
    dt_max: float = 0.08

    # ------------------------------------------------------------------
    # Initial condition randomization
    # ------------------------------------------------------------------
    init_xy_range_m: float = 5.0
    init_altitude_min_m: float = 2.0
    init_altitude_max_m: float = 6.0
    init_vel_range_mps: float = 0.25

    # ------------------------------------------------------------------
    # Baseline PID velocity controller
    # This approximates the existing precision landing outer-loop controller.
    # ------------------------------------------------------------------
    kp_xy: float = 0.45
    kd_xy: float = 0.18
    kp_z: float = 0.35
    kd_z: float = 0.12
    max_pid_xy_mps: float = 0.70
    max_pid_z_mps: float = 0.35

    # When target_valid is false, the safe behavior is to command hover/hold.
    freeze_control_when_target_invalid: bool = True
    apply_residual_when_target_invalid: bool = False

    # ------------------------------------------------------------------
    # RL residual velocity limits
    # action in [-1, 1]^3 is scaled to this residual velocity command.
    # ------------------------------------------------------------------
    residual_xy_mps: float = 0.25
    residual_z_mps: float = 0.10

    # Final velocity command saturation
    max_cmd_xy_mps: float = 0.90
    max_cmd_z_mps: float = 0.38

    # ------------------------------------------------------------------
    # PX4 velocity-loop response model
    # The value is randomized per episode and per axis.
    # alpha_eff is adjusted when dt jitter changes the actual step duration.
    # ------------------------------------------------------------------
    vel_response_alpha_min: float = 0.20
    vel_response_alpha_max: float = 0.45

    # ------------------------------------------------------------------
    # Action and observation delay models
    # action_delay: onboard inference / command application delay
    # obs_delay: camera + YOLO + depth + camera-to-NED pipeline delay
    # ------------------------------------------------------------------
    action_delay_steps_min: int = 1
    action_delay_steps_max: int = 4
    obs_delay_steps_min: int = 1
    obs_delay_steps_max: int = 4

    # ------------------------------------------------------------------
    # Disturbance models
    # ------------------------------------------------------------------
    wind_accel_xy_max_mps2: float = 0.08
    wind_accel_z_max_mps2: float = 0.02
    process_noise_vel_std_mps: float = 0.006

    # ------------------------------------------------------------------
    # Perception / target measurement model
    # target_true is [0, 0, 0]. target_measured is an absolute NED target.
    # ------------------------------------------------------------------
    target_noise_xy_std_m: float = 0.04
    target_noise_z_std_m: float = 0.03
    target_noise_max_m: float = 0.15

    target_dropout_prob: float = 0.03     # detection invalid for this frame
    target_outlier_prob: float = 0.01     # large false measurement jump
    target_outlier_xy_m: float = 1.5
    target_stale_prob: float = 0.05       # old target value is repeated

    # Observation noise for vehicle velocity
    velocity_obs_noise_std_mps: float = 0.015

    # ------------------------------------------------------------------
    # Safety limits
    # ------------------------------------------------------------------
    max_xy_error_m: float = 10.0
    max_altitude_m: float = 9.0
    max_speed_mps: float = 2.5
    below_ground_limit_m: float = 0.20

    # ------------------------------------------------------------------
    # Success condition
    # ------------------------------------------------------------------
    success_xy_m: float = 0.20
    success_altitude_m: float = 0.05
    success_vxy_mps: float = 0.20
    success_vz_mps: float = 0.15

    # ------------------------------------------------------------------
    # Reward coefficients
    # ------------------------------------------------------------------
    w_xy: float = 1.00
    w_z: float = 0.35
    w_speed: float = 0.08
    w_action: float = 0.03
    w_saturation: float = 0.70
    w_near_ground_vz: float = 1.50


class LandingEnv(gym.Env):
    

    """Observation, shape=(10,):
        [dx_obs, dy_obs, dz_obs,
         vx_obs, vy_obs, vz_obs,
         prev_action_x, prev_action_y, prev_action_z,
         target_valid]

    Action, shape=(3,):
        normalized residual velocity command in [-1, 1]^3.

    Control law inside the environment:
        v_cmd = v_pid + v_residual

    The reward and termination use the true target [0, 0, 0], while the policy
    and PID use delayed/noisy/stale/dropout target measurements.
    """

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

        # Last element is target_valid, represented as 0.0 or 1.0.
        high = np.array(
            [30.0, 30.0, 30.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            shape=(10,),
            dtype=np.float32,
        )

        self.target_true = np.zeros(3, dtype=np.float64)
        self.target_measured = np.zeros(3, dtype=np.float64)
        self.last_raw_target = np.zeros(3, dtype=np.float64)

        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)

        self.prev_action = np.zeros(3, dtype=np.float64)
        self.action_queue: list[np.ndarray] = []
        self.target_queue: list[tuple[np.ndarray, bool, str]] = []

        self.step_count = 0

        # Per-episode randomized values
        self.vel_response_alpha = np.ones(3, dtype=np.float64) * 0.35
        self.wind_accel = np.zeros(3, dtype=np.float64)
        self.action_delay_steps = 2
        self.obs_delay_steps = 2

        # Current delayed target used by both observation and PID.
        self.obs_target = np.zeros(3, dtype=np.float64)
        self.obs_target_valid = True
        self.obs_target_mode = "init"

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _sample_dt(self) -> float:
        dt = self.cfg.dt + self.np_random.normal(0.0, self.cfg.dt_jitter_std)
        return float(np.clip(dt, self.cfg.dt_min, self.cfg.dt_max))

    def _sample_raw_target_measurement(self) -> tuple[np.ndarray, bool, str]:
        """Sample a raw absolute NED target measurement.

        Returns:
            target: absolute NED target measurement
            valid: target valid flag
            mode:  string label for debugging/logging
        """
        # Dropout: detection failed. Keep the previous target value but mark invalid.
        if self.np_random.random() < self.cfg.target_dropout_prob:
            return self.last_raw_target.copy(), False, "dropout"

        # Stale target: old measurement is repeated but still marked valid.
        if self.np_random.random() < self.cfg.target_stale_prob:
            return self.last_raw_target.copy(), True, "stale"

        noise = np.array(
            [
                self.np_random.normal(0.0, self.cfg.target_noise_xy_std_m),
                self.np_random.normal(0.0, self.cfg.target_noise_xy_std_m),
                self.np_random.normal(0.0, self.cfg.target_noise_z_std_m),
            ],
            dtype=np.float64,
        )

        noise_norm = float(np.linalg.norm(noise))
        if noise_norm > self.cfg.target_noise_max_m:
            noise *= self.cfg.target_noise_max_m / (noise_norm + 1e-9)

        target = self.target_true + noise
        mode = "normal"

        if self.np_random.random() < self.cfg.target_outlier_prob:
            target[:2] += self.np_random.uniform(
                -self.cfg.target_outlier_xy_m,
                self.cfg.target_outlier_xy_m,
                size=2,
            )
            mode = "outlier"

        self.last_raw_target = target.copy()
        return target, True, mode

    def _update_target_pipeline(self) -> None:
        """Apply perception measurement model and observation delay queue."""
        raw_target, valid, mode = self._sample_raw_target_measurement()

        self.target_queue.append((raw_target.copy(), bool(valid), mode))

        max_len = max(1, int(self.obs_delay_steps) + 1)
        while len(self.target_queue) > max_len:
            self.target_queue.pop(0)

        delayed_target, delayed_valid, delayed_mode = self.target_queue[0]

        self.target_measured = delayed_target.copy()
        self.obs_target = delayed_target.copy()
        self.obs_target_valid = bool(delayed_valid)
        self.obs_target_mode = delayed_mode

    def _get_obs(self) -> np.ndarray:
        error_obs = self.obs_target - self.pos

        vel_obs = self.vel + self.np_random.normal(
            0.0,
            self.cfg.velocity_obs_noise_std_mps,
            size=3,
        )

        valid = 1.0 if self.obs_target_valid else 0.0

        obs = np.concatenate(
            [
                error_obs,
                vel_obs,
                self.prev_action,
                np.array([valid], dtype=np.float64),
            ]
        ).astype(np.float32)

        return obs

    # ------------------------------------------------------------------
    # Controller helpers
    # ------------------------------------------------------------------
    def _pid_velocity(self, target: np.ndarray, valid: bool) -> np.ndarray:
        if (not valid) and self.cfg.freeze_control_when_target_invalid:
            return np.zeros(3, dtype=np.float64)

        error = target - self.pos

        vx = self.cfg.kp_xy * error[0] - self.cfg.kd_xy * self.vel[0]
        vy = self.cfg.kp_xy * error[1] - self.cfg.kd_xy * self.vel[1]
        vz = self.cfg.kp_z * error[2] - self.cfg.kd_z * self.vel[2]

        vxy = np.array([vx, vy], dtype=np.float64)
        norm_xy = float(np.linalg.norm(vxy))
        if norm_xy > self.cfg.max_pid_xy_mps:
            vxy *= self.cfg.max_pid_xy_mps / (norm_xy + 1e-9)

        vz = float(np.clip(vz, -self.cfg.max_pid_z_mps, self.cfg.max_pid_z_mps))

        return np.array([vxy[0], vxy[1], vz], dtype=np.float64)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        return np.array(
            [
                action[0] * self.cfg.residual_xy_mps,
                action[1] * self.cfg.residual_xy_mps,
                action[2] * self.cfg.residual_z_mps,
            ],
            dtype=np.float64,
        )

    def _limit_velocity_command(self, v_cmd: np.ndarray) -> np.ndarray:
        v_cmd = np.asarray(v_cmd, dtype=np.float64).copy()

        vxy_norm = float(np.linalg.norm(v_cmd[:2]))
        if vxy_norm > self.cfg.max_cmd_xy_mps:
            v_cmd[:2] *= self.cfg.max_cmd_xy_mps / (vxy_norm + 1e-9)

        v_cmd[2] = float(np.clip(v_cmd[2], -self.cfg.max_cmd_z_mps, self.cfg.max_cmd_z_mps))
        return v_cmd

    def _apply_action_delay(self, raw_action: np.ndarray) -> np.ndarray:
        if self.action_delay_steps <= 0:
            return raw_action.copy()

        self.action_queue.append(raw_action.copy())
        return self.action_queue.pop(0)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

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

        self.prev_action = np.zeros(3, dtype=np.float64)

        self.action_delay_steps = int(
            self.np_random.integers(
                self.cfg.action_delay_steps_min,
                self.cfg.action_delay_steps_max + 1,
            )
        )
        self.obs_delay_steps = int(
            self.np_random.integers(
                self.cfg.obs_delay_steps_min,
                self.cfg.obs_delay_steps_max + 1,
            )
        )

        self.action_queue = [
            np.zeros(3, dtype=np.float64)
            for _ in range(max(0, self.action_delay_steps))
        ]
        self.target_queue = []

        self.vel_response_alpha = self.np_random.uniform(
            self.cfg.vel_response_alpha_min,
            self.cfg.vel_response_alpha_max,
            size=3,
        ).astype(np.float64)

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

        # Initialize target measurement pipeline with one clean-ish measurement.
        self.last_raw_target = self.target_true.copy()
        self.obs_target = self.target_true.copy()
        self.obs_target_valid = True
        self.obs_target_mode = "reset"
        self.target_measured = self.obs_target.copy()

        # Fill the perception delay queue so that the first observations are stable.
        for _ in range(max(1, self.obs_delay_steps + 1)):
            self._update_target_pipeline()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        raw_action = np.asarray(action, dtype=np.float64)
        raw_action = np.clip(raw_action, -1.0, 1.0)
        self.prev_action = raw_action.copy()

        dt = self._sample_dt()

        # The action was selected from the current observation. Therefore both
        # PID and residual RL use the currently stored delayed target.
        control_target = self.obs_target.copy()
        control_target_valid = bool(self.obs_target_valid)

        applied_action = self._apply_action_delay(raw_action)

        v_pid = self._pid_velocity(control_target, control_target_valid)

        if control_target_valid or self.cfg.apply_residual_when_target_invalid:
            v_residual = self._scale_action(applied_action)
        else:
            v_residual = np.zeros(3, dtype=np.float64)

        v_cmd = self._limit_velocity_command(v_pid + v_residual)

        # Convert nominal alpha to an effective alpha under dt jitter.
        # If dt == cfg.dt, alpha_eff == vel_response_alpha.
        alpha_eff = 1.0 - np.power(
            1.0 - np.clip(self.vel_response_alpha, 1e-4, 0.9999),
            dt / max(self.cfg.dt, 1e-6),
        )
        alpha_eff = np.clip(alpha_eff, 0.0, 1.0)

        self.vel = self.vel + alpha_eff * (v_cmd - self.vel)
        self.vel = self.vel + self.wind_accel * dt

        # Process noise. Keep scale roughly consistent with the nominal step.
        noise_scale = math.sqrt(max(dt, 1e-6) / max(self.cfg.dt, 1e-6))
        self.vel += self.np_random.normal(
            0.0,
            self.cfg.process_noise_vel_std_mps * noise_scale,
            size=3,
        )

        self.pos = self.pos + self.vel * dt

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

        saturation = np.maximum(np.abs(raw_action) - 0.80, 0.0)
        reward += -self.cfg.w_saturation * float(np.sum(saturation ** 2))

        if xy_error < 0.5:
            reward += 0.5
        if z_error < 0.5:
            reward += 0.5

        # NED: z velocity > 0 means descending. Near the ground, any fast
        # vertical motion is treated as dangerous.
        if z_error < 0.8:
            reward += -self.cfg.w_near_ground_vz * vz_abs

        success = (
            xy_error < self.cfg.success_xy_m
            and z_error < self.cfg.success_altitude_m
            and vxy < self.cfg.success_vxy_mps
            and vz_abs < self.cfg.success_vz_mps
        )

        failure_reason = "none"
        altitude = -float(self.pos[2])

        if xy_error > self.cfg.max_xy_error_m:
            failure_reason = "xy_error_limit"
        elif altitude > self.cfg.max_altitude_m:
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
            "failure_reason": failure_reason,
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "target_true": self.target_true.copy(),
            "target_measured": self.target_measured.copy(),
            "target_valid": control_target_valid,
            "target_mode": self.obs_target_mode,
            "dt": dt,
            "vel_response_alpha": self.vel_response_alpha.copy(),
            "alpha_eff": alpha_eff.copy(),
            "wind_accel": self.wind_accel.copy(),
            "action_delay_steps": self.action_delay_steps,
            "obs_delay_steps": self.obs_delay_steps,
            "applied_action": applied_action.copy(),
            "raw_action": raw_action.copy(),
            "v_pid": v_pid.copy(),
            "v_residual": v_residual.copy(),
            "v_cmd": v_cmd.copy(),
        }

        # Prepare next observation after the state update.
        self._update_target_pipeline()
        next_obs = self._get_obs()

        return next_obs, float(reward), terminated, truncated, info
