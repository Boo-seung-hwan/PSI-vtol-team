import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from landing_rl.controllers import BaselineController
from landing_rl.disturbances import DisturbanceModel
from landing_rl.dynamics import ProcessNoiseSampler, ResponseAlphas
from landing_rl.envs.action_latency import ActionLatency
from landing_rl.envs.initial_state import InitialStateSampler
from landing_rl.envs.loop_timing import LoopTiming
from landing_rl.perception import (
    ObsLatency,
    ObservationNoiseSampler,
    TargetMeasurementModel,
)


def wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class LandingConfig:

    # ------------------------------------------------------------------
    # Simulation timing
    # ------------------------------------------------------------------
    dt: float = 0.05                 # nominal control period [s], 20 Hz
    max_steps: int = 420             # 21 s episode at 20 Hz
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


    # ------------------------------------------------------------------
    # Landing descent helper
    # ------------------------------------------------------------------
    descent_xy_gate_m: float = 0.45
    descent_start_height_m: float = 1.20
    landing_descent_bias_mps: float = 0.10

    # When target_valid is false, the safe behavior is to command hover/hold.
    freeze_control_when_target_invalid: bool = True
    apply_residual_when_target_invalid: bool = False

    # ------------------------------------------------------------------
    # RL residual velocity limits
    # action in [-1, 1]^3 is scaled to this residual velocity command.
    # ------------------------------------------------------------------
    residual_xy_mps: float = 0.25
    residual_z_mps: float = 0.05

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
    wind_accel_xy_max_mps2: float = 0.00             #0.08
    wind_accel_z_max_mps2: float = 0.00              #0.02
    process_noise_vel_std_mps: float = 0.006

    # ------------------------------------------------------------------
    # Perception / target measurement model
    # target_true is [0, 0, 0]. target_measured is an absolute NED target.
    # ------------------------------------------------------------------
    target_noise_xy_std_m: float = 0.00              #0.04
    target_noise_z_std_m: float = 0.00               #0.03
    target_noise_max_m: float = 0.15

    target_dropout_prob: float = 0.03     # detection invalid for this frame
    target_outlier_prob: float = 0.01     # large false measurement jump
    target_outlier_xy_m: float = 1.5
    target_stale_prob: float = 0.05       # old target value is repeated

    # Observation noise for vehicle velocity / acceleration
    velocity_obs_noise_std_mps: float = 0.015
    acceleration_obs_noise_std_mps2: float = 0.05
    max_acceleration_obs_mps2: float = 4.0

    # ------------------------------------------------------------------
    # Attitude / heading model
    # This is a lightweight real2sim approximation: velocity-loop commands
    # imply lateral acceleration demand, which implies roll/pitch tilt.
    # Yaw is modeled as the vehicle heading tracking a fixed target yaw.
    # ------------------------------------------------------------------
    target_yaw_rad: float = 0.0
    init_roll_pitch_range_rad: float = math.radians(5.0)
    init_yaw_range_rad: float = 0.0

    attitude_response_alpha_min: float = 0.25
    attitude_response_alpha_max: float = 0.55
    max_tilt_target_rad: float = math.radians(25.0)
    max_tilt_rad: float = math.radians(45.0)

    yaw_align_kp: float = 1.4
    max_yaw_rate_radps: float = math.radians(90.0)

    attitude_obs_noise_std_rad: float = math.radians(1.0)
    attitude_process_noise_std_rad: float = math.radians(0.25)

    # ------------------------------------------------------------------
    # Rigid-body approximation
    # velocity command -> desired acceleration -> attitude/thrust setpoint
    # -> body-rate/thrust first-order response -> NED acceleration.
    # This is still much simpler than PX4+motor dynamics, but it preserves
    # the landing-relevant delay chain that the old velocity first-order
    # model compressed away.
    # ------------------------------------------------------------------
    gravity_mps2: float = 9.8065
    vel_cmd_tau_s: float = 0.45
    max_cmd_accel_xy_mps2: float = 1.8
    max_cmd_accel_z_mps2: float = 1.0

    attitude_time_constant_s: float = 0.22
    body_rate_response_alpha_min: float = 0.25
    body_rate_response_alpha_max: float = 0.55
    max_roll_pitch_rate_radps: float = math.radians(220.0)
    max_yaw_rate_response_radps: float = math.radians(120.0)
    body_rate_process_noise_std_radps: float = math.radians(1.5)

    thrust_response_alpha_min: float = 0.18
    thrust_response_alpha_max: float = 0.45
    min_thrust_accel_mps2: float = 0.25 * 9.8065
    max_thrust_accel_mps2: float = 1.80 * 9.8065
    thrust_process_noise_std_mps2: float = 0.08

    # Aerodynamic/drag damping. Kept intentionally small because PX4 inner
    # loops already stabilize the platform in the real system.
    linear_drag_xy: float = 0.10
    linear_drag_z: float = 0.06

    # ------------------------------------------------------------------
    # Near-ground / touchdown model
    # ground_z_m follows PX4 local NED convention: z=0 is the pad/ground,
    # negative z is above the ground. This block adds landing-specific physics
    # that the old velocity-response model could not represent: ground effect,
    # contact impulse, bounce, touchdown quality, and motor cutoff after a soft
    # touchdown.
    # ------------------------------------------------------------------
    ground_z_m: float = 0.0
    ground_effect_height_m: float = 0.80
    ground_effect_gain: float = 0.00        #0.18
    ground_effect_max_factor: float = 1.25

    contact_enabled: bool = True
    success_requires_contact: bool = True
    contact_success_hold_steps: int = 1

    touchdown_vz_soft_mps: float = 0.22       # NED +z/downward impact speed
    touchdown_vxy_soft_mps: float = 0.30
    touchdown_tilt_soft_rad: float = math.radians(12.0)
    hard_touchdown_vz_mps: float = 0.75
    hard_touchdown_vxy_mps: float = 0.90
    hard_touchdown_tilt_rad: float = math.radians(28.0)

    bounce_vz_threshold_mps: float = 0.25
    bounce_restitution_min: float = 0.08
    bounce_restitution_max: float = 0.35
    max_bounce_count: int = 2
    ground_friction_xy: float = 0.55

    motor_cutoff_on_soft_contact: bool = True
    motor_cutoff_thrust_accel_mps2: float = 0.0

    w_touchdown_impact: float = 5.0
    w_touchdown_lateral: float = 1.5
    w_touchdown_tilt: float = 4.0
    w_bounce: float = 30.0

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
    success_tilt_rad: float = math.radians(10.0)
    success_yaw_error_rad: float = math.pi

    # ------------------------------------------------------------------
    # Reward coefficients
    # ------------------------------------------------------------------
    w_xy: float = 1.00
    w_z: float = 0.35
    w_speed: float = 0.08
    w_accel: float = 0.03
    w_action: float = 0.03
    w_action_delta: float = 0.05
    w_progress: float = 5.00
    w_saturation: float = 0.70
    w_near_ground_vz: float = 1.50
    w_tilt: float = 0.20
    w_yaw: float = 0.00             #0.05
    w_near_ground_tilt: float = 0.80


class LandingEnv(gym.Env):
    

    """Observation, shape=(16,):
        [dx_obs, dy_obs, dz_obs,
         vx_obs, vy_obs, vz_obs,
         ax_obs, ay_obs, az_obs,
         roll, pitch, yaw_error,
         prev_action_x, prev_action_y, prev_action_z,
         target_valid]

    Action, shape=(3,):
        normalized residual velocity command in [-1, 1]^3.

    Control law inside the environment:
        v_cmd = v_pid + v_residual. The inner plant model then maps v_cmd to
        desired acceleration, attitude/thrust setpoints, delayed body-rate and
        thrust response, NED rigid-body acceleration, and landing contact.

    The reward and termination use the true target [0, 0, 0], while the policy
    and PID use delayed/noisy/stale/dropout target measurements.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: LandingConfig | None = None):
        super().__init__()
        self.cfg = config or LandingConfig()

        # Phase 2: deterministic velocity-control arithmetic lives here now.
        # The controller owns only cfg; it holds no episode state and no RNG.
        self.controller = BaselineController(self.cfg)

        # Phase 3: per-episode constant wind lives here now. Owns only cfg and
        # the wind vector; holds no RNG. self.wind_accel stays a view of the
        # model's canonical vector (never a second, independent vector).
        self.disturbance = DisturbanceModel(self.cfg)

        # Phase 4: control-step dt sampling lives here now. Owns only cfg;
        # holds no RNG and no clock/episode state. step() decides when to call.
        self.loop_timing = LoopTiming(self.cfg)

        # Phase 5: action-path (command) latency lives here now. Owns only cfg,
        # action_delay_steps, and the FIFO. Holds no RNG; reset() consumes one
        # np_random.integers(...) draw. self.action_delay_steps / self.action_queue
        # stay env attributes, aliased to this helper's objects after reset().
        self.action_latency = ActionLatency(self.cfg)

        # Phase 6: the raw target-measurement stochastic model (dropout / stale /
        # noise / outlier) lives here now. Owns only cfg and last_raw_target;
        # holds no RNG. sample() takes np_random and has data-dependent draw
        # counts. self.last_raw_target is kept as a compatibility mirror of the
        # model's canonical value.
        self.target_measurement_model = TargetMeasurementModel(self.cfg)

        # Phase 7: the target observation-delay state + FIFO queue live here now.
        # Owns only cfg, obs_delay_steps, and target_queue; holds no RNG.
        # reset() consumes one np_random.integers(...) draw -- the SECOND integer
        # draw in env reset, kept AFTER action_latency.reset. self.obs_delay_steps
        # / self.target_queue stay env attributes, aliased to this helper.
        self.obs_latency = ObsLatency(self.cfg)

        # Phase 10: the three per-_get_obs() observation-noise draws (velocity,
        # acceleration, attitude) live here now. Owns only cfg; keeps no noise
        # and no RNG. sample() consumes three np_random.normal(0.0, std, size=3)
        # draws in that order, still only from _get_obs(). All deterministic
        # observation assembly stays in _get_obs().
        self.observation_noise = ObservationNoiseSampler(self.cfg)

        # Phase 8: the four per-episode response-alpha draws (vel, attitude,
        # body_rate, thrust) live here now. Owns only cfg + the four alpha
        # fields; holds no RNG. reset() consumes four np_random.uniform(...)
        # draws in that order. self.*_response_alpha stay env attributes,
        # aliased to this component after reset(). vel_/attitude_response_alpha
        # are info-only but remain part of the frozen RNG stream.
        self.response_alphas = ResponseAlphas(self.cfg)

        # Phase 11: the three in-step process-noise draws (body-rate, thrust,
        # translational) live here now. Owns only cfg; keeps no noise, no RNG,
        # no dt/noise_scale. Called from _update_rigid_body_dynamics() via three
        # explicit methods at their original separated call sites. noise_scale
        # is still computed in that method and passed in.
        self.process_noise = ProcessNoiseSampler(self.cfg)

        # Phase 9: the randomized initial (pos, vel, attitude) draws live here
        # now. Owns only cfg; keeps no pos/vel/attitude/RNG. sample() consumes
        # seven np_random.uniform(...) draws (x, y, altitude, vel size=3, roll,
        # pitch, yaw) at the TOP of reset(). LandingEnv still owns self.pos /
        # self.vel / self.attitude and all deterministic reset assignments.
        self.initial_state_sampler = InitialStateSampler(self.cfg)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Last element is target_valid, represented as 0.0 or 1.0.
        high = np.array(
            [
                30.0, 30.0, 30.0,      # position error [m]
                6.0, 6.0, 6.0,         # velocity [m/s]
                self.cfg.max_acceleration_obs_mps2,
                self.cfg.max_acceleration_obs_mps2,
                self.cfg.max_acceleration_obs_mps2,  # acceleration [m/s^2]
                math.pi, math.pi, math.pi,  # roll, pitch, yaw_error [rad]
                1.0, 1.0, 1.0,         # previous normalized action
                1.0,                   # target_valid
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            shape=(16,),
            dtype=np.float32,
        )

        self.target_true = np.zeros(3, dtype=np.float64)
        self.target_measured = np.zeros(3, dtype=np.float64)
        self.last_raw_target = np.zeros(3, dtype=np.float64)

        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.accel = np.zeros(3, dtype=np.float64)
        self.prev_accel = np.zeros(3, dtype=np.float64)

        # attitude = [roll, pitch, yaw] in radians.
        # roll/pitch are coupled to lateral acceleration demand; yaw tracks target_yaw.
        self.attitude = np.zeros(3, dtype=np.float64)
        self.target_yaw = float(self.cfg.target_yaw_rad)
        self.yaw_rate = 0.0
        self.body_rates = np.zeros(3, dtype=np.float64)  # [p, q, r] proxy [rad/s]
        self.thrust_accel = float(self.cfg.gravity_mps2)  # collective thrust per mass [m/s^2]
        self.attitude_setpoint = np.zeros(3, dtype=np.float64)
        self.thrust_accel_setpoint = float(self.cfg.gravity_mps2)
        self.accel_cmd = np.zeros(3, dtype=np.float64)

        self.prev_action = np.zeros(3, dtype=np.float64)
        self.prev_potential = 0.0
        self.action_queue: list[np.ndarray] = []
        self.target_queue: list[tuple[np.ndarray, bool, str]] = []

        self.step_count = 0

        # Per-episode randomized values
        self.vel_response_alpha = np.ones(3, dtype=np.float64) * 0.35  # kept for backward-compatible info fields
        self.attitude_response_alpha = np.ones(2, dtype=np.float64) * 0.40  # kept for backward-compatible info fields
        self.body_rate_response_alpha = np.ones(3, dtype=np.float64) * 0.40
        self.thrust_response_alpha = 0.30
        self.wind_accel = self.disturbance.wind_accel  # canonical vector, shared
        self.action_delay_steps = 0      #2
        self.obs_delay_steps = 2

        # Current delayed target used by both observation and PID.
        self.obs_target = np.zeros(3, dtype=np.float64)
        self.obs_target_valid = True
        self.obs_target_mode = "init"

        # Landing-contact state. These are reset every episode and updated by
        # the contact model after position integration.
        self.ground_contact = False
        self.contact_count = 0
        self.bounce_count = 0
        self.motor_cutoff = False
        self.ground_effect_factor = 1.0
        self.contact_event = False
        self.soft_contact = False
        self.hard_contact = False
        self.bounced = False
        self.touchdown_quality = "none"
        self.last_impact_vz = 0.0
        self.last_touchdown_vxy = 0.0
        self.last_bounce_speed = 0.0

    # ------------------------------------------------------------------
    # Sampling helpers
    #
    # Phase 4: _sample_dt was moved verbatim to
    # landing_rl/envs/loop_timing.py as LoopTiming.sample_dt(rng). step() calls
    # self.loop_timing.sample_dt(self.np_random) at the same point.
    #
    # Phase 6: _sample_raw_target_measurement was moved verbatim to
    # landing_rl/perception/target_measurement.py as
    # TargetMeasurementModel.sample(rng, target_true).
    # Phase 7: the obs-delay queue block was moved verbatim to
    # landing_rl/perception/obs_latency.py as ObsLatency.push_and_get.
    # _update_target_pipeline stays as the orchestration seam below.
    # ------------------------------------------------------------------

    def _update_target_pipeline(self) -> None:
        """Apply perception measurement model and observation delay queue."""
        raw_target, valid, mode = self.target_measurement_model.sample(
            self.np_random,
            self.target_true,
        )
        # Compatibility mirror: keep env.last_raw_target equal to the model's
        # canonical value. The legacy code reassigns (not mutates)
        # last_raw_target on the normal/outlier path, so this re-alias after
        # every sample keeps the two in lockstep (identity holds at every
        # observable point; see target_measurement.py).
        self.last_raw_target = self.target_measurement_model.last_raw_target

        delayed_target, delayed_valid, delayed_mode = self.obs_latency.push_and_get(
            raw_target, valid, mode
        )

        self.target_measured = delayed_target.copy()
        self.obs_target = delayed_target.copy()
        self.obs_target_valid = bool(delayed_valid)
        self.obs_target_mode = delayed_mode

    def _yaw_error(self) -> float:
        return wrap_pi(self.target_yaw - float(self.attitude[2]))

    def _get_obs(self) -> np.ndarray:
        error_obs = self.obs_target - self.pos

        # Phase 10: the three observation-noise draws live in
        # ObservationNoiseSampler now -- same three np_random.normal(0.0, std,
        # size=3) calls in the same order (velocity, acceleration, attitude).
        # Only the deterministic transforms below stay in _get_obs. Folding the
        # draws to one call here is RNG-neutral: the np.clip and attitude_obs
        # construction between the legacy draws consume no RNG.
        vel_noise, accel_noise, attitude_noise = self.observation_noise.sample(
            self.np_random
        )

        vel_obs = self.vel + vel_noise

        accel_obs = self.accel + accel_noise
        accel_obs = np.clip(
            accel_obs,
            -self.cfg.max_acceleration_obs_mps2,
            self.cfg.max_acceleration_obs_mps2,
        )

        attitude_obs = np.array(
            [
                self.attitude[0],
                self.attitude[1],
                self._yaw_error(),
            ],
            dtype=np.float64,
        )
        attitude_obs += attitude_noise
        attitude_obs[2] = wrap_pi(float(attitude_obs[2]))

        valid = 1.0 if self.obs_target_valid else 0.0

        obs = np.concatenate(
            [
                error_obs,
                vel_obs,
                accel_obs,
                attitude_obs,
                self.prev_action,
                np.array([valid], dtype=np.float64),
            ]
        ).astype(np.float32)

        return obs

    # ------------------------------------------------------------------
    # Controller helpers
    #
    # Phase 2: _pid_velocity, _scale_action, and _limit_velocity_command were
    # moved verbatim to landing_rl/controllers/baseline_controller.py as
    # BaselineController.pid_velocity / scale_action / combine_and_limit.
    # step() calls self.controller for those, plus apply_descent_gate for the
    # descent gate #2 that used to be inline in step().
    # ------------------------------------------------------------------
    # Phase 5: _apply_action_delay was moved verbatim to
    # landing_rl/envs/action_latency.py as ActionLatency.apply. step() calls
    # self.action_latency.apply(raw_action) at the identical location.

    def _altitude_agl(self) -> float:
        """Altitude above ground in meters. In NED, negative z is above ground."""
        return max(0.0, float(self.cfg.ground_z_m - self.pos[2]))
    
    def _potential(self) -> float:
        err = self.target_true - self.pos
        xy = float(np.linalg.norm(err[:2]))
        z = abs(float(err[2]))
        speed = float(np.linalg.norm(self.vel))

        return -(1.0 * xy + 0.6 * z + 0.15 * speed)

    def _compute_ground_effect_factor(self) -> float:
        """Return thrust multiplier caused by ground effect near the pad."""
        h = self._altitude_agl()
        height = max(float(self.cfg.ground_effect_height_m), 1e-6)

        if h >= height or self.ground_contact:
            return 1.0

        closeness = 1.0 - h / height
        factor = 1.0 + float(self.cfg.ground_effect_gain) * closeness * closeness
        return float(np.clip(factor, 1.0, self.cfg.ground_effect_max_factor))

    def _reset_contact_step_flags(self) -> None:
        self.contact_event = False
        self.soft_contact = False
        self.hard_contact = False
        self.bounced = False
        self.touchdown_quality = "none"
        self.last_impact_vz = 0.0
        self.last_touchdown_vxy = 0.0
        self.last_bounce_speed = 0.0

    def _apply_ground_contact(self, vel_before_step: np.ndarray) -> None:
        """Resolve simple ground contact and bounce at z=ground_z_m.

        NED convention: z velocity > 0 means descending toward the ground.
        A soft touchdown pins the vehicle to the ground and optionally cuts
        thrust. A faster touchdown creates an upward rebound. A very fast or
        highly tilted touchdown is marked as hard contact and will terminate
        the episode as a failed landing.
        """
        if not self.cfg.contact_enabled:
            return

        ground_z = float(self.cfg.ground_z_m)
        if self.pos[2] < ground_z:
            self.ground_contact = False
            self.contact_count = 0
            return

        self.contact_event = True
        self.ground_contact = True
        self.contact_count += 1
        self.pos[2] = ground_z

        impact_vz = max(float(self.vel[2]), float(vel_before_step[2]), 0.0)
        touchdown_vxy = float(np.linalg.norm(self.vel[:2]))
        tilt = float(np.linalg.norm(self.attitude[:2]))

        self.last_impact_vz = impact_vz
        self.last_touchdown_vxy = touchdown_vxy

        self.hard_contact = (
            impact_vz > self.cfg.hard_touchdown_vz_mps
            or touchdown_vxy > self.cfg.hard_touchdown_vxy_mps
            or tilt > self.cfg.hard_touchdown_tilt_rad
        )

        self.soft_contact = (
            impact_vz <= self.cfg.touchdown_vz_soft_mps
            and touchdown_vxy <= self.cfg.touchdown_vxy_soft_mps
            and tilt <= self.cfg.touchdown_tilt_soft_rad
            and not self.hard_contact
        )

        if self.hard_contact:
            self.touchdown_quality = "hard"
        elif self.soft_contact:
            self.touchdown_quality = "soft"
        else:
            self.touchdown_quality = "rough"

        # Tangential ground friction removes lateral sliding at contact.
        self.vel[:2] *= float(np.clip(1.0 - self.cfg.ground_friction_xy, 0.0, 1.0))

        should_bounce = (
            impact_vz > self.cfg.bounce_vz_threshold_mps
            and not self.soft_contact
        )

        if should_bounce:
            restitution = float(self.np_random.uniform(
                self.cfg.bounce_restitution_min,
                self.cfg.bounce_restitution_max,
            ))
            bounce_speed = restitution * impact_vz
            self.vel[2] = -bounce_speed  # negative NED z means rebound upward
            self.bounced = True
            self.bounce_count += 1
            self.contact_count = 0
            self.last_bounce_speed = bounce_speed
            if not self.hard_contact:
                self.touchdown_quality = "bounce"
        else:
            # No rebound: vehicle stays on the ground.
            self.vel[2] = 0.0
            self.last_bounce_speed = 0.0

        if self.soft_contact and self.cfg.motor_cutoff_on_soft_contact:
            self.motor_cutoff = True
            self.thrust_accel_setpoint = float(self.cfg.motor_cutoff_thrust_accel_mps2)
            self.thrust_accel = float(self.cfg.motor_cutoff_thrust_accel_mps2)

    def _rotation_body_to_ned(self) -> np.ndarray:
        """Body-to-NED rotation matrix using aerospace roll/pitch/yaw."""
        roll, pitch, yaw = [float(v) for v in self.attitude]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    def _velocity_command_to_inner_loop_setpoints(
        self,
        v_cmd: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Convert velocity command to attitude/thrust setpoints.

        This approximates the PX4 cascade:
            velocity command -> acceleration demand -> attitude/thrust setpoint.

        NED convention is used. Positive z acceleration means downward
        acceleration, so it is produced by reducing collective thrust below
        hover thrust.
        """
        dt_safe = max(float(dt), 1e-6)
        g = float(self.cfg.gravity_mps2)

        # Desired NED acceleration from velocity error. This replaces the old
        # direct first-order velocity response with a physically interpretable
        # acceleration request.
        accel_cmd = (np.asarray(v_cmd, dtype=np.float64) - self.vel) / max(
            self.cfg.vel_cmd_tau_s,
            dt_safe,
        )

        axy_norm = float(np.linalg.norm(accel_cmd[:2]))
        if axy_norm > self.cfg.max_cmd_accel_xy_mps2:
            accel_cmd[:2] *= self.cfg.max_cmd_accel_xy_mps2 / (axy_norm + 1e-9)
        accel_cmd[2] = float(np.clip(
            accel_cmd[2],
            -self.cfg.max_cmd_accel_z_mps2,
            self.cfg.max_cmd_accel_z_mps2,
        ))

        # Near-hover multicopter tilt approximation in NED.
        # +roll produces +East acceleration. +pitch produces -North acceleration.
        denom = max(g - float(accel_cmd[2]), 1e-3)
        roll_sp = math.atan2(float(accel_cmd[1]), denom)
        pitch_sp = math.atan2(-float(accel_cmd[0]), denom)

        roll_sp = float(np.clip(
            roll_sp,
            -self.cfg.max_tilt_target_rad,
            self.cfg.max_tilt_target_rad,
        ))
        pitch_sp = float(np.clip(
            pitch_sp,
            -self.cfg.max_tilt_target_rad,
            self.cfg.max_tilt_target_rad,
        ))

        yaw_sp = float(self.target_yaw)
        thrust_sp = float(np.clip(
            g - float(accel_cmd[2]),
            self.cfg.min_thrust_accel_mps2,
            self.cfg.max_thrust_accel_mps2,
        ))

        attitude_sp = np.array([roll_sp, pitch_sp, yaw_sp], dtype=np.float64)
        return attitude_sp, thrust_sp, accel_cmd

    def _update_rigid_body_dynamics(self, v_cmd: np.ndarray, dt: float) -> None:
        """Update attitude, thrust, acceleration, velocity, and position.

        This is a compact 6DoF-inspired plant model. It does not simulate motor
        mixing or rotor angular momentum, but it exposes the key delay chain:
            v_cmd -> attitude/thrust setpoint -> body-rate/thrust response
            -> thrust vector in NED -> acceleration -> velocity -> position.
        """
        dt_safe = max(float(dt), 1e-6)
        g = float(self.cfg.gravity_mps2)

        self._reset_contact_step_flags()
        vel_before = self.vel.copy()

        attitude_sp, thrust_sp, accel_cmd = self._velocity_command_to_inner_loop_setpoints(v_cmd, dt_safe)
        if self.motor_cutoff:
            thrust_sp = float(self.cfg.motor_cutoff_thrust_accel_mps2)

        self.attitude_setpoint = attitude_sp.copy()
        self.thrust_accel_setpoint = float(thrust_sp)
        self.accel_cmd = accel_cmd.copy()

        # Attitude/rate controller proxy.
        att_error = np.array(
            [
                attitude_sp[0] - self.attitude[0],
                attitude_sp[1] - self.attitude[1],
                wrap_pi(float(attitude_sp[2] - self.attitude[2])),
            ],
            dtype=np.float64,
        )

        rate_cmd = att_error / max(self.cfg.attitude_time_constant_s, dt_safe)
        rate_cmd[0] = float(np.clip(
            rate_cmd[0],
            -self.cfg.max_roll_pitch_rate_radps,
            self.cfg.max_roll_pitch_rate_radps,
        ))
        rate_cmd[1] = float(np.clip(
            rate_cmd[1],
            -self.cfg.max_roll_pitch_rate_radps,
            self.cfg.max_roll_pitch_rate_radps,
        ))
        rate_cmd[2] = float(np.clip(
            self.cfg.yaw_align_kp * att_error[2],
            -self.cfg.max_yaw_rate_response_radps,
            self.cfg.max_yaw_rate_response_radps,
        ))

        rate_alpha_eff = 1.0 - np.power(
            1.0 - np.clip(self.body_rate_response_alpha, 1e-4, 0.9999),
            dt_safe / max(self.cfg.dt, 1e-6),
        )
        rate_alpha_eff = np.clip(rate_alpha_eff, 0.0, 1.0)

        self.body_rates = self.body_rates + rate_alpha_eff * (rate_cmd - self.body_rates)

        noise_scale = math.sqrt(dt_safe / max(self.cfg.dt, 1e-6))
        # Phase 11: body-rate process noise (draw 1 of 3). Verbatim expression;
        # noise_scale stays computed here.
        self.body_rates += self.process_noise.sample_body_rate(
            self.np_random, noise_scale
        )

        self.attitude[0] = float(np.clip(
            self.attitude[0] + self.body_rates[0] * dt_safe,
            -math.pi,
            math.pi,
        ))
        self.attitude[1] = float(np.clip(
            self.attitude[1] + self.body_rates[1] * dt_safe,
            -math.pi,
            math.pi,
        ))
        self.attitude[2] = wrap_pi(float(self.attitude[2] + self.body_rates[2] * dt_safe))
        self.yaw_rate = float(self.body_rates[2])

        # Collective thrust/motor response proxy.
        thrust_alpha_eff = 1.0 - (1.0 - np.clip(self.thrust_response_alpha, 1e-4, 0.9999)) ** (
            dt_safe / max(self.cfg.dt, 1e-6)
        )
        thrust_alpha_eff = float(np.clip(thrust_alpha_eff, 0.0, 1.0))
        self.thrust_accel = float(
            self.thrust_accel
            + thrust_alpha_eff * (thrust_sp - self.thrust_accel)
            + self.process_noise.sample_thrust(self.np_random, noise_scale)  # draw 2 of 3 (scalar)
        )
        thrust_min = (
            float(self.cfg.motor_cutoff_thrust_accel_mps2)
            if self.motor_cutoff
            else float(self.cfg.min_thrust_accel_mps2)
        )
        self.thrust_accel = float(np.clip(
            self.thrust_accel,
            thrust_min,
            self.cfg.max_thrust_accel_mps2,
        ))

        # Rigid-body translational dynamics in NED.
        r_bn = self._rotation_body_to_ned()
        body_z_in_ned = r_bn[:, 2]
        gravity_accel = np.array([0.0, 0.0, g], dtype=np.float64)

        self.ground_effect_factor = self._compute_ground_effect_factor()
        effective_thrust_accel = self.thrust_accel * self.ground_effect_factor
        thrust_accel_ned = -effective_thrust_accel * body_z_in_ned
        drag_accel = -np.array(
            [
                self.cfg.linear_drag_xy * self.vel[0],
                self.cfg.linear_drag_xy * self.vel[1],
                self.cfg.linear_drag_z * self.vel[2],
            ],
            dtype=np.float64,
        )
        # Phase 11: translational-acceleration process noise (draw 3 of 3).
        process_accel_noise = self.process_noise.sample_translational_accel(
            self.np_random, noise_scale, dt_safe
        )

        self.prev_accel = self.accel.copy()
        self.accel = gravity_accel + thrust_accel_ned + drag_accel + self.wind_accel + process_accel_noise

        self.vel = self.vel + self.accel * dt_safe
        self.pos = self.pos + self.vel * dt_safe

        self._apply_ground_contact(vel_before)

        # Keep the acceleration channel consistent with the actual velocity change.
        # This is the value the policy observes as ax/ay/az.
        self.accel = (self.vel - vel_before) / dt_safe

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        # Phase 9: the seven randomized initial-state draws live in
        # InitialStateSampler now -- same order (x, y, altitude, vel size=3,
        # roll, pitch, yaw), same bounds / size / dtype / z = -altitude NED
        # mapping, at this same point in the reset RNG sequence (the very first
        # draws of reset). The deterministic zero/const assignments between the
        # legacy velocity draw and attitude draws (accel, prev_accel,
        # prev_action, target_yaw) consume no RNG and stay in this env.
        pos, vel, attitude = self.initial_state_sampler.sample(self.np_random)
        self.pos = pos
        self.vel = vel
        self.accel = np.zeros(3, dtype=np.float64)
        self.prev_accel = np.zeros(3, dtype=np.float64)

        self.prev_action = np.zeros(3, dtype=np.float64)

        self.target_yaw = float(self.cfg.target_yaw_rad)
        self.attitude = attitude
        self.yaw_rate = 0.0
        self.body_rates = np.zeros(3, dtype=np.float64)
        self.thrust_accel = float(self.cfg.gravity_mps2)
        self.attitude_setpoint = self.attitude.copy()
        self.thrust_accel_setpoint = float(self.cfg.gravity_mps2)
        self.accel_cmd = np.zeros(3, dtype=np.float64)

        self.ground_contact = False
        self.contact_count = 0
        self.bounce_count = 0
        self.motor_cutoff = False
        self.ground_effect_factor = 1.0
        self._reset_contact_step_flags()

        self.prev_potential = self._potential()

        # Phase 5/7: the two delay draws stay in this exact order --
        # action_delay_steps FIRST (ActionLatency), obs_delay_steps SECOND
        # (ObsLatency) -- each one np_random.integers(...) call, never merged or
        # vectorized. self.action_queue / self.target_queue / self.action_delay_steps
        # / self.obs_delay_steps remain env attributes aliased to the helpers.
        self.action_delay_steps = self.action_latency.reset(self.np_random)
        self.obs_delay_steps = self.obs_latency.reset(self.np_random)

        self.action_queue = self.action_latency.action_queue
        self.target_queue = self.obs_latency.target_queue

        # Phase 8: the four per-episode response-alpha draws live in
        # ResponseAlphas now -- same four np_random.uniform(...) calls in the
        # same order (vel, attitude, body_rate, thrust), same bounds / sizes /
        # .astype(float64) / float(...), at the same point in the reset RNG
        # sequence (after ObsLatency.reset, before DisturbanceModel.reset).
        # vel_/attitude_response_alpha are info-only but MUST still be sampled.
        # self.*_response_alpha stay env attributes aliased to the component.
        self.response_alphas.reset(self.np_random)
        self.vel_response_alpha = self.response_alphas.vel_response_alpha
        self.attitude_response_alpha = self.response_alphas.attitude_response_alpha
        self.body_rate_response_alpha = self.response_alphas.body_rate_response_alpha
        self.thrust_response_alpha = self.response_alphas.thrust_response_alpha

        # Phase 3: same three scalar np_random.uniform draws (x, y, z order,
        # same bounds), at the same point in the reset RNG sequence.
        self.wind_accel = self.disturbance.reset(self.np_random)

        # Initialize target measurement pipeline with one clean-ish measurement.
        # Phase 6: the measurement history lives in target_measurement_model now;
        # reset() consumes no RNG. self.last_raw_target stays a mirror of it.
        self.target_measurement_model.reset(self.target_true)
        self.last_raw_target = self.target_measurement_model.last_raw_target
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

        old_action = self.prev_action.copy()
        raw_action = np.asarray(action, dtype=np.float64)
        raw_action = np.clip(raw_action, -1.0, 1.0)
        action_delta = raw_action - old_action
        self.prev_action = raw_action.copy()

        dt = self.loop_timing.sample_dt(self.np_random)

        # The action was selected from the current observation.
        # Therefore both PID and residual RL use the currently stored delayed target.
        control_target = self.obs_target.copy()
        control_target_valid = bool(self.obs_target_valid)

        altitude_agl_before = self._altitude_agl()
        control_error = control_target - self.pos
        control_xy_error = float(np.linalg.norm(control_error[:2]))

        applied_action = self.action_latency.apply(raw_action)

        v_pid = self.controller.pid_velocity(
            control_target,
            control_target_valid,
            self.pos,
            self.vel,
            altitude_agl_before,
        )

        if control_target_valid or self.cfg.apply_residual_when_target_invalid:
            v_residual = self.controller.scale_action(applied_action)
        else:
            v_residual = np.zeros(3, dtype=np.float64)

        v_cmd = self.controller.combine_and_limit(v_pid, v_residual)

        # 착륙 gate 안에서는 PPO residual이 PID의 하강 bias를 죽이지 못하게 함.
        # NED 기준 +z velocity = 아래로 하강.
        v_cmd = self.controller.apply_descent_gate(
            v_cmd,
            control_target_valid,
            control_xy_error,
            altitude_agl_before,
            self.ground_contact,
        )

        self._update_rigid_body_dynamics(v_cmd, dt)

        # dynamics update 이후의 고도. reward 계산에는 이 값을 써야 함.
        altitude_agl = self._altitude_agl()

        reward = 0.0
        potential = self._potential()
        progress_reward = self.cfg.w_progress * (potential - self.prev_potential)
        reward += progress_reward
        self.prev_potential = potential

        alpha_eff = np.array([math.nan, math.nan, math.nan], dtype=np.float64)

        true_error = self.target_true - self.pos
        xy_error = float(np.linalg.norm(true_error[:2]))
        z_error = abs(float(true_error[2]))
        vxy = float(np.linalg.norm(self.vel[:2]))
        vz_abs = abs(float(self.vel[2]))
        speed = float(np.linalg.norm(self.vel))
        accel_mag = float(np.linalg.norm(self.accel))
        jerk_mag = float(np.linalg.norm((self.accel - self.prev_accel) / max(float(dt), 1e-6)))
        roll = float(self.attitude[0])
        pitch = float(self.attitude[1])
        yaw = float(self.attitude[2])
        yaw_error = self._yaw_error()
        yaw_error_abs = abs(float(yaw_error))
        tilt = float(np.linalg.norm(self.attitude[:2]))

        reward += -self.cfg.w_xy * xy_error
        reward += -self.cfg.w_z * z_error
        reward += -self.cfg.w_speed * speed
        reward += -self.cfg.w_accel * accel_mag
        reward += -self.cfg.w_tilt * tilt
        reward += -self.cfg.w_yaw * yaw_error_abs
        reward += -self.cfg.w_action * float(np.linalg.norm(raw_action))
        reward += -self.cfg.w_action_delta * float(np.sum(action_delta ** 2))

        saturation = np.maximum(np.abs(raw_action) - 0.80, 0.0)
        reward += -self.cfg.w_saturation * float(np.sum(saturation ** 2))
        

        if xy_error < 0.5 and altitude_agl < 1.0:
            reward += 0.2
        if xy_error < 0.35 and altitude_agl > self.cfg.success_altitude_m:
        # NED 기준 vel[2] > 0 이 하강
            descent_rate = max(float(self.vel[2]), 0.0)
            safe_descent = min(descent_rate, 0.18)
            reward += 1.0 * safe_descent

        # NED: z velocity > 0 means descending. Near the ground, any fast
        # vertical motion is treated as dangerous.
        if z_error < 0.8:
            safe_vz = self.cfg.success_vz_mps
            excess_vz = max(vz_abs - safe_vz, 0.0)

            reward += -self.cfg.w_near_ground_vz * excess_vz
            reward += -self.cfg.w_near_ground_tilt * tilt

        if self.contact_event:
            reward += -self.cfg.w_touchdown_impact * (self.last_impact_vz ** 2)
            reward += -self.cfg.w_touchdown_lateral * (self.last_touchdown_vxy ** 2)
            reward += -self.cfg.w_touchdown_tilt * (tilt ** 2)

        if self.bounced:
            reward += -self.cfg.w_bounce * (1.0 + self.last_bounce_speed)

        kinematic_success = (
            xy_error < self.cfg.success_xy_m
            and z_error < self.cfg.success_altitude_m
            and vxy < self.cfg.success_vxy_mps
            and vz_abs < self.cfg.success_vz_mps
            and tilt < self.cfg.success_tilt_rad
            and yaw_error_abs < self.cfg.success_yaw_error_rad
        )

        contact_success = (
            self.ground_contact
            and not self.bounced
            and not self.hard_contact
            and self.touchdown_quality == "soft"
            and self.contact_count >= self.cfg.contact_success_hold_steps
            and kinematic_success
        )

        success = contact_success if self.cfg.success_requires_contact else kinematic_success

        failure_reason = "none"
        altitude = -float(self.pos[2])

        if xy_error > self.cfg.max_xy_error_m:
            failure_reason = "xy_error_limit"
        elif altitude > self.cfg.max_altitude_m:
            failure_reason = "altitude_limit"
        elif speed > self.cfg.max_speed_mps:
            failure_reason = "speed_limit"
        elif tilt > self.cfg.max_tilt_rad:
            failure_reason = "attitude_tilt_limit"
        elif self.hard_contact:
            failure_reason = "hard_landing"
        elif self.bounce_count > self.cfg.max_bounce_count:
            failure_reason = "excessive_bounce"
        elif self.pos[2] > self.cfg.below_ground_limit_m:
            failure_reason = "below_ground_limit"

        failed = failure_reason != "none"

        terminated = False
        if success:
            reward += 500.0
            terminated = True
        elif failed:
            reward -= 300.0
            terminated = True

        truncated = (self.step_count >= self.cfg.max_steps) and not terminated

        if truncated:
            reward -= 150.0
            reward -= 50.0 * min(xy_error / self.cfg.success_xy_m, 5.0)
            reward -= 50.0 * min(z_error / self.cfg.success_altitude_m, 5.0)

        info = {
            "xy_error": xy_error,
            "z_error": z_error,
            "vxy": vxy,
            "vz_abs": vz_abs,
            "accel_mag": accel_mag,
            "jerk_mag": jerk_mag,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "yaw_error": yaw_error,
            "tilt": tilt,
            "yaw_rate": self.yaw_rate,
            "body_rates": self.body_rates.copy(),
            "thrust_accel": self.thrust_accel,
            "thrust_accel_setpoint": self.thrust_accel_setpoint,
            "attitude_setpoint": self.attitude_setpoint.copy(),
            "accel_cmd": self.accel_cmd.copy(),
            "success": success,
            "failed": failed,
            "failure_reason": failure_reason,
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "accel": self.accel.copy(),
            "prev_accel": self.prev_accel.copy(),
            "target_true": self.target_true.copy(),
            "target_measured": self.target_measured.copy(),
            "target_valid": control_target_valid,
            "target_mode": self.obs_target_mode,
            "dt": dt,
            "vel_response_alpha": self.vel_response_alpha.copy(),
            "attitude_response_alpha": self.attitude_response_alpha.copy(),
            "body_rate_response_alpha": self.body_rate_response_alpha.copy(),
            "thrust_response_alpha": self.thrust_response_alpha,
            "alpha_eff": alpha_eff.copy(),
            "wind_accel": self.wind_accel.copy(),
            "ground_effect_factor": self.ground_effect_factor,
            "ground_contact": self.ground_contact,
            "contact_event": self.contact_event,
            "contact_count": self.contact_count,
            "bounce_count": self.bounce_count,
            "soft_contact": self.soft_contact,
            "hard_contact": self.hard_contact,
            "bounced": self.bounced,
            "touchdown_quality": self.touchdown_quality,
            "last_impact_vz": self.last_impact_vz,
            "last_touchdown_vxy": self.last_touchdown_vxy,
            "last_bounce_speed": self.last_bounce_speed,
            "motor_cutoff": self.motor_cutoff,
            "altitude_agl": self._altitude_agl(),
            "action_delay_steps": self.action_delay_steps,
            "obs_delay_steps": self.obs_delay_steps,
            "progress_reward": progress_reward,
            "action_delta": action_delta.copy(),
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
