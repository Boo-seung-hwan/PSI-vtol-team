"""MuJoCo-backed precision-landing Gymnasium environment.

Preserves the existing RL/control contract EXACTLY:

    observation_space.shape == (16,)   [dx,dy,dz,vx,vy,vz,ax,ay,az,
                                          roll,pitch,yaw_error,
                                          prev_action_x,y,z, target_valid]
    action_space.shape == (3,) in [-1,1]^3   (normalized residual velocity)

by reusing, UNMODIFIED, the following components from ``landing_rl``:
``BaselineController`` (PID + residual + descent gates),
``DisturbanceModel`` (wind), ``LoopTiming`` (dt jitter),
``ActionLatency`` / ``ObsLatency`` (the two separate delay FIFOs -- neither
is the physical inner-loop delay, which lives inside
``landing_mujoco.dynamics.actuation_model`` instead, per task spec section
11), ``TargetMeasurementModel`` / ``ObservationNoiseSampler`` (perception),
``InitialStateSampler`` (episode-start randomization), and
``InnerLoopCommandModel`` (v_cmd -> attitude/thrust setpoint, used inside
``MuJoCoDynamics``).

Replaced by MuJoCo: the rigid-body integrator (was
``LegacyVehicleDynamics.advance_free_flight``, Python Euler integration) and
the manual bounce/contact model (was ``ContactModel``) -- both now live in
``landing_mujoco.dynamics.mujoco_dynamics.MuJoCoDynamics``.

The reward, success/failure logic, and 16-D observation assembly are ported
here with the SAME formulas/thresholds as ``landing_rl.envs.landing_env
.LandingEnv`` (same ``LandingConfig`` fields, untouched) -- this is a
physics-plant swap, not a reward or RL-task redesign, per the task's
structural-freeze requirement.

Landing-gear standoff (pad_target_ned vs. vehicle_touchdown_target_ned)
------------------------------------------------------------------------
The legacy environment's contact model implicitly assumes ZERO landing-gear
standoff: it snaps the CG straight to ``pos[2] = ground_z_m`` on contact, so
``target_true`` (the pad position) doubles as the vehicle's own resting-CG
target without anyone ever separating the two concepts. MuJoCo's real
landing-gear geometry means the CG actually rests at
``z ~= ground_z_m - (gear standoff)`` -- so this environment explicitly
distinguishes:

    pad_target_ned                 -- the physical landing-pad position
                                       (what perception measures/estimates;
                                       identical role to legacy `target_true`
                                       / `obs_target`)
    vehicle_touchdown_target_ned   -- the desired VEHICLE-REFERENCE (CG)
                                       position at a correct level touchdown

related by (task spec's own derivation):

    p_landing^N = p_vehicle^N + R_B^N r_landing^B
    =>  vehicle_touchdown_target_ned = pad_target_ned - R_desired @ r_landing_body

where ``r_landing_body`` (``landing_mujoco.configs.param_schema
.landing_reference_point_body_m``) is the nominal landing-gear contact
point relative to CG -- explicitly derived from the actual per-leg
geometry, NEVER assumed equal to ``ground_clearance_m`` -- and
``R_desired`` is the LEVEL (roll=pitch=0, yaw=target_yaw) touchdown
attitude. dx/dy/dz (the observation) and the reward/success xy_error/
z_error are computed against ``vehicle_touchdown_target_ned``, not the raw
pad position, so a correct stationary touchdown gives xy_error~=0,
z_error~=0 AND ground_contact=True simultaneously -- restoring the
semantic meaning the legacy zero-standoff model had, without assuming zero
standoff. ``success_altitude_m`` is UNCHANGED (0.05 m) -- this fix changes
what z_error is measured against, not the tolerance itself.

``altitude_agl`` is a SEPARATE, physical-clearance concept: the landing-gear
reference point's height above the ground plane using the vehicle's
CURRENT (not desired) attitude. It reaches zero at actual physical contact
independently of z_error (e.g. a tilted vehicle sitting at the correct
target z has z_error~=0 but altitude_agl>0, since its actual gear point is
off the ground due to tilt). See ``MUJOCO_MODEL.md``.

``landing_rl.envs.landing_env.LandingEnv`` (and the frozen
``mujoco_rl/envs/env_prototype.py``) are UNCHANGED by this -- the
zero-standoff behavior there remains exactly as before.
"""

from __future__ import annotations

import math
from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from landing_mujoco.configs.param_schema import UAVPhysicalParams, landing_reference_point_body_m
from landing_mujoco.configs.simulation_config import SimulationConfig
from landing_mujoco.coordinates.transforms import rotation_body_to_world_ned
from landing_mujoco.dynamics.mujoco_dynamics import MuJoCoDynamics
from landing_rl.controllers import BaselineController
from landing_rl.disturbances import DisturbanceModel
from landing_rl.dynamics.vehicle_state import VehicleState
from landing_rl.envs.action_latency import ActionLatency
from landing_rl.envs.initial_state import InitialStateSampler
from landing_rl.envs.landing_env import LandingConfig, wrap_pi
from landing_rl.envs.loop_timing import LoopTiming
from landing_rl.perception import ObsLatency, ObservationNoiseSampler, TargetMeasurementModel


def deterministic_overrides(cfg: LandingConfig) -> LandingConfig:
    """Return a copy of ``cfg`` with every stochastic channel collapsed to a
    fixed value (task spec section 25: no wind, no observation noise, no
    target dropout/stale/outlier, no stochastic actuator noise, fixed
    delays, fixed initial state). Does not mutate ``cfg``."""
    return replace(
        cfg,
        dt_jitter_std=0.0,
        init_xy_range_m=0.0,
        init_vel_range_mps=0.0,
        init_roll_pitch_range_rad=0.0,
        init_yaw_range_rad=0.0,
        init_altitude_max_m=cfg.init_altitude_min_m,
        wind_accel_xy_max_mps2=0.0,
        wind_accel_z_max_mps2=0.0,
        process_noise_vel_std_mps=0.0,
        target_noise_xy_std_m=0.0,
        target_noise_z_std_m=0.0,
        target_dropout_prob=0.0,
        target_outlier_prob=0.0,
        target_stale_prob=0.0,
        velocity_obs_noise_std_mps=0.0,
        acceleration_obs_noise_std_mps2=0.0,
        attitude_obs_noise_std_rad=0.0,
        attitude_process_noise_std_rad=0.0,
        body_rate_process_noise_std_radps=0.0,
        thrust_process_noise_std_mps2=0.0,
        action_delay_steps_max=cfg.action_delay_steps_min,
        obs_delay_steps_max=cfg.obs_delay_steps_min,
    )


class MujocoLandingEnv(gym.Env):
    """Observation/action contract identical to
    ``landing_rl.envs.landing_env.LandingEnv`` -- see that class's
    docstring. The physical plant is MuJoCo instead of the legacy analytic
    model."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        physical_params: UAVPhysicalParams,
        config: LandingConfig | None = None,
        sim_config: SimulationConfig | None = None,
        deterministic_physics: bool = False,
    ):
        super().__init__()
        self.physical_params = physical_params
        self.cfg = config or LandingConfig()
        if deterministic_physics:
            self.cfg = deterministic_overrides(self.cfg)
        self.deterministic_physics = deterministic_physics
        self.sim_cfg = sim_config or SimulationConfig(
            control_dt=self.cfg.dt, deterministic_physics=deterministic_physics
        )

        self.controller = BaselineController(self.cfg)
        self.disturbance = DisturbanceModel(self.cfg)
        self.loop_timing = LoopTiming(self.cfg)
        self.action_latency = ActionLatency(self.cfg)
        self.target_measurement_model = TargetMeasurementModel(self.cfg)
        self.obs_latency = ObsLatency(self.cfg)
        self.observation_noise = ObservationNoiseSampler(self.cfg)
        self.initial_state_sampler = InitialStateSampler(self.cfg)

        self.mujoco_dynamics = MuJoCoDynamics(physical_params, self.cfg, self.sim_cfg)

        # Landing-gear standoff: the nominal landing-gear contact point
        # relative to CG, explicitly derived from geometry (never assumed
        # equal to ground_clearance_m) -- see the module docstring.
        self.r_landing_body = landing_reference_point_body_m(physical_params.geometry)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        high = np.array(
            [
                30.0, 30.0, 30.0,
                6.0, 6.0, 6.0,
                self.cfg.max_acceleration_obs_mps2,
                self.cfg.max_acceleration_obs_mps2,
                self.cfg.max_acceleration_obs_mps2,
                math.pi, math.pi, math.pi,
                1.0, 1.0, 1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, shape=(16,), dtype=np.float32)

        self.target_true = np.zeros(3, dtype=np.float64)
        self.target_measured = np.zeros(3, dtype=np.float64)

        self._vehicle_state = self._fresh_vehicle_state(
            np.zeros(3), np.zeros(3), np.zeros(3)
        )
        self.prev_action = np.zeros(3, dtype=np.float64)
        self.prev_potential = 0.0
        self.step_count = 0

        self.wind_accel = self.disturbance.wind_accel
        self.action_delay_steps = 0
        self.obs_delay_steps = 0
        self.obs_target = np.zeros(3, dtype=np.float64)
        self.obs_target_valid = True
        self.obs_target_mode = "init"

    def _fresh_vehicle_state(self, pos, vel, attitude) -> VehicleState:
        return VehicleState(
            pos=pos.astype(np.float64),
            vel=vel.astype(np.float64),
            accel=np.zeros(3, dtype=np.float64),
            prev_accel=np.zeros(3, dtype=np.float64),
            attitude=attitude.astype(np.float64),
            yaw_rate=0.0,
            body_rates=np.zeros(3, dtype=np.float64),
            thrust_accel=float(self.cfg.gravity_mps2),
            attitude_setpoint=attitude.astype(np.float64).copy(),
            thrust_accel_setpoint=float(self.cfg.gravity_mps2),
            accel_cmd=np.zeros(3, dtype=np.float64),
            ground_effect_factor=1.0,
        )

    # ------------------------------------------------------------------
    def _update_target_pipeline(self) -> None:
        raw_target, valid, mode = self.target_measurement_model.sample(self.np_random, self.target_true)
        delayed_target, delayed_valid, delayed_mode = self.obs_latency.push_and_get(raw_target, valid, mode)
        self.target_measured = delayed_target.copy()
        self.obs_target = delayed_target.copy()
        self.obs_target_valid = bool(delayed_valid)
        self.obs_target_mode = delayed_mode

    def _yaw_error(self) -> float:
        return wrap_pi(self.target_yaw - float(self._vehicle_state.attitude[2]))

    # ------------------------------------------------------------------
    # Landing-gear standoff: pad_target_ned vs. vehicle_touchdown_target_ned
    # ------------------------------------------------------------------
    def _touchdown_offset_ned(self) -> np.ndarray:
        """R_desired @ r_landing_body, where R_desired is the LEVEL
        (roll=pitch=0) touchdown attitude at this episode's fixed
        target_yaw. From the task's own derivation:
        p_landing^N = p_vehicle^N + R_B^N r_landing^B, evaluated at the
        desired touchdown pose, gives the vehicle-reference offset from
        the pad target."""
        r_desired = rotation_body_to_world_ned(0.0, 0.0, self.target_yaw)
        return r_desired @ self.r_landing_body

    def _vehicle_touchdown_target(self, pad_target_ned: np.ndarray) -> np.ndarray:
        """vehicle_touchdown_target_ned = pad_target_ned - R_desired @ r_landing_body."""
        return pad_target_ned - self._touchdown_offset_ned()

    def _altitude_agl(self) -> float:
        """Physical clearance of the landing-gear reference point above the
        ground plane -- a DIFFERENT concept from z_error. Uses the
        vehicle's CURRENT (not desired/level) attitude, since this is a
        physical clearance/contact-related quantity, not a task target: a
        tilted vehicle sitting exactly at the touchdown target z can still
        have nonzero altitude_agl."""
        st = self._vehicle_state
        r_current = rotation_body_to_world_ned(*st.attitude) @ self.r_landing_body
        gear_z = float(st.pos[2] + r_current[2])
        return max(0.0, float(self.cfg.ground_z_m - gear_z))

    def _get_obs(self) -> np.ndarray:
        st = self._vehicle_state
        # dx,dy,dz is vehicle-reference-to-touchdown-target error, not raw
        # pad-to-CG distance (landing-gear standoff fix).
        error_obs = self._vehicle_touchdown_target(self.obs_target) - st.pos

        vel_noise, accel_noise, attitude_noise = self.observation_noise.sample(self.np_random)
        vel_obs = st.vel + vel_noise
        accel_obs = np.clip(
            st.accel + accel_noise,
            -self.cfg.max_acceleration_obs_mps2,
            self.cfg.max_acceleration_obs_mps2,
        )
        attitude_obs = np.array([st.attitude[0], st.attitude[1], self._yaw_error()], dtype=np.float64)
        attitude_obs += attitude_noise
        attitude_obs[2] = wrap_pi(float(attitude_obs[2]))

        valid = 1.0 if self.obs_target_valid else 0.0
        obs = np.concatenate(
            [error_obs, vel_obs, accel_obs, attitude_obs, self.prev_action, np.array([valid])]
        ).astype(np.float32)
        return obs

    def _potential(self) -> float:
        err = self._vehicle_touchdown_target(self.target_true) - self._vehicle_state.pos
        xy = float(np.linalg.norm(err[:2]))
        z = abs(float(err[2]))
        speed = float(np.linalg.norm(self._vehicle_state.vel))
        return -(1.0 * xy + 0.6 * z + 0.15 * speed)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        pos, vel, attitude = self.initial_state_sampler.sample(self.np_random)
        self.prev_action = np.zeros(3, dtype=np.float64)
        self.target_yaw = float(self.cfg.target_yaw_rad)
        self._vehicle_state = self._fresh_vehicle_state(pos, vel, attitude)

        self.mujoco_dynamics.reset(pos, vel, attitude)

        self.prev_potential = self._potential()

        self.action_delay_steps = self.action_latency.reset(self.np_random)
        self.obs_delay_steps = self.obs_latency.reset(self.np_random)
        self.wind_accel = self.disturbance.reset(self.np_random)

        self.target_measurement_model.reset(self.target_true)
        self.obs_target = self.target_true.copy()
        self.obs_target_valid = True
        self.obs_target_mode = "reset"
        self.target_measured = self.obs_target.copy()

        for _ in range(max(1, self.obs_delay_steps + 1)):
            self._update_target_pipeline()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        st = self._vehicle_state

        old_action = self.prev_action.copy()
        raw_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action_delta = raw_action - old_action
        self.prev_action = raw_action.copy()

        dt = self.loop_timing.sample_dt(self.np_random)

        # PID/RL control against the vehicle-reference touchdown target,
        # not the raw (noisy/delayed) pad measurement (landing-gear
        # standoff fix).
        control_target = self._vehicle_touchdown_target(self.obs_target.copy())
        control_target_valid = bool(self.obs_target_valid)

        altitude_agl_before = self._altitude_agl()
        control_error = control_target - st.pos
        control_xy_error = float(np.linalg.norm(control_error[:2]))

        applied_action = self.action_latency.apply(raw_action)

        v_pid = self.controller.pid_velocity(
            control_target, control_target_valid, st.pos, st.vel, altitude_agl_before
        )
        if control_target_valid or self.cfg.apply_residual_when_target_invalid:
            v_residual = self.controller.scale_action(applied_action)
        else:
            v_residual = np.zeros(3, dtype=np.float64)
        v_cmd = self.controller.combine_and_limit(v_pid, v_residual)
        v_cmd = self.controller.apply_descent_gate(
            v_cmd, control_target_valid, control_xy_error, altitude_agl_before,
            self.mujoco_dynamics.contact_state.ground_contact,
        )

        motor_cutoff = self.mujoco_dynamics.contact_state.motor_cutoff
        ground_contact_prev = self.mujoco_dynamics.contact_state.ground_contact

        contact_result = self.mujoco_dynamics.step(
            st, v_cmd, dt, self.np_random, self.wind_accel, self.target_yaw,
            motor_cutoff, ground_contact_prev,
        )

        altitude_agl = self._altitude_agl()

        reward = 0.0
        potential = self._potential()
        progress_reward = self.cfg.w_progress * (potential - self.prev_potential)
        reward += progress_reward
        self.prev_potential = potential

        # Reward/success against the TRUE vehicle touchdown target (uses
        # the true, undelayed pad position, exactly mirroring the legacy
        # convention that reward/termination use ground truth while PID/RL
        # use delayed/noisy measurements -- only the standoff offset is new).
        vehicle_touchdown_target_true = self._vehicle_touchdown_target(self.target_true)
        true_error = vehicle_touchdown_target_true - st.pos
        xy_error = float(np.linalg.norm(true_error[:2]))
        z_error = abs(float(true_error[2]))
        vxy = float(np.linalg.norm(st.vel[:2]))
        vz_abs = abs(float(st.vel[2]))
        speed = float(np.linalg.norm(st.vel))
        accel_mag = float(np.linalg.norm(st.accel))
        jerk_mag = float(np.linalg.norm((st.accel - st.prev_accel) / max(float(dt), 1e-6)))
        roll = float(st.attitude[0])
        pitch = float(st.attitude[1])
        yaw = float(st.attitude[2])
        yaw_error = self._yaw_error()
        yaw_error_abs = abs(float(yaw_error))
        tilt = float(np.linalg.norm(st.attitude[:2]))

        reward += -self.cfg.w_xy * xy_error
        reward += -self.cfg.w_z * z_error
        reward += -self.cfg.w_speed * speed
        reward += -self.cfg.w_accel * accel_mag
        reward += -self.cfg.w_tilt * tilt
        reward += -self.cfg.w_yaw * yaw_error_abs
        reward += -self.cfg.w_action * float(np.linalg.norm(raw_action))
        reward += -self.cfg.w_action_delta * float(np.sum(action_delta**2))

        saturation = np.maximum(np.abs(raw_action) - 0.80, 0.0)
        reward += -self.cfg.w_saturation * float(np.sum(saturation**2))

        if xy_error < 0.5 and altitude_agl < 1.0:
            reward += 0.2
        if xy_error < 0.35 and altitude_agl > self.cfg.success_altitude_m:
            descent_rate = max(float(st.vel[2]), 0.0)
            safe_descent = min(descent_rate, 0.18)
            reward += 1.0 * safe_descent

        if z_error < 0.8:
            safe_vz = self.cfg.success_vz_mps
            excess_vz = max(vz_abs - safe_vz, 0.0)
            reward += -self.cfg.w_near_ground_vz * excess_vz
            reward += -self.cfg.w_near_ground_tilt * tilt

        if contact_result.contact_event:
            reward += -self.cfg.w_touchdown_impact * (contact_result.last_impact_vz**2)
            reward += -self.cfg.w_touchdown_lateral * (contact_result.last_touchdown_vxy**2)
            reward += -self.cfg.w_touchdown_tilt * (tilt**2)

        if contact_result.bounced:
            reward += -self.cfg.w_bounce * (1.0 + contact_result.last_bounce_speed)

        kinematic_success = (
            xy_error < self.cfg.success_xy_m
            and z_error < self.cfg.success_altitude_m
            and vxy < self.cfg.success_vxy_mps
            and vz_abs < self.cfg.success_vz_mps
            and tilt < self.cfg.success_tilt_rad
            and yaw_error_abs < self.cfg.success_yaw_error_rad
        )
        contact_success = (
            self.mujoco_dynamics.contact_state.ground_contact
            and not contact_result.bounced
            and not contact_result.hard_contact
            and contact_result.touchdown_quality == "soft"
            and self.mujoco_dynamics.contact_state.contact_count >= self.cfg.contact_success_hold_steps
            and kinematic_success
        )
        success = contact_success if self.cfg.success_requires_contact else kinematic_success

        failure_reason = "none"
        altitude = -float(st.pos[2])
        if xy_error > self.cfg.max_xy_error_m:
            failure_reason = "xy_error_limit"
        elif altitude > self.cfg.max_altitude_m:
            failure_reason = "altitude_limit"
        elif speed > self.cfg.max_speed_mps:
            failure_reason = "speed_limit"
        elif tilt > self.cfg.max_tilt_rad:
            failure_reason = "attitude_tilt_limit"
        elif contact_result.hard_contact:
            failure_reason = "hard_landing"
        elif self.mujoco_dynamics.contact_state.bounce_count > self.cfg.max_bounce_count:
            failure_reason = "excessive_bounce"
        elif st.pos[2] > self.cfg.below_ground_limit_m:
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
            "parameter_set": self.physical_params.parameter_set.value,
            "vehicle_name": self.physical_params.vehicle_name,
            "xy_error": xy_error, "z_error": z_error, "vxy": vxy, "vz_abs": vz_abs,
            "accel_mag": accel_mag, "jerk_mag": jerk_mag,
            "roll": roll, "pitch": pitch, "yaw": yaw, "yaw_error": yaw_error, "tilt": tilt,
            "body_rates": st.body_rates.copy(),
            "thrust_accel": st.thrust_accel,
            "pos": st.pos.copy(), "vel": st.vel.copy(), "accel": st.accel.copy(),
            "success": success, "failed": failed, "failure_reason": failure_reason,
            "target_true": self.target_true.copy(), "target_measured": self.target_measured.copy(),
            "pad_target_ned": self.target_true.copy(),
            "vehicle_touchdown_target_ned": vehicle_touchdown_target_true.copy(),
            "target_valid": control_target_valid, "target_mode": self.obs_target_mode,
            "raw_action": raw_action.copy(), "applied_action": applied_action.copy(),
            "v_pid": v_pid.copy(), "v_residual": v_residual.copy(), "v_cmd": v_cmd.copy(),
            "ground_contact": self.mujoco_dynamics.contact_state.ground_contact,
            "contact_count": self.mujoco_dynamics.contact_state.contact_count,
            "bounce_count": self.mujoco_dynamics.contact_state.bounce_count,
            "touchdown_quality": contact_result.touchdown_quality,
            "last_impact_vz": contact_result.last_impact_vz,
            "last_touchdown_vxy": contact_result.last_touchdown_vxy,
            "last_bounce_speed": contact_result.last_bounce_speed,
            "motor_cutoff": self.mujoco_dynamics.contact_state.motor_cutoff,
            "altitude_agl": altitude_agl,
            "action_delay_steps": self.action_delay_steps,
            "obs_delay_steps": self.obs_delay_steps,
            "dt": dt,
            "wind_accel": self.wind_accel.copy(),
            "ground_effect_factor": st.ground_effect_factor,
            # MuJoCo-specific diagnostics (task spec section 22).
            "mujoco_qpos": self.mujoco_dynamics.data.qpos.copy(),
            "mujoco_qvel": self.mujoco_dynamics.data.qvel.copy(),
            "applied_body_force": self.mujoco_dynamics.last_applied_force_world.copy(),
            "applied_body_torque": self.mujoco_dynamics.last_applied_torque_world.copy(),
            "collective_thrust": self.mujoco_dynamics.last_collective_thrust_n,
            "desired_collective_thrust": self.mujoco_dynamics.last_desired_collective_thrust_n,
            "desired_body_rates": self.mujoco_dynamics.last_desired_body_rates.copy(),
            "delayed_body_rate_command": self.mujoco_dynamics.last_delayed_body_rate_command.copy(),
            "contact_force": float(np.sum(np.abs(self.mujoco_dynamics.data.efc_force)))
            if self.mujoco_dynamics.data.ncon > 0
            else 0.0,
            "number_of_contacts": int(self.mujoco_dynamics.data.ncon),
            "physics_dt": self.sim_cfg.physics_dt,
            "physics_substeps": self.sim_cfg.physics_substeps(dt),
        }

        self._update_target_pipeline()
        return self._get_obs(), float(reward), terminated, truncated, info
