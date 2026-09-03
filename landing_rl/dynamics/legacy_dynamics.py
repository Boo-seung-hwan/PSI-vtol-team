"""Legacy free-flight / PX4-proxy rigid-body dynamics.

Phase 13C-1 extraction. This module holds the FREE-FLIGHT portion of
``LandingEnv._update_rigid_body_dynamics`` lifted verbatim from
``mujoco_rl/envs/env_prototype.py`` (the immutable reference), specifically the
D4-D25 span of the Phase-12 execution audit:

    _velocity_command_to_inner_loop_setpoints   (PX4-cascade proxy)
    motor-cutoff thrust-setpoint override
    attitude/rate controller proxy + body-rate first-order response
    body-rate process noise            (ProcessNoiseSampler, draw 1 of 3)
    attitude integration + yaw_rate
    collective-thrust first-order response
    thrust process noise               (ProcessNoiseSampler, draw 2 of 3)
    motor-cutoff-aware thrust clipping
    body-to-NED rotation, gravity, ground effect, drag, wind
    translational process noise        (ProcessNoiseSampler, draw 3 of 3)
    prev_accel snapshot
    provisional accel / velocity / position integration

Copied character-for-character, with only:
    self.<vehicle field>  ->  state.<field>          (VehicleState, mutated in place)
    self.np_random        ->  rng                    (LandingEnv.np_random, passed in)
    self.target_yaw       ->  target_yaw             (explicit scalar input)
    self.wind_accel       ->  wind_accel             (DisturbanceModel output, passed in)
    self.body_rate_response_alpha / self.thrust_response_alpha
                          ->  explicit inputs
    self.motor_cutoff     ->  motor_cutoff           (explicit bool from ContactState)
    self.ground_contact   ->  ground_contact         (explicit bool: the PREVIOUS /
                                                      persistent contact state that
                                                      ground effect reads at D18)
    self._reset_contact_step_flags / _rotation_body_to_ned /
    _velocity_command_to_inner_loop_setpoints / _compute_ground_effect_factor /
    _altitude_agl  ->  local equivalents

What is NOT here (Phase 13C-1 boundary -- these stay in ``LandingEnv``)
--------------------------------------------------------------------
* ``contact.begin_step()`` (D2) and the ``vel_before`` snapshot.
* ``ContactModel.apply(...)`` (D26).
* The post-contact final-acceleration recompute
  ``state.accel = (state.vel - vel_before) / dt_safe`` (D27).
* ``PlantModel`` -- not created in this phase. ``LandingEnv`` still orchestrates
  begin_step -> vel_before -> advance_free_flight -> contact.apply -> finalize.

Ownership
---------
``LegacyVehicleDynamics`` owns ONLY ``cfg`` and the already-extracted
``ProcessNoiseSampler`` (``process_noise``). It does NOT own an RNG, a
``VehicleState``, a ``ContactModel``, a ``LandingEnv`` reference, wind state,
response-alpha state, or target state. ``noise_scale`` is a local of
``advance_free_flight`` (unchanged expression
``sqrt(dt_safe / max(cfg.dt, 1e-6))``); the sampler still receives it as an
argument. ``cfg.attitude_process_noise_std_rad`` remains unused -- the legacy
dynamics performs exactly three process-noise draws.

No behavior change. The OLD-vs-NEW exact parity gate
(``test_legacy_regression_contract.py``) and the PPO / VecNormalize checkpoint
gate (``test_checkpoint_compatibility.py``) remain authoritative.
"""

from __future__ import annotations

import math

import numpy as np

from landing_rl.dynamics.vehicle_state import VehicleState


def wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]. Verbatim copy of the env-module helper."""
    return math.atan2(math.sin(angle), math.cos(angle))


class LegacyVehicleDynamics:
    """Free-flight rigid-body integrator + PX4-cascade proxy. Stateless beyond
    ``cfg`` and ``process_noise``; ``advance_free_flight`` mutates the
    ``VehicleState`` handed in and takes ``LandingEnv.np_random`` as ``rng``."""

    def __init__(self, cfg, process_noise):
        self.cfg = cfg
        self.process_noise = process_noise

    # ------------------------------------------------------------------
    # verbatim helpers (self.<field> -> state.<field>)
    # ------------------------------------------------------------------
    def _altitude_agl(self, state: VehicleState) -> float:
        """Altitude above ground in meters. In NED, negative z is above ground.
        Verbatim copy of ``LandingEnv._altitude_agl`` (``self.pos`` ->
        ``state.pos``)."""
        return max(0.0, float(self.cfg.ground_z_m - state.pos[2]))

    def _compute_ground_effect_factor(
        self, state: VehicleState, ground_contact: bool
    ) -> float:
        """Return thrust multiplier caused by ground effect near the pad.

        Verbatim copy of ``LandingEnv._compute_ground_effect_factor`` with
        ``self.ground_contact`` -> the explicit ``ground_contact`` input. That
        bool is the PREVIOUS / persistent contact state (this step's D26 contact
        resolution has not run yet), exactly as at the legacy D18 call site.
        """
        h = self._altitude_agl(state)
        height = max(float(self.cfg.ground_effect_height_m), 1e-6)

        if h >= height or ground_contact:
            return 1.0

        closeness = 1.0 - h / height
        factor = 1.0 + float(self.cfg.ground_effect_gain) * closeness * closeness
        return float(np.clip(factor, 1.0, self.cfg.ground_effect_max_factor))

    def _rotation_body_to_ned(self, state: VehicleState) -> np.ndarray:
        """Body-to-NED rotation matrix using aerospace roll/pitch/yaw. Verbatim
        copy of ``LandingEnv._rotation_body_to_ned`` (``self.attitude`` ->
        ``state.attitude``)."""
        roll, pitch, yaw = [float(v) for v in state.attitude]
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
        state: VehicleState,
        v_cmd: np.ndarray,
        dt: float,
        target_yaw: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Convert velocity command to attitude/thrust setpoints.

        Verbatim copy of ``LandingEnv._velocity_command_to_inner_loop_setpoints``
        with ``self.vel`` -> ``state.vel`` and ``self.target_yaw`` ->
        ``target_yaw``.

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
        accel_cmd = (np.asarray(v_cmd, dtype=np.float64) - state.vel) / max(
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

        yaw_sp = float(target_yaw)
        thrust_sp = float(np.clip(
            g - float(accel_cmd[2]),
            self.cfg.min_thrust_accel_mps2,
            self.cfg.max_thrust_accel_mps2,
        ))

        attitude_sp = np.array([roll_sp, pitch_sp, yaw_sp], dtype=np.float64)
        return attitude_sp, thrust_sp, accel_cmd

    # ------------------------------------------------------------------
    # free-flight integration -- verbatim D4-D25 of _update_rigid_body_dynamics
    # ------------------------------------------------------------------
    def advance_free_flight(
        self,
        state: VehicleState,
        v_cmd: np.ndarray,
        dt: float,
        rng,
        wind_accel: np.ndarray,
        target_yaw: float,
        body_rate_response_alpha: np.ndarray,
        thrust_response_alpha: float,
        motor_cutoff: bool,
        ground_contact: bool,
    ) -> None:
        """Advance attitude, thrust, acceleration, velocity, and position for
        one control step, up to but NOT including ground contact.

        ``state`` is mutated in place. ``rng`` must be ``LandingEnv.np_random``;
        it is consumed by exactly the three ``ProcessNoiseSampler`` draws
        (body-rate ``normal(size=3)`` -> thrust scalar ``normal`` ->
        translational ``normal(size=3)``), in that order.

        This is a compact 6DoF-inspired plant model. It does not simulate motor
        mixing or rotor angular momentum, but it exposes the key delay chain:
            v_cmd -> attitude/thrust setpoint -> body-rate/thrust response
            -> thrust vector in NED -> acceleration -> velocity -> position.
        """
        dt_safe = max(float(dt), 1e-6)
        g = float(self.cfg.gravity_mps2)

        attitude_sp, thrust_sp, accel_cmd = self._velocity_command_to_inner_loop_setpoints(
            state, v_cmd, dt_safe, target_yaw
        )
        if motor_cutoff:
            thrust_sp = float(self.cfg.motor_cutoff_thrust_accel_mps2)

        state.attitude_setpoint = attitude_sp.copy()
        state.thrust_accel_setpoint = float(thrust_sp)
        state.accel_cmd = accel_cmd.copy()

        # Attitude/rate controller proxy.
        att_error = np.array(
            [
                attitude_sp[0] - state.attitude[0],
                attitude_sp[1] - state.attitude[1],
                wrap_pi(float(attitude_sp[2] - state.attitude[2])),
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
            1.0 - np.clip(body_rate_response_alpha, 1e-4, 0.9999),
            dt_safe / max(self.cfg.dt, 1e-6),
        )
        rate_alpha_eff = np.clip(rate_alpha_eff, 0.0, 1.0)

        state.body_rates = state.body_rates + rate_alpha_eff * (rate_cmd - state.body_rates)

        noise_scale = math.sqrt(dt_safe / max(self.cfg.dt, 1e-6))
        # Phase 11: body-rate process noise (draw 1 of 3). Verbatim expression;
        # noise_scale stays computed here.
        state.body_rates += self.process_noise.sample_body_rate(
            rng, noise_scale
        )

        state.attitude[0] = float(np.clip(
            state.attitude[0] + state.body_rates[0] * dt_safe,
            -math.pi,
            math.pi,
        ))
        state.attitude[1] = float(np.clip(
            state.attitude[1] + state.body_rates[1] * dt_safe,
            -math.pi,
            math.pi,
        ))
        state.attitude[2] = wrap_pi(float(state.attitude[2] + state.body_rates[2] * dt_safe))
        state.yaw_rate = float(state.body_rates[2])

        # Collective thrust/motor response proxy.
        thrust_alpha_eff = 1.0 - (1.0 - np.clip(thrust_response_alpha, 1e-4, 0.9999)) ** (
            dt_safe / max(self.cfg.dt, 1e-6)
        )
        thrust_alpha_eff = float(np.clip(thrust_alpha_eff, 0.0, 1.0))
        state.thrust_accel = float(
            state.thrust_accel
            + thrust_alpha_eff * (thrust_sp - state.thrust_accel)
            + self.process_noise.sample_thrust(rng, noise_scale)  # draw 2 of 3 (scalar)
        )
        thrust_min = (
            float(self.cfg.motor_cutoff_thrust_accel_mps2)
            if motor_cutoff
            else float(self.cfg.min_thrust_accel_mps2)
        )
        state.thrust_accel = float(np.clip(
            state.thrust_accel,
            thrust_min,
            self.cfg.max_thrust_accel_mps2,
        ))

        # Rigid-body translational dynamics in NED.
        r_bn = self._rotation_body_to_ned(state)
        body_z_in_ned = r_bn[:, 2]
        gravity_accel = np.array([0.0, 0.0, g], dtype=np.float64)

        state.ground_effect_factor = self._compute_ground_effect_factor(
            state, ground_contact
        )
        effective_thrust_accel = state.thrust_accel * state.ground_effect_factor
        thrust_accel_ned = -effective_thrust_accel * body_z_in_ned
        drag_accel = -np.array(
            [
                self.cfg.linear_drag_xy * state.vel[0],
                self.cfg.linear_drag_xy * state.vel[1],
                self.cfg.linear_drag_z * state.vel[2],
            ],
            dtype=np.float64,
        )
        # Phase 11: translational-acceleration process noise (draw 3 of 3).
        process_accel_noise = self.process_noise.sample_translational_accel(
            rng, noise_scale, dt_safe
        )

        state.prev_accel = state.accel.copy()
        state.accel = gravity_accel + thrust_accel_ned + drag_accel + wind_accel + process_accel_noise

        state.vel = state.vel + state.accel * dt_safe
        state.pos = state.pos + state.vel * dt_safe
