"""Actuation model abstraction (task spec section 13).

    ActuationModel (Protocol)
        IdentifiedWrenchActuation   -- v0, collective force + body torque
        RotorLevelActuation         -- NOT implemented (reserved)

v0 deliberately applies a single collective thrust force + a single body
torque (no per-rotor force/torque decomposition) because no measured
thrust/torque coefficient exists for any rotor on this vehicle -- inventing
one would violate CLAUDE.md's "do not invent precise rotor coefficients"
rule. Motor positions and spin directions ARE still recorded (in
``UAVPhysicalParams.motors``) so a future ``RotorLevelActuation`` can be
built without re-measuring anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from landing_mujoco.configs.param_schema import UAVPhysicalParams
from landing_mujoco.dynamics.delay_buffer import DelayBuffer, steps_from_seconds
from landing_mujoco.dynamics.identified_inner_loop import (
    attitude_error_to_rate_cmd,
    body_torque_from_rate_response,
    rate_response_omega_dot,
    thrust_response_step,
)


@dataclass
class ActuationOutput:
    """One control-step-averaged summary, used for MuJoCo diagnostics
    (task spec section 22): the wrench applied at the FINAL physics
    substep of the step, plus the setpoints that produced it."""

    torque_body_frd: np.ndarray
    thrust_accel: float
    desired_body_rates: np.ndarray
    delayed_body_rate_command: np.ndarray
    desired_collective_thrust_n: float
    collective_thrust_n: float


class ActuationModel(Protocol):
    def reset(self) -> None: ...

    def compute_substep(
        self,
        *,
        attitude: np.ndarray,
        attitude_setpoint: np.ndarray,
        omega_frd: np.ndarray,
        thrust_accel_setpoint: float,
        physics_dt: float,
        control_cfg,
        motor_cutoff: bool = False,
    ) -> ActuationOutput: ...


class IdentifiedWrenchActuation:
    """Identified closed-loop response (task spec sections 9/10): attitude
    error -> rate command (outer surrogate, reused control-tuning cascade)
    -> identified per-axis rate response (K/tau/T_delay) -> body torque via
    measured/provisional inertia -> plus an identified first-order thrust
    response, clipped to the vehicle's real T_max.

    Owns: per-axis delay buffers (roll/pitch/yaw rate command + thrust
    setpoint) and the thrust actuator's persistent state (a scalar, exactly
    like the legacy ``VehicleState.thrust_accel`` -- MuJoCo has no notion of
    this, it is genuinely actuator-internal state).
    """

    def __init__(self, params: UAVPhysicalParams, physics_dt: float):
        self.params = params
        ir = params.identified_response
        mp = params.mass_properties

        self.j_diag = np.array([mp.ixx, mp.iyy, mp.izz], dtype=np.float64)
        self.tau_roll = float(ir.tau_roll_s)
        self.tau_pitch = float(ir.tau_pitch_s)
        self.tau_thrust = float(ir.tau_thrust_s)
        # No yaw system identification is assumed available (task spec
        # section 12): fall back to the existing (unmodified) attitude
        # outer-loop time constant as a conservative, documented
        # approximation rather than inventing a yaw tau.
        self.tau_yaw = float(ir.tau_yaw_s) if ir.tau_yaw_s is not None else None
        self.k_roll = ir.gain("roll")
        self.k_pitch = ir.gain("pitch")
        self.k_thrust = ir.gain("thrust")
        self.k_yaw = ir.gain("yaw")

        self.mass_kg = float(mp.mass_kg)
        self.max_thrust_accel_physical = float(params.thrust.max_collective_thrust_n) / self.mass_kg

        delay_steps = steps_from_seconds(ir.delay_s, physics_dt)
        self._delay_roll = DelayBuffer(delay_steps, 0.0)
        self._delay_pitch = DelayBuffer(delay_steps, 0.0)
        self._delay_yaw = DelayBuffer(delay_steps, 0.0)
        self._delay_thrust = DelayBuffer(delay_steps, float(mp.mass_kg) * 9.80665 / self.mass_kg)

        self._hover_accel = 9.80665
        self.thrust_accel = self._hover_accel

    def reset(self, hover_accel: float) -> None:
        self._hover_accel = float(hover_accel)
        self.thrust_accel = self._hover_accel
        self._delay_roll.reset(0.0)
        self._delay_pitch.reset(0.0)
        self._delay_yaw.reset(0.0)
        self._delay_thrust.reset(self._hover_accel)

    def compute_substep(
        self,
        *,
        attitude: np.ndarray,
        attitude_setpoint: np.ndarray,
        omega_frd: np.ndarray,
        thrust_accel_setpoint: float,
        physics_dt: float,
        control_cfg,
        motor_cutoff: bool = False,
    ) -> ActuationOutput:
        rate_cmd = attitude_error_to_rate_cmd(attitude, attitude_setpoint, control_cfg)

        delayed_rate_cmd = np.array(
            [
                self._delay_roll.push_and_get(rate_cmd[0]),
                self._delay_pitch.push_and_get(rate_cmd[1]),
                self._delay_yaw.push_and_get(rate_cmd[2]),
            ],
            dtype=np.float64,
        )

        yaw_tau = self.tau_yaw if self.tau_yaw is not None else max(
            float(control_cfg.attitude_time_constant_s), 1e-6
        )
        tau_axes = np.array([self.tau_roll, self.tau_pitch, yaw_tau], dtype=np.float64)
        k_axes = np.array([self.k_roll, self.k_pitch, self.k_yaw], dtype=np.float64)

        omega_dot_des = rate_response_omega_dot(omega_frd, delayed_rate_cmd, tau_axes, k_axes)
        torque = body_torque_from_rate_response(self.j_diag, omega_frd, omega_dot_des)

        delayed_thrust_sp = self._delay_thrust.push_and_get(float(thrust_accel_setpoint))

        max_thrust_accel = min(
            float(control_cfg.max_thrust_accel_mps2), self.max_thrust_accel_physical
        )
        # Motor-cutoff (post soft-touchdown) lowers the floor to the
        # cutoff value instead of the normal-operation minimum -- matching
        # the legacy behavior (LegacyVehicleDynamics.advance_free_flight's
        # thrust_min selection) -- otherwise thrust could never actually
        # reach zero after touchdown, nor in a free-fall/motor-cutoff test.
        min_thrust_accel = (
            float(control_cfg.motor_cutoff_thrust_accel_mps2)
            if motor_cutoff
            else max(float(control_cfg.min_thrust_accel_mps2), 0.0)
        )

        self.thrust_accel = thrust_response_step(
            self.thrust_accel,
            delayed_thrust_sp,
            self.tau_thrust,
            self.k_thrust,
            physics_dt,
            min_thrust_accel,
            max_thrust_accel,
        )

        return ActuationOutput(
            torque_body_frd=torque,
            thrust_accel=self.thrust_accel,
            desired_body_rates=rate_cmd,
            delayed_body_rate_command=delayed_rate_cmd,
            desired_collective_thrust_n=float(thrust_accel_setpoint) * self.mass_kg,
            collective_thrust_n=self.thrust_accel * self.mass_kg,
        )


class RotorLevelActuation:
    """Reserved for a future per-rotor force/torque model once thrust/torque
    coefficients are actually measured. NOT implemented in v0 -- do not
    fabricate rotor coefficients to make this work."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RotorLevelActuation requires measured rotor thrust/torque "
            "coefficients, which do not exist yet for this vehicle. Use "
            "IdentifiedWrenchActuation for v0."
        )
