"""Identified closed-loop inner-loop response math.

Pure functions, no MuJoCo, no RNG, no I/O -- these implement exactly the
model described in the task spec section 9/10:

    omega_dot_des = (K_axis * omega_sp_delayed - omega_axis) / tau_axis
    tau_body = J * omega_dot_des + omega x (J * omega)

    dT/dt = (K_thrust * T_sp_delayed - T) / tau_thrust   (T in accel units,
                                                            m/s^2, consistent
                                                            with the existing
                                                            LandingConfig
                                                            thrust_accel_*
                                                            semantics)

Everything here operates in the FRD/NED policy convention (angles in
radians, rates in rad/s, "thrust" as specific force in m/s^2) -- the ONLY
place that knows about MuJoCo's own frame is
``landing_mujoco/coordinates/transforms.py``, applied by the caller
(``mujoco_dynamics.py``) at the point forces/torques are handed to MuJoCo.

Attitude-to-rate surrogate
--------------------------
``InnerLoopCommandModel`` (reused unchanged from ``landing_rl``) produces an
ATTITUDE setpoint, not a body-rate setpoint. Per the task spec ("If the
existing controller naturally produces attitude setpoints rather than
body-rate setpoints, create an explicit attitude-to-rate outer inner-loop
surrogate before applying the identified rate dynamics"),
``attitude_error_to_rate_cmd`` below is exactly the outer cascade already
used by the legacy environment
(``landing_rl/dynamics/legacy_dynamics.py::advance_free_flight``, the
"Attitude/rate controller proxy" block) -- reused via the same
``LandingConfig`` fields (``attitude_time_constant_s``,
``max_roll_pitch_rate_radps``, ``yaw_align_kp``,
``max_yaw_rate_response_radps``). These are CONTROL-tuning parameters (not
physical/identified) and are left untouched, per the structural-freeze rule.
"""

from __future__ import annotations

import numpy as np

from landing_rl.envs.landing_env import wrap_pi


def attitude_error_to_rate_cmd(attitude: np.ndarray, attitude_setpoint: np.ndarray, cfg) -> np.ndarray:
    """(roll, pitch, yaw) attitude error -> desired body rate (p, q, r).

    Verbatim reuse of the legacy attitude/rate controller proxy's math
    (``LegacyVehicleDynamics.advance_free_flight``), parameterized by the
    existing (unmodified) ``LandingConfig`` control-tuning fields.
    """
    att_error = np.array(
        [
            attitude_setpoint[0] - attitude[0],
            attitude_setpoint[1] - attitude[1],
            wrap_pi(float(attitude_setpoint[2] - attitude[2])),
        ],
        dtype=np.float64,
    )
    tau_c = max(float(cfg.attitude_time_constant_s), 1e-6)
    rate_cmd = att_error / tau_c
    rate_cmd[0] = float(np.clip(rate_cmd[0], -cfg.max_roll_pitch_rate_radps, cfg.max_roll_pitch_rate_radps))
    rate_cmd[1] = float(np.clip(rate_cmd[1], -cfg.max_roll_pitch_rate_radps, cfg.max_roll_pitch_rate_radps))
    rate_cmd[2] = float(
        np.clip(
            cfg.yaw_align_kp * att_error[2],
            -cfg.max_yaw_rate_response_radps,
            cfg.max_yaw_rate_response_radps,
        )
    )
    return rate_cmd


def rate_response_omega_dot(
    omega: np.ndarray,
    rate_cmd_delayed: np.ndarray,
    tau_axes: np.ndarray,
    k_axes: np.ndarray,
) -> np.ndarray:
    """omega_dot_des[i] = (K_i * rate_cmd_delayed[i] - omega[i]) / tau_i, per
    axis (roll, pitch, yaw)."""
    tau_safe = np.clip(np.asarray(tau_axes, dtype=np.float64), 1e-6, None)
    return (np.asarray(k_axes, dtype=np.float64) * np.asarray(rate_cmd_delayed, dtype=np.float64) - np.asarray(omega, dtype=np.float64)) / tau_safe


def body_torque_from_rate_response(j_diag: np.ndarray, omega: np.ndarray, omega_dot_des: np.ndarray) -> np.ndarray:
    """tau_body = J * omega_dot_des + omega x (J * omega), J diagonal.

    Because ``tau_body`` already includes the gyroscopic/Coriolis term
    ``omega x (J omega)``, applying it as the external torque to a MuJoCo
    rigid body whose own equations of motion are ``J*alpha + omega x (J
    omega) = tau_applied`` makes the ACTUAL angular acceleration equal
    ``omega_dot_des`` -- i.e. the identified rate law becomes the vehicle's
    real dynamics, rather than fighting MuJoCo's own rigid-body terms.
    """
    j_diag = np.asarray(j_diag, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    j_omega = j_diag * omega
    coriolis = np.cross(omega, j_omega)
    return j_diag * np.asarray(omega_dot_des, dtype=np.float64) + coriolis


def thrust_response_step(
    thrust_accel: float,
    thrust_sp_delayed: float,
    tau_thrust: float,
    k_thrust: float,
    physics_dt: float,
    min_thrust_accel: float,
    max_thrust_accel: float,
) -> float:
    """One explicit-Euler step of dT/dt = (K*T_sp_delayed - T) / tau_thrust,
    in specific-force units (m/s^2), then clipped to
    [min_thrust_accel, max_thrust_accel] (the latter being
    min(cfg.max_thrust_accel_mps2, T_max/mass) -- see actuation_model.py)."""
    tau_c = max(float(tau_thrust), 1e-6)
    d_thrust = (float(k_thrust) * float(thrust_sp_delayed) - thrust_accel) / tau_c
    updated = thrust_accel + d_thrust * float(physics_dt)
    return float(np.clip(updated, min_thrust_accel, max_thrust_accel))


def force_and_point_torque(r_body: np.ndarray, force_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a force applied AT body-frame point ``r_body`` (relative to
    CG), return the equivalent (force_at_cg, torque_at_cg) pair:

        F_cg = F
        tau_cg = r_body x F

    Pure rigid-body statics, independent of MuJoCo -- used by
    ``test_cg_force_moment.py`` and reserved for a future rotor-level
    actuation model (task spec section 13)."""
    r_body = np.asarray(r_body, dtype=np.float64)
    force_body = np.asarray(force_body, dtype=np.float64)
    torque = np.cross(r_body, force_body)
    return force_body.copy(), torque
