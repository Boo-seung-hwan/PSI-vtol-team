"""Version-tolerant signal schema for the SI preprocessing pipeline.

Scope is deliberately narrow: only the signals the current SI plan needs
(velocity-controller telemetry, attitude loop, rate loop, translational /
thrust support). It is NOT a general PX4 ingester.

Observed PX4-version differences that this schema tolerates (see the
2026-09-07 read-only ULog audit, four OTHER-PROJECT sample logs on PX4
``main`` / ``v1.17.0-alpha``):

  * ``vehicle_attitude_setpoint`` has only ``q_d`` + ``thrust_body`` (no
    ``roll_body`` / ``pitch_body`` / ``yaw_body``). Euler setpoints are ALWAYS
    derived from ``q_d`` here, never read from a version-specific field.
  * ``vehicle_attitude`` Euler is always derived from ``q``.
  * ``vehicle_rates_setpoint`` carries scalar ``roll`` / ``pitch`` / ``yaw``
    (rad/s). An array form, if a future log has one, is handled by
    ``rates_setpoint_fields``.
  * ``esc_status`` may be entirely absent -> optional, never fatal.
  * ``battery_status`` and ``vehicle_local_position.dist_bottom`` are treated
    as optional support signals (filled with NaN when unavailable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

SCHEMA_VERSION = "0.2.0"          # 0.2.0: block-specific masks; setpoint-pairing
#                                   validity; spec_thrust_recon renamed to
#                                   body_z_specific_force_proxy + near_level gate
PREPROCESSING_VERSION = "0.2.0"

# Tilt below which the body-z specific force is ~ the vertical specific thrust.
# Mirrored in masks.NEAR_LEVEL_MAX_RAD (single source here).
NEAR_LEVEL_MAX_RAD = math.radians(10.0)
NEAR_LEVEL_MAX_RAD_DEG = 10.0

# ---------------------------------------------------------------------------
# Frame / unit vocabulary (kept as plain strings; see frames.py for helpers)
# ---------------------------------------------------------------------------
FRAME_NED = "NED"          # earth-fixed North-East-Down
FRAME_FRD = "FRD"          # body Forward-Right-Down
FRAME_QUAT_NED_FROM_BODY = "q(NED<-FRD)"
FRAME_NONE = "-"

# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------
# Topics that MUST be present for the pipeline to run at all.
REQUIRED_TOPICS = (
    "vehicle_local_position",
    "vehicle_local_position_setpoint",
    "vehicle_attitude",
    "vehicle_attitude_setpoint",
    "vehicle_angular_velocity",
    "vehicle_rates_setpoint",
    "vehicle_acceleration",
    "actuator_motors",
    "vehicle_control_mode",
    "vehicle_land_detected",
)

# Topics that improve the datasets but must never cause a hard failure.
OPTIONAL_TOPICS = (
    "trajectory_setpoint",       # provenance only (position-mode sample logs)
    "vehicle_thrust_setpoint",   # redundant with attitude_setpoint.thrust_body on a quad
    "vehicle_torque_setpoint",
    "hover_thrust_estimate",
    "offboard_control_mode",
    "actuator_outputs",
    "battery_status",
    "esc_status",
)

ALL_TOPICS = REQUIRED_TOPICS + OPTIONAL_TOPICS

# Measurement topics whose ``timestamp_sample`` (when present) is the physical
# sample instant and should be preferred over ``timestamp``.
PREFER_TIMESTAMP_SAMPLE = frozenset({
    "vehicle_local_position",
    "vehicle_attitude",
    "vehicle_angular_velocity",
    "vehicle_acceleration",
    "vehicle_thrust_setpoint",
    "vehicle_torque_setpoint",
    "actuator_motors",
    "hover_thrust_estimate",
})

# ---------------------------------------------------------------------------
# Loop definitions -- each gets its OWN canonical timebase. The pipeline never
# resamples every signal onto the single fastest rate.
# ---------------------------------------------------------------------------
LOOP_VELOCITY = "velocity"        # ~10 Hz, master = vehicle_local_position
LOOP_ATTITUDE = "attitude"        # ~20 Hz, master = vehicle_attitude
LOOP_RATE = "rate"                # ~50 Hz, master = vehicle_angular_velocity
LOOP_TRANSLATION = "translation"  # ~20 Hz, master = vehicle_acceleration

LOOP_MASTER_TOPIC = {
    LOOP_VELOCITY: "vehicle_local_position",
    LOOP_ATTITUDE: "vehicle_attitude",
    LOOP_RATE: "vehicle_angular_velocity",
    LOOP_TRANSLATION: "vehicle_acceleration",
}

LOOP_NOMINAL_HZ = {
    LOOP_VELOCITY: 10.0,
    LOOP_ATTITUDE: 20.0,
    LOOP_RATE: 50.0,
    LOOP_TRANSLATION: 20.0,
}

ALL_LOOPS = (LOOP_VELOCITY, LOOP_ATTITUDE, LOOP_RATE, LOOP_TRANSLATION)


@dataclass(frozen=True)
class ColumnSpec:
    """One derived-dataset column: where it comes from and what it means."""

    name: str
    unit: str
    frame: str
    source: str          # human-readable "topic.field" or derivation description
    optional: bool = False


# ---------------------------------------------------------------------------
# Per-loop column specs. These describe the OUTPUT dataframe columns. The
# actual array assembly lives in dataset.py; this is the single source of
# truth for (name, unit, frame, source).
# ---------------------------------------------------------------------------
VELOCITY_COLUMNS = (
    ColumnSpec("t", "s", FRAME_NONE, "master timebase (vehicle_local_position stamp - t0)"),
    ColumnSpec("v_sp_N", "m/s", FRAME_NED, "vehicle_local_position_setpoint.vx (PX4 _vel_sp, post position-P)"),
    ColumnSpec("v_sp_E", "m/s", FRAME_NED, "vehicle_local_position_setpoint.vy"),
    ColumnSpec("v_sp_D", "m/s", FRAME_NED, "vehicle_local_position_setpoint.vz"),
    ColumnSpec("v_N", "m/s", FRAME_NED, "vehicle_local_position.vx"),
    ColumnSpec("v_E", "m/s", FRAME_NED, "vehicle_local_position.vy"),
    ColumnSpec("v_D", "m/s", FRAME_NED, "vehicle_local_position.vz"),
    ColumnSpec("velocity_error_N", "m/s", FRAME_NED, "v_sp_N - v_N"),
    ColumnSpec("velocity_error_E", "m/s", FRAME_NED, "v_sp_E - v_E"),
    ColumnSpec("velocity_error_D", "m/s", FRAME_NED, "v_sp_D - v_D"),
    ColumnSpec("a_sp_N", "m/s^2", FRAME_NED, "vehicle_local_position_setpoint.acceleration[0] (PX4 accel demand)"),
    ColumnSpec("a_sp_E", "m/s^2", FRAME_NED, "vehicle_local_position_setpoint.acceleration[1]"),
    ColumnSpec("a_sp_D", "m/s^2", FRAME_NED, "vehicle_local_position_setpoint.acceleration[2] (may carry no-fly sentinel)"),
    ColumnSpec("thrust_sp_N", "1(norm)", FRAME_NED, "vehicle_local_position_setpoint.thrust[0] (hover~=hover_thrust)"),
    ColumnSpec("thrust_sp_E", "1(norm)", FRAME_NED, "vehicle_local_position_setpoint.thrust[1]"),
    ColumnSpec("thrust_sp_D", "1(norm)", FRAME_NED, "vehicle_local_position_setpoint.thrust[2] (up = negative)"),
    ColumnSpec("thrust_sp_mag", "1(norm)", FRAME_NONE, "||vehicle_local_position_setpoint.thrust||"),
    ColumnSpec("hover_thrust", "1(norm)", FRAME_NONE, "hover_thrust_estimate.hover_thrust (ZOH)", optional=True),
    ColumnSpec("hover_thrust_valid", "bool", FRAME_NONE, "hover_thrust_estimate.valid (ZOH)", optional=True),
)

ATTITUDE_COLUMNS = (
    ColumnSpec("t", "s", FRAME_NONE, "master timebase (vehicle_attitude stamp - t0)"),
    ColumnSpec("q_sp_w", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude_setpoint.q_d[0] (ZOH)"),
    ColumnSpec("q_sp_x", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude_setpoint.q_d[1] (ZOH)"),
    ColumnSpec("q_sp_y", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude_setpoint.q_d[2] (ZOH)"),
    ColumnSpec("q_sp_z", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude_setpoint.q_d[3] (ZOH)"),
    ColumnSpec("roll_sp", "rad", FRAME_NED, "euler_xyz(q_sp) roll  (derived, never a version Euler field)"),
    ColumnSpec("pitch_sp", "rad", FRAME_NED, "euler_xyz(q_sp) pitch (derived)"),
    ColumnSpec("yaw_sp", "rad", FRAME_NED, "euler_xyz(q_sp) yaw (derived)"),
    ColumnSpec("q_w", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude.q[0]"),
    ColumnSpec("q_x", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude.q[1]"),
    ColumnSpec("q_y", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude.q[2]"),
    ColumnSpec("q_z", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude.q[3]"),
    ColumnSpec("roll", "rad", FRAME_NED, "euler_xyz(q) roll (derived)"),
    ColumnSpec("pitch", "rad", FRAME_NED, "euler_xyz(q) pitch (derived)"),
    ColumnSpec("yaw", "rad", FRAME_NED, "euler_xyz(q) yaw (derived)"),
    ColumnSpec("tilt_sp", "rad", FRAME_NONE, "angle(body_z_sp, world_down) from q_sp (derived)"),
    ColumnSpec("tilt", "rad", FRAME_NONE, "angle(body_z, world_down) from q (derived)"),
    ColumnSpec("thrust_sp_mag", "1(norm)", FRAME_NONE, "-vehicle_attitude_setpoint.thrust_body[2] (ZOH)"),
    ColumnSpec("near_level_valid", "bool", FRAME_NONE,
               f"tilt <= {NEAR_LEVEL_MAX_RAD_DEG:g} deg -- gate for the thrust block"),
)

RATE_COLUMNS = (
    ColumnSpec("t", "s", FRAME_NONE, "master timebase (vehicle_angular_velocity stamp - t0)"),
    ColumnSpec("p_sp", "rad/s", FRAME_FRD, "vehicle_rates_setpoint roll rate (ZOH)"),
    ColumnSpec("q_sp", "rad/s", FRAME_FRD, "vehicle_rates_setpoint pitch rate (ZOH)"),
    ColumnSpec("r_sp", "rad/s", FRAME_FRD, "vehicle_rates_setpoint yaw rate (ZOH)"),
    ColumnSpec("p", "rad/s", FRAME_FRD, "vehicle_angular_velocity.xyz[0]"),
    ColumnSpec("q", "rad/s", FRAME_FRD, "vehicle_angular_velocity.xyz[1]"),
    ColumnSpec("r", "rad/s", FRAME_FRD, "vehicle_angular_velocity.xyz[2]"),
    ColumnSpec("thrust_sp_mag", "1(norm)", FRAME_NONE, "-vehicle_rates_setpoint.thrust_body[2] (ZOH)"),
)

TRANSLATION_COLUMNS = (
    ColumnSpec("t", "s", FRAME_NONE, "master timebase (vehicle_acceleration stamp - t0)"),
    ColumnSpec("acc_b_x", "m/s^2", FRAME_FRD, "vehicle_acceleration.xyz[0] (specific force, incl. gravity)"),
    ColumnSpec("acc_b_y", "m/s^2", FRAME_FRD, "vehicle_acceleration.xyz[1] (specific force)"),
    ColumnSpec("acc_b_z", "m/s^2", FRAME_FRD, "vehicle_acceleration.xyz[2] (specific force, ~ -g level)"),
    ColumnSpec("acc_N", "m/s^2", FRAME_NED, "vehicle_local_position.ax (ZOH; EKF, gravity removed)"),
    ColumnSpec("acc_E", "m/s^2", FRAME_NED, "vehicle_local_position.ay (ZOH)"),
    ColumnSpec("acc_D", "m/s^2", FRAME_NED, "vehicle_local_position.az (ZOH)"),
    ColumnSpec("v_N", "m/s", FRAME_NED, "vehicle_local_position.vx (ZOH)"),
    ColumnSpec("v_E", "m/s", FRAME_NED, "vehicle_local_position.vy (ZOH)"),
    ColumnSpec("v_D", "m/s", FRAME_NED, "vehicle_local_position.vz (ZOH)"),
    ColumnSpec("pos_N", "m", FRAME_NED, "vehicle_local_position.x (ZOH)"),
    ColumnSpec("pos_E", "m", FRAME_NED, "vehicle_local_position.y (ZOH)"),
    ColumnSpec("pos_D", "m", FRAME_NED, "vehicle_local_position.z (ZOH)"),
    ColumnSpec("q_w", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude.q[0] (ZOH)"),
    ColumnSpec("q_x", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude.q[1] (ZOH)"),
    ColumnSpec("q_y", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude.q[2] (ZOH)"),
    ColumnSpec("q_z", "1", FRAME_QUAT_NED_FROM_BODY, "vehicle_attitude.q[3] (ZOH)"),
    ColumnSpec("tilt", "rad", FRAME_NONE, "angle(body_z, world_down) from q (derived)"),
    ColumnSpec(
        "body_z_specific_force_proxy", "m/s^2", FRAME_FRD,
        "= -vehicle_acceleration.xyz[2]. CRUDE proxy for vertical specific "
        "thrust, ONLY meaningful where near_level_valid is True (tilt <= "
        f"{NEAR_LEVEL_MAX_RAD_DEG:g} deg). NOT a calibrated thrust measurement; "
        "do not treat as a generally valid thrust signal.",
    ),
    ColumnSpec("near_level_valid", "bool", FRAME_NONE,
               f"tilt <= {NEAR_LEVEL_MAX_RAD_DEG:g} deg -- gate for body_z_specific_force_proxy"),
    ColumnSpec("hover_thrust", "1(norm)", FRAME_NONE, "hover_thrust_estimate.hover_thrust (ZOH)", optional=True),
    ColumnSpec("battery_v", "V", FRAME_NONE, "battery_status.voltage_v (ZOH)", optional=True),
    ColumnSpec("battery_i", "A", FRAME_NONE, "battery_status.current_a (ZOH)", optional=True),
    ColumnSpec("agl", "m", FRAME_NONE, "vehicle_local_position.dist_bottom (ZOH; rangefinder AGL)", optional=True),
)

LOOP_COLUMNS = {
    LOOP_VELOCITY: VELOCITY_COLUMNS,
    LOOP_ATTITUDE: ATTITUDE_COLUMNS,
    LOOP_RATE: RATE_COLUMNS,
    LOOP_TRANSLATION: TRANSLATION_COLUMNS,
}

# ---------------------------------------------------------------------------
# PX4 parameter subsets recorded for provenance. These are the controller
# parameters of the LOGGED vehicle. For the sample logs they belong to the
# OTHER-PROJECT UAV and are provenance metadata only -- never UGRP values.
# ---------------------------------------------------------------------------
MPC_PARAM_KEYS = (
    "MPC_ACC_DECOUPLE", "MPC_TILTMAX_AIR", "MPC_TILTMAX_LND", "MPC_USE_HTE",
    "MPC_THR_HOVER", "MPC_THR_MIN", "MPC_THR_MAX", "MPC_THR_XY_MARG",
    "MPC_XY_VEL_P_ACC", "MPC_XY_VEL_I_ACC", "MPC_XY_VEL_D_ACC",
    "MPC_Z_VEL_P_ACC", "MPC_Z_VEL_I_ACC", "MPC_Z_VEL_D_ACC",
    "MPC_XY_P", "MPC_Z_P", "MPC_XY_VEL_MAX", "MPC_Z_VEL_MAX_UP",
    "MPC_Z_VEL_MAX_DN", "MPC_LAND_SPEED",
)
MC_PARAM_KEYS = (
    "MC_ROLLRATE_P", "MC_PITCHRATE_P", "MC_YAWRATE_P",
    "MC_ROLLRATE_I", "MC_PITCHRATE_I", "MC_YAWRATE_I",
    "MC_ROLLRATE_D", "MC_PITCHRATE_D", "MC_YAWRATE_D",
    "MC_ROLL_P", "MC_PITCH_P", "MC_YAW_P",
)
AIRFRAME_PARAM_KEYS = ("SYS_AUTOSTART", "MAV_TYPE", "SDLOG_PROFILE")


@dataclass(frozen=True)
class RatesSetpointFields:
    """Resolved field names for vehicle_rates_setpoint (version tolerant)."""

    roll: str
    pitch: str
    yaw: str
    thrust_z: Optional[str]


def rates_setpoint_fields(available_fields) -> RatesSetpointFields:
    """Resolve roll/pitch/yaw rate field names across PX4 message variants.

    Observed (PX4 main / v1.17-alpha sample logs): scalar ``roll`` / ``pitch``
    / ``yaw``. Older/other variants sometimes expose ``xyz[0..2]``. Raise a
    clear error if neither shape is present.
    """
    fields = set(available_fields)
    if {"roll", "pitch", "yaw"} <= fields:
        thr = "thrust_body[2]" if "thrust_body[2]" in fields else None
        return RatesSetpointFields("roll", "pitch", "yaw", thr)
    if {"xyz[0]", "xyz[1]", "xyz[2]"} <= fields:
        thr = "thrust_body[2]" if "thrust_body[2]" in fields else None
        return RatesSetpointFields("xyz[0]", "xyz[1]", "xyz[2]", thr)
    raise SchemaError(
        "vehicle_rates_setpoint has neither scalar roll/pitch/yaw nor xyz[0..2]; "
        f"available fields: {sorted(fields)}"
    )


class SchemaError(Exception):
    """Raised when a required topic/field cannot be resolved for this log."""
