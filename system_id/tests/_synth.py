"""Synthetic in-memory ULog fixtures for preprocessing unit tests.

No real ``.ulg`` file is created. We build :class:`LoadedULog` /
:class:`TopicData` directly so the tests are fast and version-stable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from system_id.preprocessing.ulog_loader import LoadedULog, TopicData


def make_topic(name: str, data: Dict[str, np.ndarray]) -> TopicData:
    data = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
    fields = tuple(sorted(data))
    n = len(next(iter(data.values()))) if data else 0
    return TopicData(
        name=name,
        multi_id=0,
        fields=fields,
        data=data,
        n=n,
        has_timestamp_sample="timestamp_sample" in data,
    )


def _grid(hz: float, dur_s: float, t0_us: float = 1_000_000.0):
    n = int(round(hz * dur_s))
    t = t0_us + (np.arange(n) / hz) * 1e6
    return t, n


def build_synthetic_loaded(
    dur_s: float = 20.0,
    hover_thrust: float = 0.55,
    drop_lps_window_s: tuple = None,
    big_tilt: bool = False,
) -> LoadedULog:
    """A minimal but schema-complete synthetic log.

    ``drop_lps_window_s=(a,b)``: delete vehicle_local_position_setpoint samples
    with a <= t < b so some vehicle_local_position samples have no partner
    within the +/-15 ms pairing tolerance (exercises unmatched-pair rejection).
    ``big_tilt=True``: roll setpoint step of ~0.6 rad (~34 deg) so tilt clearly
    exceeds NEAR_LEVEL_MAX_RAD (exercises the near-level gate on the thrust block).

    Timeline (seconds): [0,3) disarmed on ground, [3,5) armed climbing,
    [5, dur-3) airborne offboard with a roll doublet + vertical step,
    [dur-3, dur) descending / ground contact.
    """
    rng = np.random.default_rng(0)
    topics: Dict[str, TopicData] = {}

    # --- fast grids
    t50, n50 = _grid(50.0, dur_s)
    t20, n20 = _grid(20.0, dur_s)
    t10, n10 = _grid(10.0, dur_s)
    t5, n5 = _grid(5.0, dur_s)

    def phase_mask(t_us):
        ts = (t_us - t_us[0]) / 1e6
        armed = ts >= 3.0
        airborne = (ts >= 5.0) & (ts < dur_s - 3.0)
        gcontact = ts >= dur_s - 2.0
        return ts, armed, airborne, gcontact

    # ---------------- vehicle_control_mode (on-change; sample sparsely) ------
    tcm = np.array([0, 3.0, 5.0, dur_s - 2.5]) * 1e6 + t10[0]
    cm = dict(
        timestamp=tcm,
        flag_armed=np.array([0, 1, 1, 1]),
        flag_control_offboard_enabled=np.array([0, 0, 1, 1]),
        flag_control_velocity_enabled=np.array([0, 0, 1, 1]),
        flag_multicopter_position_control_enabled=np.array([1, 1, 1, 1]),
    )
    topics["vehicle_control_mode"] = make_topic("vehicle_control_mode", cm)

    tld = np.array([0, 5.0, dur_s - 2.0]) * 1e6 + t10[0]
    ld = dict(
        timestamp=tld,
        landed=np.array([1, 0, 1]),
        ground_contact=np.array([1, 0, 1]),
        maybe_landed=np.array([1, 0, 1]),
        in_ground_effect=np.array([1, 0, 1]),
    )
    topics["vehicle_land_detected"] = make_topic("vehicle_land_detected", ld)

    # ---------------- actuator_motors (10 Hz) -------------------------------
    ts10, armed10, air10, gc10 = phase_mask(t10)
    base = np.where(armed10, hover_thrust, 0.0)
    # brief saturation burst near t=8s
    burst = (ts10 >= 8.0) & (ts10 < 8.4)
    m0 = np.clip(base + 0.02 * rng.standard_normal(n10) + np.where(burst, 0.5, 0.0), 0.0, 1.0)
    am = dict(timestamp=t10, timestamp_sample=t10 - 90.0)
    for i in range(4):
        am[f"control[{i}]"] = m0 if i == 0 else np.clip(base + 0.02 * rng.standard_normal(n10), 0.0, 1.0)
    topics["actuator_motors"] = make_topic("actuator_motors", am)
    topics["actuator_outputs"] = make_topic("actuator_outputs", dict(
        timestamp=t10, **{f"output[{i}]": 1000 + 900 * am[f"control[{i}]"] for i in range(4)}
    ))

    # ---------------- vehicle_local_position (10 Hz) ----------------------
    v_cmd = np.where((ts10 >= 6.0) & (ts10 < 9.0), 0.6, 0.0)  # forward step
    v_meas = 0.0 * v_cmd
    for k in range(1, n10):
        v_meas[k] = v_meas[k - 1] + 0.05 * (v_cmd[k - 1] - v_meas[k - 1]) * 2.0
    vlp = dict(
        timestamp=t10, timestamp_sample=t10 - 450.0,
        vx=v_meas + 0.01 * rng.standard_normal(n10),
        vy=0.01 * rng.standard_normal(n10),
        vz=np.where((ts10 >= 10.0) & (ts10 < 12.0), 0.5, 0.0) + 0.01 * rng.standard_normal(n10),
        ax=0.02 * rng.standard_normal(n10), ay=0.02 * rng.standard_normal(n10),
        az=0.02 * rng.standard_normal(n10),
        x=np.cumsum(v_meas) * 0.1, y=np.zeros(n10),
        z=np.where(air10, -5.0, -0.1),
        heading=0.05 * rng.standard_normal(n10),
        dist_bottom=np.where(air10, 5.0, 0.05),
    )
    topics["vehicle_local_position"] = make_topic("vehicle_local_position", vlp)

    # ---------------- vehicle_local_position_setpoint (10 Hz, +0.5ms) -------
    a_sp = 1.8 * (v_cmd - v_meas)
    a_sp_z = np.where(air10, 0.1 * rng.standard_normal(n10), 100.0)  # sentinel on ground
    lps = dict(
        timestamp=t10 + 0.5,  # published just after the vlp sample
        vx=v_cmd, vy=np.zeros(n10),
        vz=np.where((ts10 >= 10.0) & (ts10 < 12.0), 0.5, 0.0),
        x=np.full(n10, np.nan), y=np.full(n10, np.nan), z=np.full(n10, np.nan),
        **{"acceleration[0]": a_sp, "acceleration[1]": np.zeros(n10), "acceleration[2]": a_sp_z},
        **{"thrust[0]": 0.03 * rng.standard_normal(n10), "thrust[1]": 0.01 * rng.standard_normal(n10),
           "thrust[2]": np.where(air10, -hover_thrust, -0.01)},
        yaw=np.zeros(n10), yawspeed=np.zeros(n10),
    )
    if drop_lps_window_s is not None:
        a, b = drop_lps_window_s
        keep = ~(((t10 - t10[0]) / 1e6 >= a) & ((t10 - t10[0]) / 1e6 < b))
        lps = {k: np.asarray(v)[keep] for k, v in lps.items()}
    topics["vehicle_local_position_setpoint"] = make_topic("vehicle_local_position_setpoint", lps)

    # ---------------- attitude (20 Hz) -----------------------------------
    ts20, armed20, air20, gc20 = phase_mask(t20)
    _amp = 0.60 if big_tilt else 0.20
    roll_sp = np.where((ts20 >= 7.0) & (ts20 < 7.5), _amp,
                       np.where((ts20 >= 7.5) & (ts20 < 8.0), -_amp, 0.0))
    roll = np.zeros(n20)
    for k in range(1, n20):
        roll[k] = roll[k - 1] + 0.20 * (roll_sp[k - 1] - roll[k - 1])
    pitch_sp = 0.05 * np.sin(2 * np.pi * 0.3 * ts20)
    pitch = 0.9 * pitch_sp
    yaw = 0.3 * ts20  # ramp -> crosses pi

    def q_from_rpy(r, p, y):
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        cp, sp = np.cos(p / 2), np.sin(p / 2)
        cy, sy = np.cos(y / 2), np.sin(y / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        yq = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.stack([w, x, yq, z], axis=-1)

    q = q_from_rpy(roll, pitch, yaw)
    qd = q_from_rpy(roll_sp, pitch_sp, yaw)
    topics["vehicle_attitude"] = make_topic("vehicle_attitude", dict(
        timestamp=t20, timestamp_sample=t20 - 200.0,
        **{f"q[{i}]": q[:, i] for i in range(4)},
    ))
    topics["vehicle_attitude_setpoint"] = make_topic("vehicle_attitude_setpoint", dict(
        timestamp=t20,
        **{f"q_d[{i}]": qd[:, i] for i in range(4)},
        **{"thrust_body[2]": np.where(air20, -hover_thrust, -0.01)},
        yaw_sp_move_rate=np.zeros(n20),
    ))
    topics["vehicle_acceleration"] = make_topic("vehicle_acceleration", dict(
        timestamp=t20, timestamp_sample=t20 - 70.0,
        **{"xyz[0]": 0.1 * rng.standard_normal(n20), "xyz[1]": 0.1 * rng.standard_normal(n20),
           "xyz[2]": np.where(air20, -9.8, -0.2) + 0.1 * rng.standard_normal(n20)},
    ))

    # ---------------- rate (50 Hz) --------------------------------------
    ts50, armed50, air50, gc50 = phase_mask(t50)
    p_sp = np.where((ts50 >= 7.0) & (ts50 < 7.05), 2.0, 0.0)
    p = np.zeros(n50)
    for k in range(1, n50):
        p[k] = p[k - 1] + 0.5 * (p_sp[k - 1] - p[k - 1])
    q_sp = 0.3 * np.sin(2 * np.pi * 1.0 * ts50)
    qb = 0.8 * q_sp
    r_sp = 0.2 * np.sign(np.sin(2 * np.pi * 0.2 * ts50))
    rb = 0.85 * r_sp
    topics["vehicle_rates_setpoint"] = make_topic("vehicle_rates_setpoint", dict(
        timestamp=t50, roll=p_sp, pitch=q_sp, yaw=r_sp,
        **{"thrust_body[2]": np.where(air50, -hover_thrust, -0.01)},
        reset_integral=np.zeros(n50),
    ))
    topics["vehicle_angular_velocity"] = make_topic("vehicle_angular_velocity", dict(
        timestamp=t50, timestamp_sample=t50 - 60.0,
        **{"xyz[0]": p + 0.01 * rng.standard_normal(n50),
           "xyz[1]": qb + 0.01 * rng.standard_normal(n50),
           "xyz[2]": rb + 0.01 * rng.standard_normal(n50)},
    ))
    topics["vehicle_thrust_setpoint"] = make_topic("vehicle_thrust_setpoint", dict(
        timestamp=t50, timestamp_sample=t50 - 70.0,
        **{"xyz[0]": np.zeros(n50), "xyz[1]": np.zeros(n50),
           "xyz[2]": np.where(air50, -hover_thrust, -0.01)},
    ))

    # ---------------- support ------------------------------------------
    topics["hover_thrust_estimate"] = make_topic("hover_thrust_estimate", dict(
        timestamp=t10, timestamp_sample=t10 - 400.0,
        hover_thrust=np.full(n10, hover_thrust), valid=np.ones(n10),
    ))
    topics["offboard_control_mode"] = make_topic("offboard_control_mode", dict(
        timestamp=t10, position=np.zeros(n10), velocity=np.ones(n10),
        acceleration=np.zeros(n10), attitude=np.zeros(n10), body_rate=np.zeros(n10),
        direct_actuator=np.zeros(n10), thrust_and_torque=np.zeros(n10),
    ))
    topics["battery_status"] = make_topic("battery_status", dict(
        timestamp=t5, voltage_v=np.full(n5, 22.2), current_a=np.full(n5, 10.0),
    ))
    topics["trajectory_setpoint"] = make_topic("trajectory_setpoint", dict(
        timestamp=t5, **{f"velocity[{i}]": np.zeros(n5) for i in range(3)},
        **{f"position[{i}]": np.zeros(n5) for i in range(3)},
        **{f"acceleration[{i}]": np.full(n5, np.nan) for i in range(3)},
        yaw=np.zeros(n5), yawspeed=np.full(n5, np.nan),
    ))
    # NOTE: esc_status deliberately absent -> optional-topic handling under test.

    start = min(float(td.data["timestamp"][0]) for td in topics.values())
    last = max(float(td.data["timestamp"][-1]) for td in topics.values())
    return LoadedULog(
        path="<synthetic>",
        sha256="0" * 64,
        px4={"ver_sw_branch": "synthetic", "ver_sw_release": "0", "ver_hw": "SYNTH"},
        params={
            "MPC_XY_VEL_P_ACC": 1.8, "MPC_Z_VEL_P_ACC": 4.0, "MPC_ACC_DECOUPLE": 1,
            "MPC_THR_HOVER": 0.5, "MC_ROLLRATE_P": 0.15, "SYS_AUTOSTART": 4001, "MAV_TYPE": 2,
        },
        topics=topics,
        start_us=start, last_us=last, dropout_us=0.0, n_dropouts=0,
        present_topics=tuple(sorted(topics)),
        missing_required=(),
        missing_optional=("esc_status",),
        multi_instance_topics=(),
        warnings=["optional topics absent: esc_status"],
    )
