"""Per-loop dataset assembly.

Four datasets, each on its OWN native timebase:

  velocity     ~10 Hz  PX4 velocity-controller telemetry (KNOWN controller;
                       validation/consistency data, not an opaque plant block)
  attitude     ~20 Hz  attitude-setpoint -> attitude (IDENTIFIED response)
  rate         ~50 Hz  rate-setpoint -> body-rate (IDENTIFIED response)
  translation  ~20 Hz  attitude + thrust support for translational response

Every column's (name, unit, frame, source) comes from schema.LOOP_COLUMNS.
NED acceleration and body specific force are kept in separate, clearly named
columns and never added together.

Masks
-----
Each dataset carries the raw :class:`MaskResult` (``flight_base`` + reason
masks) PLUS ``block_masks`` -- one final bool[n] per SI block the loop serves
(see ``masks.LOOP_BLOCKS``). A block mask is::

    flight_base
      [& ~in_ground_effect  if the block is ground-effect-sensitive]
      [& near_level_valid   if the block is the thrust block]
      & (required input/output columns all finite)
      [& setpoint_pairing_matched   for the velocity blocks]

so a row whose required I/O could not be assembled (NaN, or an unmatched
``vehicle_local_position_setpoint`` pairing) is False for that block and can
never be handed to an identifier as valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from system_id.preprocessing import frames
from system_id.preprocessing.masks import (
    LOOP_BLOCKS,
    NEAR_LEVEL_MAX_RAD,
    MaskResult,
    compute_masks,
)
from system_id.preprocessing.schema import (
    LOOP_ATTITUDE,
    LOOP_COLUMNS,
    LOOP_RATE,
    LOOP_TRANSLATION,
    LOOP_VELOCITY,
    rates_setpoint_fields,
)
from system_id.preprocessing.timebase import (
    CYCLE_PAIR_TOL_US,
    LoopTimebase,
    build_loop_timebase,
    choose_stamp_us,
    log_t0_us,
    nearest_match,
    sample_aligned,
    zoh,
)
from system_id.preprocessing.ulog_loader import LoadedULog


@dataclass
class LoopDataset:
    loop: str
    columns: Dict[str, np.ndarray]
    column_meta: Dict[str, Dict[str, str]]
    timebase: LoopTimebase
    mask: MaskResult
    n: int
    block_masks: Dict[str, np.ndarray] = field(default_factory=dict)   # block -> bool[n] final valid
    aux: Dict[str, np.ndarray] = field(default_factory=dict)           # supporting bool[n] terms
    pairing: Dict[str, float] = field(default_factory=dict)            # setpoint<->measurement match stats

    def to_dataframe(self):  # pragma: no cover - convenience only
        import pandas as pd

        return pd.DataFrame(self.columns)


def _col(td_get, key, n):
    try:
        v = np.asarray(td_get(key), dtype=np.float64)
        if v.shape[0] != n:
            return np.full(n, np.nan)
        return v
    except Exception:
        return np.full(n, np.nan)


def _finite_all(*arrays) -> np.ndarray:
    out = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        out &= np.isfinite(np.asarray(a, dtype=np.float64))
    return out


def _meta_for(loop: str) -> Dict[str, Dict[str, str]]:
    return {
        c.name: {"unit": c.unit, "frame": c.frame, "source": c.source, "optional": str(c.optional)}
        for c in LOOP_COLUMNS[loop]
    }


# --------------------------------------------------------------------------- #
# velocity-controller dataset
# --------------------------------------------------------------------------- #
def build_velocity_dataset(loaded: LoadedULog, t0_us: float) -> LoopDataset:
    tb = build_loop_timebase(loaded, LOOP_VELOCITY, t0_us)
    t_us, n = tb.t_us, tb.n
    vlp = loaded.topic("vehicle_local_position")
    lps = loaded.topic("vehicle_local_position_setpoint")
    lps_t, _ = choose_stamp_us(lps)

    v_N = _col(vlp.get, "vx", n)
    v_E = _col(vlp.get, "vy", n)
    v_D = _col(vlp.get, "vz", n)

    # local_position_setpoint is produced in the SAME MPC cycle as this
    # vehicle_local_position sample -> nearest-match within +/- CYCLE_PAIR_TOL_US.
    pair_idx = nearest_match(t_us, lps_t, CYCLE_PAIR_TOL_US)
    pair_matched = pair_idx >= 0

    def sp(key):
        return sample_aligned(t_us, lps_t, np.asarray(lps.get(key), dtype=np.float64),
                              CYCLE_PAIR_TOL_US)

    v_sp_N, v_sp_E, v_sp_D = sp("vx"), sp("vy"), sp("vz")
    a_sp_N, a_sp_E, a_sp_D = sp("acceleration[0]"), sp("acceleration[1]"), sp("acceleration[2]")
    thr_N, thr_E, thr_D = sp("thrust[0]"), sp("thrust[1]"), sp("thrust[2]")

    cols: Dict[str, np.ndarray] = {
        "t": tb.t_s,
        "v_sp_N": v_sp_N, "v_sp_E": v_sp_E, "v_sp_D": v_sp_D,
        "v_N": v_N, "v_E": v_E, "v_D": v_D,
        "velocity_error_N": v_sp_N - v_N,
        "velocity_error_E": v_sp_E - v_E,
        "velocity_error_D": v_sp_D - v_D,
        "a_sp_N": a_sp_N, "a_sp_E": a_sp_E, "a_sp_D": a_sp_D,
        "thrust_sp_N": thr_N, "thrust_sp_E": thr_E, "thrust_sp_D": thr_D,
        "thrust_sp_mag": frames.thrust_vector_magnitude(thr_N, thr_E, thr_D),
    }
    if loaded.has_topic("hover_thrust_estimate"):
        hte = loaded.topic("hover_thrust_estimate")
        hte_t, _ = choose_stamp_us(hte)
        cols["hover_thrust"] = zoh(hte_t, np.asarray(hte.get("hover_thrust"), dtype=np.float64), t_us)
        cols["hover_thrust_valid"] = zoh(hte_t, np.asarray(hte.get("valid"), dtype=np.float64), t_us)
    else:
        cols["hover_thrust"] = np.full(n, np.nan)
        cols["hover_thrust_valid"] = np.full(n, np.nan)

    mask = compute_masks(loaded, t_us, LOOP_VELOCITY)

    xy_io_finite = _finite_all(cols["velocity_error_N"], cols["velocity_error_E"],
                               cols["a_sp_N"], cols["a_sp_E"])
    z_io_finite = _finite_all(cols["velocity_error_D"], cols["a_sp_D"])

    ds = LoopDataset(LOOP_VELOCITY, cols, _meta_for(LOOP_VELOCITY), tb, mask, n)
    ds.aux = {"setpoint_pairing_matched": pair_matched,
              "xy_io_finite": xy_io_finite, "z_io_finite": z_io_finite}
    ds.block_masks = {
        "velocity_xy": mask.combine_block("velocity_xy", pair_matched, xy_io_finite),
        "velocity_z": mask.combine_block("velocity_z", pair_matched, z_io_finite),
    }
    matched = int(pair_matched.sum())
    ds.pairing = {
        "topic_pair": "vehicle_local_position_setpoint <-> vehicle_local_position",
        "tol_ms": CYCLE_PAIR_TOL_US / 1e3,
        "n": n, "matched": matched, "unmatched": n - matched,
        "match_fraction": round(matched / n, 6) if n else 0.0,
    }
    return ds


# --------------------------------------------------------------------------- #
# attitude dataset
# --------------------------------------------------------------------------- #
def build_attitude_dataset(loaded: LoadedULog, t0_us: float) -> LoopDataset:
    tb = build_loop_timebase(loaded, LOOP_ATTITUDE, t0_us)
    t_us, n = tb.t_us, tb.n
    att = loaded.topic("vehicle_attitude")
    asp = loaded.topic("vehicle_attitude_setpoint")
    asp_t, _ = choose_stamp_us(asp)

    q = np.column_stack([_col(att.get, f"q[{i}]", n) for i in range(4)])
    roll, pitch, yaw = frames.euler_xyz_columns(q)
    tilt = frames.tilt_from_quat(q)

    qsp = np.column_stack([
        zoh(asp_t, np.asarray(asp.get(f"q_d[{i}]"), dtype=np.float64), t_us) for i in range(4)
    ])
    roll_sp, pitch_sp, yaw_sp = frames.euler_xyz_columns(qsp)
    tilt_sp = frames.tilt_from_quat(qsp)

    if asp.has("thrust_body[2]"):
        thr_mag = -zoh(asp_t, np.asarray(asp.get("thrust_body[2]"), dtype=np.float64), t_us)
    else:
        thr_mag = np.full(n, np.nan)

    near_level = np.isfinite(tilt) & (tilt <= NEAR_LEVEL_MAX_RAD)

    cols = {
        "t": tb.t_s,
        "q_sp_w": qsp[:, 0], "q_sp_x": qsp[:, 1], "q_sp_y": qsp[:, 2], "q_sp_z": qsp[:, 3],
        "roll_sp": roll_sp, "pitch_sp": pitch_sp, "yaw_sp": yaw_sp,
        "q_w": q[:, 0], "q_x": q[:, 1], "q_y": q[:, 2], "q_z": q[:, 3],
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "tilt_sp": tilt_sp, "tilt": tilt,
        "thrust_sp_mag": thr_mag,
        "near_level_valid": near_level.astype(np.float64),
    }
    mask = compute_masks(loaded, t_us, LOOP_ATTITUDE)

    att_io_finite = _finite_all(cols["roll_sp"], cols["pitch_sp"], cols["roll"], cols["pitch"])
    thr_io_finite = _finite_all(cols["thrust_sp_mag"])

    ds = LoopDataset(LOOP_ATTITUDE, cols, _meta_for(LOOP_ATTITUDE), tb, mask, n)
    ds.aux = {"near_level_valid": near_level, "attitude_io_finite": att_io_finite}
    ds.block_masks = {
        "attitude": mask.combine_block("attitude", att_io_finite),
        "thrust": mask.combine_block("thrust", near_level, thr_io_finite),
    }
    return ds


# --------------------------------------------------------------------------- #
# rate dataset
# --------------------------------------------------------------------------- #
def build_rate_dataset(loaded: LoadedULog, t0_us: float) -> LoopDataset:
    tb = build_loop_timebase(loaded, LOOP_RATE, t0_us)
    t_us, n = tb.t_us, tb.n
    av = loaded.topic("vehicle_angular_velocity")
    rsp = loaded.topic("vehicle_rates_setpoint")
    rf = rates_setpoint_fields(rsp.fields)
    rsp_t, _ = choose_stamp_us(rsp)

    p = _col(av.get, "xyz[0]", n)
    q_ = _col(av.get, "xyz[1]", n)
    r = _col(av.get, "xyz[2]", n)
    p_sp = zoh(rsp_t, np.asarray(rsp.get(rf.roll), dtype=np.float64), t_us)
    q_sp = zoh(rsp_t, np.asarray(rsp.get(rf.pitch), dtype=np.float64), t_us)
    r_sp = zoh(rsp_t, np.asarray(rsp.get(rf.yaw), dtype=np.float64), t_us)
    thr_mag = (
        -zoh(rsp_t, np.asarray(rsp.get(rf.thrust_z), dtype=np.float64), t_us)
        if rf.thrust_z else np.full(n, np.nan)
    )

    cols = {
        "t": tb.t_s,
        "p_sp": p_sp, "q_sp": q_sp, "r_sp": r_sp,
        "p": p, "q": q_, "r": r,
        "thrust_sp_mag": thr_mag,
    }
    mask = compute_masks(loaded, t_us, LOOP_RATE)
    rate_io_finite = _finite_all(p_sp, q_sp, r_sp, p, q_, r)

    ds = LoopDataset(LOOP_RATE, cols, _meta_for(LOOP_RATE), tb, mask, n)
    ds.aux = {"rate_io_finite": rate_io_finite}
    ds.block_masks = {"rate": mask.combine_block("rate", rate_io_finite)}
    return ds


# --------------------------------------------------------------------------- #
# translational / thrust-support dataset
# --------------------------------------------------------------------------- #
def build_translation_dataset(loaded: LoadedULog, t0_us: float) -> LoopDataset:
    tb = build_loop_timebase(loaded, LOOP_TRANSLATION, t0_us)
    t_us, n = tb.t_us, tb.n
    acc = loaded.topic("vehicle_acceleration")
    vlp = loaded.topic("vehicle_local_position")
    att = loaded.topic("vehicle_attitude")
    vlp_t, _ = choose_stamp_us(vlp)
    att_t, _ = choose_stamp_us(att)

    acc_b_x = _col(acc.get, "xyz[0]", n)
    acc_b_y = _col(acc.get, "xyz[1]", n)
    acc_b_z = _col(acc.get, "xyz[2]", n)

    def v_zoh(key):
        return zoh(vlp_t, np.asarray(vlp.get(key), dtype=np.float64), t_us)

    q = np.column_stack([
        zoh(att_t, np.asarray(att.get(f"q[{i}]"), dtype=np.float64), t_us) for i in range(4)
    ])
    tilt = frames.tilt_from_quat(q)
    near_level = np.isfinite(tilt) & (tilt <= NEAR_LEVEL_MAX_RAD)

    cols = {
        "t": tb.t_s,
        "acc_b_x": acc_b_x, "acc_b_y": acc_b_y, "acc_b_z": acc_b_z,
        "acc_N": v_zoh("ax"), "acc_E": v_zoh("ay"), "acc_D": v_zoh("az"),
        "v_N": v_zoh("vx"), "v_E": v_zoh("vy"), "v_D": v_zoh("vz"),
        "pos_N": v_zoh("x"), "pos_E": v_zoh("y"), "pos_D": v_zoh("z"),
        "q_w": q[:, 0], "q_x": q[:, 1], "q_y": q[:, 2], "q_z": q[:, 3],
        "tilt": tilt,
        # CRUDE proxy -- only meaningful where near_level_valid is True.
        "body_z_specific_force_proxy": -acc_b_z,
        "near_level_valid": near_level.astype(np.float64),
    }
    if loaded.has_topic("hover_thrust_estimate"):
        hte = loaded.topic("hover_thrust_estimate")
        hte_t, _ = choose_stamp_us(hte)
        cols["hover_thrust"] = zoh(hte_t, np.asarray(hte.get("hover_thrust"), dtype=np.float64), t_us)
    else:
        cols["hover_thrust"] = np.full(n, np.nan)
    if loaded.has_topic("battery_status"):
        bat = loaded.topic("battery_status")
        bat_t, _ = choose_stamp_us(bat)
        cols["battery_v"] = zoh(bat_t, _col(bat.get, "voltage_v", bat.n), t_us)
        cols["battery_i"] = zoh(bat_t, _col(bat.get, "current_a", bat.n), t_us)
    else:
        cols["battery_v"] = np.full(n, np.nan)
        cols["battery_i"] = np.full(n, np.nan)
    cols["agl"] = v_zoh("dist_bottom") if vlp.has("dist_bottom") else np.full(n, np.nan)

    mask = compute_masks(loaded, t_us, LOOP_TRANSLATION)
    tr_io_finite = _finite_all(acc_b_x, acc_b_y, acc_b_z)
    thr_io_finite = _finite_all(cols["body_z_specific_force_proxy"])

    ds = LoopDataset(LOOP_TRANSLATION, cols, _meta_for(LOOP_TRANSLATION), tb, mask, n)
    ds.aux = {"near_level_valid": near_level, "translation_io_finite": tr_io_finite}
    ds.block_masks = {
        "translation": mask.combine_block("translation", tr_io_finite),
        "thrust": mask.combine_block("thrust", near_level, thr_io_finite),
    }
    return ds


def build_all(loaded: LoadedULog) -> Dict[str, LoopDataset]:
    t0 = log_t0_us(loaded)
    return {
        LOOP_VELOCITY: build_velocity_dataset(loaded, t0),
        LOOP_ATTITUDE: build_attitude_dataset(loaded, t0),
        LOOP_RATE: build_rate_dataset(loaded, t0),
        LOOP_TRANSLATION: build_translation_dataset(loaded, t0),
    }
