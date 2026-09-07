"""Data-quality / excitation report.

A GATE on data usability, NOT a system-identification result. It fits nothing
and never declares a physical model "identified".

Structure: per log -> per-loop timebase diagnostics + per-SI-BLOCK
(attitude / rate / velocity_xy / velocity_z / thrust / translation) valid
duration, segments, and coarse excitation grade (GOOD / MARGINAL / POOR from
signal std / range / input-output correlation).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from system_id.preprocessing.dataset import LoopDataset
from system_id.preprocessing.masks import BLOCK_NAMES
from system_id.preprocessing.schema import (
    ALL_TOPICS,
    LOOP_ATTITUDE,
    LOOP_RATE,
    LOOP_TRANSLATION,
    LOOP_VELOCITY,
)
from system_id.preprocessing.segments import Segment, segments_table
from system_id.preprocessing.ulog_loader import LoadedULog

REPORT_DISCLAIMER = (
    "DATA QUALITY GATE ONLY. This report does NOT declare any physical model "
    "identified and contains no fitted system-identification parameters."
)

# Which loop dataset each SI block is reported from.
BLOCK_LOOP: Dict[str, str] = {
    "attitude": LOOP_ATTITUDE,
    "rate": LOOP_RATE,
    "velocity_xy": LOOP_VELOCITY,
    "velocity_z": LOOP_VELOCITY,
    "thrust": LOOP_TRANSLATION,
    "translation": LOOP_TRANSLATION,
}

EXC_THRESHOLDS = {
    # channel-family : (marginal_std, good_std, good_range)
    "velocity_error": (0.04, 0.10, 0.5),   # m/s
    "attitude_sp": (0.02, 0.05, 0.20),     # rad
    "rate_sp": (0.06, 0.15, 0.60),         # rad/s
    "thrust_sp": (0.025, 0.06, 0.15),      # normalized / m/s^2 proxy
}
CORR_POOR = 0.30
CORR_MARGINAL = 0.50

_GRADE_RANK = {"POOR": 0, "MARGINAL": 1, "GOOD": 2}
_RANK_GRADE = {v: k for k, v in _GRADE_RANK.items()}


def _grade(std: float, rng: float, key: str, corr: float = None) -> str:
    m, g, gr = EXC_THRESHOLDS[key]
    if not np.isfinite(std):
        return "POOR"
    if std >= g and rng >= gr:
        grade = "GOOD"
    elif std >= m:
        grade = "MARGINAL"
    else:
        grade = "POOR"
    if corr is not None and np.isfinite(corr):
        if abs(corr) < CORR_POOR:
            grade = "POOR"
        elif abs(corr) < CORR_MARGINAL:
            grade = _RANK_GRADE[min(_GRADE_RANK[grade], _GRADE_RANK["MARGINAL"])]
    return grade


def _signal_stats(x: np.ndarray, mask: np.ndarray, key: str) -> Dict[str, object]:
    xm = x[mask]
    xm = xm[np.isfinite(xm)]
    if xm.size < 30:
        return {"n": int(xm.size), "cmd_std": None, "cmd_range": None,
                "corr": None, "excitation": "POOR"}
    std = float(np.std(xm))
    rng = float(np.ptp(xm))
    return {"n": int(xm.size), "cmd_std": round(std, 4), "cmd_range": round(rng, 4),
            "corr": None, "excitation": _grade(std, rng, key)}


def _pair_stats(x: np.ndarray, y: np.ndarray, mask: np.ndarray, key: str) -> Dict[str, object]:
    xm = x[mask]
    ym = y[mask]
    ok = np.isfinite(xm) & np.isfinite(ym)
    xm, ym = xm[ok], ym[ok]
    if xm.size < 30:
        return {"n": int(xm.size), "cmd_std": None, "cmd_range": None,
                "corr": None, "excitation": "POOR"}
    xu = np.unwrap(xm) if (key == "attitude_sp" and xm.size > 1) else xm
    std = float(np.std(xu))
    rng = float(np.ptp(xu))
    corr = float(np.corrcoef(xm, ym)[0, 1]) if np.std(ym) > 0 else None
    return {
        "n": int(xm.size),
        "cmd_std": round(std, 4),
        "cmd_range": round(rng, 4),
        "corr": None if corr is None else round(corr, 3),
        "excitation": _grade(std, rng, key, corr),
    }


def _effective_rates(loaded: LoadedULog) -> Dict[str, float]:
    out = {}
    for name in ALL_TOPICS:
        if not loaded.has_topic(name):
            continue
        td = loaded.topic(name)
        t = np.asarray(td.data.get("timestamp", []), dtype=np.float64)
        if t.size > 1:
            span = (t[-1] - t[0]) / 1e6
            out[name] = round((t.size - 1) / span, 2) if span > 0 else 0.0
    return out


def _excitation_for_block(block: str, datasets: Dict[str, LoopDataset]) -> Dict[str, object]:
    ds = datasets[BLOCK_LOOP[block]]
    c = ds.columns
    m = ds.block_masks[block]
    if block == "velocity_xy":
        return {"verr_N->a_sp_N": _pair_stats(c["velocity_error_N"], c["a_sp_N"], m, "velocity_error"),
                "verr_E->a_sp_E": _pair_stats(c["velocity_error_E"], c["a_sp_E"], m, "velocity_error")}
    if block == "velocity_z":
        return {"verr_D->a_sp_D": _pair_stats(c["velocity_error_D"], c["a_sp_D"], m, "velocity_error")}
    if block == "attitude":
        return {"roll_sp->roll": _pair_stats(c["roll_sp"], c["roll"], m, "attitude_sp"),
                "pitch_sp->pitch": _pair_stats(c["pitch_sp"], c["pitch"], m, "attitude_sp"),
                "yaw_sp->yaw": _pair_stats(c["yaw_sp"], c["yaw"], m, "attitude_sp")}
    if block == "rate":
        return {"p_sp->p": _pair_stats(c["p_sp"], c["p"], m, "rate_sp"),
                "q_sp->q": _pair_stats(c["q_sp"], c["q"], m, "rate_sp"),
                "r_sp->r": _pair_stats(c["r_sp"], c["r"], m, "rate_sp")}
    if block == "thrust":
        out = {"body_z_sf_proxy(var)": _signal_stats(c["body_z_specific_force_proxy"], m, "thrust_sp")}
        att = datasets[LOOP_ATTITUDE]
        if "thrust" in att.block_masks:
            out["thrust_sp_mag(var)"] = _signal_stats(
                att.columns["thrust_sp_mag"], att.block_masks["thrust"], "thrust_sp")
        return out
    if block == "translation":
        return {"body_z_sf_proxy(var)": _signal_stats(c["body_z_specific_force_proxy"], m, "thrust_sp")}
    return {}


def per_log_report(
    loaded: LoadedULog,
    datasets: Dict[str, LoopDataset],
    block_segments: Dict[str, List[Segment]],
    source_project: str,
    dataset_role: str,
) -> Dict[str, object]:
    loops_diag = {}
    for name, ds in datasets.items():
        dt = ds.timebase.dt_median_ms / 1e3
        loops_diag[name] = {
            "master_topic": ds.timebase.master_topic,
            "stamp_kind": ds.timebase.stamp_kind,
            "n_samples": ds.n,
            "effective_hz": round(ds.timebase.effective_hz, 2),
            "dt_median_ms": round(ds.timebase.dt_median_ms, 3),
            "dt_std_ms": round(ds.timebase.dt_std_ms, 3),
            "nonmonotonic_count": ds.timebase.nonmonotonic_count,
            "flight_base_s": round(float(np.sum(ds.mask.flight_base)) * dt, 2),
            "flight_base_fraction": round(float(np.mean(ds.mask.flight_base)), 4) if ds.n else 0.0,
            "in_ground_effect_fraction": round(float(np.mean(ds.mask.ground_effect)), 4) if ds.n else 0.0,
            "rejection_fraction_by_reason": {k: round(v, 4) for k, v in ds.mask.summary.items()},
        }
        if ds.pairing:
            loops_diag[name]["setpoint_pairing"] = ds.pairing

    blocks = {}
    for block in BLOCK_NAMES:
        ds = datasets[BLOCK_LOOP[block]]
        m = ds.block_masks[block]
        dt = ds.timebase.dt_median_ms / 1e3
        blocks[block] = {
            "from_loop": BLOCK_LOOP[block],
            "ge_sensitive": block in {"velocity_z", "thrust", "translation"},
            "near_level_gated": block == "thrust",
            "valid_s": round(float(np.sum(m)) * dt, 2),
            "valid_fraction": round(float(np.mean(m)), 4) if ds.n else 0.0,
            "segments": segments_table(block_segments.get(block, [])),
            "excitation": _excitation_for_block(block, datasets),
        }

    sat_frac = datasets[LOOP_ATTITUDE].mask.summary.get("motor_saturated", 0.0)
    return {
        "disclaimer": REPORT_DISCLAIMER,
        "source_project": source_project,
        "dataset_role": dataset_role,
        "log": loaded.path,
        "sha256": loaded.sha256,
        "px4": loaded.px4,
        "log_window_s": round((loaded.last_us - loaded.start_us) / 1e6, 2),
        "dropouts": {"n": loaded.n_dropouts, "seconds": round(loaded.dropout_us / 1e6, 4)},
        "available_topics": list(loaded.present_topics),
        "missing_optional_topics": list(loaded.missing_optional),
        "effective_sample_rates_hz": _effective_rates(loaded),
        "motor_saturation_fraction": round(float(sat_frac), 4),
        "loops": loops_diag,
        "blocks": blocks,
    }


def render_text(report: Dict[str, object]) -> str:
    L: List[str] = []
    L.append("=" * 78)
    L.append(f"LOG  {report['log']}")
    L.append(f"  {report['source_project']} / {report['dataset_role']}")
    L.append(f"  PX4 {report['px4'].get('ver_sw_branch')} {report['px4'].get('ver_sw_release')} "
             f"hw={report['px4'].get('ver_hw')}  window={report['log_window_s']}s  "
             f"dropouts={report['dropouts']['n']}")
    L.append(f"  motor-saturation fraction: {report['motor_saturation_fraction']:.3f}")
    miss = report["missing_optional_topics"]
    L.append(f"  missing optional topics: {', '.join(miss) if miss else '(none)'}")
    for lname, lr in report["loops"].items():
        line = (f"  loop {lname:11s} master={lr['master_topic']} ({lr['stamp_kind']}) "
                f"~{lr['effective_hz']}Hz n={lr['n_samples']} flight_base={lr['flight_base_s']}s "
                f"({lr['flight_base_fraction']*100:.1f}%) ge={lr['in_ground_effect_fraction']*100:.1f}%")
        if "setpoint_pairing" in lr:
            p = lr["setpoint_pairing"]
            line += f"  pairing matched={p['match_fraction']*100:.2f}% (unmatched {p['unmatched']})"
        L.append(line)
    for bname, br in report["blocks"].items():
        L.append(f"  --- block {bname:12s} from {br['from_loop']:11s} "
                 f"valid={br['valid_s']}s ({br['valid_fraction']*100:.1f}%)  "
                 f"segs={br['segments']['n_segments']} longest={br['segments']['longest_valid_s']}s"
                 f"{'  [GE-excluded]' if br['ge_sensitive'] else ''}"
                 f"{'  [near-level-gated]' if br['near_level_gated'] else ''}")
        for ch, st in br["excitation"].items():
            L.append(f"      exc {ch:22s} std={st['cmd_std']} range={st['cmd_range']} "
                     f"corr={st['corr']}  -> {st['excitation']}")
    L.append(f"  NOTE: {report['disclaimer']}")
    return "\n".join(L)
