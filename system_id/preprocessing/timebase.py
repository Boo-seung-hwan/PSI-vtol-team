"""Per-loop timebase construction and zero-order-hold resampling.

Design rules (from the 2026-09-07 ULog audit):

  * Each control loop keeps its OWN canonical grid at its native master-topic
    rate. The pipeline never resamples every signal onto the single fastest
    rate.
  * Measurement topics use ``timestamp_sample`` when present (physical sample
    instant); setpoint/mode topics use ``timestamp``.
  * ``vehicle_local_position_setpoint`` is produced in the same MPC cycle as
    the ``vehicle_local_position`` sample that drove it, and is stamped
    ~0.5-1 ms later. It is therefore aligned to the local-position sample by
    NEAREST match within a small tolerance, not by plain ZOH (which would pick
    the previous cycle).
  * Commands/setpoints may be zero-order-held onto a faster measurement grid.
    Slow measurements are never upsampled to fabricate bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from system_id.preprocessing.schema import (
    LOOP_MASTER_TOPIC,
    LOOP_NOMINAL_HZ,
    PREFER_TIMESTAMP_SAMPLE,
)
from system_id.preprocessing.ulog_loader import LoadedULog, TopicData

# Nearest-match tolerance when pairing local_position_setpoint <-> local_position.
CYCLE_PAIR_TOL_US = 15_000.0  # 15 ms (half a 10 Hz period is 50 ms)


@dataclass
class LoopTimebase:
    loop: str
    master_topic: str
    stamp_kind: str            # "timestamp_sample" or "timestamp"
    t_us: np.ndarray           # master stamps (float64 microseconds)
    t0_us: float               # subtracted origin (log-wide)
    t_s: np.ndarray            # (t_us - t0_us) / 1e6
    dt_median_ms: float
    dt_std_ms: float
    dt_min_ms: float
    dt_max_ms: float
    nonmonotonic_count: int
    effective_hz: float
    n: int


def choose_stamp_us(td: TopicData) -> Tuple[np.ndarray, str]:
    """Return (stamps_us_float64, kind) for a topic per the audit rule."""
    prefer_sample = td.name in PREFER_TIMESTAMP_SAMPLE
    if prefer_sample and td.has_timestamp_sample:
        return np.asarray(td.data["timestamp_sample"], dtype=np.float64), "timestamp_sample"
    return np.asarray(td.data["timestamp"], dtype=np.float64), "timestamp"


def zoh(src_t_us: np.ndarray, src_v: np.ndarray, query_t_us: np.ndarray) -> np.ndarray:
    """Zero-order hold: value of ``src_v`` in effect at each ``query_t_us``.

    Before the first sample, the first value is held (documented, deterministic).
    ``src_t_us`` must be sorted ascending.
    """
    src_t_us = np.asarray(src_t_us, dtype=np.float64)
    src_v = np.asarray(src_v)
    if src_t_us.size == 0:
        return np.full(np.shape(query_t_us), np.nan, dtype=np.float64)
    idx = np.searchsorted(src_t_us, np.asarray(query_t_us, dtype=np.float64), side="right") - 1
    idx = np.clip(idx, 0, src_v.shape[0] - 1)
    return src_v[idx]


def nearest_match(
    ref_t_us: np.ndarray, other_t_us: np.ndarray, tol_us: float = CYCLE_PAIR_TOL_US
) -> np.ndarray:
    """For each ``ref_t_us`` return the index into ``other_t_us`` of the nearest
    sample, or -1 if none is within ``tol_us``.
    """
    ref_t_us = np.asarray(ref_t_us, dtype=np.float64)
    other_t_us = np.asarray(other_t_us, dtype=np.float64)
    if other_t_us.size == 0:
        return np.full(ref_t_us.shape, -1, dtype=np.int64)
    pos = np.searchsorted(other_t_us, ref_t_us)
    lo = np.clip(pos - 1, 0, other_t_us.size - 1)
    hi = np.clip(pos, 0, other_t_us.size - 1)
    d_lo = np.abs(ref_t_us - other_t_us[lo])
    d_hi = np.abs(ref_t_us - other_t_us[hi])
    take_hi = d_hi < d_lo
    idx = np.where(take_hi, hi, lo)
    dist = np.where(take_hi, d_hi, d_lo)
    idx[dist > tol_us] = -1
    return idx.astype(np.int64)


def sample_aligned(
    ref_t_us: np.ndarray, other_t_us: np.ndarray, other_v: np.ndarray,
    tol_us: float = CYCLE_PAIR_TOL_US,
) -> np.ndarray:
    """Nearest-match ``other_v`` onto ``ref_t_us``; unmatched rows -> NaN."""
    other_v = np.asarray(other_v, dtype=np.float64)
    idx = nearest_match(ref_t_us, other_t_us, tol_us)
    out = np.full(ref_t_us.shape, np.nan, dtype=np.float64)
    ok = idx >= 0
    out[ok] = other_v[idx[ok]]
    return out


def build_loop_timebase(loaded: LoadedULog, loop: str, t0_us: float) -> LoopTimebase:
    master_name = LOOP_MASTER_TOPIC[loop]
    td = loaded.topic(master_name)
    t_us, kind = choose_stamp_us(td)
    t_us = np.asarray(t_us, dtype=np.float64)
    dt = np.diff(t_us)
    nonmono = int(np.count_nonzero(dt <= 0))
    dt_ms = dt / 1e3 if dt.size else np.array([np.nan])
    span_s = (t_us[-1] - t_us[0]) / 1e6 if t_us.size > 1 else 0.0
    eff_hz = (t_us.size - 1) / span_s if span_s > 0 else 0.0
    return LoopTimebase(
        loop=loop,
        master_topic=master_name,
        stamp_kind=kind,
        t_us=t_us,
        t0_us=float(t0_us),
        t_s=(t_us - t0_us) / 1e6,
        dt_median_ms=float(np.nanmedian(dt_ms)),
        dt_std_ms=float(np.nanstd(dt_ms)),
        dt_min_ms=float(np.nanmin(dt_ms)),
        dt_max_ms=float(np.nanmax(dt_ms)),
        nonmonotonic_count=nonmono,
        effective_hz=float(eff_hz),
        n=int(t_us.size),
    )


def log_t0_us(loaded: LoadedULog) -> float:
    """Single log-wide origin: min chosen stamp across the master topics.

    Using master-topic stamps (not the raw pre-arm boot stamps of every topic)
    keeps ``t`` small and comparable across loops.
    """
    candidates = []
    for loop, name in LOOP_MASTER_TOPIC.items():
        if loaded.has_topic(name):
            t_us, _ = choose_stamp_us(loaded.topic(name))
            if t_us.size:
                candidates.append(float(t_us[0]))
    return min(candidates) if candidates else float(loaded.start_us)
