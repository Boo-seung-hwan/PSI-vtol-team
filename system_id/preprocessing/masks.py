"""Per-sample validity masks with explicit rejection reasons.

There is ONE common flight-valid base mask plus BLOCK-SPECIFIC masks.

Base (``flight_base``) -- a sample is in normal offboard multicopter flight:
armed, offboard, multicopter-position-control enabled, not landed /
ground-contact / maybe-landed, no motor at its rail, and the logged
acceleration setpoint is not the PX4 "no-fly" sentinel ``Vector3f(0,0,100)``.

Block masks start from ``flight_base`` and add ONLY the extra rejections that
can actually contaminate that block's physical output:

    attitude_valid      = flight_base
    rate_valid          = flight_base
    velocity_xy_valid   = flight_base
    velocity_z_valid    = flight_base  & ~in_ground_effect
    thrust_valid        = flight_base  & ~in_ground_effect  & near_level
    translation_valid   = flight_base  & ~in_ground_effect

Ground effect augments vertical lift; it does not distort the
attitude-setpoint->attitude or rate-setpoint->body-rate response, so low
altitude alone (no contact, no saturation) is NOT a reason to drop
attitude/rate data.

The ``near_level`` term and the required-I/O-finite / setpoint-pairing terms
are supplied by ``dataset.py`` (they need the assembled columns) and combined
via :meth:`MaskResult.combine_block`.

Every threshold is a named module constant, never an inline literal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from system_id.preprocessing.schema import NEAR_LEVEL_MAX_RAD
from system_id.preprocessing.timebase import choose_stamp_us, zoh
from system_id.preprocessing.ulog_loader import LoadedULog

# --- named thresholds -----------------------------------------------------
MOTOR_SAT_HIGH = 0.98          # actuator_motors.control >= this -> upper-rail
MOTOR_SAT_LOW = 0.02           # actuator_motors.control <= this -> lower-rail
MOTOR_SAT_DILATION_S = 0.03    # widen each saturated sample by +/- this
N_MOTORS_CHECKED = 4           # quad; control[0..3]
ACCEL_SENTINEL_ABS_MPS2 = 30.0  # |vehicle_local_position_setpoint.acceleration[2]| above this
#                                 is PX4's Vector3f(0,0,100) no-thrust injection, not a demand
# NEAR_LEVEL_MAX_RAD is imported from schema (single source of truth).

REJECTION_REASONS: Tuple[str, ...] = (
    "not_armed",
    "not_offboard",
    "not_mc_pos",
    "landed",
    "ground_contact",
    "maybe_landed",
    "motor_saturated",
    "accel_sentinel",
    "in_ground_effect",
)
# The subset that defines flight_base (order = first_reason priority).
BASE_REASONS: Tuple[str, ...] = tuple(r for r in REJECTION_REASONS if r != "in_ground_effect")

# --- block model --------------------------------------------------------
BLOCK_NAMES: Tuple[str, ...] = (
    "attitude", "rate", "velocity_xy", "velocity_z", "thrust", "translation",
)
BLOCK_GE_SENSITIVE = frozenset({"velocity_z", "thrust", "translation"})
BLOCK_NEAR_LEVEL_REQUIRED = frozenset({"thrust"})
# Which blocks each loop dataset can serve (a loop may feed more than one).
LOOP_BLOCKS: Dict[str, Tuple[str, ...]] = {
    "velocity": ("velocity_xy", "velocity_z"),
    "attitude": ("attitude", "thrust"),
    "rate": ("rate",),
    "translation": ("translation", "thrust"),
}


@dataclass
class MaskResult:
    loop: str
    n: int
    reason_masks: Dict[str, np.ndarray]     # reason -> bool[n], True == this reason rejects the sample
    flight_base: np.ndarray                 # bool[n]: BASE_REASONS all clear
    ground_effect: np.ndarray               # bool[n]: True == in_ground_effect
    first_reason: np.ndarray                # object[n]: first failing BASE reason, else "" (GE not counted here)
    summary: Dict[str, float] = field(default_factory=dict)   # reason -> fraction rejected

    # Backward-compatible aliases (flight_base == old valid_core).
    @property
    def valid_core(self) -> np.ndarray:
        return self.flight_base

    @property
    def valid_strict(self) -> np.ndarray:
        return self.flight_base & ~self.ground_effect

    def block_base(self, block: str) -> np.ndarray:
        """flight_base plus the ground-effect term for GE-sensitive blocks.

        The near-level and finite/pairing terms are added by
        :meth:`combine_block` in dataset.py (they need assembled columns).
        """
        if block not in BLOCK_NAMES:
            raise KeyError(f"unknown SI block {block!r}; known: {BLOCK_NAMES}")
        m = self.flight_base.copy()
        if block in BLOCK_GE_SENSITIVE:
            m &= ~self.ground_effect
        return m

    def combine_block(self, block: str, *extra: np.ndarray) -> np.ndarray:
        """block_base(block) AND every extra bool[n] mask (near-level, finite,
        setpoint-pairing). Any False in an extra term rejects the sample."""
        m = self.block_base(block)
        for e in extra:
            if e is None:
                continue
            m = m & np.asarray(e, dtype=bool)
        return m


def _step_zoh(loaded: LoadedULog, topic: str, field_name: str, t_us: np.ndarray) -> np.ndarray:
    td = loaded.topic(topic)
    src_t, _ = choose_stamp_us(td)
    return zoh(src_t, np.asarray(td.get(field_name), dtype=np.float64), t_us) > 0.5


def motor_saturation_series(loaded: LoadedULog) -> Tuple[np.ndarray, np.ndarray]:
    """Return (stamp_us, sat_bool) at the actuator_motors rate.

    ``sat_bool[k]`` is True if any of control[0..N_MOTORS_CHECKED-1] is at or
    beyond a rail at sample k.
    """
    td = loaded.topic("actuator_motors")
    t_us, _ = choose_stamp_us(td)
    cols = []
    for i in range(N_MOTORS_CHECKED):
        key = f"control[{i}]"
        if td.has(key):
            cols.append(np.asarray(td.get(key), dtype=np.float64))
    if not cols:
        return t_us, np.zeros(t_us.shape, dtype=bool)
    m = np.column_stack(cols)
    finite = np.isfinite(m)
    hi = np.where(finite, m >= MOTOR_SAT_HIGH, False).any(axis=1)
    lo = np.where(finite, m <= MOTOR_SAT_LOW, False).any(axis=1)
    return t_us, (hi | lo)


def _dilate_bool(t_us: np.ndarray, flag: np.ndarray, query_t_us: np.ndarray, half_width_s: float) -> np.ndarray:
    """ZOH ``flag`` onto ``query_t_us`` then OR-expand by +/- half_width_s."""
    base = zoh(t_us, flag.astype(np.float64), query_t_us) > 0.5
    if half_width_s <= 0 or not base.any():
        return base
    hw_us = half_width_s * 1e6
    out = base.copy()
    qi = np.asarray(query_t_us, dtype=np.float64)
    src_true_t = t_us[flag.astype(bool)] if flag.any() else np.array([])
    for tt in src_true_t:
        out |= np.abs(qi - tt) <= hw_us
    return out


def compute_masks(loaded: LoadedULog, t_us: np.ndarray, loop: str) -> MaskResult:
    """Build the reason masks + ``flight_base`` on the loop's own timebase."""
    t_us = np.asarray(t_us, dtype=np.float64)
    n = int(t_us.size)

    armed = _step_zoh(loaded, "vehicle_control_mode", "flag_armed", t_us)
    offb = _step_zoh(loaded, "vehicle_control_mode", "flag_control_offboard_enabled", t_us)
    mcpos = _step_zoh(loaded, "vehicle_control_mode", "flag_multicopter_position_control_enabled", t_us)
    landed = _step_zoh(loaded, "vehicle_land_detected", "landed", t_us)
    gcontact = _step_zoh(loaded, "vehicle_land_detected", "ground_contact", t_us)
    mlanded = _step_zoh(loaded, "vehicle_land_detected", "maybe_landed", t_us)
    try:
        ige = _step_zoh(loaded, "vehicle_land_detected", "in_ground_effect", t_us)
    except KeyError:
        ige = np.zeros(n, dtype=bool)

    sat_t, sat_flag = motor_saturation_series(loaded)
    sat = _dilate_bool(sat_t, sat_flag, t_us, MOTOR_SAT_DILATION_S)

    lps = loaded.topic("vehicle_local_position_setpoint")
    lps_t, _ = choose_stamp_us(lps)
    a_sp_d = zoh(lps_t, np.asarray(lps.get("acceleration[2]"), dtype=np.float64), t_us)
    sentinel = ~np.isfinite(a_sp_d) | (np.abs(a_sp_d) > ACCEL_SENTINEL_ABS_MPS2)

    reasons: Dict[str, np.ndarray] = {
        "not_armed": ~armed,
        "not_offboard": ~offb,
        "not_mc_pos": ~mcpos,
        "landed": landed,
        "ground_contact": gcontact,
        "maybe_landed": mlanded,
        "motor_saturated": sat,
        "accel_sentinel": sentinel,
        "in_ground_effect": ige,
    }

    reject_base = np.zeros(n, dtype=bool)
    for r in BASE_REASONS:
        reject_base |= reasons[r]
    flight_base = ~reject_base

    first_reason = np.empty(n, dtype=object)
    first_reason[:] = ""
    remaining = reject_base.copy()
    for r in BASE_REASONS:
        hit = remaining & reasons[r]
        first_reason[hit] = r
        remaining &= ~reasons[r]

    summary = {r: (float(np.mean(m)) if n else 0.0) for r, m in reasons.items()}

    return MaskResult(
        loop=loop,
        n=n,
        reason_masks=reasons,
        flight_base=flight_base,
        ground_effect=reasons["in_ground_effect"],
        first_reason=first_reason,
        summary=summary,
    )
