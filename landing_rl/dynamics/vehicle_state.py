"""Canonical vehicle-dynamics state container.

Phase 13C-1 of the structural migration. ``VehicleState`` is a deliberately
plain mutable dataclass holding ONLY the physical / rigid-body dynamics fields
that ``LandingEnv._update_rigid_body_dynamics`` reads and writes. It is the
canonical owner of those fields from Phase 13C-1 onward; ``LandingEnv`` keeps
flat ``self.*`` attributes as compatibility mirrors (see
``LandingEnv._sync_vehicle_state_compatibility_fields``).

Field inventory (exactly the 12 identified in the Phase 12 audit):

    pos, vel, accel, prev_accel          -- NED position / velocity /
                                            current+previous acceleration [m, m/s, m/s^2]
    attitude                             -- [roll, pitch, yaw] [rad]
    yaw_rate                             -- scalar [rad/s]
    body_rates                           -- [p, q, r] proxy [rad/s]
    thrust_accel                         -- collective thrust per mass [m/s^2]
    attitude_setpoint                    -- [roll_sp, pitch_sp, yaw_sp] [rad]
    thrust_accel_setpoint                -- scalar [m/s^2]
    accel_cmd                            -- NED acceleration demand [m/s^2]
    ground_effect_factor                 -- thrust multiplier near the pad [-]

Deliberately NOT here (separate owners / explicit inputs):
    target_yaw, wind_accel, motor_cutoff, ground_contact, contact_count,
    bounce_count, the eight contact transient flags, the response alphas, any
    RNG, any cfg.

Types / shapes match the legacy environment exactly: the four position/velocity
/acceleration fields and ``attitude`` / ``body_rates`` / ``attitude_setpoint``
/ ``accel_cmd`` are ``np.ndarray`` shape ``(3,)`` float64; ``yaw_rate`` /
``thrust_accel`` / ``thrust_accel_setpoint`` / ``ground_effect_factor`` are
Python floats.

No behavior change. The OLD-vs-NEW exact parity gate
(``test_legacy_regression_contract.py``) and the PPO / VecNormalize checkpoint
gate (``test_checkpoint_compatibility.py``) remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleState:
    """Mutable rigid-body dynamics state. Plain data -- no methods, no RNG,
    no cfg, no contact / disturbance / perception fields."""

    pos: np.ndarray
    vel: np.ndarray
    accel: np.ndarray
    prev_accel: np.ndarray
    attitude: np.ndarray
    yaw_rate: float
    body_rates: np.ndarray
    thrust_accel: float
    attitude_setpoint: np.ndarray
    thrust_accel_setpoint: float
    accel_cmd: np.ndarray
    ground_effect_factor: float
