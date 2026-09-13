"""Ground-contact CLASSIFICATION (not resolution).

MuJoCo resolves the actual contact physics (penetration, normal/tangential
impulse, whether the vehicle rebounds) natively via the landing-gear contact
geoms configured in ``mjcf_builder.py`` -- this module does NOT touch
position or velocity. It only reads the kinematic outcome MuJoCo already
produced and classifies it into the same soft/hard/rough/bounce categories
``landing_rl.contact.contact_model.ContactModel`` uses, with the same
threshold fields from the (unmodified) ``LandingConfig``.

Reuses ``ContactState`` / ``ContactResult`` from ``landing_rl`` directly (the
plain per-episode/per-step bookkeeping dataclasses) so the info-field shape
stays identical -- only the ground-truth kinematics feeding the
classification differ (MuJoCo-resolved instead of manually resolved).
"""

from __future__ import annotations

import numpy as np

from landing_rl.contact.contact_model import ContactResult, ContactState


def begin_step(result: ContactResult) -> None:
    """Clear the eight transient per-step result fields. Verbatim
    equivalent of ``ContactModel.begin_step``."""
    result.contact_event = False
    result.soft_contact = False
    result.hard_contact = False
    result.bounced = False
    result.touchdown_quality = "none"
    result.last_impact_vz = 0.0
    result.last_touchdown_vxy = 0.0
    result.last_bounce_speed = 0.0


def classify_touchdown(
    cfg,
    ground_contact_now: bool,
    vel_before_ned: np.ndarray,
    vel_after_ned: np.ndarray,
    attitude: np.ndarray,
    state: ContactState,
    result: ContactResult,
) -> None:
    """Classify the outcome of one control step given whether MuJoCo
    resolved a landing-gear/ground contact this step, and the vehicle's
    velocity immediately before/after. Mutates ``state``/``result`` in
    place. Does NOT touch pos/vel/attitude -- MuJoCo already integrated
    those; this is read-only classification."""
    if not ground_contact_now:
        state.ground_contact = False
        state.contact_count = 0
        return

    result.contact_event = True
    state.ground_contact = True
    state.contact_count += 1

    impact_vz = max(float(vel_after_ned[2]), float(vel_before_ned[2]), 0.0)
    touchdown_vxy = float(np.linalg.norm(vel_after_ned[:2]))
    tilt = float(np.linalg.norm(attitude[:2]))

    result.last_impact_vz = impact_vz
    result.last_touchdown_vxy = touchdown_vxy

    result.hard_contact = (
        impact_vz > cfg.hard_touchdown_vz_mps
        or touchdown_vxy > cfg.hard_touchdown_vxy_mps
        or tilt > cfg.hard_touchdown_tilt_rad
    )
    result.soft_contact = (
        impact_vz <= cfg.touchdown_vz_soft_mps
        and touchdown_vxy <= cfg.touchdown_vxy_soft_mps
        and tilt <= cfg.touchdown_tilt_soft_rad
        and not result.hard_contact
    )

    if result.hard_contact:
        result.touchdown_quality = "hard"
    elif result.soft_contact:
        result.touchdown_quality = "soft"
    else:
        result.touchdown_quality = "rough"

    # Whether the vehicle rebounds is MuJoCo's own contact-resolved outcome
    # (its solref/friction, not a Python restitution formula) -- we only
    # read the resulting vertical speed. NED: negative vel_z = upward.
    rebound_speed = -float(vel_after_ned[2])
    if rebound_speed > cfg.bounce_vz_threshold_mps and not result.soft_contact:
        result.bounced = True
        state.bounce_count += 1
        state.contact_count = 0
        result.last_bounce_speed = rebound_speed
        if not result.hard_contact:
            result.touchdown_quality = "bounce"
    else:
        result.last_bounce_speed = 0.0

    if result.soft_contact and cfg.motor_cutoff_on_soft_contact:
        state.motor_cutoff = True
