"""Ground-contact model.

Phase 13B extraction. This module holds the legacy ground-contact logic and
contact bookkeeping lifted verbatim from
``mujoco_rl/envs/env_prototype.py`` (the immutable reference), specifically:

    LandingEnv._reset_contact_step_flags()   ->  ContactModel.begin_step()
    LandingEnv._apply_ground_contact(...)     ->  ContactModel.apply(...)
    LandingEnv.reset() contact-state init     ->  ContactModel.reset()

No behavior change. The OLD-vs-NEW exact parity gate
(``test_legacy_regression_contract.py``) and the PPO / VecNormalize checkpoint
gate (``test_checkpoint_compatibility.py``) remain authoritative.

What is NOT here (Phase 13B is contact-only)
-------------------------------------------
* The rigid-body free-flight integrator, the PX4-cascade proxy, ``noise_scale``,
  the three ``ProcessNoiseSampler`` draws, and the post-contact
  ``accel = (vel - vel_before) / dt_safe`` recompute all stay in
  ``LandingEnv._update_rigid_body_dynamics``. Topology stays transitional
  "B" (env orchestrates: free-flight -> ContactModel.apply -> finalize accel).
* No ``VehicleState`` / ``PlantModel`` / ``LegacyVehicleDynamics``. Those belong
  to the later dynamics migration (Phase 13C).
* ``_compute_ground_effect_factor`` stays in ``LandingEnv`` and still reads the
  env's (synchronized) ``ground_contact`` mirror at D18, before this step's D26
  contact resolution.

Ownership
---------
``ContactModel`` owns ONLY ``cfg``, one ``ContactState`` (the four persistent
episode fields), and one ``ContactResult`` (the eight transient per-step
fields). It does NOT own an RNG, position, velocity, attitude, thrust state, or
any environment reference. The constructor and ``reset()`` consume zero random
numbers. ``apply()`` consumes exactly one ``rng.uniform(...)`` draw, and only
on the legacy ``should_bounce`` branch.

The vehicle arrays ``pos`` / ``vel`` / ``attitude`` are passed into ``apply()``
by reference and mutated in place exactly as the legacy method did
(``pos[2] = ground_z``, ``vel[:2] *= ...``, ``vel[2] = ...``). The scalar
thrust values are passed by value and returned, because a Python ``float``
cannot be mutated through a reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContactState:
    """Persistent per-episode contact bookkeeping.

    These four fields survive across steps; they are initialized by
    ``ContactModel.reset()`` and mutated only by ``ContactModel.apply()``.
    They are NOT touched by ``begin_step()``.
    """

    ground_contact: bool = False
    contact_count: int = 0
    bounce_count: int = 0
    motor_cutoff: bool = False


@dataclass
class ContactResult:
    """Transient per-step contact outputs.

    These are exactly the eight fields the legacy
    ``_reset_contact_step_flags()`` zeroed at the top of every dynamics step.
    Defaults match the legacy reset values verbatim.
    """

    contact_event: bool = False
    soft_contact: bool = False
    hard_contact: bool = False
    bounced: bool = False
    touchdown_quality: str = "none"
    last_impact_vz: float = 0.0
    last_touchdown_vxy: float = 0.0
    last_bounce_speed: float = 0.0


class ContactModel:
    """Legacy ground-contact resolution + bookkeeping.

    Stateless beyond ``cfg``, ``state`` (persistent), and ``result``
    (transient). No RNG is owned; ``apply()`` takes ``LandingEnv.np_random``.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = ContactState()
        self.result = ContactResult()

    # ------------------------------------------------------------------
    # reset / begin_step
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Restore the legacy per-episode contact-state init.

        Mirrors ``LandingEnv.reset()`` lines that set ``ground_contact=False``,
        ``contact_count=0``, ``bounce_count=0``, ``motor_cutoff=False`` and then
        call ``_reset_contact_step_flags()``. Consumes zero RNG.
        """
        self.state.ground_contact = False
        self.state.contact_count = 0
        self.state.bounce_count = 0
        self.state.motor_cutoff = False
        self.begin_step()

    def begin_step(self) -> None:
        """Verbatim ``_reset_contact_step_flags()``: zero the eight transient
        result fields. Does NOT touch any persistent ``ContactState`` field."""
        self.result.contact_event = False
        self.result.soft_contact = False
        self.result.hard_contact = False
        self.result.bounced = False
        self.result.touchdown_quality = "none"
        self.result.last_impact_vz = 0.0
        self.result.last_touchdown_vxy = 0.0
        self.result.last_bounce_speed = 0.0

    # ------------------------------------------------------------------
    # apply -- verbatim _apply_ground_contact
    # ------------------------------------------------------------------
    def apply(
        self,
        *,
        pos: np.ndarray,
        vel: np.ndarray,
        attitude: np.ndarray,
        vel_before_step: np.ndarray,
        thrust_accel: float,
        thrust_accel_setpoint: float,
        rng,
    ):
        """Resolve simple ground contact and bounce at ``z = ground_z_m``.

        ``pos`` / ``vel`` / ``attitude`` are the caller's arrays and are mutated
        in place exactly as the legacy method did. ``rng`` must be
        ``LandingEnv.np_random``; it is consumed only on the ``should_bounce``
        branch.

        Returns ``(result, thrust_accel, thrust_accel_setpoint)``. The two
        scalars are returned because contact may overwrite them on a soft
        touchdown and a Python ``float`` cannot be mutated by reference.
        """
        if not self.cfg.contact_enabled:
            return self.result, thrust_accel, thrust_accel_setpoint

        ground_z = float(self.cfg.ground_z_m)
        if pos[2] < ground_z:
            self.state.ground_contact = False
            self.state.contact_count = 0
            return self.result, thrust_accel, thrust_accel_setpoint

        self.result.contact_event = True
        self.state.ground_contact = True
        self.state.contact_count += 1
        pos[2] = ground_z

        impact_vz = max(float(vel[2]), float(vel_before_step[2]), 0.0)
        touchdown_vxy = float(np.linalg.norm(vel[:2]))
        tilt = float(np.linalg.norm(attitude[:2]))

        self.result.last_impact_vz = impact_vz
        self.result.last_touchdown_vxy = touchdown_vxy

        self.result.hard_contact = (
            impact_vz > self.cfg.hard_touchdown_vz_mps
            or touchdown_vxy > self.cfg.hard_touchdown_vxy_mps
            or tilt > self.cfg.hard_touchdown_tilt_rad
        )

        self.result.soft_contact = (
            impact_vz <= self.cfg.touchdown_vz_soft_mps
            and touchdown_vxy <= self.cfg.touchdown_vxy_soft_mps
            and tilt <= self.cfg.touchdown_tilt_soft_rad
            and not self.result.hard_contact
        )

        if self.result.hard_contact:
            self.result.touchdown_quality = "hard"
        elif self.result.soft_contact:
            self.result.touchdown_quality = "soft"
        else:
            self.result.touchdown_quality = "rough"

        # Tangential ground friction removes lateral sliding at contact.
        vel[:2] *= float(np.clip(1.0 - self.cfg.ground_friction_xy, 0.0, 1.0))

        should_bounce = (
            impact_vz > self.cfg.bounce_vz_threshold_mps
            and not self.result.soft_contact
        )

        if should_bounce:
            restitution = float(rng.uniform(
                self.cfg.bounce_restitution_min,
                self.cfg.bounce_restitution_max,
            ))
            bounce_speed = restitution * impact_vz
            vel[2] = -bounce_speed  # negative NED z means rebound upward
            self.result.bounced = True
            self.state.bounce_count += 1
            self.state.contact_count = 0
            self.result.last_bounce_speed = bounce_speed
            if not self.result.hard_contact:
                self.result.touchdown_quality = "bounce"
        else:
            # No rebound: vehicle stays on the ground.
            vel[2] = 0.0
            self.result.last_bounce_speed = 0.0

        if self.result.soft_contact and self.cfg.motor_cutoff_on_soft_contact:
            self.state.motor_cutoff = True
            thrust_accel_setpoint = float(self.cfg.motor_cutoff_thrust_accel_mps2)
            thrust_accel = float(self.cfg.motor_cutoff_thrust_accel_mps2)

        return self.result, thrust_accel, thrust_accel_setpoint
