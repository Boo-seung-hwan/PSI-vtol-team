"""landing_rl contact package.

Phase 13B of the structural migration: the legacy ground-contact logic and
contact bookkeeping extracted from ``LandingEnv`` (see ``contact_model``). No
behavior change; the OLD-vs-NEW exact parity gate and the PPO / VecNormalize
checkpoint gate remain authoritative.

The rigid-body free-flight integrator, the PX4-cascade proxy, the three
process-noise draws, ground effect, and the post-contact acceleration
recompute all deliberately stay inside ``LandingEnv`` for now. ``VehicleState``
/ ``PlantModel`` / ``LegacyVehicleDynamics`` are not introduced here.
"""

from landing_rl.contact.contact_model import (
    ContactModel,
    ContactResult,
    ContactState,
)

__all__ = ["ContactModel", "ContactResult", "ContactState"]
