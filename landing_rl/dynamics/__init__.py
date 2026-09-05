"""landing_rl dynamics package.

Phase 8: the four per-episode response-alpha samples extracted from
``LandingEnv.reset()`` (see ``response_alphas``).
Phase 11: the three in-step process-noise draws (body-rate / thrust /
translational) extracted from ``LandingEnv._update_rigid_body_dynamics()``
(see ``process_noise``).
Phase 13C-1: the canonical ``VehicleState`` container (see ``vehicle_state``)
and the free-flight / PX4-proxy rigid-body integration -- D4-D25 of
``LandingEnv._update_rigid_body_dynamics`` -- extracted into
``LegacyVehicleDynamics`` (see ``legacy_dynamics``).
Phase 13C-2: the surrounding D2/D3/D26/D27 orchestration -- previously inline
in ``LandingEnv._update_rigid_body_dynamics`` -- extracted into ``PlantModel``
(see ``plant_model``), which coordinates ``LegacyVehicleDynamics`` (free
flight) and ``ContactModel`` (ground contact) in the same order.

No behavior change; the OLD-vs-NEW exact parity gate and the PPO/VecNormalize
checkpoint gate remain authoritative.

``LandingEnv`` still owns the canonical ``VehicleState``, the ``ContactModel``
instance, and the compatibility-mirror sync calls; it calls
``PlantModel.step(...)`` once per control step and re-syncs its flat mirrors
from the (mutated in place) ``VehicleState`` / ``ContactModel`` afterward.
"""

from landing_rl.dynamics.legacy_dynamics import LegacyVehicleDynamics
from landing_rl.dynamics.plant_model import PlantModel
from landing_rl.dynamics.process_noise import ProcessNoiseSampler
from landing_rl.dynamics.response_alphas import ResponseAlphas
from landing_rl.dynamics.vehicle_state import VehicleState

__all__ = [
    "LegacyVehicleDynamics",
    "PlantModel",
    "ProcessNoiseSampler",
    "ResponseAlphas",
    "VehicleState",
]
