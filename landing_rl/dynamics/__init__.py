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

No behavior change; the OLD-vs-NEW exact parity gate and the PPO/VecNormalize
checkpoint gate remain authoritative.

Still inside ``LandingEnv`` (NOT here): the D2 ``contact.begin_step()`` +
``vel_before`` snapshot, the D26 ``ContactModel.apply(...)`` call, and the D27
post-contact ``accel = (vel - vel_before) / dt_safe`` recompute. No
``PlantModel`` -- ``LandingEnv`` still orchestrates the free-flight / contact /
finalize sequence.
"""

from landing_rl.dynamics.legacy_dynamics import LegacyVehicleDynamics
from landing_rl.dynamics.process_noise import ProcessNoiseSampler
from landing_rl.dynamics.response_alphas import ResponseAlphas
from landing_rl.dynamics.vehicle_state import VehicleState

__all__ = [
    "LegacyVehicleDynamics",
    "ProcessNoiseSampler",
    "ResponseAlphas",
    "VehicleState",
]
