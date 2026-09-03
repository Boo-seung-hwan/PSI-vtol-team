"""landing_rl dynamics package.

Phase 8: the four per-episode response-alpha samples extracted from
``LandingEnv.reset()`` (see ``response_alphas``).
Phase 11: the three in-step process-noise draws (body-rate / thrust /
translational) extracted from ``LandingEnv._update_rigid_body_dynamics()``
(see ``process_noise``).

No behavior change; the OLD-vs-NEW exact parity gate and the PPO/VecNormalize
checkpoint gate remain authoritative.

The rigid-body integrator, the PX4-cascade proxy, the ``noise_scale``
computation, ground effect, and contact all deliberately stay inside
``LandingEnv`` for now.
"""

from landing_rl.dynamics.process_noise import ProcessNoiseSampler
from landing_rl.dynamics.response_alphas import ResponseAlphas

__all__ = ["ProcessNoiseSampler", "ResponseAlphas"]
