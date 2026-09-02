"""landing_rl dynamics package.

Phase 8 of the structural migration: the four per-episode response-alpha
samples extracted from ``LandingEnv.reset()`` (see ``response_alphas``). No
behavior change; the OLD-vs-NEW exact parity gate and the PPO/VecNormalize
checkpoint gate remain authoritative.

The rigid-body integrator, the PX4-cascade proxy, process noise, ground
effect, and contact all deliberately stay inside ``LandingEnv`` for now.
"""

from landing_rl.dynamics.response_alphas import ResponseAlphas

__all__ = ["ResponseAlphas"]
