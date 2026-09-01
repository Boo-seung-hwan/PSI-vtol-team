"""landing_rl disturbances package.

Phase 3 of the structural migration: per-episode constant wind acceleration
extracted from ``LandingEnv`` (see ``disturbance_model``). No behavior change;
the OLD-vs-NEW exact parity gate remains authoritative.
"""

from landing_rl.disturbances.disturbance_model import DisturbanceModel

__all__ = ["DisturbanceModel"]
