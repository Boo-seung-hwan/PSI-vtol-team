"""landing_rl environments package.

Phase 1 of the structural migration. ``landing_env`` is currently a
byte-for-byte copy of ``mujoco_rl/envs/env_prototype.py`` (the frozen legacy
reference). No behavior is changed in this phase; only the module location
differs. Modular decomposition (controller / dynamics / perception /
disturbance / contact / reward) is deferred to later phases.
"""

from landing_rl.envs.landing_env import LandingConfig, LandingEnv

__all__ = ["LandingConfig", "LandingEnv"]
