"""landing_rl perception package.

Phase 6 of the structural migration: the raw target-measurement stochastic
model extracted from ``LandingEnv`` (see ``target_measurement``). No behavior
change; the OLD-vs-NEW exact parity gate and the PPO/VecNormalize checkpoint
gate remain authoritative.

The observation-delay queue (``obs_delay_steps`` / ``target_queue`` /
``obs_target*``) deliberately stays inside ``LandingEnv`` for now.
"""

from landing_rl.perception.target_measurement import TargetMeasurementModel

__all__ = ["TargetMeasurementModel"]
