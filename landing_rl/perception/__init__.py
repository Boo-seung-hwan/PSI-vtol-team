"""landing_rl perception package.

Phase 6: the raw target-measurement stochastic model extracted from
``LandingEnv`` (see ``target_measurement``).
Phase 7: the target observation-delay state + FIFO queue extracted from
``LandingEnv`` (see ``obs_latency``).
Phase 10: the vehicle-state observation-noise draws (vel / accel / attitude)
extracted from ``LandingEnv._get_obs`` (see ``observation_noise``).

No behavior change; the OLD-vs-NEW exact parity gate and the PPO/VecNormalize
checkpoint gate remain authoritative.

``obs_target`` / ``obs_target_valid`` / ``obs_target_mode`` / ``target_measured``
and all deterministic observation assembly (accel clip, yaw wrap, concatenate,
float32 cast) deliberately stay inside ``LandingEnv._get_obs``.
"""

from landing_rl.perception.obs_latency import ObsLatency
from landing_rl.perception.observation_noise import ObservationNoiseSampler
from landing_rl.perception.target_measurement import TargetMeasurementModel

__all__ = ["ObsLatency", "ObservationNoiseSampler", "TargetMeasurementModel"]
