"""Raw target-measurement model (perception front-end).

Phase 6 extraction. This helper holds ONLY the raw absolute-NED
target-measurement stochastic model that previously lived on ``LandingEnv``:

    LandingEnv._sample_raw_target_measurement()  ->  TargetMeasurementModel.sample(rng, target_true)

The boundary is:

    TRUE TARGET  ->  TargetMeasurementModel  ->  (raw_target, valid, mode)

The observation-delay queue (``obs_delay_steps`` / ``target_queue`` /
``obs_target`` / ``obs_target_valid`` / ``obs_target_mode`` /
``target_measured``) stays in ``LandingEnv`` -- it is NOT part of this
component in Phase 6.

Verbatim copy
------------
The body of ``sample`` is copied character-for-character from
``mujoco_rl/envs/env_prototype.py`` (the immutable reference), with only the
receiver rebindings ``self.np_random -> rng`` and ``self.target_true ->
target_true``. The branch order, the conditional early returns, the three
scalar ``normal`` draws (not vectorized), the noise-norm clamp
``* max / (norm + 1e-9)``, the ``random()`` gates (not ``uniform``), and the
``uniform(..., size=2)`` outlier draw are unchanged. RNG consumption is
DATA-DEPENDENT and branch-dependent exactly as before.

Critical semantics (unchanged):

    dropout        -> returns last_raw_target.copy(), valid=False, mode="dropout"
                      last_raw_target is NOT updated
    stale          -> returns last_raw_target.copy(), valid=True,  mode="stale"
                      last_raw_target is NOT updated
    normal/outlier -> last_raw_target = target.copy(); returns target, True, mode

Ownership
---------
Owns ONLY ``cfg`` and ``last_raw_target``. Does NOT own an RNG,
``obs_delay_steps``, ``target_queue``, ``obs_target*``, position, velocity,
attitude, the controller, or any environment reference. The constructor and
``reset`` consume zero random numbers. ``LandingEnv`` passes its own
``np_random`` into ``sample`` and remains the sole RNG owner.
"""

from __future__ import annotations

import numpy as np


class TargetMeasurementModel:
    """Dropout / stale / noisy / outlier raw target-measurement model."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_raw_target = np.zeros(3, dtype=np.float64)

    def reset(self, target_true) -> None:
        """Reset the measurement history to the true target. Consumes no RNG.

        ``last_raw_target`` is rebound to ``target_true.copy()`` (same as the
        legacy ``self.last_raw_target = self.target_true.copy()`` in
        ``LandingEnv.reset``).
        """
        self.last_raw_target = target_true.copy()

    def sample(self, rng, target_true) -> tuple[np.ndarray, bool, str]:
        """Sample a raw absolute NED target measurement.

        ``rng`` must be ``LandingEnv.np_random``. Verbatim copy of the legacy
        ``LandingEnv._sample_raw_target_measurement`` (see module docstring).

        Returns:
            target: absolute NED target measurement
            valid: target valid flag
            mode:  string label for debugging/logging
        """
        # Dropout: detection failed. Keep the previous target value but mark invalid.
        if rng.random() < self.cfg.target_dropout_prob:
            return self.last_raw_target.copy(), False, "dropout"

        # Stale target: old measurement is repeated but still marked valid.
        if rng.random() < self.cfg.target_stale_prob:
            return self.last_raw_target.copy(), True, "stale"

        noise = np.array(
            [
                rng.normal(0.0, self.cfg.target_noise_xy_std_m),
                rng.normal(0.0, self.cfg.target_noise_xy_std_m),
                rng.normal(0.0, self.cfg.target_noise_z_std_m),
            ],
            dtype=np.float64,
        )

        noise_norm = float(np.linalg.norm(noise))
        if noise_norm > self.cfg.target_noise_max_m:
            noise *= self.cfg.target_noise_max_m / (noise_norm + 1e-9)

        target = target_true + noise
        mode = "normal"

        if rng.random() < self.cfg.target_outlier_prob:
            target[:2] += rng.uniform(
                -self.cfg.target_outlier_xy_m,
                self.cfg.target_outlier_xy_m,
                size=2,
            )
            mode = "outlier"

        self.last_raw_target = target.copy()
        return target, True, mode
