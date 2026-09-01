"""Focused tests for the Phase 3 DisturbanceModel (per-episode wind).

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py``, which remains authoritative. These
tests localize a wind-sampling regression by comparing ``DisturbanceModel``
against a verbatim copy of the legacy wind block, using independent generators
started from the same state.

Comparison is exact (``np.array_equal`` / identical RNG state). No tolerance.
"""

from __future__ import annotations

import unittest

import numpy as np

from test_legacy_regression_contract import (
    CONFIG_SPECS,
    NewLandingConfig,
    _rng_state_equal,
)

from landing_rl.disturbances import DisturbanceModel


def _legacy_wind_sample(cfg, rng):
    """Verbatim copy of the wind_accel sampling block from
    mujoco_rl/envs/env_prototype.py :: LandingEnv.reset()."""
    return np.array(
        [
            rng.uniform(
                -cfg.wind_accel_xy_max_mps2,
                cfg.wind_accel_xy_max_mps2,
            ),
            rng.uniform(
                -cfg.wind_accel_xy_max_mps2,
                cfg.wind_accel_xy_max_mps2,
            ),
            rng.uniform(
                -cfg.wind_accel_z_max_mps2,
                cfg.wind_accel_z_max_mps2,
            ),
        ],
        dtype=np.float64,
    )


def _overrides(name):
    for cfg_name, overrides, _doc in CONFIG_SPECS:
        if cfg_name == name:
            return overrides
    raise KeyError(name)


class DisturbanceModelTest(unittest.TestCase):
    maxDiff = None

    # -- construction consumes no RNG ----------------------------------

    def test_constructor_consumes_no_rng(self):
        rng = np.random.default_rng(7)
        before = rng.bit_generator.state
        model = DisturbanceModel(NewLandingConfig())
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        self.assertIsInstance(model.wind_accel, np.ndarray)
        self.assertEqual(model.wind_accel.shape, (3,))
        self.assertEqual(model.wind_accel.dtype, np.dtype(np.float64))
        self.assertTrue(np.array_equal(model.wind_accel, np.zeros(3)))
        self.assertFalse(hasattr(model, "np_random"))
        self.assertFalse(hasattr(model, "rng"))

    # -- zero-wind config -------------------------------------------

    def test_zero_wind_config(self):
        # default LandingConfig has wind_accel_*_max == 0.0
        cfg = NewLandingConfig()
        self.assertEqual(cfg.wind_accel_xy_max_mps2, 0.0)
        self.assertEqual(cfg.wind_accel_z_max_mps2, 0.0)

        model = DisturbanceModel(cfg)
        w = model.reset(np.random.default_rng(0))
        self.assertEqual(w.shape, (3,))
        self.assertEqual(w.dtype, np.dtype(np.float64))
        self.assertTrue(np.array_equal(w, np.zeros(3)))
        self.assertIs(w, model.wind_accel)

    # -- non-zero wind config -------------------------------------

    def test_nonzero_wind_config_within_bounds(self):
        cfg = NewLandingConfig(**_overrides("stage2_eval"))
        self.assertGreater(cfg.wind_accel_xy_max_mps2, 0.0)
        self.assertGreater(cfg.wind_accel_z_max_mps2, 0.0)

        model = DisturbanceModel(cfg)
        for seed in (0, 1, 5000, 12345):
            w = model.reset(np.random.default_rng(seed))
            self.assertEqual(w.shape, (3,))
            self.assertEqual(w.dtype, np.dtype(np.float64))
            self.assertLessEqual(abs(float(w[0])), cfg.wind_accel_xy_max_mps2)
            self.assertLessEqual(abs(float(w[1])), cfg.wind_accel_xy_max_mps2)
            self.assertLessEqual(abs(float(w[2])), cfg.wind_accel_z_max_mps2)
        # at least one seed produced a genuinely non-zero vector
        any_nonzero = any(
            not np.array_equal(
                DisturbanceModel(cfg).reset(np.random.default_rng(s)), np.zeros(3)
            )
            for s in (0, 1, 5000, 12345)
        )
        self.assertTrue(any_nonzero)

    # -- exact parity vs legacy block, same starting RNG state --------

    def test_matches_legacy_block_exactly(self):
        for cfg_name in ("default", "stage0_eval", "stage2_eval", "stage2_train"):
            overrides = {} if cfg_name == "default" else _overrides(cfg_name)
            cfg = NewLandingConfig(**overrides)
            for seed in (0, 1, 5000, 999):
                with self.subTest(config=cfg_name, seed=seed):
                    rng_legacy = np.random.default_rng(seed)
                    rng_model = np.random.default_rng(seed)

                    w_legacy = _legacy_wind_sample(cfg, rng_legacy)
                    w_model = DisturbanceModel(cfg).reset(rng_model)

                    self.assertEqual(w_legacy.dtype, w_model.dtype)
                    self.assertEqual(w_legacy.shape, w_model.shape)
                    self.assertTrue(
                        np.array_equal(w_legacy, w_model),
                        f"[{cfg_name} seed={seed}] wind mismatch: "
                        f"OLD={w_legacy!r} NEW={w_model!r}",
                    )
                    self.assertTrue(
                        _rng_state_equal(
                            rng_legacy.bit_generator.state,
                            rng_model.bit_generator.state,
                        ),
                        f"[{cfg_name} seed={seed}] RNG state diverged after "
                        f"wind sampling",
                    )

    # -- exact draw count (three scalar uniforms) --------------------

    def test_consumes_exactly_three_draws(self):
        cfg = NewLandingConfig(**_overrides("stage2_train"))
        for seed in (0, 1, 5000):
            with self.subTest(seed=seed):
                rng_model = np.random.default_rng(seed)
                DisturbanceModel(cfg).reset(rng_model)

                rng_ref = np.random.default_rng(seed)
                rng_ref.uniform(-cfg.wind_accel_xy_max_mps2, cfg.wind_accel_xy_max_mps2)
                rng_ref.uniform(-cfg.wind_accel_xy_max_mps2, cfg.wind_accel_xy_max_mps2)
                rng_ref.uniform(-cfg.wind_accel_z_max_mps2, cfg.wind_accel_z_max_mps2)

                self.assertTrue(
                    _rng_state_equal(
                        rng_model.bit_generator.state, rng_ref.bit_generator.state
                    )
                )

    # -- regression matrix exercises non-zero wind ------------------

    def test_regression_matrix_has_nonzero_wind_configs(self):
        for cfg_name in ("stage2_eval", "stage2_train"):
            overrides = _overrides(cfg_name)
            self.assertGreater(
                overrides["wind_accel_xy_max_mps2"], 0.0,
                f"{cfg_name} must exercise non-zero wind_accel_xy_max_mps2",
            )
            self.assertGreater(
                overrides["wind_accel_z_max_mps2"], 0.0,
                f"{cfg_name} must exercise non-zero wind_accel_z_max_mps2",
            )


if __name__ == "__main__":
    unittest.main()
