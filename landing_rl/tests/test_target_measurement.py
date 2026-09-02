"""Focused tests for the Phase 6 TargetMeasurementModel (raw target sampling).

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py`` and the checkpoint gate in
``test_checkpoint_compatibility.py``, both of which remain authoritative.

This is the first extracted component with DATA-DEPENDENT RNG consumption, so
the tests focus on branch order and per-branch draw counts. Each branch is
forced via probability overrides and compared against a reference
``np.random.Generator`` that performs exactly the expected draw sequence;
RNG-state equality after the call is the draw-count assertion. A verbatim
standalone replica of the legacy method is also compared over long random
sequences on independent generators started from the same state.

Comparison is exact (``np.array_equal`` / ``==`` / identical RNG state).
No tolerance.
"""

from __future__ import annotations

import unittest

import numpy as np

from test_legacy_regression_contract import (
    CONFIG_SPECS,
    NewLandingConfig,
    NewLandingEnv,
    _rng_state_equal,
)

from landing_rl.perception import TargetMeasurementModel


# ---------------------------------------------------------------------------
# Verbatim standalone replica of
# mujoco_rl/envs/env_prototype.py :: LandingEnv._sample_raw_target_measurement
# (+ the reset assignment), with receiver rebindings only
# (self.np_random -> rng, self.target_true -> target_true).
# ---------------------------------------------------------------------------

class _LegacyMeasurement:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_raw_target = np.zeros(3, dtype=np.float64)

    def reset(self, target_true):
        self.last_raw_target = target_true.copy()

    def sample(self, rng, target_true):
        if rng.random() < self.cfg.target_dropout_prob:
            return self.last_raw_target.copy(), False, "dropout"

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


def _overrides(name):
    for cfg_name, overrides, _doc in CONFIG_SPECS:
        if cfg_name == name:
            return overrides
    raise KeyError(name)


_CONFIG_NAMES = ("default", "stage0_eval", "stage2_eval", "stage2_train")


def _cfg(**extra):
    """A config with real non-zero target-noise magnitudes (from stage2_train),
    plus any branch-forcing probability overrides."""
    base = dict(_overrides("stage2_train"))
    base.update(extra)
    return NewLandingConfig(**base)


_TT = np.array([1.25, -0.75, 2.0], dtype=np.float64)  # an arbitrary true target


class TargetMeasurementModelTest(unittest.TestCase):
    maxDiff = None

    # -- A. constructor consumes zero RNG ----------------------------

    def test_constructor_consumes_no_rng(self):
        rng = np.random.default_rng(4)
        before = rng.bit_generator.state
        model = TargetMeasurementModel(NewLandingConfig())
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        self.assertEqual(set(vars(model)), {"cfg", "last_raw_target"})
        self.assertEqual(model.last_raw_target.dtype, np.dtype(np.float64))
        self.assertEqual(model.last_raw_target.shape, (3,))
        self.assertTrue(np.array_equal(model.last_raw_target, np.zeros(3)))
        self.assertFalse(hasattr(model, "np_random"))
        self.assertFalse(hasattr(model, "rng"))

    # -- B. reset(target_true): zero RNG, last_raw_target == target_true

    def test_reset_consumes_no_rng_and_copies_target(self):
        rng = np.random.default_rng(9)
        before = rng.bit_generator.state
        model = TargetMeasurementModel(_cfg())
        model.reset(_TT)
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        self.assertTrue(np.array_equal(model.last_raw_target, _TT))
        self.assertIsNot(model.last_raw_target, _TT)  # a copy, not the same array
        # mutating the source afterwards must not touch the stored history
        tt2 = _TT.copy()
        model.reset(tt2)
        tt2 += 5.0
        self.assertTrue(np.array_equal(model.last_raw_target, _TT))

    # -- C. forced dropout branch ---------------------------------

    def test_forced_dropout_branch(self):
        cfg = _cfg(target_dropout_prob=1.0, target_stale_prob=0.0,
                   target_outlier_prob=0.0)
        for seed in (0, 1, 5000, 77):
            with self.subTest(seed=seed):
                model = TargetMeasurementModel(cfg)
                model.reset(_TT)
                prev = model.last_raw_target.copy()

                rng_model = np.random.default_rng(seed)
                out, valid, mode = model.sample(rng_model, _TT)

                self.assertEqual(mode, "dropout")
                self.assertFalse(valid)
                self.assertTrue(np.array_equal(out, prev))
                self.assertIsNot(out, model.last_raw_target)  # returns a copy
                self.assertTrue(np.array_equal(model.last_raw_target, prev))  # unchanged

                # exactly one random() draw consumed
                rng_ref = np.random.default_rng(seed)
                rng_ref.random()
                self.assertTrue(
                    _rng_state_equal(
                        rng_model.bit_generator.state, rng_ref.bit_generator.state
                    ),
                    f"[seed={seed}] dropout branch must consume exactly one random()",
                )

    # -- D. forced stale branch ---------------------------------

    def test_forced_stale_branch(self):
        cfg = _cfg(target_dropout_prob=0.0, target_stale_prob=1.0,
                   target_outlier_prob=0.0)
        for seed in (0, 1, 5000, 77):
            with self.subTest(seed=seed):
                model = TargetMeasurementModel(cfg)
                model.reset(_TT)
                prev = model.last_raw_target.copy()

                rng_model = np.random.default_rng(seed)
                out, valid, mode = model.sample(rng_model, _TT)

                self.assertEqual(mode, "stale")
                self.assertTrue(valid)
                self.assertTrue(np.array_equal(out, prev))
                self.assertTrue(np.array_equal(model.last_raw_target, prev))  # unchanged

                # exactly two random() draws consumed (dropout check, stale check)
                rng_ref = np.random.default_rng(seed)
                rng_ref.random()
                rng_ref.random()
                self.assertTrue(
                    _rng_state_equal(
                        rng_model.bit_generator.state, rng_ref.bit_generator.state
                    ),
                    f"[seed={seed}] stale branch must consume exactly two random()",
                )

    # -- E. normal measurement branch --------------------------

    def test_forced_normal_branch(self):
        cfg = _cfg(target_dropout_prob=0.0, target_stale_prob=0.0,
                   target_outlier_prob=0.0,
                   target_noise_xy_std_m=0.04, target_noise_z_std_m=0.03,
                   target_noise_max_m=1e9)  # effectively no clipping
        for seed in (0, 1, 5000, 77):
            with self.subTest(seed=seed):
                model = TargetMeasurementModel(cfg)
                model.reset(_TT)

                rng_model = np.random.default_rng(seed)
                out, valid, mode = model.sample(rng_model, _TT)

                self.assertEqual(mode, "normal")
                self.assertTrue(valid)

                # reference: 2x random() + 3x normal() + 1x random()
                rng_ref = np.random.default_rng(seed)
                rng_ref.random()
                rng_ref.random()
                nx = rng_ref.normal(0.0, cfg.target_noise_xy_std_m)
                ny = rng_ref.normal(0.0, cfg.target_noise_xy_std_m)
                nz = rng_ref.normal(0.0, cfg.target_noise_z_std_m)
                rng_ref.random()
                exp_noise = np.array([nx, ny, nz], dtype=np.float64)
                exp = _TT + exp_noise

                self.assertTrue(np.array_equal(out, exp), f"{out!r} != {exp!r}")
                self.assertEqual(out.dtype, np.dtype(np.float64))
                self.assertTrue(np.array_equal(model.last_raw_target, exp))
                self.assertIsNot(model.last_raw_target, out)  # stored is a copy
                self.assertTrue(
                    _rng_state_equal(
                        rng_model.bit_generator.state, rng_ref.bit_generator.state
                    ),
                    f"[seed={seed}] normal branch draw count mismatch",
                )

    # -- F. measurement-noise clipping branch -----------------

    def test_forced_noise_clipping_branch(self):
        # huge std + tiny max => the clip path is taken every time
        cfg = _cfg(target_dropout_prob=0.0, target_stale_prob=0.0,
                   target_outlier_prob=0.0,
                   target_noise_xy_std_m=50.0, target_noise_z_std_m=50.0,
                   target_noise_max_m=0.15)
        for seed in (0, 1, 5000, 77):
            with self.subTest(seed=seed):
                model = TargetMeasurementModel(cfg)
                model.reset(_TT)

                rng_model = np.random.default_rng(seed)
                out, valid, mode = model.sample(rng_model, _TT)
                self.assertEqual(mode, "normal")

                rng_ref = np.random.default_rng(seed)
                rng_ref.random()
                rng_ref.random()
                noise = np.array(
                    [
                        rng_ref.normal(0.0, cfg.target_noise_xy_std_m),
                        rng_ref.normal(0.0, cfg.target_noise_xy_std_m),
                        rng_ref.normal(0.0, cfg.target_noise_z_std_m),
                    ],
                    dtype=np.float64,
                )
                nn = float(np.linalg.norm(noise))
                self.assertGreater(nn, cfg.target_noise_max_m)  # clip really engaged
                noise *= cfg.target_noise_max_m / (nn + 1e-9)
                rng_ref.random()
                exp = _TT + noise

                self.assertTrue(np.array_equal(out, exp), f"{out!r} != {exp!r}")
                # clipped magnitude is (numerically) at the cap
                self.assertLessEqual(
                    float(np.linalg.norm(out - _TT)), cfg.target_noise_max_m
                )
                self.assertTrue(
                    _rng_state_equal(
                        rng_model.bit_generator.state, rng_ref.bit_generator.state
                    )
                )

    # -- G. outlier branch ------------------------------------

    def test_forced_outlier_branch(self):
        cfg = _cfg(target_dropout_prob=0.0, target_stale_prob=0.0,
                   target_outlier_prob=1.0,
                   target_noise_xy_std_m=0.04, target_noise_z_std_m=0.03,
                   target_noise_max_m=1e9, target_outlier_xy_m=1.5)
        for seed in (0, 1, 5000, 77):
            with self.subTest(seed=seed):
                model = TargetMeasurementModel(cfg)
                model.reset(_TT)

                rng_model = np.random.default_rng(seed)
                out, valid, mode = model.sample(rng_model, _TT)
                self.assertEqual(mode, "outlier")
                self.assertTrue(valid)

                # reference: 2x random() + 3x normal() + 1x random() + uniform(size=2)
                rng_ref = np.random.default_rng(seed)
                rng_ref.random()
                rng_ref.random()
                noise = np.array(
                    [
                        rng_ref.normal(0.0, cfg.target_noise_xy_std_m),
                        rng_ref.normal(0.0, cfg.target_noise_xy_std_m),
                        rng_ref.normal(0.0, cfg.target_noise_z_std_m),
                    ],
                    dtype=np.float64,
                )
                rng_ref.random()
                exp = _TT + noise
                bump = rng_ref.uniform(
                    -cfg.target_outlier_xy_m, cfg.target_outlier_xy_m, size=2
                )
                exp = exp.copy()
                exp[:2] += bump

                self.assertTrue(np.array_equal(out, exp), f"{out!r} != {exp!r}")
                self.assertTrue(np.array_equal(model.last_raw_target, exp))
                self.assertTrue(
                    _rng_state_equal(
                        rng_model.bit_generator.state, rng_ref.bit_generator.state
                    ),
                    f"[seed={seed}] outlier branch draw count mismatch",
                )

    def test_normal_branch_does_not_draw_outlier_uniform(self):
        """The outlier gate is a random() that fails -> no uniform() draw on
        the normal path. Draw order: random, random, normal, normal, normal,
        random."""
        cfg = _cfg(target_dropout_prob=0.0, target_stale_prob=0.0,
                   target_outlier_prob=0.0,
                   target_noise_xy_std_m=0.04, target_noise_z_std_m=0.03,
                   target_noise_max_m=1e9)
        rng_model = np.random.default_rng(123)
        model = TargetMeasurementModel(cfg)
        model.reset(_TT)
        model.sample(rng_model, _TT)

        rng_ref = np.random.default_rng(123)
        rng_ref.random()
        rng_ref.random()
        rng_ref.normal(0.0, cfg.target_noise_xy_std_m)
        rng_ref.normal(0.0, cfg.target_noise_xy_std_m)
        rng_ref.normal(0.0, cfg.target_noise_z_std_m)
        rng_ref.random()
        self.assertTrue(
            _rng_state_equal(
                rng_model.bit_generator.state, rng_ref.bit_generator.state
            )
        )

    # -- H. long sequence vs verbatim legacy replica ------------

    def test_long_sequence_matches_legacy_replica_exactly(self):
        for cfg_name in _CONFIG_NAMES:
            overrides = {} if cfg_name == "default" else dict(_overrides(cfg_name))
            cfg = NewLandingConfig(**overrides)
            for seed in (0, 1, 5000, 999):
                with self.subTest(config=cfg_name, seed=seed):
                    rng_model = np.random.default_rng(seed)
                    rng_leg = np.random.default_rng(seed)

                    model = TargetMeasurementModel(cfg)
                    legacy = _LegacyMeasurement(cfg)
                    model.reset(_TT)
                    legacy.reset(_TT)

                    # move the true target around a bit each step
                    tt = _TT.copy()
                    for i in range(400):
                        tt = tt + np.array(
                            [0.001 * ((i % 7) - 3), -0.0005 * (i % 5), 0.0002 * (i % 3)],
                            dtype=np.float64,
                        )
                        o_m, v_m, md_m = model.sample(rng_model, tt)
                        o_l, v_l, md_l = legacy.sample(rng_leg, tt)

                        self.assertEqual((v_m, md_m), (v_l, md_l), f"step {i}")
                        self.assertEqual(o_m.dtype, o_l.dtype)
                        self.assertEqual(o_m.shape, o_l.shape)
                        self.assertTrue(
                            np.array_equal(o_m, o_l),
                            f"[{cfg_name} seed={seed}] step {i} ({md_m}): "
                            f"{o_m!r} != {o_l!r}",
                        )
                        self.assertTrue(
                            np.array_equal(
                                model.last_raw_target, legacy.last_raw_target
                            ),
                            f"[{cfg_name} seed={seed}] step {i}: last_raw_target diverged",
                        )
                        self.assertTrue(
                            _rng_state_equal(
                                rng_model.bit_generator.state,
                                rng_leg.bit_generator.state,
                            ),
                            f"[{cfg_name} seed={seed}] step {i}: RNG state diverged",
                        )
                    # the long run must have actually visited the rare branches
                    # for the stage config that has non-zero probs
                    if cfg_name == "stage2_train":
                        modes = set()
                        rng2 = np.random.default_rng(seed)
                        m2 = TargetMeasurementModel(cfg)
                        m2.reset(_TT)
                        for _ in range(400):
                            modes.add(m2.sample(rng2, _TT)[2])
                        self.assertIn("dropout", modes)
                        self.assertIn("stale", modes)

    # -- I. env compatibility mirror ---------------------------

    def test_env_last_raw_target_mirrors_model(self):
        for cfg_name in _CONFIG_NAMES:
            for seed in (0, 1, 5000):
                with self.subTest(config=cfg_name, seed=seed):
                    overrides = {} if cfg_name == "default" else dict(_overrides(cfg_name))
                    env = NewLandingEnv(NewLandingConfig(**overrides))
                    try:
                        env.reset(seed=seed)
                        self.assertIs(
                            env.last_raw_target,
                            env.target_measurement_model.last_raw_target,
                        )
                        for _ in range(30):
                            env.step(np.zeros(3, dtype=np.float32))
                            self.assertTrue(
                                np.array_equal(
                                    env.last_raw_target,
                                    env.target_measurement_model.last_raw_target,
                                ),
                                f"[{cfg_name} seed={seed}] mirror value diverged",
                            )
                            self.assertIs(
                                env.last_raw_target,
                                env.target_measurement_model.last_raw_target,
                                f"[{cfg_name} seed={seed}] mirror identity lost",
                            )
                    finally:
                        env.close()

    # -- coverage guard: matrix has non-zero dropout/stale/outlier/noise

    def test_regression_matrix_exercises_all_branches(self):
        st = _overrides("stage2_train")
        self.assertGreater(st["target_dropout_prob"], 0.0)
        self.assertGreater(st["target_stale_prob"], 0.0)
        self.assertGreater(st["target_outlier_prob"], 0.0)
        self.assertGreater(st["target_noise_xy_std_m"], 0.0)
        self.assertGreater(st["target_noise_z_std_m"], 0.0)
        # default dataclass also has low but non-zero dropout/stale/outlier
        d = NewLandingConfig()
        self.assertGreater(d.target_dropout_prob, 0.0)
        self.assertGreater(d.target_stale_prob, 0.0)
        self.assertGreater(d.target_outlier_prob, 0.0)


if __name__ == "__main__":
    unittest.main()
