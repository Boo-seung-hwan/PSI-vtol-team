"""Focused tests for the Phase 4 LoopTiming helper (control-step dt sampling).

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py``, which remains authoritative. These
tests localize a dt-sampling regression by comparing ``LoopTiming.sample_dt``
against a verbatim copy of the legacy ``LandingEnv._sample_dt`` expression,
using independent generators started from the same state.

Comparison is exact (``==`` on the dt float, identical RNG state). No tolerance.
"""

from __future__ import annotations

import unittest

import numpy as np

from test_legacy_regression_contract import (
    CONFIG_SPECS,
    NewLandingConfig,
    _rng_state_equal,
)

from landing_rl.envs.loop_timing import LoopTiming


def _legacy_sample_dt(cfg, rng):
    """Verbatim copy of LandingEnv._sample_dt from
    mujoco_rl/envs/env_prototype.py."""
    dt = cfg.dt + rng.normal(0.0, cfg.dt_jitter_std)
    return float(np.clip(dt, cfg.dt_min, cfg.dt_max))


class _FixedNormalRng:
    """Minimal stand-in exposing only the ``normal(loc, scale)`` call contract
    used by LoopTiming.sample_dt. Records the calls it receives. Not used by
    production code."""

    def __init__(self, value):
        self._value = value
        self.calls = []

    def normal(self, loc, scale):
        self.calls.append((loc, scale))
        return self._value


def _overrides(name):
    for cfg_name, overrides, _doc in CONFIG_SPECS:
        if cfg_name == name:
            return overrides
    raise KeyError(name)


_CONFIG_NAMES = ("default", "stage0_eval", "stage2_eval", "stage2_train")


def _cfg(name):
    return NewLandingConfig(**({} if name == "default" else _overrides(name)))


class LoopTimingTest(unittest.TestCase):
    maxDiff = None

    # -- A. constructor consumes no RNG ------------------------------

    def test_constructor_consumes_no_rng(self):
        rng = np.random.default_rng(11)
        before = rng.bit_generator.state
        timing = LoopTiming(NewLandingConfig())
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        self.assertFalse(hasattr(timing, "np_random"))
        self.assertFalse(hasattr(timing, "rng"))
        # owns only cfg
        self.assertEqual(set(vars(timing)), {"cfg"})

    # -- B. exact parity vs legacy expression, same starting RNG -----

    def test_matches_legacy_sample_dt_exactly(self):
        for cfg_name in _CONFIG_NAMES:
            cfg = _cfg(cfg_name)
            for seed in (0, 1, 5000, 999, 424242):
                with self.subTest(config=cfg_name, seed=seed):
                    rng_legacy = np.random.default_rng(seed)
                    rng_new = np.random.default_rng(seed)

                    dt_legacy = _legacy_sample_dt(cfg, rng_legacy)
                    dt_new = LoopTiming(cfg).sample_dt(rng_new)

                    self.assertIsInstance(dt_new, float)
                    self.assertEqual(
                        dt_legacy, dt_new,
                        f"[{cfg_name} seed={seed}] dt mismatch: "
                        f"OLD={dt_legacy!r} NEW={dt_new!r}",
                    )
                    self.assertTrue(
                        _rng_state_equal(
                            rng_legacy.bit_generator.state,
                            rng_new.bit_generator.state,
                        ),
                        f"[{cfg_name} seed={seed}] RNG state diverged after dt draw",
                    )

    def test_consumes_exactly_one_normal_draw(self):
        cfg = _cfg("stage2_train")
        for seed in (0, 1, 5000):
            with self.subTest(seed=seed):
                rng_new = np.random.default_rng(seed)
                LoopTiming(cfg).sample_dt(rng_new)

                rng_ref = np.random.default_rng(seed)
                rng_ref.normal(0.0, cfg.dt_jitter_std)

                self.assertTrue(
                    _rng_state_equal(
                        rng_new.bit_generator.state, rng_ref.bit_generator.state
                    )
                )

    # -- C. zero jitter --------------------------------------------

    def test_zero_jitter(self):
        cfg = NewLandingConfig(dt_jitter_std=0.0)
        rng = np.random.default_rng(3)
        before_state = np.random.default_rng(3).bit_generator.state

        dt = LoopTiming(cfg).sample_dt(rng)

        # dt is exactly cfg.dt (0.05 is inside [dt_min, dt_max]); one draw of
        # normal(0.0, 0.0) is still consumed.
        self.assertEqual(dt, float(np.clip(cfg.dt, cfg.dt_min, cfg.dt_max)))
        self.assertEqual(dt, cfg.dt)
        self.assertFalse(_rng_state_equal(before_state, rng.bit_generator.state))

    # -- D. non-zero jitter --------------------------------------

    def test_nonzero_jitter_within_bounds(self):
        cfg = NewLandingConfig(dt_jitter_std=0.010)
        self.assertGreater(cfg.dt_jitter_std, 0.0)
        seen = set()
        for seed in range(40):
            dt = LoopTiming(cfg).sample_dt(np.random.default_rng(seed))
            self.assertGreaterEqual(dt, cfg.dt_min)
            self.assertLessEqual(dt, cfg.dt_max)
            seen.add(dt)
        self.assertGreater(len(seen), 1)  # jitter actually varies dt

    # -- E. clipping at dt_min / dt_max --------------------------

    def test_clips_to_dt_max(self):
        cfg = NewLandingConfig()
        rng = _FixedNormalRng(+1000.0)
        dt = LoopTiming(cfg).sample_dt(rng)
        self.assertIsInstance(dt, float)
        self.assertEqual(dt, cfg.dt_max)
        self.assertEqual(rng.calls, [(0.0, cfg.dt_jitter_std)])

    def test_clips_to_dt_min(self):
        cfg = NewLandingConfig()
        rng = _FixedNormalRng(-1000.0)
        dt = LoopTiming(cfg).sample_dt(rng)
        self.assertIsInstance(dt, float)
        self.assertEqual(dt, cfg.dt_min)
        self.assertEqual(rng.calls, [(0.0, cfg.dt_jitter_std)])

    def test_no_clip_when_in_range(self):
        cfg = NewLandingConfig()
        rng = _FixedNormalRng(0.0)
        dt = LoopTiming(cfg).sample_dt(rng)
        self.assertEqual(dt, cfg.dt)  # cfg.dt in [dt_min, dt_max]

    # -- existing regression matrix exercises non-zero jitter -----

    def test_regression_matrix_has_nonzero_jitter_configs(self):
        jitters = {}
        for cfg_name in _CONFIG_NAMES:
            jitters[cfg_name] = _cfg(cfg_name).dt_jitter_std
        # default (no override) and every stage config use non-zero jitter
        self.assertEqual(jitters["default"], 0.008)
        self.assertEqual(jitters["stage0_eval"], 0.004)
        self.assertEqual(jitters["stage2_eval"], 0.010)
        self.assertEqual(jitters["stage2_train"], 0.010)
        for name, j in jitters.items():
            self.assertGreater(j, 0.0, f"{name} must exercise non-zero dt jitter")


if __name__ == "__main__":
    unittest.main()
