"""Focused tests for the Phase 5 ActionLatency helper (command-path delay).

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py`` and the checkpoint gate in
``test_checkpoint_compatibility.py``, both of which remain authoritative.
These tests localize an action-latency regression by comparing ``ActionLatency``
against verbatim copies of the legacy ``integers(...)`` draw and the legacy
``_apply_action_delay`` FIFO, using independent generators started from the
same state.

Comparison is exact (``np.array_equal`` / ``==`` / identical RNG state). No
tolerance.
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

from landing_rl.envs.action_latency import ActionLatency


def _legacy_action_delay_sample(cfg, rng) -> int:
    """Verbatim copy of the action_delay_steps draw from
    mujoco_rl/envs/env_prototype.py :: LandingEnv.reset()."""
    return int(
        rng.integers(
            cfg.action_delay_steps_min,
            cfg.action_delay_steps_max + 1,
        )
    )


class _LegacyActionDelay:
    """Verbatim copy of LandingEnv._apply_action_delay + its queue init, as a
    standalone reference. Not used by production code."""

    def __init__(self, action_delay_steps):
        self.action_delay_steps = int(action_delay_steps)
        self.action_queue = [
            np.zeros(3, dtype=np.float64)
            for _ in range(max(0, self.action_delay_steps))
        ]

    def apply(self, raw_action):
        if self.action_delay_steps <= 0:
            return raw_action.copy()
        self.action_queue.append(raw_action.copy())
        return self.action_queue.pop(0)


def _overrides(name):
    for cfg_name, overrides, _doc in CONFIG_SPECS:
        if cfg_name == name:
            return overrides
    raise KeyError(name)


_CONFIG_NAMES = ("default", "stage0_eval", "stage2_eval", "stage2_train")


def _cfg(name, **extra):
    base = {} if name == "default" else dict(_overrides(name))
    base.update(extra)
    return NewLandingConfig(**base)


def _act(*xyz):
    return np.asarray(xyz, dtype=np.float64)


class ActionLatencyTest(unittest.TestCase):
    maxDiff = None

    # -- A. constructor consumes zero RNG -----------------------------

    def test_constructor_consumes_no_rng(self):
        rng = np.random.default_rng(13)
        before = rng.bit_generator.state
        lat = ActionLatency(NewLandingConfig())
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        self.assertFalse(hasattr(lat, "np_random"))
        self.assertFalse(hasattr(lat, "rng"))
        self.assertEqual(set(vars(lat)), {"cfg", "action_delay_steps", "action_queue"})
        self.assertEqual(lat.action_delay_steps, 0)
        self.assertEqual(lat.action_queue, [])

    # -- B. reset(rng) == legacy integers expression, same RNG after ---

    def test_reset_matches_legacy_integers_draw_exactly(self):
        for cfg_name in _CONFIG_NAMES:
            cfg = _cfg(cfg_name)
            for seed in (0, 1, 5000, 999, 424242):
                with self.subTest(config=cfg_name, seed=seed):
                    rng_legacy = np.random.default_rng(seed)
                    rng_new = np.random.default_rng(seed)

                    n_legacy = _legacy_action_delay_sample(cfg, rng_legacy)
                    n_new = ActionLatency(cfg).reset(rng_new)

                    self.assertIsInstance(n_new, int)
                    self.assertEqual(
                        n_legacy, n_new,
                        f"[{cfg_name} seed={seed}] action_delay_steps mismatch: "
                        f"OLD={n_legacy} NEW={n_new}",
                    )
                    self.assertTrue(
                        _rng_state_equal(
                            rng_legacy.bit_generator.state,
                            rng_new.bit_generator.state,
                        ),
                        f"[{cfg_name} seed={seed}] RNG state diverged after "
                        f"action-delay draw",
                    )

    def test_reset_consumes_exactly_one_integers_draw(self):
        cfg = _cfg("stage2_train")  # 1..5
        for seed in (0, 1, 5000):
            with self.subTest(seed=seed):
                rng_new = np.random.default_rng(seed)
                ActionLatency(cfg).reset(rng_new)

                rng_ref = np.random.default_rng(seed)
                rng_ref.integers(
                    cfg.action_delay_steps_min, cfg.action_delay_steps_max + 1
                )

                self.assertTrue(
                    _rng_state_equal(
                        rng_new.bit_generator.state, rng_ref.bit_generator.state
                    )
                )

    def test_reset_preserves_order_before_obs_delay_draw(self):
        """action-delay draw BEFORE obs-delay draw: doing ActionLatency.reset
        then a manual obs-delay integers draw leaves the generator in the same
        state as the legacy two-draw sequence."""
        cfg = _cfg("stage2_train")
        for seed in (0, 1, 5000, 7):
            with self.subTest(seed=seed):
                rng_new = np.random.default_rng(seed)
                a_new = ActionLatency(cfg).reset(rng_new)
                o_new = int(
                    rng_new.integers(
                        cfg.obs_delay_steps_min, cfg.obs_delay_steps_max + 1
                    )
                )

                rng_legacy = np.random.default_rng(seed)
                a_leg = int(
                    rng_legacy.integers(
                        cfg.action_delay_steps_min,
                        cfg.action_delay_steps_max + 1,
                    )
                )
                o_leg = int(
                    rng_legacy.integers(
                        cfg.obs_delay_steps_min, cfg.obs_delay_steps_max + 1
                    )
                )

                self.assertEqual((a_new, o_new), (a_leg, o_leg))
                self.assertTrue(
                    _rng_state_equal(
                        rng_new.bit_generator.state,
                        rng_legacy.bit_generator.state,
                    )
                )

    # -- C. minimum delay / D. maximum delay -------------------------

    def test_minimum_delay_bound(self):
        cfg = _cfg("default", action_delay_steps_min=2, action_delay_steps_max=2)
        for seed in range(20):
            lat = ActionLatency(cfg)
            n = lat.reset(np.random.default_rng(seed))
            self.assertEqual(n, 2)
            self.assertEqual(len(lat.action_queue), 2)

    def test_delay_within_declared_range_and_hits_max(self):
        cfg = _cfg("default", action_delay_steps_min=1, action_delay_steps_max=5)
        seen = set()
        for seed in range(200):
            n = ActionLatency(cfg).reset(np.random.default_rng(seed))
            self.assertGreaterEqual(n, 1)
            self.assertLessEqual(n, 5)
            seen.add(n)
        self.assertEqual(seen, {1, 2, 3, 4, 5})  # inclusive upper bound reached

    # -- E. delay == 0 identity behavior ---------------------------

    def test_delay_zero_identity(self):
        cfg = _cfg("default", action_delay_steps_min=0, action_delay_steps_max=0)
        lat = ActionLatency(cfg)
        n = lat.reset(np.random.default_rng(1))
        self.assertEqual(n, 0)
        self.assertEqual(lat.action_queue, [])

        raw = _act(0.2, -0.4, 0.9)
        out = lat.apply(raw)
        self.assertTrue(np.array_equal(out, raw))
        self.assertIsNot(out, raw)                 # returns a copy
        self.assertEqual(lat.action_queue, [])     # queue untouched

        raw2 = _act(-1.0, 1.0, 0.0)
        out2 = lat.apply(raw2)
        self.assertTrue(np.array_equal(out2, raw2))
        self.assertEqual(lat.action_queue, [])

    # -- F. delay == 1 behavior ----------------------------------

    def test_delay_one_fifo(self):
        cfg = _cfg("default", action_delay_steps_min=1, action_delay_steps_max=1)
        lat = ActionLatency(cfg)
        lat.reset(np.random.default_rng(0))
        self.assertEqual(len(lat.action_queue), 1)
        self.assertTrue(np.array_equal(lat.action_queue[0], np.zeros(3)))

        a1, a2, a3 = _act(1, 2, 3), _act(4, 5, 6), _act(7, 8, 9)
        ref = _LegacyActionDelay(1)

        for a in (a1, a2, a3):
            got = lat.apply(a.copy())
            exp = ref.apply(a.copy())
            self.assertTrue(np.array_equal(got, exp), f"{got!r} != {exp!r}")

        # first output was the initial zero, then a1, then a2
        # (verified elementwise against the reference above)

    # -- G. delay > 1 FIFO behavior across a sequence -------------

    def test_delay_three_fifo_sequence(self):
        cfg = _cfg("default", action_delay_steps_min=3, action_delay_steps_max=3)
        lat = ActionLatency(cfg)
        lat.reset(np.random.default_rng(0))
        self.assertEqual(len(lat.action_queue), 3)

        ref = _LegacyActionDelay(3)
        seq = [_act(i, i + 0.5, -i) for i in range(1, 8)]
        outs, exps = [], []
        for a in seq:
            outs.append(lat.apply(a.copy()))
            exps.append(ref.apply(a.copy()))

        for k, (g, e) in enumerate(zip(outs, exps)):
            self.assertTrue(np.array_equal(g, e), f"step {k}: {g!r} != {e!r}")
        # first three outputs are the zero-initialized slots
        for k in range(3):
            self.assertTrue(np.array_equal(outs[k], np.zeros(3)), f"step {k}")
        # thereafter outputs lag the input by exactly 3
        for k in range(3, len(seq)):
            self.assertTrue(np.array_equal(outs[k], seq[k - 3]), f"step {k}")

    # -- H. reset re-initializes the queue with exact zero arrays ---

    def test_reset_reinitializes_queue(self):
        cfg = _cfg("default", action_delay_steps_min=2, action_delay_steps_max=2)
        lat = ActionLatency(cfg)
        lat.reset(np.random.default_rng(0))
        first_queue = lat.action_queue
        lat.apply(_act(1, 1, 1))
        lat.apply(_act(2, 2, 2))
        self.assertFalse(np.array_equal(lat.action_queue[-1], np.zeros(3)))

        lat.reset(np.random.default_rng(0))
        self.assertIsNot(lat.action_queue, first_queue)   # fresh list
        self.assertEqual(len(lat.action_queue), 2)
        for elem in lat.action_queue:
            self.assertTrue(np.array_equal(elem, np.zeros(3)))

    # -- I. queue dtype float64, shape (3,) -----------------------

    def test_queue_element_dtype_and_shape(self):
        cfg = _cfg("default", action_delay_steps_min=4, action_delay_steps_max=4)
        lat = ActionLatency(cfg)
        lat.reset(np.random.default_rng(0))
        for elem in lat.action_queue:
            self.assertEqual(elem.dtype, np.dtype(np.float64))
            self.assertEqual(elem.shape, (3,))

        out = lat.apply(_act(0.1, 0.2, 0.3))
        self.assertEqual(out.dtype, np.dtype(np.float64))
        self.assertEqual(out.shape, (3,))

    # -- J. env compatibility alias -----------------------------

    def test_env_queue_is_helper_queue_after_reset(self):
        for cfg_name in _CONFIG_NAMES:
            for seed in (0, 1, 5000):
                with self.subTest(config=cfg_name, seed=seed):
                    env = NewLandingEnv(_cfg(cfg_name))
                    try:
                        env.reset(seed=seed)
                        self.assertIs(
                            env.action_queue, env.action_latency.action_queue,
                            f"[{cfg_name} seed={seed}] env.action_queue is not "
                            f"the helper's queue object",
                        )
                        self.assertEqual(
                            env.action_delay_steps,
                            env.action_latency.action_delay_steps,
                        )
                        self.assertEqual(
                            len(env.action_queue),
                            max(0, env.action_delay_steps),
                        )
                        # a second reset re-aliases to the new queue object
                        old_q = env.action_queue
                        env.reset(seed=seed + 1)
                        self.assertIs(
                            env.action_queue, env.action_latency.action_queue
                        )
                        self.assertIsNot(env.action_queue, old_q)
                    finally:
                        env.close()

    def test_regression_matrix_covers_multiple_delay_regimes(self):
        """The OLD-vs-NEW matrix exercises fixed delay==1 and randomized 1..5."""
        se = _overrides("stage2_eval")
        self.assertEqual(se["action_delay_steps_min"], 1)
        self.assertEqual(se["action_delay_steps_max"], 1)
        st = _overrides("stage2_train")
        self.assertEqual(st["action_delay_steps_min"], 1)
        self.assertEqual(st["action_delay_steps_max"], 5)
        s0 = _overrides("stage0_eval")
        self.assertEqual(s0["action_delay_steps_min"], 0)
        self.assertEqual(s0["action_delay_steps_max"], 2)
        # default dataclass range
        self.assertEqual(NewLandingConfig().action_delay_steps_min, 1)
        self.assertEqual(NewLandingConfig().action_delay_steps_max, 4)


if __name__ == "__main__":
    unittest.main()
