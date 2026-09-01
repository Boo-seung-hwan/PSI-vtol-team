"""OLD-vs-NEW landing environment parity contract (Phase 1).

Purpose
-------
Prove that the new environment

    landing_rl/envs/landing_env.py :: LandingEnv        (NEW)

is behaviorally identical to the frozen canonical reference

    mujoco_rl/envs/env_prototype.py :: LandingEnv        (OLD)

In Phase 1 ``landing_env.py`` is a byte-for-byte copy of ``env_prototype.py``;
this suite is the regression gate that keeps the two in exact lockstep as the
structural migration proceeds. It also keeps a reduced OLD-vs-OLD run as a
self-consistency sanity check (the mechanism inherited from Phase 0).

What is compared, at reset and after every step
----------------------------------------------
    * observation           (exact, incl. shape==(16,) and dtype==float32)
    * reward                (exact)
    * terminated / truncated (exact)
    * every ``info`` channel (exact value, exact dtype/shape) + frozen key-set
    * ``env.np_random.bit_generator.state``  (identical)
    * observation_space / action_space  (equal between OLD and NEW)

Comparison is EXACT. ``np.array_equal`` / ``==`` / identical RNG state. The only
use of ``equal_nan`` is so an all-NaN channel (``alpha_eff``) compares equal to
itself. No numerical tolerance is present. If exact parity ever fails, the
failure message reports the first differing field with OLD and NEW values and
the RNG-state equality before/after the failing step -- do not add a tolerance.

Not in scope for Phase 1
------------------------
    * No modular decomposition (controller / dynamics / perception /
      disturbance / contact / reward / latency / config split).
    * No PPO checkpoint / VecNormalize regression, no ``mujoco_rl/runs/`` access.
    * No full large-seed regression campaign.

Run from the worktree root:

    python3 -m unittest discover -s landing_rl/tests -p "test_*.py"
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Locate paths (independent of cwd / PYTHONPATH).
# ---------------------------------------------------------------------------

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parents[1]  # landing_rl/tests -> landing_rl -> <worktree root>
_ENV_PROTOTYPE_PATH = _REPO_ROOT / "mujoco_rl" / "envs" / "env_prototype.py"

# Make ``import landing_rl`` resolvable when this suite is run by a runner that
# does not already put the repo root on sys.path. This does not modify
# PYTHONPATH, requirements, or packaging -- it is a test-local path fix only.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# OLD: load the canonical legacy environment directly from its file path.
#
# Loading by file path (not ``import``) keeps the OLD side pinned to the exact
# reference file regardless of import layout, and keeps it a genuinely separate
# module object from the NEW package (so this is a real OLD-vs-NEW comparison,
# not the same class twice).
# ---------------------------------------------------------------------------

def _load_legacy_module():
    if not _ENV_PROTOTYPE_PATH.is_file():
        raise FileNotFoundError(
            f"legacy reference implementation not found: {_ENV_PROTOTYPE_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "phase1_legacy_env_prototype", _ENV_PROTOTYPE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # help dataclass / pickling introspection
    spec.loader.exec_module(module)
    return module


_OLD = _load_legacy_module()
OldLandingEnv = _OLD.LandingEnv
OldLandingConfig = _OLD.LandingConfig

# ---------------------------------------------------------------------------
# NEW: the landing_rl package environment (Phase 1 = byte copy of OLD).
# ---------------------------------------------------------------------------

from landing_rl.envs import LandingConfig as NewLandingConfig  # noqa: E402
from landing_rl.envs import LandingEnv as NewLandingEnv  # noqa: E402


LEGACY_IMPL = ("legacy", OldLandingEnv, OldLandingConfig)
NEW_IMPL = ("landing_rl", NewLandingEnv, NewLandingConfig)
IMPLS = (LEGACY_IMPL, NEW_IMPL)


# ---------------------------------------------------------------------------
# Frozen contract constants (baseline = current mujoco_rl/envs/env_prototype.py)
# ---------------------------------------------------------------------------

# Every key present in the dict returned by ``LandingEnv.step`` (lines ~1156-1215
# of env_prototype.py). ``reset`` returns ``{}`` and is checked separately.
EXPECTED_INFO_KEYS = frozenset(
    {
        "xy_error", "z_error", "vxy", "vz_abs", "accel_mag", "jerk_mag",
        "roll", "pitch", "yaw", "yaw_error", "tilt", "yaw_rate",
        "body_rates", "thrust_accel", "thrust_accel_setpoint",
        "attitude_setpoint", "accel_cmd",
        "success", "failed", "failure_reason",
        "pos", "vel", "accel", "prev_accel",
        "target_true", "target_measured", "target_valid", "target_mode",
        "dt",
        "vel_response_alpha", "attitude_response_alpha",
        "body_rate_response_alpha", "thrust_response_alpha", "alpha_eff",
        "wind_accel", "ground_effect_factor",
        "ground_contact", "contact_event", "contact_count", "bounce_count",
        "soft_contact", "hard_contact", "bounced", "touchdown_quality",
        "last_impact_vz", "last_touchdown_vxy", "last_bounce_speed",
        "motor_cutoff", "altitude_agl",
        "action_delay_steps", "obs_delay_steps",
        "progress_reward", "action_delta", "applied_action", "raw_action",
        "v_pid", "v_residual", "v_cmd",
    }
)

# observation_space.high literal from env_prototype.py __init__ (indices 6-8 are
# cfg.max_acceleration_obs_mps2, whose current default is 4.0; the rest are hard
# literals). observation_space.low is -high. dtype float32.
EXPECTED_OBS_HIGH = np.array(
    [30.0, 30.0, 30.0,
     6.0, 6.0, 6.0,
     4.0, 4.0, 4.0,
     math.pi, math.pi, math.pi,
     1.0, 1.0, 1.0,
     1.0],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Config specs. Each entry: (name, overrides_dict, source_doc).
#
# ``overrides_dict`` is applied as ``LandingConfig(**overrides)`` to BOTH the
# OLD and the NEW config dataclass, so the ``default`` (empty override) case
# also verifies that every default field is identical between the two classes.
# Values are reproduced verbatim from the named source file (Phase 1: do not
# split / rename / alter config).
# ---------------------------------------------------------------------------

CONFIG_SPECS = (
    (
        "default",
        {},
        "mujoco_rl/envs/env_prototype.py :: LandingConfig() dataclass defaults; "
        "also used by mujoco_rl/scripts/plot_trajectory.py. contact_enabled=True.",
    ),
    (
        "stage0_eval",
        dict(
            init_xy_range_m=4.0,
            init_altitude_min_m=2.0,
            init_altitude_max_m=5.0,
            init_vel_range_mps=0.20,
            dt_jitter_std=0.004,
            action_delay_steps_min=0,
            action_delay_steps_max=2,
            obs_delay_steps_min=0,
            obs_delay_steps_max=2,
            target_dropout_prob=0.01,
            target_outlier_prob=0.0,
            target_stale_prob=0.02,
            target_noise_xy_std_m=0.02,
            target_noise_z_std_m=0.015,
            wind_accel_xy_max_mps2=0.0,
            wind_accel_z_max_mps2=0.0,
            contact_enabled=False,
            success_requires_contact=False,
            init_yaw_range_rad=0.0,
            w_yaw=0.0,
        ),
        "mujoco_rl/eval_env_stage0.py :: make_config() (== train_ppo_v3_long.py). "
        "Contact disabled; delay range 0-2.",
    ),
    (
        "stage2_eval",
        dict(
            max_steps=500,
            residual_xy_mps=0.25,
            residual_z_mps=0.05,
            init_xy_range_m=6.0,
            init_altitude_min_m=2.0,
            init_altitude_max_m=7.0,
            init_vel_range_mps=0.60,
            dt_jitter_std=0.010,
            action_delay_steps_min=1,
            action_delay_steps_max=1,
            obs_delay_steps_min=1,
            obs_delay_steps_max=1,
            target_dropout_prob=0.00,
            target_outlier_prob=0.00,
            target_stale_prob=0.0,
            target_noise_xy_std_m=0.00,
            target_noise_z_std_m=0.0,
            wind_accel_xy_max_mps2=0.1,
            wind_accel_z_max_mps2=0.02,
            ground_effect_gain=0.005,
            landing_descent_bias_mps=0.14,
            contact_enabled=True,
            success_requires_contact=True,
            init_yaw_range_rad=0.5,
            w_yaw=0.05,
        ),
        "mujoco_rl/eval_compare_v2.py :: make_config(). Final-stage eval config "
        "for the primary checkpoint ppo_landing_residual_v4_stage2_contact_final.",
    ),
    (
        "stage2_train",
        dict(
            max_steps=500,
            residual_xy_mps=0.25,
            residual_z_mps=0.05,
            init_xy_range_m=6.0,
            init_altitude_min_m=2.0,
            init_altitude_max_m=7.0,
            init_vel_range_mps=0.60,
            dt_jitter_std=0.010,
            action_delay_steps_min=1,
            action_delay_steps_max=5,
            obs_delay_steps_min=1,
            obs_delay_steps_max=5,
            target_dropout_prob=0.05,
            target_outlier_prob=0.02,
            target_stale_prob=0.08,
            target_noise_xy_std_m=0.06,
            target_noise_z_std_m=0.04,
            wind_accel_xy_max_mps2=0.10,
            wind_accel_z_max_mps2=0.02,
            ground_effect_gain=0.005,
            landing_descent_bias_mps=0.14,
            contact_enabled=True,
            success_requires_contact=True,
            init_yaw_range_rad=0.5,
            w_yaw=0.05,
        ),
        "mujoco_rl/train_ppo_stage2_contact.py :: make_config(). Widest "
        "randomization: delay 1-5, non-zero dropout/stale/outlier/target noise.",
    ),
)

SEEDS = (0, 1, 5000)
PATTERNS = ("zero", "pseudo_random", "saturation")

# Fast development budget (not the final gate). Episodes that terminate or
# truncate earlier stop early.
MAIN_STEPS = 100

# Longer budget for the contact-path scenario only. The default episode length
# is 420 steps, so this still runs at most one full episode per env.
CONTACT_STEPS = 420


# ---------------------------------------------------------------------------
# Fixed action sequences.
#
# CRITICAL: the action-sequence RNG must NOT be either environment's
# ``np_random``. ``pseudo_random`` uses a standalone ``np.random.default_rng``
# seeded with a constant unrelated to any env seed.
# ---------------------------------------------------------------------------

_ACTION_SEQ_RNG_SEED = 20260902  # standalone; never passed to env.reset()


def make_action_sequence(pattern: str, n_steps: int) -> np.ndarray:
    """Return a deterministic ``(n_steps, 3)`` float32 action array."""
    if pattern == "zero":
        # PID-only path: both descent gates, contact, v_cmd descent-bias override.
        return np.zeros((n_steps, 3), dtype=np.float32)

    if pattern == "pseudo_random":
        rng = np.random.default_rng(_ACTION_SEQ_RNG_SEED)
        return rng.uniform(-1.0, 1.0, size=(n_steps, 3)).astype(np.float32)

    if pattern == "saturation":
        # Out-of-range magnitudes every step: input clip, saturation reward term
        # (|a| - 0.80), residual saturation, and _limit_velocity_command.
        seq = np.empty((n_steps, 3), dtype=np.float32)
        seq[0::2] = np.array([1.5, -1.5, 1.2], dtype=np.float32)
        seq[1::2] = np.array([-1.5, 1.5, -1.2], dtype=np.float32)
        return seq

    if pattern == "descend_z":
        # Sustained full-scale downward residual; drives to touchdown so the
        # contact model runs within the default episode.
        seq = np.zeros((n_steps, 3), dtype=np.float32)
        seq[:, 2] = 1.0
        return seq

    raise ValueError(f"unknown action pattern: {pattern!r}")


# ---------------------------------------------------------------------------
# Exact comparators (unchanged from Phase 0; no tolerance).
# ---------------------------------------------------------------------------

def _rng_state_equal(a, b) -> bool:
    """Recursively compare two ``bit_generator.state`` structures.

    For the default PCG64 generator the state is a nested dict of Python ints;
    this also handles ndarray-valued entries used by other bit generators.
    """
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        if a.keys() != b.keys():
            return False
        return all(_rng_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def _floating_array(x) -> bool:
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating)


def _value_equal(x, y) -> bool:
    """Exact equality for one ``info`` value.

    ``equal_nan`` is enabled only for floating data so that a channel that is
    NaN in the reference (e.g. ``alpha_eff``) compares equal to the same NaN
    channel. No numerical tolerance is applied.
    """
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            return False
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if _floating_array(x):
            return bool(np.array_equal(x, y, equal_nan=True))
        return bool(np.array_equal(x, y))
    if isinstance(x, float) and isinstance(y, float):
        if math.isnan(x) and math.isnan(y):
            return True
        return x == y
    return x == y


def _obs_equal(a_obs, b_obs) -> bool:
    return (
        isinstance(a_obs, np.ndarray)
        and isinstance(b_obs, np.ndarray)
        and a_obs.dtype == b_obs.dtype == np.float32
        and a_obs.shape == b_obs.shape == (16,)
        and bool(np.array_equal(a_obs, b_obs, equal_nan=True))
    )


def _rng_state(env):
    return env.np_random.bit_generator.state


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

class LandingEnvParityContractTest(unittest.TestCase):
    """OLD (mujoco_rl) vs NEW (landing_rl) landing environment parity."""

    maxDiff = None

    # -- shared assertions ---------------------------------------------------

    def _assert_reset_equal(self, old_reset, new_reset, old_env, new_env, ctx):
        old_obs, old_info = old_reset
        new_obs, new_info = new_reset
        problems = []

        if old_info != {}:
            problems.append(f"OLD reset info is not empty: {old_info!r}")
        if new_info != {}:
            problems.append(f"NEW reset info is not empty: {new_info!r}")
        if not _obs_equal(old_obs, new_obs):
            problems.append(
                f"reset observation mismatch:\n  OLD={old_obs!r}\n  NEW={new_obs!r}"
            )
        if not _rng_state_equal(_rng_state(old_env), _rng_state(new_env)):
            problems.append(
                "reset np_random.bit_generator.state diverged:\n"
                f"  OLD={_rng_state(old_env)}\n  NEW={_rng_state(new_env)}"
            )

        if problems:
            self.fail(f"[{ctx}] reset contract broken:\n" + "\n".join(problems))

    def _assert_states_equal(
        self, old_out, new_out, old_env, new_env, ctx, rng_equal_before
    ):
        old_obs, old_rew, old_term, old_trunc, old_info = old_out
        new_obs, new_rew, new_term, new_trunc, new_info = new_out
        problems = []

        if not _obs_equal(old_obs, new_obs):
            problems.append(
                f"observation mismatch:\n  OLD={old_obs!r}\n  NEW={new_obs!r}"
            )

        if not _value_equal(float(old_rew), float(new_rew)):
            problems.append(f"reward mismatch: OLD={old_rew!r} NEW={new_rew!r}")

        if (old_term, old_trunc) != (new_term, new_trunc):
            problems.append(
                "terminated/truncated mismatch: "
                f"OLD=(term={old_term}, trunc={old_trunc}) "
                f"NEW=(term={new_term}, trunc={new_trunc})"
            )

        old_keys, new_keys = set(old_info), set(new_info)
        if old_keys != new_keys:
            problems.append(
                "info key-set mismatch OLD vs NEW: "
                f"OLD-only={sorted(old_keys - new_keys)} "
                f"NEW-only={sorted(new_keys - old_keys)}"
            )
        if old_keys != EXPECTED_INFO_KEYS:
            problems.append(
                "OLD info key-set drifted from frozen contract: "
                f"missing={sorted(EXPECTED_INFO_KEYS - old_keys)} "
                f"unexpected={sorted(old_keys - EXPECTED_INFO_KEYS)}"
            )
        if new_keys != EXPECTED_INFO_KEYS:
            problems.append(
                "NEW info key-set drifted from frozen contract: "
                f"missing={sorted(EXPECTED_INFO_KEYS - new_keys)} "
                f"unexpected={sorted(new_keys - EXPECTED_INFO_KEYS)}"
            )

        for key in sorted(old_keys & new_keys):
            if not _value_equal(old_info[key], new_info[key]):
                problems.append(
                    f"info[{key!r}] mismatch: OLD={old_info[key]!r} "
                    f"NEW={new_info[key]!r}"
                )

        # alpha_eff must remain the [nan, nan, nan] sentinel on both sides.
        for label, info in (("OLD", old_info), ("NEW", new_info)):
            ae = info.get("alpha_eff")
            if not (
                isinstance(ae, np.ndarray)
                and ae.shape == (3,)
                and np.all(np.isnan(ae))
            ):
                problems.append(f"{label} alpha_eff sentinel changed: {ae!r}")

        rng_equal_after = _rng_state_equal(_rng_state(old_env), _rng_state(new_env))
        if not rng_equal_after:
            problems.append(
                "np_random.bit_generator.state diverged:\n"
                f"  OLD={_rng_state(old_env)}\n  NEW={_rng_state(new_env)}"
            )

        if problems:
            self.fail(
                f"[{ctx}] OLD-vs-NEW parity broken:\n"
                f"  RNG state equal BEFORE this step: {rng_equal_before}\n"
                f"  RNG state equal AFTER this step:  {rng_equal_after}\n"
                + "\n".join(problems)
            )

    def _run_paired(
        self,
        overrides,
        seed,
        pattern,
        n_steps,
        left_impl=LEGACY_IMPL,
        right_impl=NEW_IMPL,
    ):
        """Run two envs in lockstep. Returns True if the contact model was
        observed to activate at any compared step."""
        left_name, left_env_cls, left_cfg_cls = left_impl
        right_name, right_env_cls, right_cfg_cls = right_impl

        left_env = left_env_cls(left_cfg_cls(**overrides))
        right_env = right_env_cls(right_cfg_cls(**overrides))

        label = f"{left_name}-vs-{right_name} {pattern} seed={seed}"

        self.assertEqual(
            left_env.observation_space, right_env.observation_space,
            f"[{label}] observation_space differs",
        )
        self.assertEqual(
            left_env.action_space, right_env.action_space,
            f"[{label}] action_space differs",
        )

        left_reset = left_env.reset(seed=seed)
        right_reset = right_env.reset(seed=seed)
        self._assert_reset_equal(
            left_reset, right_reset, left_env, right_env, f"{label} @reset"
        )

        actions = make_action_sequence(pattern, n_steps)
        rng_equal_before = _rng_state_equal(
            _rng_state(left_env), _rng_state(right_env)
        )
        saw_contact = False
        try:
            for i in range(n_steps):
                act = actions[i].copy()
                left_out = left_env.step(act.copy())
                right_out = right_env.step(act.copy())
                self._assert_states_equal(
                    left_out, right_out, left_env, right_env,
                    f"{label} step={i}", rng_equal_before,
                )
                rng_equal_before = _rng_state_equal(
                    _rng_state(left_env), _rng_state(right_env)
                )
                info = left_out[4]
                if info.get("contact_event") or info.get("ground_contact"):
                    saw_contact = True
                if left_out[2] or left_out[3]:  # terminated or truncated
                    break
        finally:
            left_env.close()
            right_env.close()
        return saw_contact

    # -- import / package --------------------------------------------------

    def test_new_package_import(self):
        """``from landing_rl.envs import LandingEnv, LandingConfig`` works and
        resolves to the landing_rl.envs.landing_env module."""
        from landing_rl.envs import LandingConfig, LandingEnv

        self.assertIs(LandingEnv, NewLandingEnv)
        self.assertIs(LandingConfig, NewLandingConfig)
        self.assertEqual(LandingEnv.__module__, "landing_rl.envs.landing_env")
        self.assertEqual(LandingConfig.__module__, "landing_rl.envs.landing_env")

    # -- space / api contract -------------------------------------------

    def test_spaces_frozen_each_impl(self):
        """Both OLD and NEW expose the exact frozen space definition."""
        for impl_name, env_cls, cfg_cls in IMPLS:
            with self.subTest(impl=impl_name):
                env = env_cls(cfg_cls())
                try:
                    obs_space = env.observation_space
                    act_space = env.action_space

                    self.assertEqual(obs_space.shape, (16,))
                    self.assertEqual(obs_space.dtype, np.dtype(np.float32))
                    self.assertTrue(
                        np.array_equal(obs_space.high, EXPECTED_OBS_HIGH),
                        f"[{impl_name}] observation_space.high drifted:\n"
                        f"  got={obs_space.high!r}\n"
                        f"  expected={EXPECTED_OBS_HIGH!r}",
                    )
                    self.assertTrue(
                        np.array_equal(obs_space.low, -EXPECTED_OBS_HIGH),
                        f"[{impl_name}] observation_space.low drifted:\n"
                        f"  got={obs_space.low!r}\n"
                        f"  expected={(-EXPECTED_OBS_HIGH)!r}",
                    )

                    self.assertEqual(act_space.shape, (3,))
                    self.assertEqual(act_space.dtype, np.dtype(np.float32))
                    self.assertTrue(
                        np.array_equal(act_space.high, np.ones(3, np.float32))
                    )
                    self.assertTrue(
                        np.array_equal(act_space.low, -np.ones(3, np.float32))
                    )

                    self.assertEqual(cfg_cls().max_acceleration_obs_mps2, 4.0)
                    self.assertTrue(
                        np.array_equal(
                            obs_space.high[6:9],
                            np.full(
                                3,
                                env.cfg.max_acceleration_obs_mps2,
                                dtype=np.float32,
                            ),
                        )
                    )
                finally:
                    env.close()

    def test_old_vs_new_spaces_equal(self):
        """OLD and NEW agree on observation_space and action_space for every
        Phase 1 config."""
        for cfg_name, overrides, _doc in CONFIG_SPECS:
            with self.subTest(config=cfg_name):
                old_env = OldLandingEnv(OldLandingConfig(**overrides))
                new_env = NewLandingEnv(NewLandingConfig(**overrides))
                try:
                    self.assertEqual(
                        old_env.observation_space, new_env.observation_space
                    )
                    self.assertEqual(old_env.action_space, new_env.action_space)
                finally:
                    old_env.close()
                    new_env.close()

    def test_reset_and_step_api_contract_each_impl(self):
        """reset() -> (obs, {}); step() -> 5-tuple with obs shape/dtype fixed."""
        for impl_name, env_cls, cfg_cls in IMPLS:
            for cfg_name, overrides, _doc in CONFIG_SPECS:
                for seed in SEEDS:
                    with self.subTest(impl=impl_name, config=cfg_name, seed=seed):
                        env = env_cls(cfg_cls(**overrides))
                        try:
                            reset_out = env.reset(seed=seed)
                            self.assertIsInstance(reset_out, tuple)
                            self.assertEqual(len(reset_out), 2)
                            obs, info = reset_out
                            self.assertEqual(info, {})
                            self.assertIsInstance(obs, np.ndarray)
                            self.assertEqual(obs.shape, (16,))
                            self.assertEqual(obs.dtype, np.dtype(np.float32))

                            step_out = env.step(np.zeros(3, dtype=np.float32))
                            self.assertIsInstance(step_out, tuple)
                            self.assertEqual(len(step_out), 5)
                            s_obs, s_rew, s_term, s_trunc, s_info = step_out
                            self.assertEqual(s_obs.shape, (16,))
                            self.assertEqual(s_obs.dtype, np.dtype(np.float32))
                            self.assertIsInstance(float(s_rew), float)
                            self.assertIsInstance(s_term, bool)
                            self.assertIsInstance(s_trunc, bool)
                            self.assertEqual(set(s_info), set(EXPECTED_INFO_KEYS))
                        finally:
                            env.close()

    def test_info_key_set_frozen_each_impl(self):
        """step() info exposes exactly the frozen key-set for OLD and NEW."""
        for impl_name, env_cls, cfg_cls in IMPLS:
            with self.subTest(impl=impl_name):
                env = env_cls(cfg_cls())
                try:
                    env.reset(seed=0)
                    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
                    self.assertEqual(set(info), set(EXPECTED_INFO_KEYS))
                finally:
                    env.close()

    def test_alpha_eff_is_all_nan_sentinel_each_impl(self):
        """info['alpha_eff'] stays the current [nan, nan, nan] sentinel."""
        for impl_name, env_cls, cfg_cls in IMPLS:
            with self.subTest(impl=impl_name):
                env = env_cls(cfg_cls())
                try:
                    env.reset(seed=0)
                    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
                    ae = info["alpha_eff"]
                    self.assertIsInstance(ae, np.ndarray)
                    self.assertEqual(ae.shape, (3,))
                    self.assertTrue(np.all(np.isnan(ae)))
                finally:
                    env.close()

    # -- RNG -----------------------------------------------------------

    def test_rng_state_matches_after_reset_old_vs_new(self):
        """OLD and NEW share RNG state right after an identically-seeded reset,
        before any step consumes further draws."""
        for cfg_name, overrides, _doc in CONFIG_SPECS:
            for seed in SEEDS:
                with self.subTest(config=cfg_name, seed=seed):
                    old_env = OldLandingEnv(OldLandingConfig(**overrides))
                    new_env = NewLandingEnv(NewLandingConfig(**overrides))
                    try:
                        old_env.reset(seed=seed)
                        new_env.reset(seed=seed)
                        self.assertTrue(
                            _rng_state_equal(
                                _rng_state(old_env), _rng_state(new_env)
                            ),
                            f"[{cfg_name} seed={seed}] RNG state diverged at reset",
                        )
                    finally:
                        old_env.close()
                        new_env.close()

    # -- primary OLD-vs-NEW parity -----------------------------------

    def test_old_vs_new_paired_parity(self):
        """Full observable contract, OLD vs NEW, lockstep, for every
        config x seed x action pattern."""
        for cfg_name, overrides, _doc in CONFIG_SPECS:
            for seed in SEEDS:
                for pattern in PATTERNS:
                    with self.subTest(config=cfg_name, seed=seed, pattern=pattern):
                        self._run_paired(overrides, seed, pattern, MAIN_STEPS)

    def test_contact_path_old_vs_new(self):
        """A sustained downward residual reaches touchdown within the default
        420-step episode; OLD and NEW stay in exact lockstep through contact."""
        saw_contact = False
        for seed in (0, 1):
            with self.subTest(seed=seed):
                if self._run_paired({}, seed, "descend_z", CONTACT_STEPS):
                    saw_contact = True
        self.assertTrue(
            saw_contact,
            "expected the contact model to activate (contact_event / "
            "ground_contact) in at least one descend_z run",
        )

    # -- legacy-vs-legacy self-consistency sanity (from Phase 0) -----

    def test_legacy_self_consistency_sanity(self):
        """Reduced OLD-vs-OLD run: guards harness determinism and the RNG
        contract independently of the NEW implementation."""
        for seed in (0, 1):
            for pattern in PATTERNS:
                with self.subTest(seed=seed, pattern=pattern):
                    self._run_paired(
                        {}, seed, pattern, MAIN_STEPS,
                        left_impl=LEGACY_IMPL, right_impl=LEGACY_IMPL,
                    )


if __name__ == "__main__":
    unittest.main()
