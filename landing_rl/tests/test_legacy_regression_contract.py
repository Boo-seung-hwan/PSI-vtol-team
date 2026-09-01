"""Phase 0 regression harness: legacy-vs-legacy self-consistency.

Purpose
-------
Freeze the *observable behavior* of the current canonical environment

    mujoco_rl/envs/env_prototype.py

before any structural refactor of ``landing_rl/`` begins.

What this suite does
--------------------
It instantiates the SAME legacy ``LandingEnv`` twice ("A" and "B") with an
identical config, an identical ``reset(seed=...)``, and an identical fixed
action sequence, then asserts the two instances stay bit-for-bit identical
across, at reset and after every step:

    * observation / reward / terminated / truncated
    * every ``info`` channel (plus the frozen ``info`` key-set)
    * ``env.np_random.bit_generator.state``
    * observation_space / action_space

What this suite deliberately does NOT do (Phase 0 scope)
-------------------------------------------------------
    * It does not import or construct any ``landing_rl`` environment
      (none exists yet).
    * It does not load PPO checkpoints, VecNormalize ``.pkl`` files, or any
      ``mujoco_rl/runs/`` artifact.
    * It does not introduce a numerical tolerance. Comparison is exact
      (``np.array_equal`` / ``==`` / identical RNG state). ``equal_nan`` is
      used only so an all-NaN channel compares equal to itself.
    * It never calls ``env._get_obs()`` (that call consumes RNG); only the
      values returned by ``reset()`` / ``step()`` are inspected.

Forward compatibility
---------------------
The comparison core (:func:`_assert_states_equal`, :func:`_assert_reset_equal`)
takes two ``(env, step_output)`` pairs. A later phase can replace side "B" with
``landing_rl.envs.LandingEnv`` to obtain an OLD-vs-NEW paired regression test
without reworking the assertions.

Run from the worktree root:

    python -m unittest discover -s landing_rl/tests -p "test_*.py"
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Load the canonical legacy environment directly from its file path.
#
# We do NOT rely on sys.path / cwd (the mujoco_rl scripts use ``from
# envs.env_prototype import ...`` which only works when run from mujoco_rl/,
# and ``landing_rl/envs`` is an unrelated empty directory). Loading by file
# path keeps this harness independent of packaging and import layout, which
# section 7 / section J of the Phase 0 brief require us not to touch.
# ---------------------------------------------------------------------------

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parents[1]  # landing_rl/tests -> landing_rl -> <worktree root>
_ENV_PROTOTYPE_PATH = _REPO_ROOT / "mujoco_rl" / "envs" / "env_prototype.py"


def _load_legacy_module():
    if not _ENV_PROTOTYPE_PATH.is_file():
        raise FileNotFoundError(
            f"legacy reference implementation not found: {_ENV_PROTOTYPE_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "phase0_legacy_env_prototype", _ENV_PROTOTYPE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # help dataclass / pickling introspection
    spec.loader.exec_module(module)
    return module


_LEGACY = _load_legacy_module()
LandingEnv = _LEGACY.LandingEnv
LandingConfig = _LEGACY.LandingConfig


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
# Config factories. Each returns a fresh LandingConfig and documents the exact
# source file / callable it was copied from. Defaults, field names, and values
# are reproduced verbatim (Phase 0 section H: do not split / rename / alter).
# ---------------------------------------------------------------------------

def cfg_default():
    """``mujoco_rl/envs/env_prototype.py`` :: ``LandingConfig()`` dataclass
    defaults. Also the config used by ``mujoco_rl/scripts/plot_trajectory.py``
    (which constructs a bare ``LandingEnv()``). ``contact_enabled`` is True by
    default."""
    return LandingConfig()


def cfg_stage0_eval():
    """Copied verbatim from ``mujoco_rl/eval_env_stage0.py`` :: ``make_config()``
    (identical to ``mujoco_rl/train_ppo_v3_long.py`` :: ``make_config()``).
    Contact disabled; action/obs delay range 0-2."""
    return LandingConfig(
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
    )


def cfg_stage2_eval():
    """Copied verbatim from ``mujoco_rl/eval_compare_v2.py`` :: ``make_config()``.
    This is the final-stage evaluation config for the primary checkpoint of
    record (``ppo_landing_residual_v4_stage2_contact_final``). Contact enabled,
    fixed 1-step delays, zero perception noise, wind + ground effect on."""
    return LandingConfig(
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
    )


def cfg_stage2_train():
    """Copied verbatim from ``mujoco_rl/train_ppo_stage2_contact.py`` ::
    ``make_config()``. Widest exercised randomization: action/obs delay range
    1-5, non-zero dropout / stale / outlier / target noise, wind. Included for
    RNG-path coverage (variable per-episode draw counts)."""
    return LandingConfig(
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
    )


CONFIG_FACTORIES = (
    ("default", cfg_default),
    ("stage0_eval", cfg_stage0_eval),
    ("stage2_eval", cfg_stage2_eval),
    ("stage2_train", cfg_stage2_train),
)

SEEDS = (0, 1, 5000)
PATTERNS = ("zero", "pseudo_random", "saturation")

# Fast development budget (Phase 0 section 4: ~50-100 steps/case, not the full
# final gate). Episodes that terminate/truncate earlier stop early.
MAIN_STEPS = 100

# Longer budget for the contact-path scenario only. The default episode length
# is 420 steps, so this still runs at most one full episode per env.
CONTACT_STEPS = 420


# ---------------------------------------------------------------------------
# Fixed action sequences.
#
# CRITICAL (Phase 0 section 4/6): the action-sequence RNG must NOT be either
# environment's ``np_random``. ``pseudo_random`` uses a standalone
# ``np.random.default_rng`` seeded with a constant unrelated to any env seed.
# ---------------------------------------------------------------------------

_ACTION_SEQ_RNG_SEED = 20260902  # standalone; never passed to env.reset()


def make_action_sequence(pattern: str, n_steps: int) -> np.ndarray:
    """Return a deterministic ``(n_steps, 3)`` float32 action array."""
    if pattern == "zero":
        # Exercises the PID-only path: both descent gates, contact, and the
        # v_cmd descent-bias override with no residual contribution.
        return np.zeros((n_steps, 3), dtype=np.float32)

    if pattern == "pseudo_random":
        rng = np.random.default_rng(_ACTION_SEQ_RNG_SEED)
        return rng.uniform(-1.0, 1.0, size=(n_steps, 3)).astype(np.float32)

    if pattern == "saturation":
        # Out-of-range magnitudes on every step: exercises the input clip
        # (``np.clip(action, -1, 1)``), the saturation reward term
        # (``|a| - 0.80``), residual saturation, and ``_limit_velocity_command``.
        seq = np.empty((n_steps, 3), dtype=np.float32)
        seq[0::2] = np.array([1.5, -1.5, 1.2], dtype=np.float32)
        seq[1::2] = np.array([-1.5, 1.5, -1.2], dtype=np.float32)
        return seq

    if pattern == "descend_z":
        # Sustained full-scale downward residual; drives the vehicle to
        # touchdown within the default episode so the contact model runs.
        seq = np.zeros((n_steps, 3), dtype=np.float32)
        seq[:, 2] = 1.0
        return seq

    raise ValueError(f"unknown action pattern: {pattern!r}")


# ---------------------------------------------------------------------------
# Exact comparators
# ---------------------------------------------------------------------------

def _rng_state_equal(a, b) -> bool:
    """Recursively compare two ``bit_generator.state`` structures.

    For the default PCG64 generator the state is a nested dict of Python ints,
    but this also handles ndarray-valued entries used by other bit generators.
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


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

class LegacyRegressionContractTest(unittest.TestCase):
    """Legacy ``LandingEnv`` frozen-behavior contract."""

    maxDiff = None

    # -- shared assertions ------------------------------------------------

    def _assert_reset_equal(self, a_reset, b_reset, env_a, env_b, ctx):
        a_obs, a_info = a_reset
        b_obs, b_info = b_reset
        problems = []

        if a_info != {}:
            problems.append(f"A reset info is not empty: {a_info!r}")
        if b_info != {}:
            problems.append(f"B reset info is not empty: {b_info!r}")
        if not _obs_equal(a_obs, b_obs):
            problems.append(
                f"reset observation mismatch:\n  A={a_obs!r}\n  B={b_obs!r}"
            )
        if not _rng_state_equal(
            env_a.np_random.bit_generator.state,
            env_b.np_random.bit_generator.state,
        ):
            problems.append(
                "reset np_random.bit_generator.state diverged:\n"
                f"  A={env_a.np_random.bit_generator.state}\n"
                f"  B={env_b.np_random.bit_generator.state}"
            )

        if problems:
            self.fail(f"[{ctx}] reset contract broken:\n" + "\n".join(problems))

    def _assert_states_equal(self, a_out, b_out, env_a, env_b, ctx):
        a_obs, a_rew, a_term, a_trunc, a_info = a_out
        b_obs, b_rew, b_term, b_trunc, b_info = b_out
        problems = []

        if not _obs_equal(a_obs, b_obs):
            problems.append(
                f"observation mismatch:\n  A={a_obs!r}\n  B={b_obs!r}"
            )

        if not _value_equal(float(a_rew), float(b_rew)):
            problems.append(f"reward mismatch: A={a_rew!r} B={b_rew!r}")

        if (a_term, a_trunc) != (b_term, b_trunc):
            problems.append(
                f"terminated/truncated mismatch: "
                f"A=(term={a_term}, trunc={a_trunc}) "
                f"B=(term={b_term}, trunc={b_trunc})"
            )

        a_keys, b_keys = set(a_info), set(b_info)
        if a_keys != b_keys:
            problems.append(
                f"info key-set mismatch between A and B: "
                f"A-only={sorted(a_keys - b_keys)} "
                f"B-only={sorted(b_keys - a_keys)}"
            )
        if a_keys != EXPECTED_INFO_KEYS:
            problems.append(
                "info key-set drifted from frozen contract: "
                f"missing={sorted(EXPECTED_INFO_KEYS - a_keys)} "
                f"unexpected={sorted(a_keys - EXPECTED_INFO_KEYS)}"
            )

        for key in sorted(a_keys & b_keys):
            if not _value_equal(a_info[key], b_info[key]):
                problems.append(
                    f"info[{key!r}] mismatch: A={a_info[key]!r} B={b_info[key]!r}"
                )

        # alpha_eff must remain the [nan, nan, nan] sentinel (RNG-free, but part
        # of the frozen info contract downstream scripts read).
        ae = a_info.get("alpha_eff")
        if not (
            isinstance(ae, np.ndarray)
            and ae.shape == (3,)
            and np.all(np.isnan(ae))
        ):
            problems.append(f"alpha_eff sentinel changed: {ae!r}")

        if not _rng_state_equal(
            env_a.np_random.bit_generator.state,
            env_b.np_random.bit_generator.state,
        ):
            problems.append(
                "np_random.bit_generator.state diverged:\n"
                f"  A={env_a.np_random.bit_generator.state}\n"
                f"  B={env_b.np_random.bit_generator.state}"
            )

        if problems:
            self.fail(
                f"[{ctx}] legacy self-consistency broken:\n" + "\n".join(problems)
            )

    def _run_paired(self, cfg_factory, seed, pattern, n_steps):
        """Run two legacy envs in lockstep. Returns True if the contact model
        was observed to activate at any compared step."""
        env_a = LandingEnv(cfg_factory())
        env_b = LandingEnv(cfg_factory())

        self.assertEqual(
            env_a.observation_space, env_b.observation_space,
            "paired envs disagree on observation_space",
        )
        self.assertEqual(
            env_a.action_space, env_b.action_space,
            "paired envs disagree on action_space",
        )

        a_reset = env_a.reset(seed=seed)
        b_reset = env_b.reset(seed=seed)
        self._assert_reset_equal(
            a_reset, b_reset, env_a, env_b, f"{pattern} seed={seed} @reset"
        )

        actions = make_action_sequence(pattern, n_steps)
        saw_contact = False
        try:
            for i in range(n_steps):
                act = actions[i].copy()
                a_out = env_a.step(act.copy())
                b_out = env_b.step(act.copy())
                self._assert_states_equal(
                    a_out, b_out, env_a, env_b,
                    f"{pattern} seed={seed} step={i}",
                )
                info = a_out[4]
                if info.get("contact_event") or info.get("ground_contact"):
                    saw_contact = True
                if a_out[2] or a_out[3]:  # terminated or truncated
                    break
        finally:
            env_a.close()
            env_b.close()
        return saw_contact

    # -- tests ----------------------------------------------------------

    def test_observation_and_action_space_frozen(self):
        """Space definition matches the frozen baseline exactly (guards e.g. a
        change to ``max_acceleration_obs_mps2``, which is baked into the space)."""
        env = LandingEnv(LandingConfig())
        try:
            obs_space = env.observation_space
            act_space = env.action_space

            self.assertEqual(obs_space.shape, (16,))
            self.assertEqual(obs_space.dtype, np.dtype(np.float32))
            self.assertTrue(
                np.array_equal(obs_space.high, EXPECTED_OBS_HIGH),
                f"observation_space.high drifted:\n  got={obs_space.high!r}\n"
                f"  expected={EXPECTED_OBS_HIGH!r}",
            )
            self.assertTrue(
                np.array_equal(obs_space.low, -EXPECTED_OBS_HIGH),
                f"observation_space.low drifted:\n  got={obs_space.low!r}\n"
                f"  expected={(-EXPECTED_OBS_HIGH)!r}",
            )

            self.assertEqual(act_space.shape, (3,))
            self.assertEqual(act_space.dtype, np.dtype(np.float32))
            self.assertTrue(np.array_equal(act_space.high, np.ones(3, np.float32)))
            self.assertTrue(np.array_equal(act_space.low, -np.ones(3, np.float32)))

            # The acceleration bound in the space is sourced from config.
            self.assertEqual(LandingConfig().max_acceleration_obs_mps2, 4.0)
            self.assertTrue(
                np.array_equal(
                    obs_space.high[6:9],
                    np.full(3, env.cfg.max_acceleration_obs_mps2, dtype=np.float32),
                )
            )
        finally:
            env.close()

    def test_reset_info_is_empty_dict(self):
        """``reset`` returns ``(obs, {})`` for every Phase 0 config/seed."""
        for cfg_name, cfg_factory in CONFIG_FACTORIES:
            for seed in SEEDS:
                with self.subTest(config=cfg_name, seed=seed):
                    env = LandingEnv(cfg_factory())
                    try:
                        _, info = env.reset(seed=seed)
                        self.assertEqual(info, {})
                    finally:
                        env.close()

    def test_info_key_set_frozen(self):
        """``step`` info exposes exactly the frozen key-set (no additions or
        removals vs the current env_prototype.py baseline)."""
        env = LandingEnv(LandingConfig())
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            self.assertEqual(set(info), set(EXPECTED_INFO_KEYS))
        finally:
            env.close()

    def test_rng_state_matches_immediately_after_reset(self):
        """Two identically-seeded legacy envs share RNG state right after reset,
        before any step consumes further draws."""
        for cfg_name, cfg_factory in CONFIG_FACTORIES:
            for seed in SEEDS:
                with self.subTest(config=cfg_name, seed=seed):
                    env_a = LandingEnv(cfg_factory())
                    env_b = LandingEnv(cfg_factory())
                    try:
                        env_a.reset(seed=seed)
                        env_b.reset(seed=seed)
                        self.assertTrue(
                            _rng_state_equal(
                                env_a.np_random.bit_generator.state,
                                env_b.np_random.bit_generator.state,
                            ),
                            f"[{cfg_name} seed={seed}] RNG state diverged at reset",
                        )
                    finally:
                        env_a.close()
                        env_b.close()

    def test_paired_legacy_self_consistency(self):
        """Full observable contract, lockstep, for every config x seed x pattern."""
        for cfg_name, cfg_factory in CONFIG_FACTORIES:
            for seed in SEEDS:
                for pattern in PATTERNS:
                    with self.subTest(config=cfg_name, seed=seed, pattern=pattern):
                        self._run_paired(cfg_factory, seed, pattern, MAIN_STEPS)

    def test_contact_path_is_exercised_and_reproducible(self):
        """A sustained downward residual reaches touchdown within the default
        420-step episode; the contact model stays deterministic across the
        paired run."""
        saw_contact = False
        for seed in (0, 1):
            with self.subTest(seed=seed):
                if self._run_paired(cfg_default, seed, "descend_z", CONTACT_STEPS):
                    saw_contact = True
        self.assertTrue(
            saw_contact,
            "expected the contact model to activate (contact_event / "
            "ground_contact) in at least one descend_z run; if the plant "
            "changed, adjust the seed or step budget",
        )


if __name__ == "__main__":
    unittest.main()
