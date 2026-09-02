"""Phase 4.5 -- PPO / VecNormalize checkpoint compatibility gate.

No architectural change is made by this module. It verifies that the current

    landing_rl/envs/landing_env.py :: LandingEnv               (NEW)

is still compatible with the *actual* primary trained artifacts

    mujoco_rl/runs/ppo_landing_residual_v4_stage2_contact_final.zip
    mujoco_rl/runs/vecnormalize_v4_stage2_contact.pkl

and that OLD (``mujoco_rl/envs/env_prototype.py``) and NEW produce identical
deterministic policy rollouts under the real inference wrapper stack

    Monitor -> DummyVecEnv -> VecNormalize.load(...)   (training=False, norm_reward=False)

What is checked
---------------
  * The real PPO checkpoint loads against BOTH the OLD-wrapped and the
    NEW-wrapped env through ``PPO.load(..., env=...)`` (this genuinely invokes
    SB3 ``check_for_correct_spaces``); no observation/action-space mismatch.
  * VecNormalize loads independently for OLD and NEW from the SAME .pkl, and
    its compatibility-relevant properties are as expected (obs_rms shape (16,),
    training False, norm_obs True, norm_reward False, ...).
  * For matched seeds 5000..5004, run each episode to termination/truncation
    with ``deterministic=True`` and compare, step by step:
        - normalized observation (VecNormalize output)   exact
        - deterministic policy action OLD vs NEW          exact
        - reward, done                                    exact
        - info: success / failed / failure_reason / pos / vel / accel /
          v_pid / v_residual / v_cmd / wind_accel / dt / contact state
        - underlying env ``np_random.bit_generator.state`` (read-only access
          through the wrapper stack; no extra env calls, no ``_get_obs()``)

Comparison is EXACT (``np.array_equal`` / ``==`` / identical RNG state). Raw
OLD obs == raw NEW obs is already the Phase 1-4 gate, and both wrappers load
the same frozen ``obs_rms``, so normalized observations must match exactly;
no tolerance is introduced here.

Skip behavior
-------------
If ``stable_baselines3`` is unavailable, or the primary artifacts are not
present (``mujoco_rl/runs/`` is git-ignored and absent from the refactor
worktree; the files are read in place from the original worktree by absolute
path), the artifact-dependent tests SKIP with an explicit message. If the
artifacts ARE present the tests actually run. No checkpoint is copied, moved,
symlinked, modified, or committed.

Run from the worktree root:

    python3 -m unittest discover -s landing_rl/tests -p "test_*.py"
"""

from __future__ import annotations

import math
import pathlib
import unittest

import numpy as np

from test_legacy_regression_contract import (
    CONFIG_SPECS,
    NewLandingConfig,
    NewLandingEnv,
    OldLandingConfig,
    OldLandingEnv,
    _rng_state_equal,
)

# ---------------------------------------------------------------------------
# Primary artifacts. ``mujoco_rl/runs/`` is git-ignored and is not present in
# this refactor worktree, so read the artifacts in place from the original
# worktree by absolute path. Do NOT copy / move / symlink / modify / commit.
# ---------------------------------------------------------------------------

_PRIMARY_RUNS = pathlib.Path("/home/qntmdghkss/drone_stack/mujoco_rl/runs")
_CKPT = _PRIMARY_RUNS / "ppo_landing_residual_v4_stage2_contact_final.zip"
_VECNORM = _PRIMARY_RUNS / "vecnormalize_v4_stage2_contact.pkl"

_ARTIFACTS_OK = _CKPT.is_file() and _VECNORM.is_file()
_ARTIFACTS_MSG = (
    "primary artifacts present"
    if _ARTIFACTS_OK
    else f"primary artifact(s) missing under {_PRIMARY_RUNS}: "
    f"ckpt={_CKPT.is_file()} vecnormalize={_VECNORM.is_file()}"
)

try:  # SB3 is an existing project dependency; no new dependency is added.
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    _SB3_OK = True
    _SB3_MSG = "stable_baselines3 available"
except Exception as exc:  # pragma: no cover - environment-dependent
    _SB3_OK = False
    _SB3_MSG = f"stable_baselines3 unavailable: {exc!r}"


# Matched development-scale seeds (compatibility, not statistical performance).
SEEDS = (5000, 5001, 5002, 5003, 5004)

# info channels compared exactly OLD-vs-NEW every step (superset of the brief).
_INFO_KEYS_COMPARED = (
    "success", "failed", "failure_reason",
    "pos", "vel", "accel",
    "v_pid", "v_residual", "v_cmd",
    "wind_accel", "dt",
    "ground_contact", "contact_event", "contact_count", "bounce_count",
    "soft_contact", "hard_contact", "bounced", "touchdown_quality",
    "last_impact_vz", "last_touchdown_vxy",
    "altitude_agl", "xy_error", "z_error",
    "raw_action", "applied_action", "action_delay_steps", "obs_delay_steps",
)


def _stage2_eval_overrides() -> dict:
    """The eval config of record for the primary v4 stage2-contact checkpoint.

    Sourced verbatim (via ``CONFIG_SPECS``) from
    ``mujoco_rl/eval_compare_v2.py :: make_config()`` -- see the guard test
    ``test_config_of_record_matches_eval_compare_v2``.
    """
    for name, overrides, _doc in CONFIG_SPECS:
        if name == "stage2_eval":
            return dict(overrides)
    raise KeyError("stage2_eval spec not found in CONFIG_SPECS")


def _build_stack(env_cls, cfg_cls):
    """Monitor -> DummyVecEnv -> VecNormalize.load(same .pkl), inference mode.

    OLD and NEW each get an independent config instance and an independently
    loaded VecNormalize (no shared VecNormalize object).
    """
    overrides = _stage2_eval_overrides()

    def _make():
        return Monitor(env_cls(cfg_cls(**overrides)))

    venv = DummyVecEnv([_make])
    venv = VecNormalize.load(str(_VECNORM), venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def _inner_env(venv):
    """The base LandingEnv underneath VecNormalize -> DummyVecEnv -> Monitor.

    ``.unwrapped`` is a plain attribute walk; it triggers no environment call
    and does not touch observations.
    """
    return venv.venv.envs[0].unwrapped


def _val_equal(x, y) -> bool:
    """Exact equality for one info value (arrays compared with equal_nan so a
    NaN channel equals itself; no numerical tolerance)."""
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            return False
        if np.issubdtype(x.dtype, np.floating) or np.issubdtype(y.dtype, np.floating):
            return bool(np.array_equal(x, y, equal_nan=True))
        return bool(np.array_equal(x, y))
    if isinstance(x, float) and isinstance(y, float):
        return (math.isnan(x) and math.isnan(y)) or x == y
    return x == y


@unittest.skipUnless(_SB3_OK, _SB3_MSG)
@unittest.skipUnless(_ARTIFACTS_OK, _ARTIFACTS_MSG)
class CheckpointCompatibilityTest(unittest.TestCase):
    """OLD vs NEW compatibility with the real PPO / VecNormalize artifacts."""

    maxDiff = None

    # -- config of record -------------------------------------------------

    def test_config_of_record_matches_eval_compare_v2(self):
        """The overrides used here equal ``eval_compare_v2.py :: make_config()``
        (the eval config bound to the primary v4 stage2-contact checkpoint)."""
        ov = _stage2_eval_overrides()
        expected = dict(
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
        self.assertEqual(ov, expected)

    # -- space compatibility -------------------------------------------

    def test_checkpoint_loads_against_old_and_new(self):
        """The real checkpoint loads against BOTH wrapped envs via
        ``PPO.load(..., env=...)`` with no space mismatch."""
        old_venv = _build_stack(OldLandingEnv, OldLandingConfig)
        new_venv = _build_stack(NewLandingEnv, NewLandingConfig)
        try:
            model_old = PPO.load(str(_CKPT), env=old_venv, device="cpu")
            model_new = PPO.load(str(_CKPT), env=new_venv, device="cpu")

            # checkpoint spaces are identical regardless of which env it bound to
            self.assertEqual(model_old.observation_space, model_new.observation_space)
            self.assertEqual(model_old.action_space, model_new.action_space)

            # and they match what each wrapped env advertises
            for model, venv, tag in (
                (model_old, old_venv, "OLD"),
                (model_new, new_venv, "NEW"),
            ):
                self.assertEqual(
                    model.observation_space, venv.observation_space,
                    f"[{tag}] checkpoint observation_space != wrapped env",
                )
                self.assertEqual(
                    model.action_space, venv.action_space,
                    f"[{tag}] checkpoint action_space != wrapped env",
                )
                self.assertEqual(venv.observation_space.shape, (16,))
                self.assertEqual(venv.action_space.shape, (3,))
        finally:
            old_venv.close()
            new_venv.close()

    def test_vecnormalize_properties(self):
        """Compatibility-relevant VecNormalize properties, loaded independently
        for OLD and NEW from the same .pkl."""
        old_venv = _build_stack(OldLandingEnv, OldLandingConfig)
        new_venv = _build_stack(NewLandingEnv, NewLandingConfig)
        try:
            for venv, tag in ((old_venv, "OLD"), (new_venv, "NEW")):
                self.assertEqual(venv.obs_rms.mean.shape, (16,), f"[{tag}]")
                self.assertEqual(venv.obs_rms.var.shape, (16,), f"[{tag}]")
                self.assertFalse(venv.training, f"[{tag}]")
                self.assertTrue(venv.norm_obs, f"[{tag}]")
                self.assertFalse(venv.norm_reward, f"[{tag}]")
                self.assertEqual(float(venv.clip_obs), 10.0, f"[{tag}]")
                self.assertEqual(float(venv.clip_reward), 10.0, f"[{tag}]")
                self.assertEqual(float(venv.gamma), 0.995, f"[{tag}]")
                self.assertEqual(float(venv.epsilon), 1e-08, f"[{tag}]")

            # the two independently-loaded stat blocks are byte-identical
            self.assertTrue(
                np.array_equal(old_venv.obs_rms.mean, new_venv.obs_rms.mean)
            )
            self.assertTrue(
                np.array_equal(old_venv.obs_rms.var, new_venv.obs_rms.var)
            )
            self.assertEqual(old_venv.obs_rms.count, new_venv.obs_rms.count)
        finally:
            old_venv.close()
            new_venv.close()

    def test_normalized_reset_obs_parity(self):
        """VecNormalize-normalized reset observation is exactly equal OLD-vs-NEW
        for each matched seed."""
        old_venv = _build_stack(OldLandingEnv, OldLandingConfig)
        new_venv = _build_stack(NewLandingEnv, NewLandingConfig)
        try:
            for seed in SEEDS:
                old_venv.seed(seed)
                new_venv.seed(seed)
                old_obs = old_venv.reset()
                new_obs = new_venv.reset()
                self.assertEqual(old_obs.shape, (1, 16))
                self.assertEqual(old_obs.dtype, new_obs.dtype)
                self.assertTrue(
                    np.array_equal(old_obs, new_obs),
                    f"[seed={seed}] normalized reset obs mismatch:\n"
                    f"  OLD={old_obs!r}\n  NEW={new_obs!r}",
                )
                self.assertTrue(
                    _rng_state_equal(
                        _inner_env(old_venv).np_random.bit_generator.state,
                        _inner_env(new_venv).np_random.bit_generator.state,
                    ),
                    f"[seed={seed}] inner-env RNG state diverged at reset",
                )
        finally:
            old_venv.close()
            new_venv.close()

    # -- deterministic policy + rollout parity --------------------------

    def test_old_vs_new_deterministic_policy_parity(self):
        """Full deterministic rollout parity across 5 matched seeds: identical
        policy actions, normalized observations, rewards, done flags, info
        channels, and inner-env RNG state at every step."""
        model = PPO.load(str(_CKPT), device="cpu")

        old_venv = _build_stack(OldLandingEnv, OldLandingConfig)
        new_venv = _build_stack(NewLandingEnv, NewLandingConfig)
        episodes = 0
        total_steps = 0
        try:
            for seed in SEEDS:
                old_venv.seed(seed)
                new_venv.seed(seed)
                old_obs = old_venv.reset()
                new_obs = new_venv.reset()

                self.assertTrue(
                    np.array_equal(old_obs, new_obs),
                    f"[seed={seed}] normalized reset obs mismatch",
                )

                done = False
                step = 0
                while not done:
                    old_act, _ = model.predict(old_obs, deterministic=True)
                    new_act, _ = model.predict(new_obs, deterministic=True)
                    self.assertTrue(
                        np.array_equal(old_act, new_act),
                        f"[seed={seed} step={step}] deterministic action mismatch:\n"
                        f"  OLD={old_act!r}\n  NEW={new_act!r}\n"
                        f"  norm-obs OLD={old_obs!r}\n  norm-obs NEW={new_obs!r}",
                    )

                    old_obs, old_rew, old_done, old_infos = old_venv.step(old_act)
                    new_obs, new_rew, new_done, new_infos = new_venv.step(new_act)

                    ctx = f"[seed={seed} step={step}]"

                    self.assertTrue(
                        np.array_equal(old_obs, new_obs),
                        f"{ctx} normalized observation mismatch:\n"
                        f"  OLD={old_obs!r}\n  NEW={new_obs!r}",
                    )
                    self.assertEqual(
                        float(old_rew[0]), float(new_rew[0]),
                        f"{ctx} reward mismatch: OLD={old_rew!r} NEW={new_rew!r}",
                    )
                    self.assertEqual(
                        bool(old_done[0]), bool(new_done[0]),
                        f"{ctx} done mismatch: OLD={old_done!r} NEW={new_done!r}",
                    )

                    old_i, new_i = old_infos[0], new_infos[0]
                    self.assertEqual(
                        set(old_i) - {"episode", "terminal_observation"},
                        set(new_i) - {"episode", "terminal_observation"},
                        f"{ctx} info key-set mismatch",
                    )
                    for key in _INFO_KEYS_COMPARED:
                        self.assertIn(key, old_i, f"{ctx} OLD info missing {key!r}")
                        self.assertIn(key, new_i, f"{ctx} NEW info missing {key!r}")
                        self.assertTrue(
                            _val_equal(old_i[key], new_i[key]),
                            f"{ctx} info[{key!r}] mismatch: "
                            f"OLD={old_i[key]!r} NEW={new_i[key]!r}",
                        )
                    if "terminal_observation" in old_i or "terminal_observation" in new_i:
                        self.assertIn("terminal_observation", old_i, ctx)
                        self.assertIn("terminal_observation", new_i, ctx)
                        self.assertTrue(
                            np.array_equal(
                                old_i["terminal_observation"],
                                new_i["terminal_observation"],
                            ),
                            f"{ctx} terminal_observation mismatch",
                        )

                    self.assertTrue(
                        _rng_state_equal(
                            _inner_env(old_venv).np_random.bit_generator.state,
                            _inner_env(new_venv).np_random.bit_generator.state,
                        ),
                        f"{ctx} inner-env RNG state diverged",
                    )

                    done = bool(old_done[0])
                    step += 1
                    self.assertLess(step, 2000, f"[seed={seed}] episode did not end")

                episodes += 1
                total_steps += step

            # sanity: the matched seeds actually produced real multi-step episodes
            self.assertEqual(episodes, len(SEEDS))
            self.assertGreater(total_steps, len(SEEDS) * 5)
        finally:
            old_venv.close()
            new_venv.close()


if __name__ == "__main__":
    unittest.main()
