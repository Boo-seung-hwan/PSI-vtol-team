"""Canonical-environment entry-point migration gate.

Purpose
-------
This is the regression gate for the "mujoco_rl/envs/env_prototype.py ->
landing_rl/envs/landing_env.py" TRAIN/EVAL ENTRY-POINT migration (moving
which ``LandingEnv`` the real train/eval scripts import; no dynamics /
reward / controller / config-value change).

It does **not** re-verify that ``landing_rl.envs.landing_env.LandingEnv`` is
behaviorally identical to ``mujoco_rl/envs/env_prototype.py::LandingEnv`` --
``test_legacy_regression_contract.py`` (exact OLD-vs-NEW parity) and
``test_structural_freeze.py`` (expanded seed/config/regime matrix) already
own that, exhaustively, and are reused here rather than re-implemented
(their comparison machinery is imported, never copied). This file adds
exactly what those two do not cover: proof that the concrete, real
train/eval SCRIPT FILES under ``mujoco_rl/`` now actually import
``landing_rl``'s ``LandingEnv``/``LandingConfig`` (not ``env_prototype``'s),
that each script's own unmodified ``make_config()`` still resolves to the
exact same field values it did before the import-line edit, and that the
resulting per-script config still produces exact OLD-vs-NEW parity when
run through both implementations.

No production source (``landing_rl/`` component code) is touched or
exercised differently by this file. If anything here fails, that is a STOP
condition for the entry-point migration, not something to route around.

Run from the worktree root:

    python3 -m unittest discover -s landing_rl/tests -p "test_*.py"

or in isolation:

    python3 -m unittest landing_rl.tests.test_entry_point_migration -v
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Locate paths (independent of cwd / PYTHONPATH) -- same pattern as
# test_legacy_regression_contract.py / test_structural_freeze.py.
# ---------------------------------------------------------------------------

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parents[1]  # landing_rl/tests -> landing_rl -> <worktree root>
_MUJOCO_RL_DIR = _REPO_ROOT / "mujoco_rl"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ``test_legacy_regression_contract`` is discovered as a flat top-level
# module by ``unittest discover -s landing_rl/tests`` (no ``__init__.py`` in
# that directory), so it is imported the same way here -- mirroring the
# precedent already established by test_structural_freeze.py. This reuses
# its already-verified OLD/NEW module handles and exact comparators instead
# of re-implementing them.
import test_legacy_regression_contract as _trc  # noqa: E402
from test_legacy_regression_contract import (  # noqa: E402
    CONFIG_SPECS,
    NewLandingConfig,
    NewLandingEnv,
    OldLandingConfig,
    OldLandingEnv,
    _rng_state,
    _rng_state_equal,
    make_action_sequence,
)

# NOTE: exactly like test_structural_freeze.py, ``LandingEnvParityContractTest``
# is deliberately accessed only as ``_trc.LandingEnvParityContractTest``,
# never imported by name into this module's namespace -- importing the class
# directly would make unittest's module-level discovery re-collect it and
# run its ``test_*`` methods a SECOND time under this module.


def _new_helper():
    """A ``LandingEnvParityContractTest`` instance used purely as a holder of
    its already-verified ``_assert_reset_equal`` / ``_assert_states_equal``
    methods -- never registered with the test loader, never run as a test
    itself. Mirrors ``test_structural_freeze.py::_new_helper``."""
    return _trc.LandingEnvParityContractTest("test_new_package_import")


def _config_by_name(name: str):
    for cfg_name, overrides, _doc in CONFIG_SPECS:
        if cfg_name == name:
            return overrides
    raise KeyError(f"no CONFIG_SPECS entry named {name!r}")


def _load_script_module(relpath: str):
    """Load a real train/eval script by file path (not ``import``), exactly
    as ``test_legacy_regression_contract._load_legacy_module`` loads the OLD
    reference file. Executing the module's top-level code is what proves the
    edited import line actually resolves; neither ``main()`` nor any
    ``if __name__ == "__main__":`` block runs."""
    path = _MUJOCO_RL_DIR / relpath
    if not path.is_file():
        raise FileNotFoundError(f"entry-point script not found: {path}")
    mod_name = "migration_check_" + relpath.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The real entry-point scripts and the CONFIG_SPECS name each one's
# unmodified make_config() must resolve to (verified by value, not by
# re-typing the override dict -- see test_B_script_config_matches_frozen_spec).
# ---------------------------------------------------------------------------

SCRIPTS_WITH_CONFIG_SPEC = (
    ("eval_env_stage0.py", "stage0_eval"),
    ("train_ppo_v3_long.py", "stage0_eval"),
    ("train_ppo_stage1_1m.py", "stage0_eval"),
    ("eval_compare_v2.py", "stage2_eval"),
    ("train_ppo_stage2_contact.py", "stage2_train"),
)

# plot_trajectory.py takes no make_config() -- it calls LandingEnv() directly,
# i.e. NewLandingConfig() defaults == CONFIG_SPECS "default".
SCRIPT_WITHOUT_MAKE_CONFIG = ("scripts/plot_trajectory.py", "default")

ALL_MIGRATED_SCRIPTS = tuple(name for name, _ in SCRIPTS_WITH_CONFIG_SPEC) + (
    SCRIPT_WITHOUT_MAKE_CONFIG[0],
    "eval_robustness_paper.py",
)

# Checkpoint / VecNormalize filename literals that must be byte-identical in
# the script source text after the import-line edit (section 9 of the
# migration task: "script가 저장/로드하는 checkpoint 이름과 VecNormalize 경로도
# 변경하지 말 것").
CHECKPOINT_LITERALS = {
    "train_ppo_v3_long.py": [
        "./runs/ppo_landing_residual_v3_stage0_final.zip",
        "./runs/ppo_landing_residual_v3_stage1_final",
        "./runs/vecnormalize_v3_stage0.pkl",
        "./runs/vecnormalize_v3_stage1.pkl",
    ],
    "train_ppo_stage1_1m.py": [
        "./runs/ppo_landing_residual_v3_stage0_final.zip",
        "./runs/ppo_landing_residual_v3_stage1_rewardfix1_final",
        "./runs/vecnormalize_v3_stage0.pkl",
        "./runs/vecnormalize_v3_stage1_rewardfix1.pkl",
    ],
    "train_ppo_stage2_contact.py": [
        "./runs/ppo_landing_residual_v3_stage2_contact_final.zip",
        "./runs/ppo_landing_residual_v4_stage2_contact_final",
        "./runs/vecnormalize_v3_stage2_contact.pkl",
        "./runs/vecnormalize_v4_stage2_contact.pkl",
    ],
    "eval_compare_v2.py": [
        "./runs/ppo_landing_residual_v4_stage2_contact_final.zip",
        "./runs/vecnormalize_v4_stage2_contact.pkl",
    ],
    "eval_env_stage0.py": [
        "./runs/vecnormalize_v3_stage0.pkl",
    ],
    "eval_robustness_paper.py": [
        "./runs/ppo_landing_residual_v4_stage2_contact_final.zip",
        "./runs/vecnormalize_v4_stage2_contact.pkl",
    ],
}

SEEDS = tuple(range(10))  # 0..9, per migration task section 6 minimum
MAIN_PATTERNS = ("zero", "pseudo_random", "saturation")
N_STEPS = 80

# Additional deterministic action patterns beyond
# test_legacy_regression_contract.PATTERNS ("zero" / "pseudo_random" /
# "saturation"). Defined locally so the frozen pattern table in that file is
# never touched. Covers migration task section 6's "fixed +axis commands" /
# "fixed -axis commands".
FIXED_AXIS_STEPS = 40


def _fixed_axis_actions(sign: float, n_steps: int) -> np.ndarray:
    """Constant full-scale action every step: ``[sign, sign, sign]``. A
    sustained single-direction residual command, distinct from the
    alternating ``saturation`` pattern already in the frozen suite."""
    return np.full((n_steps, 3), float(sign), dtype=np.float32)


def _run_paired_actions(helper, overrides, seed, actions, label):
    """OLD-vs-NEW lockstep run over an explicit action array, reusing the
    already-verified ``_assert_reset_equal`` / ``_assert_states_equal``
    comparators from ``LandingEnvParityContractTest`` (via ``helper``).

    This duplicates only the loop *scaffolding* of
    ``LandingEnvParityContractTest._run_paired`` (reset once, step in
    lockstep, compare every step) -- never the comparison semantics, which
    stay owned by the imported methods. Needed because ``_run_paired`` calls
    the frozen, pattern-name-only ``make_action_sequence`` internally and so
    cannot be handed a pre-built actions array for the two local fixed-axis
    patterns.
    """
    old_env = OldLandingEnv(OldLandingConfig(**overrides))
    new_env = NewLandingEnv(NewLandingConfig(**overrides))
    try:
        old_reset = old_env.reset(seed=seed)
        new_reset = new_env.reset(seed=seed)
        helper._assert_reset_equal(old_reset, new_reset, old_env, new_env, f"{label} @reset")

        rng_equal_before = _rng_state_equal(_rng_state(old_env), _rng_state(new_env))
        for i in range(len(actions)):
            act = actions[i].copy()
            old_out = old_env.step(act.copy())
            new_out = new_env.step(act.copy())
            helper._assert_states_equal(
                old_out, new_out, old_env, new_env, f"{label} step={i}", rng_equal_before
            )
            rng_equal_before = _rng_state_equal(_rng_state(old_env), _rng_state(new_env))
            if old_out[2] or old_out[3]:  # terminated or truncated
                break
    finally:
        old_env.close()
        new_env.close()


class EntryPointMigrationTest(unittest.TestCase):
    """Proves the real train/eval scripts now import landing_rl's LandingEnv,
    that their unmodified make_config() values are unchanged, and that those
    exact configs still produce exact OLD-vs-NEW parity."""

    maxDiff = None

    # -- A: the scripts actually resolve to landing_rl now ------------------

    def test_A_all_migrated_scripts_import_landing_rl_env(self):
        """Every migrated entry-point script's ``LandingEnv``/``LandingConfig``
        names are IDENTICAL objects to ``landing_rl.envs.landing_env``'s, and
        are NOT the OLD ``env_prototype`` classes."""
        for relpath in ALL_MIGRATED_SCRIPTS:
            with self.subTest(script=relpath):
                mod = _load_script_module(relpath)
                self.assertIs(
                    mod.LandingEnv, NewLandingEnv,
                    f"{relpath} does not import landing_rl's LandingEnv",
                )
                self.assertIs(
                    mod.LandingConfig, NewLandingConfig,
                    f"{relpath} does not import landing_rl's LandingConfig",
                )
                self.assertIsNot(mod.LandingEnv, OldLandingEnv)
                self.assertIsNot(mod.LandingConfig, OldLandingConfig)
                self.assertEqual(mod.LandingEnv.__module__, "landing_rl.envs.landing_env")
                self.assertEqual(mod.LandingConfig.__module__, "landing_rl.envs.landing_env")

    # -- B: make_config() values are byte-unchanged --------------------------

    def test_B_script_config_matches_frozen_spec(self):
        """Each script's own, UNEDITED ``make_config()`` resolves to exactly
        the same field values as the corresponding frozen ``CONFIG_SPECS``
        entry -- proves the import-line edit did not also touch config
        values, by comparing resolved dataclass field values (not raw text,
        so e.g. train_ppo_stage1_1m.py's config -- which spells out
        ``max_steps=420, residual_z_mps=0.05`` explicitly, values that equal
        the ``LandingConfig`` defaults ``stage0_eval`` leaves implicit --
        still compares exactly equal)."""
        for relpath, spec_name in SCRIPTS_WITH_CONFIG_SPEC:
            with self.subTest(script=relpath, spec=spec_name):
                mod = _load_script_module(relpath)
                script_cfg = mod.make_config()
                expected_overrides = _config_by_name(spec_name)
                expected_cfg = NewLandingConfig(**expected_overrides)
                self.assertEqual(
                    dataclasses.asdict(script_cfg),
                    dataclasses.asdict(expected_cfg),
                    f"{relpath}::make_config() drifted from CONFIG_SPECS[{spec_name!r}]",
                )

        # plot_trajectory.py: LandingEnv() with no config argument.
        relpath, spec_name = SCRIPT_WITHOUT_MAKE_CONFIG
        mod = _load_script_module(relpath)
        env = mod.LandingEnv()
        try:
            expected_cfg = NewLandingConfig(**_config_by_name(spec_name))
            self.assertEqual(dataclasses.asdict(env.cfg), dataclasses.asdict(expected_cfg))
        finally:
            env.close()

    def test_B_robustness_paper_cases_unchanged(self):
        """eval_robustness_paper.py's five case configs (not covered by
        CONFIG_SPECS) are read directly from the script's own
        ``COMMON_CONFIG`` / ``CASE_OVERRIDES`` / ``make_config`` -- this test
        only proves those literals still construct valid, matching configs;
        it does not re-type the override values."""
        mod = _load_script_module("eval_robustness_paper.py")
        self.assertEqual(
            set(mod.CASE_OVERRIDES.keys()),
            {"A_nominal", "B_delay", "C_target", "D_wind", "E_mixed"},
        )
        for case in mod.CASE_OVERRIDES:
            with self.subTest(case=case):
                cfg_via_script = mod.make_config(case)
                values = dict(mod.COMMON_CONFIG)
                values.update(mod.CASE_OVERRIDES[case])
                cfg_via_literals = NewLandingConfig(**values)
                self.assertEqual(
                    dataclasses.asdict(cfg_via_script),
                    dataclasses.asdict(cfg_via_literals),
                )

    # -- C: checkpoint / VecNormalize path literals are untouched ------------

    def test_C_checkpoint_and_vecnormalize_paths_unchanged(self):
        """The exact checkpoint/.pkl filename literals used by each script
        are still present, byte-for-byte, in its source text."""
        for relpath, literals in CHECKPOINT_LITERALS.items():
            with self.subTest(script=relpath):
                text = (_MUJOCO_RL_DIR / relpath).read_text()
                for literal in literals:
                    self.assertIn(
                        literal, text,
                        f"{relpath} no longer contains expected literal {literal!r}",
                    )

    # -- D: exact OLD-vs-NEW parity for every real script config -------------

    def test_D_script_configs_exact_parity_main_patterns(self):
        """For every script config, OLD (env_prototype) vs NEW (landing_rl)
        agree exactly (observation/reward/terminated/truncated/info) across
        seeds 0-9 and the frozen zero/pseudo_random/saturation patterns."""
        helper = _new_helper()
        spec_names = sorted({spec for _, spec in SCRIPTS_WITH_CONFIG_SPEC} | {"default"})
        for spec_name in spec_names:
            overrides = _config_by_name(spec_name)
            for seed in SEEDS:
                for pattern in MAIN_PATTERNS:
                    with self.subTest(config=spec_name, seed=seed, pattern=pattern):
                        helper._run_paired(
                            overrides, seed, pattern, N_STEPS,
                            left_impl=_trc.LEGACY_IMPL, right_impl=_trc.NEW_IMPL,
                        )

    def test_D_script_configs_exact_parity_fixed_axis_patterns(self):
        """Same configs/seeds, but with sustained constant +1 / -1 actions on
        all three axes every step (not in the frozen PATTERNS table)."""
        helper = _new_helper()
        spec_names = sorted({spec for _, spec in SCRIPTS_WITH_CONFIG_SPEC} | {"default"})
        for spec_name in spec_names:
            overrides = _config_by_name(spec_name)
            for seed in SEEDS:
                for sign, label in ((+1.0, "fixed_plus"), (-1.0, "fixed_minus")):
                    with self.subTest(config=spec_name, seed=seed, pattern=label):
                        actions = _fixed_axis_actions(sign, FIXED_AXIS_STEPS)
                        _run_paired_actions(
                            helper, overrides, seed, actions,
                            f"{spec_name} {label} seed={seed}",
                        )

    def test_D_robustness_paper_cases_exact_parity(self):
        """Light parity check (seed 0, zero + saturation patterns) for the 5
        eval_robustness_paper.py cases not present in CONFIG_SPECS."""
        helper = _new_helper()
        mod = _load_script_module("eval_robustness_paper.py")
        for case in mod.CASE_OVERRIDES:
            values = dict(mod.COMMON_CONFIG)
            values.update(mod.CASE_OVERRIDES[case])
            for pattern in ("zero", "saturation"):
                with self.subTest(case=case, pattern=pattern):
                    helper._run_paired(
                        values, 0, pattern, FIXED_AXIS_STEPS,
                        left_impl=_trc.LEGACY_IMPL, right_impl=_trc.NEW_IMPL,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
