"""Evaluation paths are CWD-independent, PID-only needs no PPO artifacts, and no
machine-specific absolute path exists in the code base.

None of these tests needs a PPO model or VecNormalize file.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from landing_rl.evaluation import artifacts  # noqa: E402

try:
    import stable_baselines3  # noqa: F401

    _SB3_OK = True
except Exception:  # pragma: no cover - environment-dependent
    _SB3_OK = False

_MUJOCO_RL_DIR = _REPO_ROOT / "mujoco_rl"


def _load_script(name: str):
    path = _MUJOCO_RL_DIR / name
    spec = importlib.util.spec_from_file_location("eval_artifacts_check_" + name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in (artifacts.ARTIFACT_DIR_ENV, artifacts.REQUIRE_ARTIFACTS_ENV)}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


class ArtifactPathTest(unittest.TestCase):
    def test_pair_of_record_names(self):
        self.assertEqual(artifacts.BASELINE_MODEL_NAME, "ppo_landing_residual_v4_stage2_contact_final.zip")
        self.assertEqual(artifacts.BASELINE_VECNORM_NAME, "vecnormalize_v4_stage2_contact.pkl")

    def test_repo_root_is_this_checkout(self):
        self.assertEqual(artifacts.REPO_ROOT, _REPO_ROOT)

    def test_default_dir_is_repo_relative_and_cwd_independent(self):
        with _clean_env():
            expected = _REPO_ROOT / "mujoco_rl" / "runs"
            before = artifacts.default_artifact_dir()
            with tempfile.TemporaryDirectory() as tmp:
                old = os.getcwd()
                os.chdir(tmp)
                try:
                    after = artifacts.default_artifact_dir()
                finally:
                    os.chdir(old)
            self.assertEqual(before, expected)
            self.assertEqual(after, expected)
            self.assertEqual(artifacts.default_model_path(), expected / artifacts.BASELINE_MODEL_NAME)
            self.assertEqual(artifacts.default_vecnorm_path(), expected / artifacts.BASELINE_VECNORM_NAME)

    def test_env_var_overrides_artifact_dir_but_not_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _clean_env(**{artifacts.ARTIFACT_DIR_ENV: tmp}):
                self.assertEqual(artifacts.default_artifact_dir(), pathlib.Path(tmp))
                self.assertEqual(artifacts.default_model_path().parent, pathlib.Path(tmp))
                # results must never default into the (possibly archival) artifact dir
                self.assertNotEqual(artifacts.default_output_dir().parent, pathlib.Path(tmp))
                self.assertEqual(artifacts.default_output_dir(), _REPO_ROOT / "mujoco_rl" / "runs" / "paper_eval")

    def test_missing_artifact_message_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _clean_env(**{artifacts.ARTIFACT_DIR_ENV: tmp}):
                msg = artifacts.missing_artifact_message()
        self.assertIsNotNone(msg)
        self.assertIn(artifacts.ARTIFACT_DIR_ENV, msg)
        self.assertIn(artifacts.BASELINE_MODEL_NAME, msg)
        self.assertIn(artifacts.BASELINE_VECNORM_NAME, msg)
        self.assertIn("--policy pid", msg)

    def test_present_artifacts_report_no_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / artifacts.BASELINE_MODEL_NAME).write_bytes(b"x")
            (d / artifacts.BASELINE_VECNORM_NAME).write_bytes(b"x")
            with _clean_env(**{artifacts.ARTIFACT_DIR_ENV: tmp}):
                self.assertIsNone(artifacts.missing_artifact_message())

    def test_require_flag(self):
        with _clean_env():
            self.assertFalse(artifacts.artifacts_required())
        with _clean_env(**{artifacts.REQUIRE_ARTIFACTS_ENV: "1"}):
            self.assertTrue(artifacts.artifacts_required())


class NoMachineSpecificPathsTest(unittest.TestCase):
    """A hard-coded home-directory path makes a test/script pass only on one
    developer's machine; keep it out of code."""

    def test_no_absolute_home_paths_in_python_sources(self):
        pattern = re.compile("/" + "home" + r"/[A-Za-z0-9_.-]+/")
        offenders = []
        for top in ("landing_rl", "landing_mujoco", "mujoco_rl", "system_id"):
            for path in (_REPO_ROOT / top).rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if pattern.search(line):
                        offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [], "machine-specific absolute paths found:\n" + "\n".join(offenders))


@unittest.skipUnless(_SB3_OK, "stable_baselines3 unavailable")
class EvalCompareCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("eval_compare_v2.py")

    def test_pid_and_random_need_no_artifacts(self):
        with tempfile.TemporaryDirectory() as empty, _clean_env(**{artifacts.ARTIFACT_DIR_ENV: empty}):
            for policy in ("pid", "random"):
                args = self.mod.parse_args(["--policy", policy])
                self.assertFalse(args.needs_ppo)

    def test_ppo_without_artifacts_fails_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as empty, _clean_env(**{artifacts.ARTIFACT_DIR_ENV: empty}):
            for argv in (["--policy", "ppo"], []):  # [] == default policy 'all'
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
                    self.mod.parse_args(argv)
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn(artifacts.ARTIFACT_DIR_ENV, err.getvalue())

    def test_explicit_paths_are_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            m, v = pathlib.Path(tmp) / "m.zip", pathlib.Path(tmp) / "v.pkl"
            m.write_bytes(b"x")
            v.write_bytes(b"x")
            args = self.mod.parse_args(["--policy", "ppo", "--model", str(m), "--vecnorm", str(v)])
            self.assertEqual((args.model, args.vecnorm), (str(m), str(v)))

    def test_defaults_match_documented_evaluation(self):
        args = self.mod.parse_args(["--policy", "pid"])
        self.assertEqual((args.n_eval, args.seed_start), (200, 5000))

    def test_pid_only_run_works_without_any_artifact(self):
        with tempfile.TemporaryDirectory() as empty, _clean_env(**{artifacts.ARTIFACT_DIR_ENV: empty}):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.mod.main(["--policy", "pid", "--n-eval", "2"])
        text = out.getvalue()
        self.assertIn("PID only", text)
        self.assertIn("Success rate:", text)
        self.assertNotIn("model        :", text)


@unittest.skipUnless(_SB3_OK, "stable_baselines3 unavailable")
class RobustnessCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("eval_robustness_paper.py")

    def test_defaults_match_documented_evaluation(self):
        with _clean_env():
            args = self.mod.parse_args([])
        self.assertEqual((args.n_eval, args.seed_start), (200, 7000))
        self.assertEqual(args.policy, "all")
        self.assertEqual(args.cases, ["A_nominal", "B_delay", "C_target", "D_wind", "E_mixed"])

    def test_vecnormalize_alias_still_accepted(self):
        a = self.mod.parse_args(["--vecnorm", "x.pkl"])
        b = self.mod.parse_args(["--vecnormalize", "x.pkl"])
        self.assertEqual(a.vecnormalize, "x.pkl")
        self.assertEqual(b.vecnormalize, "x.pkl")

    def test_ppo_mode_without_artifacts_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as empty, tempfile.TemporaryDirectory() as out:
            with _clean_env(**{artifacts.ARTIFACT_DIR_ENV: empty}):
                with self.assertRaises(FileNotFoundError) as ctx:
                    self.mod.main(["--n-eval", "1", "--cases", "A_nominal", "--output-dir", out])
        self.assertIn(artifacts.ARTIFACT_DIR_ENV, str(ctx.exception))

    def test_pid_only_run_needs_no_artifacts_and_writes_no_pairing(self):
        with tempfile.TemporaryDirectory() as empty, tempfile.TemporaryDirectory() as out:
            with _clean_env(**{artifacts.ARTIFACT_DIR_ENV: empty}), contextlib.redirect_stdout(io.StringIO()):
                self.mod.main(["--policy", "pid", "--n-eval", "2", "--cases", "A_nominal", "E_mixed", "--output-dir", out])
            files = sorted(p.name for p in pathlib.Path(out).iterdir())
            self.assertIn("robustness_summary.csv", files)
            self.assertIn("robustness_episodes.csv", files)
            self.assertNotIn("robustness_paired_pid_vs_ppo.csv", files)
            with open(pathlib.Path(out) / "robustness_summary.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([(r["case"], r["controller"]) for r in rows], [("A_nominal", "PID"), ("E_mixed", "PID")])
            self.assertEqual({r["seed_start"] for r in rows}, {"7000"})


if __name__ == "__main__":
    unittest.main()
