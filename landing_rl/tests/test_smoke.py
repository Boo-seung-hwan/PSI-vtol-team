"""The quickstart smoke check (python -m landing_rl.evaluation.smoke) stays green."""

import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from landing_rl.evaluation.smoke import run_smoke  # noqa: E402


class SmokeTest(unittest.TestCase):
    def test_smoke_runs_and_is_deterministic(self):
        first = run_smoke(seed=0, n_steps=10)
        second = run_smoke(seed=0, n_steps=10)
        self.assertEqual(first, second)
        self.assertEqual(first["obs_dim"], 16)
        self.assertEqual(first["action_dim"], 3)
        self.assertGreater(first["steps"], 0)

    def test_smoke_other_seed(self):
        self.assertGreater(run_smoke(seed=7, n_steps=5)["steps"], 0)


if __name__ == "__main__":
    unittest.main()
