import unittest

import numpy as np

from landing_mujoco.envs.mujoco_landing_env import MujocoLandingEnv
from landing_mujoco.tests._helpers import load_reference_params
from landing_rl.envs.landing_env import LandingConfig


class ResetDeterminismTest(unittest.TestCase):
    def _run(self, seed, actions, deterministic_physics):
        env = MujocoLandingEnv(
            load_reference_params(), config=LandingConfig(),
            deterministic_physics=deterministic_physics,
        )
        obs, _ = env.reset(seed=seed)
        trace = [obs.copy()]
        for a in actions:
            obs, reward, term, trunc, info = env.step(a)
            trace.append(obs.copy())
            if term or trunc:
                break
        return trace

    def test_same_seed_same_actions_reproduce_identical_trajectory(self):
        rng = np.random.default_rng(42)
        actions = [rng.uniform(-1, 1, size=3).astype(np.float32) for _ in range(80)]

        trace_a = self._run(seed=7, actions=actions, deterministic_physics=False)
        trace_b = self._run(seed=7, actions=actions, deterministic_physics=False)

        self.assertEqual(len(trace_a), len(trace_b))
        for obs_a, obs_b in zip(trace_a, trace_b):
            np.testing.assert_array_equal(obs_a, obs_b)

    def test_deterministic_physics_mode_is_also_reproducible(self):
        actions = [np.zeros(3, dtype=np.float32) for _ in range(60)]
        trace_a = self._run(seed=3, actions=actions, deterministic_physics=True)
        trace_b = self._run(seed=3, actions=actions, deterministic_physics=True)
        for obs_a, obs_b in zip(trace_a, trace_b):
            np.testing.assert_array_equal(obs_a, obs_b)

    def test_different_seed_diverges(self):
        actions = [np.zeros(3, dtype=np.float32) for _ in range(30)]
        trace_a = self._run(seed=1, actions=actions, deterministic_physics=False)
        trace_b = self._run(seed=2, actions=actions, deterministic_physics=False)
        # At least the final observation should differ given different
        # noise/target/dropout/delay draws.
        self.assertFalse(np.array_equal(trace_a[-1], trace_b[-1]))


if __name__ == "__main__":
    unittest.main()
