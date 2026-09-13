import unittest

import numpy as np

from landing_mujoco.envs.mujoco_landing_env import MujocoLandingEnv
from landing_mujoco.tests._helpers import load_reference_params
from landing_rl.envs.landing_env import LandingConfig


def make_env(deterministic=True):
    return MujocoLandingEnv(
        load_reference_params(), config=LandingConfig(), deterministic_physics=deterministic
    )


class ObservationContractTest(unittest.TestCase):
    def test_observation_space_shape(self):
        env = make_env()
        self.assertEqual(env.observation_space.shape, (16,))

    def test_reset_observation_matches_contract(self):
        env = make_env()
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (16,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertFalse(np.isnan(obs).any())
        self.assertFalse(np.isinf(obs).any())

    def test_no_nan_or_inf_over_full_episode(self):
        env = make_env()
        obs, _ = env.reset(seed=1)
        rng = np.random.default_rng(2)
        for _ in range(150):
            action = rng.uniform(-1, 1, size=3).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            self.assertFalse(np.isnan(obs).any())
            self.assertFalse(np.isinf(obs).any())
            self.assertFalse(np.isnan(reward))
            self.assertFalse(np.isinf(reward))
            if term or trunc:
                break

    def test_target_valid_is_binary(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        self.assertIn(obs[-1], (0.0, 1.0))


class ActionContractTest(unittest.TestCase):
    def test_action_space_shape_and_bounds(self):
        env = make_env()
        self.assertEqual(env.action_space.shape, (3,))
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0, 1.0])

    def test_out_of_range_action_is_clipped_not_rejected(self):
        env = make_env()
        env.reset(seed=0)
        obs, reward, term, trunc, info = env.step(np.array([5.0, -5.0, 5.0], dtype=np.float32))
        np.testing.assert_array_equal(info["raw_action"], [1.0, -1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
