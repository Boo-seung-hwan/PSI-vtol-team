import unittest

import numpy as np

from landing_mujoco.envs.mujoco_landing_env import MujocoLandingEnv
from landing_mujoco.tests._helpers import load_reference_params
from landing_rl.envs.landing_env import LandingConfig

VALID_QUALITIES = {"none", "soft", "hard", "rough", "bounce"}


class GroundContactTest(unittest.TestCase):
    def test_descent_eventually_makes_ground_contact_with_valid_classification(self):
        cfg = LandingConfig(init_altitude_min_m=1.0, init_altitude_max_m=1.0)
        env = MujocoLandingEnv(load_reference_params(), config=cfg, deterministic_physics=True)
        env.reset(seed=0)

        saw_contact = False
        for _ in range(cfg.max_steps):
            obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            self.assertIn(info["touchdown_quality"], VALID_QUALITIES)
            if info["ground_contact"]:
                saw_contact = True
            if term or trunc:
                break

        self.assertTrue(saw_contact, "vehicle never reached the ground within max_steps")
        self.assertGreaterEqual(info["contact_count"], 1)

    def test_soft_contact_triggers_motor_cutoff(self):
        cfg = LandingConfig(init_altitude_min_m=1.0, init_altitude_max_m=1.0)
        env = MujocoLandingEnv(load_reference_params(), config=cfg, deterministic_physics=True)
        env.reset(seed=0)

        motor_cutoff_seen = False
        for _ in range(cfg.max_steps):
            obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            if info["motor_cutoff"]:
                motor_cutoff_seen = True
                break
            if term or trunc:
                break

        self.assertTrue(motor_cutoff_seen, "expected a soft touchdown to latch motor_cutoff")

    def test_number_of_contacts_nonnegative_and_consistent(self):
        cfg = LandingConfig(init_altitude_min_m=1.0, init_altitude_max_m=1.0)
        env = MujocoLandingEnv(load_reference_params(), config=cfg, deterministic_physics=True)
        env.reset(seed=0)
        for _ in range(cfg.max_steps):
            obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            self.assertGreaterEqual(info["number_of_contacts"], 0)
            if info["ground_contact"]:
                self.assertGreater(info["number_of_contacts"], 0)
            if term or trunc:
                break


if __name__ == "__main__":
    unittest.main()
