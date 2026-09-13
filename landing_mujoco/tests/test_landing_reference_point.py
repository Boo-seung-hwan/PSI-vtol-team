"""Landing-gear standoff semantics: pad_target_ned vs.
vehicle_touchdown_target_ned (see MUJOCO_MODEL.md and the module docstring
of ``landing_mujoco.envs.mujoco_landing_env``).

Covers, at minimum, the cases the user specified:

  1. a level vehicle resting normally on its landing gear has
     ground_contact=True, and at that pose z_error ~= 0.
  2. changing landing-gear length changes the required CG/body resting
     height but NOT the semantic success tolerance (success_altitude_m).
  3. altitude_agl approaches zero at physical contact independently of
     z_error (they are different concepts, demonstrated by a tilted
     vehicle sitting exactly at the touchdown-target z).
  4. the OLD legacy environment's zero-standoff behavior is unchanged.
  5. no observation dimension/order change.
"""

import copy
import unittest

import numpy as np

from landing_mujoco.configs.param_schema import landing_reference_point_body_m
from landing_mujoco.envs.mujoco_landing_env import MujocoLandingEnv
from landing_mujoco.tests._helpers import load_reference_params
from landing_rl.envs.landing_env import LandingConfig
from landing_rl.envs.landing_env import LandingEnv as OldLandingEnv


def run_zero_action_descent(env, cfg, n_steps=None):
    env.reset(seed=0)
    n_steps = n_steps or cfg.max_steps
    info = None
    for _ in range(n_steps):
        obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if term or trunc:
            break
    return info


class LevelTouchdownTest(unittest.TestCase):
    def test_level_touchdown_gives_zero_z_error_and_ground_contact(self):
        params = load_reference_params()
        cfg = LandingConfig(init_altitude_min_m=1.0, init_altitude_max_m=1.0)
        env = MujocoLandingEnv(params, config=cfg, deterministic_physics=True)
        info = run_zero_action_descent(env, cfg)

        self.assertTrue(info["ground_contact"])
        self.assertLess(info["z_error"], cfg.success_altitude_m)
        self.assertTrue(info["success"], "clean level touchdown should satisfy kinematic + contact success")
        self.assertIn(info["touchdown_quality"], {"soft"})

    def test_vehicle_touchdown_target_is_shifted_from_pad_by_gear_standoff(self):
        params = load_reference_params()
        cfg = LandingConfig()
        env = MujocoLandingEnv(params, config=cfg, deterministic_physics=True)
        env.reset(seed=0)

        pad = env.target_true.copy()
        touchdown_target = env._vehicle_touchdown_target(pad)
        # r_landing_body is [0,0,0.12] for the reference config (FRD, down
        # positive) -> the vehicle-reference target must sit ABOVE the pad
        # in NED (more negative z) by that standoff.
        np.testing.assert_allclose(touchdown_target[:2], pad[:2], atol=1e-9)
        self.assertAlmostEqual(touchdown_target[2], pad[2] - env.r_landing_body[2], places=9)
        self.assertLess(touchdown_target[2], pad[2])


class GearLengthChangeTest(unittest.TestCase):
    """2. Changing gear length changes the required resting height, not the
    success tolerance."""

    def _params_with_gear_clearance(self, clearance_m: float):
        params = copy.deepcopy(load_reference_params())
        pts = params.geometry.landing_gear_points_body_m.copy()
        pts[:, 2] = clearance_m
        params.geometry.landing_gear_points_body_m = pts
        params.geometry.ground_clearance_m = clearance_m
        return params

    def test_success_altitude_m_is_unchanged_across_gear_lengths(self):
        cfg = LandingConfig(init_altitude_min_m=1.0, init_altitude_max_m=1.0)

        params_short = self._params_with_gear_clearance(0.08)
        params_long = self._params_with_gear_clearance(0.24)

        r_short = landing_reference_point_body_m(params_short.geometry)
        r_long = landing_reference_point_body_m(params_long.geometry)
        self.assertNotAlmostEqual(float(r_short[2]), float(r_long[2]))

        env_short = MujocoLandingEnv(params_short, config=cfg, deterministic_physics=True)
        env_long = MujocoLandingEnv(params_long, config=cfg, deterministic_physics=True)

        info_short = run_zero_action_descent(env_short, cfg)
        info_long = run_zero_action_descent(env_long, cfg)

        # Both must succeed against the SAME literal success_altitude_m --
        # the tolerance itself never changed.
        self.assertEqual(cfg.success_altitude_m, 0.05)
        self.assertLess(info_short["z_error"], cfg.success_altitude_m)
        self.assertLess(info_long["z_error"], cfg.success_altitude_m)
        self.assertTrue(info_short["success"])
        self.assertTrue(info_long["success"])

        # But the required CG resting altitude genuinely differs by
        # (approximately) the gear-length difference.
        resting_altitude_short = -float(info_short["pos"][2])
        resting_altitude_long = -float(info_long["pos"][2])
        self.assertAlmostEqual(
            resting_altitude_long - resting_altitude_short,
            0.24 - 0.08,
            delta=0.03,
        )


class AltitudeAglVsZErrorTest(unittest.TestCase):
    """3. altitude_agl and z_error are different concepts: a tilted vehicle
    sitting exactly at the touchdown-target z still has nonzero
    altitude_agl, even though z_error ~= 0."""

    def test_tilt_at_target_z_gives_zero_z_error_but_nonzero_altitude_agl(self):
        params = load_reference_params()
        cfg = LandingConfig()
        env = MujocoLandingEnv(params, config=cfg, deterministic_physics=True)
        env.reset(seed=0)

        touchdown_target = env._vehicle_touchdown_target(env.target_true)
        st = env._vehicle_state
        st.pos[:] = touchdown_target  # CG exactly at the desired touchdown position
        # A large roll tilt: rotating r_landing_body=[0,0,h] about the roll
        # (X) axis moves it mostly into Y, and its Z-projection change is
        # h*(1-cos(roll)) -- a small-angle tilt barely moves the Z
        # projection (second order), so use a large, clearly-tilted pose to
        # get an unambiguous signal.
        st.attitude[:] = [1.0, 0.0, 0.0]  # ~57 degrees roll -- not level

        z_error = abs(float((touchdown_target - st.pos)[2]))
        self.assertAlmostEqual(z_error, 0.0, places=9)

        altitude_agl = env._altitude_agl()
        self.assertGreater(altitude_agl, 0.03, "tilted vehicle at target z should not read zero clearance")

    def test_altitude_agl_zero_at_level_rest(self):
        params = load_reference_params()
        cfg = LandingConfig()
        env = MujocoLandingEnv(params, config=cfg, deterministic_physics=True)
        env.reset(seed=0)

        touchdown_target = env._vehicle_touchdown_target(env.target_true)
        st = env._vehicle_state
        st.pos[:] = touchdown_target
        st.attitude[:] = [0.0, 0.0, 0.0]
        self.assertAlmostEqual(env._altitude_agl(), 0.0, places=9)


class LegacyUnchangedTest(unittest.TestCase):
    """4. The old legacy environment's zero-standoff behavior remains
    unchanged (CG snaps directly to ground_z_m on contact)."""

    def test_old_env_cg_reaches_ground_z_on_contact(self):
        cfg = LandingConfig(init_altitude_min_m=1.0, init_altitude_max_m=1.0)
        env = OldLandingEnv(config=cfg)
        info = run_zero_action_descent(env, cfg)
        self.assertTrue(info["ground_contact"])
        # Legacy: CG position IS the ground plane on contact (zero standoff).
        self.assertAlmostEqual(float(info["pos"][2]), cfg.ground_z_m, places=6)
        self.assertLess(info["z_error"], cfg.success_altitude_m)


class ObservationContractUnchangedTest(unittest.TestCase):
    """5. No observation dimension/order change."""

    def test_observation_shape_and_action_shape_unchanged(self):
        params = load_reference_params()
        env = MujocoLandingEnv(params, config=LandingConfig(), deterministic_physics=True)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (16,))
        self.assertEqual(env.action_space.shape, (3,))

    def test_observation_error_channel_uses_touchdown_target_not_pad(self):
        params = load_reference_params()
        env = MujocoLandingEnv(params, config=LandingConfig(), deterministic_physics=True)
        obs, _ = env.reset(seed=0)
        st = env._vehicle_state
        expected_error = env._vehicle_touchdown_target(env.obs_target) - st.pos
        np.testing.assert_allclose(obs[0:3], expected_error, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
