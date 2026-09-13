"""Visual ON vs visual OFF must be physically identical.

Structural: exact equality of mass, inertial reference, inertia tensor, DoF
structure, joints, actuators, simulation options, and every physics-model
geom (including the landing-gear contact geoms).

Trajectories (rtol=0, atol=1e-12): free fall, hover, known body torque,
landing/contact, and deterministic (plus seeded stochastic) command
trajectories through the full Gymnasium environment.

Every scenario starts from ``mujoco.mj_resetData`` on BOTH models, so no
solver warm-start or clock history is shared between scenarios.
"""

import unittest
from dataclasses import replace

import mujoco
import numpy as np

from landing_mujoco.configs.simulation_config import SimulationConfig
from landing_mujoco.configs.visualization_config import REPO_ROOT, load_visualization_config
from landing_mujoco.dynamics.mujoco_dynamics import MuJoCoDynamics
from landing_mujoco.envs.mujoco_landing_env import MujocoLandingEnv
from landing_mujoco.tests._helpers import deterministic_control_cfg, fresh_state, load_reference_params
from landing_mujoco.visualization.x500_shell import PREFIX
from landing_rl.envs.landing_env import LandingConfig

X500_YAML = REPO_ROOT / "landing_mujoco" / "configs" / "x500_visualization.yaml"
TIGHT = dict(rtol=0.0, atol=1e-12)


def assert_same(testcase, a, b, what):
    np.testing.assert_allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                               err_msg=what, **TIGHT)


def physics_geom_ids(model):
    return [i for i in range(model.ngeom)
            if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(PREFIX)]


class StructuralInvarianceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = LandingConfig()
        sim = SimulationConfig(control_dt=cfg.dt)
        params = load_reference_params()
        cls.off = MuJoCoDynamics(params, cfg, sim).model
        cls.on = MuJoCoDynamics(params, cfg, sim, visualization=load_visualization_config(X500_YAML)).model

    def _eq(self, attr):
        a, b = getattr(self.off, attr), getattr(self.on, attr)
        self.assertTrue(np.array_equal(a, b), f"{attr}: {a} != {b}")

    def test_mass_inertia_and_inertial_reference(self):
        for attr in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_subtreemass",
                     "body_pos", "body_quat", "body_parentid", "body_rootid", "body_weldid"):
            self._eq(attr)

    def test_dof_joint_and_actuator_structure(self):
        for attr in ("nbody", "njnt", "nq", "nv", "nu", "na", "neq", "ntendon", "nsensor"):
            self.assertEqual(getattr(self.off, attr), getattr(self.on, attr), attr)
        for attr in ("jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid", "qpos0",
                     "dof_bodyid", "dof_jntid", "dof_parentid", "dof_armature", "dof_damping",
                     "dof_frictionloss", "dof_M0", "body_jntnum", "body_dofnum",
                     "actuator_trntype", "actuator_gear", "actuator_trnid", "actuator_gainprm", "actuator_biasprm"):
            self._eq(attr)

    def test_simulation_options(self):
        for attr in ("timestep", "gravity", "integrator", "cone", "solver", "iterations", "tolerance",
                     "wind", "density", "viscosity", "o_margin", "disableflags", "enableflags"):
            a, b = getattr(self.off.opt, attr), getattr(self.on.opt, attr)
            self.assertTrue(np.array_equal(a, b), attr)
        self.assertEqual(self.off.stat.meaninertia, self.on.stat.meaninertia)

    def test_physics_model_geoms_identical(self):
        ids_off, ids_on = physics_geom_ids(self.off), physics_geom_ids(self.on)
        names = lambda m, ids: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in ids]
        self.assertEqual(names(self.off, ids_off), names(self.on, ids_on))
        self.assertEqual(ids_off, ids_on)
        for attr in ("geom_type", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority",
                     "geom_bodyid", "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref",
                     "geom_solimp", "geom_solmix", "geom_margin", "geom_gap", "geom_rgba"):
            a, b = getattr(self.off, attr)[ids_off], getattr(self.on, attr)[ids_on]
            self.assertTrue(np.array_equal(a, b), attr)
        # geom_group (viewer visibility only): identical for every collision-enabled geom;
        # only the collision-inert placeholder markers move to a hidden group with the shell.
        for i in ids_off:
            name = mujoco.mj_id2name(self.off, mujoco.mjtObj.mjOBJ_GEOM, i)
            if self.off.geom_contype[i] or self.off.geom_conaffinity[i]:
                self.assertEqual(self.off.geom_group[i], self.on.geom_group[i], name)
            elif self.off.geom_group[i] != self.on.geom_group[i]:
                self.assertTrue(name == "body_box" or name.startswith(("arm_", "motor_")), name)
        # collision-enabled geoms: exactly the ground + the physical landing gear, in both
        collide = lambda m: sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)
                                   if m.geom_contype[i] or m.geom_conaffinity[i])
        self.assertEqual(collide(self.off), collide(self.on))
        self.assertEqual(collide(self.on), ["ground", "leg_0", "leg_1", "leg_2", "leg_3"])


class TrajectoryInvarianceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        params = load_reference_params()
        vis = load_visualization_config(X500_YAML)
        cls.cfg = deterministic_control_cfg()
        sim = SimulationConfig(control_dt=cls.cfg.dt)
        cls.dyn_off = MuJoCoDynamics(params, cls.cfg, sim)
        cls.dyn_on = MuJoCoDynamics(params, cls.cfg, sim, visualization=vis)
        cls.nodrag_off = MuJoCoDynamics(params, replace(cls.cfg, linear_drag_xy=0.0, linear_drag_z=0.0), sim)
        cls.nodrag_on = MuJoCoDynamics(params, replace(cls.cfg, linear_drag_xy=0.0, linear_drag_z=0.0), sim,
                                       visualization=vis)
        cls.params, cls.vis = params, vis

    def _run_dynamics(self, dyn, pos_ned, n_steps, dt, v_cmd, motor_cutoff=False, thrust_accel=None):
        mujoco.mj_resetData(dyn.model, dyn.data)
        dyn.reset(np.asarray(pos_ned, float), np.zeros(3), np.zeros(3))
        if thrust_accel is not None:
            dyn.actuation.thrust_accel = thrust_accel
        state = fresh_state(pos=pos_ned)
        rng = np.random.default_rng(7)
        rows = []
        cutoff = motor_cutoff
        for _ in range(n_steps):
            result = dyn.step(state, v_cmd=np.asarray(v_cmd, float), dt=dt, rng=rng, wind_accel_ned=np.zeros(3),
                              target_yaw=0.0, motor_cutoff=cutoff, ground_contact_prev=dyn.contact_state.ground_contact)
            if not motor_cutoff:
                cutoff = dyn.contact_state.motor_cutoff
            rows.append(np.concatenate([dyn.data.qpos, dyn.data.qvel, dyn.data.qacc, state.accel, state.attitude,
                                        state.body_rates, [state.thrust_accel, float(dyn.data.ncon),
                                                           float(dyn.contact_state.ground_contact),
                                                           float(dyn.contact_state.contact_count),
                                                           float(dyn.contact_state.motor_cutoff),
                                                           float(result.last_impact_vz)]]))
        return np.array(rows)

    def _pair(self, off, on, **kw):
        a, b = self._run_dynamics(off, **kw), self._run_dynamics(on, **kw)
        self.assertEqual(a.shape, b.shape)
        assert_same(self, a, b, f"trajectory mismatch for {kw}")
        return a, b

    def test_free_fall(self):
        a, _ = self._pair(self.nodrag_off, self.nodrag_on, pos_ned=[0, 0, -200.0], n_steps=120, dt=0.01,
                          v_cmd=[0, 0, 0], motor_cutoff=True, thrust_accel=0.0)
        self.assertLess(a[-1, 7 + 2], -5.0)  # qvel[2]: MuJoCo world z-velocity (Z-up) -> falling fast

    def test_hover(self):
        self._pair(self.dyn_off, self.dyn_on, pos_ned=[0, 0, -3.0], n_steps=100, dt=0.05, v_cmd=[0, 0, 0])

    def test_known_body_torque(self):
        def run(dyn):
            mujoco.mj_resetData(dyn.model, dyn.data)
            dyn.reset(np.array([0.0, 0.0, -10.0]), np.zeros(3), np.zeros(3))
            rows = []
            for k in range(400):
                dyn.data.xfrc_applied[dyn.body_id, :3] = 0.0
                dyn.data.xfrc_applied[dyn.body_id, 3:] = [0.05, -0.03, 0.02] if k < 200 else [0.0, 0.0, 0.0]
                mujoco.mj_step(dyn.model, dyn.data)
                rows.append(np.concatenate([dyn.data.qpos, dyn.data.qvel, dyn.data.qacc]))
            dyn.data.xfrc_applied[:] = 0.0
            return np.array(rows)
        a, b = run(self.dyn_off), run(self.dyn_on)
        assert_same(self, a, b, "known body torque")
        self.assertGreater(float(np.max(np.abs(a[:, 10:13]))), 1e-2)  # it actually rotated

    def test_landing_and_contact(self):
        a, _ = self._pair(self.dyn_off, self.dyn_on, pos_ned=[0.3, -0.2, -1.2], n_steps=260, dt=0.05,
                          v_cmd=[-0.1, 0.05, 0.35])
        ncon_col = 7 + 6 + 6 + 3 + 3 + 3 + 1
        self.assertGreater(float(np.max(a[:, ncon_col])), 0.0, "scenario never made ground contact")

    def _run_env(self, visualization, deterministic, seed, n_steps=420):
        cfg = LandingConfig(init_altitude_min_m=1.5, init_altitude_max_m=1.5)
        env = MujocoLandingEnv(self.params, config=cfg, deterministic_physics=deterministic,
                               visualization=visualization)
        mujoco.mj_resetData(env.mujoco_dynamics.model, env.mujoco_dynamics.data)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(1234)
        rows, flags, labels = [obs.astype(np.float64)], [], []
        for _ in range(n_steps):
            action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            d = env.mujoco_dynamics.data
            rows.append(np.concatenate([
                obs.astype(np.float64), [reward], d.qpos, d.qvel, info["pos"], info["vel"], info["accel"],
                info["body_rates"], [info["thrust_accel"], info["xy_error"], info["z_error"], info["altitude_agl"],
                                     info["tilt"], info["contact_count"], info["bounce_count"],
                                     info["last_impact_vz"], info["contact_force"], info["number_of_contacts"]],
                info["v_cmd"], info["applied_body_force"], info["applied_body_torque"]]))
            flags.append((term, trunc, info["success"], info["failed"], info["ground_contact"], info["motor_cutoff"]))
            labels.append((info["failure_reason"], info["touchdown_quality"], info["target_mode"]))
            if term or trunc:
                break
        return rows, flags, labels

    def _compare_env(self, deterministic, seed):
        ra, fa, la = self._run_env(None, deterministic, seed)
        rb, fb, lb = self._run_env(self.vis, deterministic, seed)
        self.assertEqual(len(ra), len(rb))
        self.assertEqual(fa, fb)
        self.assertEqual(la, lb)
        for k, (x, y) in enumerate(zip(ra, rb)):
            assert_same(self, x, y, f"env step {k}")
        return fa

    def test_deterministic_command_trajectory(self):
        flags = self._compare_env(deterministic=True, seed=0)
        self.assertTrue(any(f[4] for f in flags), "trajectory never reached the ground")

    def test_seeded_stochastic_command_trajectory(self):
        self._compare_env(deterministic=False, seed=3)


if __name__ == "__main__":
    unittest.main()
