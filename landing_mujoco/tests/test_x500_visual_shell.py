"""x500 visual shell: native reconstruction, layer separation, OFF-mode MJCF
identity, and visual-only geom properties."""

import hashlib
import itertools
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from landing_mujoco.configs.simulation_config import SimulationConfig
from landing_mujoco.configs.visualization_config import REPO_ROOT, load_visualization_config
from landing_mujoco.dynamics.mjcf_builder import PLACEHOLDER_MARKER_GROUP_WITH_SHELL
from landing_mujoco.dynamics.mujoco_dynamics import MuJoCoDynamics
from landing_mujoco.tests._helpers import load_reference_params
from landing_mujoco.visualization.x500_shell import PREFIX, VISUAL_GEOM_GROUP, load_native_assembly
from landing_rl.envs.landing_env import LandingConfig

X500_YAML = REPO_ROOT / "landing_mujoco" / "configs" / "x500_visualization.yaml"

# sha256 of the physics-only MJCF generated for tarot680b_reference.yaml with
# LandingConfig() defaults, recorded immediately BEFORE the visual shell was
# integrated. Pins that visual-OFF output is byte-for-byte unchanged. Update
# only for a deliberate, separately reviewed change to physical MJCF output.
PRE_VISUAL_PHYSICS_MJCF_SHA256 = "e2c5cb7af8399785660ae3bd9509e90f42b30ec093c87a0182d580f98aa2399f"


def make_dyn(vis=None):
    cfg = LandingConfig()
    return MuJoCoDynamics(load_reference_params(), cfg, SimulationConfig(control_dt=cfg.dt), visualization=vis)


def visual_geom_ids(model):
    return [i for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(PREFIX)]


def mesh_frame_pose_in_body(dyn, gid):
    """Undo MuJoCo's per-mesh recentering (mesh_pos/mesh_quat) so the pose of
    the mesh's own coordinate frame can be compared with the layered math."""
    m, d = dyn.model, dyn.data
    rb, pb = d.xmat[dyn.body_id].reshape(3, 3), d.xpos[dyn.body_id]
    rg, pg = rb.T @ d.geom_xmat[gid].reshape(3, 3), rb.T @ (d.geom_xpos[gid] - pb)
    mid = m.geom_dataid[gid]
    rm = np.zeros(9)
    mujoco.mju_quat2Mat(rm, m.mesh_quat[mid])
    r_mesh = rg @ rm.reshape(3, 3).T
    return pg - r_mesh @ m.mesh_pos[mid], r_mesh


def body_frame_vertices(dyn, gid):
    m, d = dyn.model, dyn.data
    mid = m.geom_dataid[gid]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    world = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
    return (world - d.xpos[dyn.body_id]) @ d.xmat[dyn.body_id].reshape(3, 3)


class VisualOffIdentityTest(unittest.TestCase):
    def test_no_visualization_mjcf_is_byte_identical_to_pre_integration(self):
        self.assertEqual(hashlib.sha256(make_dyn().xml.encode()).hexdigest(), PRE_VISUAL_PHYSICS_MJCF_SHA256)

    def test_disabled_config_is_identical_to_no_config(self):
        dyn = make_dyn(replace(load_visualization_config(X500_YAML), enabled=False))
        self.assertIsNone(dyn.visual_shell)
        self.assertEqual(hashlib.sha256(dyn.xml.encode()).hexdigest(), PRE_VISUAL_PHYSICS_MJCF_SHA256)


class VisualShellStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_visualization_config(X500_YAML)
        cls.off = make_dyn()
        cls.on = make_dyn(cls.cfg)
        cls.native = make_dyn(replace(cls.cfg, uniform_scale=1.0))
        for dyn in (cls.off, cls.on, cls.native):
            dyn.reset(np.array([0.0, 0.0, -3.0]), np.zeros(3), np.zeros(3))

    def test_visual_geoms_are_inert_and_grouped(self):
        ids = visual_geom_ids(self.on.model)
        expected = sum(len(i.meshes) for i in self.on.visual_shell.assembly.instances)
        self.assertEqual(len(ids), expected)
        self.assertEqual(expected, 22)  # 6 frame upper-structure + 4 motor bases + 4 props + 4x2 bell components
        for gid in ids:
            self.assertEqual(self.on.model.geom_contype[gid], 0)
            self.assertEqual(self.on.model.geom_conaffinity[gid], 0)
            self.assertEqual(self.on.model.geom_group[gid], VISUAL_GEOM_GROUP)
        self.assertEqual(visual_geom_ids(self.off.model), [])

    def test_split_shell_shows_physical_landing_gear(self):
        """x500 landing gear is not in the shell; the physical contact geoms
        stay in the default-visible group; placeholder markers are hidden."""
        names = lambda m: {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(m.ngeom)}
        on, off = names(self.on.model), names(self.off.model)
        self.assertFalse(any("landing" in n for n in on if n.startswith(PREFIX)))
        for n in ("ground", "leg_0", "leg_1", "leg_2", "leg_3"):
            self.assertEqual(self.on.model.geom_group[on[n]], 0, n)
            self.assertEqual(self.off.model.geom_group[off[n]], 0, n)
        markers = [n for n in off if n == "body_box" or n.startswith(("arm_", "motor_"))]
        self.assertEqual(len(markers), 9)
        for n in markers:
            self.assertEqual(self.off.model.geom_group[off[n]], 0, n)
            self.assertEqual(self.on.model.geom_group[on[n]], PLACEHOLDER_MARKER_GROUP_WITH_SHELL, n)
        # nothing in the shell reaches down to the physical contact level
        lowest_shell = min(body_frame_vertices(self.on, g)[:, 2].min() for g in visual_geom_ids(self.on.model))
        legs = [on[f"leg_{k}"] for k in range(4)]
        contact_bottom = min(self.on.model.geom_pos[g][2] - self.on.model.geom_size[g][0] for g in legs)
        self.assertGreater(lowest_shell, contact_bottom + 0.05)

    def test_no_bodies_joints_or_actuators_added(self):
        for attr in ("nbody", "njnt", "nq", "nv", "nu", "na", "nsensor", "neq"):
            self.assertEqual(getattr(self.on.model, attr), getattr(self.off.model, attr), attr)

    def test_mujoco_frames_match_python_layer_composition(self):
        shell = self.on.visual_shell
        for inst in shell.assembly.instances:
            pos, rot, mesh_scale = shell.geom_pose_in_body(inst)
            for mesh in inst.meshes:
                gid = mujoco.mj_name2id(self.on.model, mujoco.mjtObj.mjOBJ_GEOM, f"{PREFIX}{inst.name}__{mesh}")
                p_mesh, r_mesh = mesh_frame_pose_in_body(self.on, gid)
                np.testing.assert_allclose(p_mesh, pos, rtol=0, atol=1e-12)
                np.testing.assert_allclose(r_mesh, rot, rtol=0, atol=1e-12)
                mid = self.on.model.geom_dataid[gid]
                np.testing.assert_allclose(self.on.model.mesh_scale[mid], mesh_scale, rtol=0, atol=1e-12)
                np.testing.assert_allclose(mesh_scale, shell.alignment.uniform_scale
                                           * np.asarray(inst.mesh_scale_xyz), rtol=0, atol=1e-15)

    def test_native_reconstruction_matches_source_sdf(self):
        """At uniform_scale=1 the shell reproduces the x500 SDF placement."""
        dyn, shell = self.native, self.native.visual_shell
        by_name = {i.name: i for i in shell.assembly.instances}
        rotors = shell.assembly.rotor_link_positions
        # motor bases: SDF poses (+-0.174, +-0.174, 0.032), yaw -0.45
        for k, (x, y) in enumerate([(0.174, 0.174), (-0.174, 0.174), (0.174, -0.174), (-0.174, -0.174)]):
            inst = by_name[f"5010_motor_base_{k}"]
            gid = mujoco.mj_name2id(dyn.model, mujoco.mjtObj.mjOBJ_GEOM, f"{PREFIX}{inst.name}__{inst.meshes[0]}")
            p, r = mesh_frame_pose_in_body(dyn, gid)
            np.testing.assert_allclose(p, [x, y, 0.032], atol=1e-12)
            self.assertAlmostEqual(float(np.arctan2(r[1, 0], r[0, 0])), -0.45, places=12)
        # frame: z +0.025, yaw pi
        inst = by_name["base_link_visual"]
        p, r = mesh_frame_pose_in_body(
            dyn, mujoco.mj_name2id(dyn.model, mujoco.mjtObj.mjOBJ_GEOM, f"{PREFIX}{inst.name}__{inst.meshes[0]}"))
        np.testing.assert_allclose(p, [0, 0, 0.025], atol=1e-12)
        np.testing.assert_allclose(r, np.diag([-1.0, -1.0, 1.0]), atol=1e-9)
        # props: SDF mesh scale 0.84615..., hub centered on the rotor axis; bells on the axis
        for k in range(4):
            link = f"rotor_{k}"
            prop = by_name[f"rotor_{k}_visual"]
            self.assertAlmostEqual(prop.mesh_scale_xyz[0], 0.8461538461538461, places=15)
            gid = mujoco.mj_name2id(dyn.model, mujoco.mjtObj.mjOBJ_GEOM, f"{PREFIX}{prop.name}__{prop.meshes[0]}")
            v = body_frame_vertices(dyn, gid)
            center = (v.min(0) + v.max(0)) / 2
            np.testing.assert_allclose(center[:2], rotors[link][:2], atol=1e-3)
            span = float(np.max(np.ptp(v[:, :2], axis=0)))
            self.assertAlmostEqual(span, 0.346 * 0.8461538461538461, delta=1e-3)
            bell = [i for i in shell.assembly.instances if i.link == link and "motor_" in i.name][0]
            gid = mujoco.mj_name2id(dyn.model, mujoco.mjtObj.mjOBJ_GEOM, f"{PREFIX}{bell.name}__{bell.meshes[0]}")
            p, _ = mesh_frame_pose_in_body(dyn, gid)
            np.testing.assert_allclose(p, np.asarray(rotors[link]) + [0, 0, -0.032], atol=1e-12)

    def test_display_layer_is_uniform_scale_of_native(self):
        """Every visual vertex: p_display = t + R (s * p_native), for a
        non-trivial translation/rotation, i.e. the whole assembly (meshes AND
        component offsets) is scaled consistently."""
        cfg = replace(self.cfg, translation_body_m=(0.03, -0.02, -0.15), rotation_rpy_rad=(0.02, -0.03, 0.4))
        disp = make_dyn(cfg)
        disp.reset(np.array([0.0, 0.0, -3.0]), np.zeros(3), np.zeros(3))
        a = disp.visual_shell.alignment
        for gid_native, gid_disp in zip(visual_geom_ids(self.native.model), visual_geom_ids(disp.model)):
            self.assertEqual(mujoco.mj_id2name(self.native.model, mujoco.mjtObj.mjOBJ_GEOM, gid_native),
                             mujoco.mj_id2name(disp.model, mujoco.mjtObj.mjOBJ_GEOM, gid_disp))
            vn = body_frame_vertices(self.native, gid_native)
            vd = body_frame_vertices(disp, gid_disp)
            expected = a.translation_mujoco_body_m + a.uniform_scale * vn @ a.rotation_mujoco_body.T
            np.testing.assert_allclose(vd, expected, rtol=0, atol=2e-6)

    def test_scaled_wheelbase_equals_target(self):
        shell = self.on.visual_shell
        s = shell.alignment.uniform_scale
        hubs = [s * np.asarray(p[:2]) for p in shell.assembly.rotor_link_positions.values()]
        wb = max(np.linalg.norm(a - b) for a, b in itertools.combinations(hubs, 2))
        self.assertAlmostEqual(wb, 0.737, places=12)

    def test_manifest_physics_keys_are_rejected(self):
        src = load_visualization_config(X500_YAML).assets_dir
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "bundle"
            shutil.copytree(src, dst)
            manifest = json.loads((dst / "x500_native_assembly.json").read_text())
            manifest["visual_instances"][0]["mass"] = 2.0
            (dst / "x500_native_assembly.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_native_assembly(dst)


if __name__ == "__main__":
    unittest.main()
