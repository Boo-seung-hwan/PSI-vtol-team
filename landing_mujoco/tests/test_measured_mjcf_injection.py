"""Measured UGRP mass / CG / inertia / motor + skid-gear geometry -> the COMPILED
MuJoCo model.

Scope: these tests prove the measured physical parameters actually reach
``mjModel`` -- body mass, principal inertia, inertial frame, motor marker
positions and the four landing-gear contact spheres -- and that MuJoCo did not
silently rewrite them at compile time.

The measured config is still missing its propulsion / closed-loop-response
fields (max collective thrust, tau_*, actuator delay). ``MuJoCoDynamics`` /
``MujocoLandingEnv`` therefore cannot be built from it, and no value is invented
here to change that. What CAN be built -- and is tested -- is the rigid-body
MJCF:

    load_uav_params(validate=False) -> build_mjcf -> MjModel

so the physics sanity checks below use NO propulsion model: force/torque-level
wrenches go straight through ``xfrc_applied``, and the landed / tilted contact
tests only let the vehicle fall under gravity onto the skid spheres.
"""

import math
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import yaml

from dataclasses import replace

from landing_mujoco.configs.param_schema import landing_reference_point_body_m, load_uav_params
from landing_mujoco.configs.visualization_config import REPO_ROOT, load_visualization_config
from landing_mujoco.coordinates.transforms import (
    REFLECT,
    frd_vector_to_mujoco_body,
    mujoco_body_vector_to_frd,
    mujoco_quaternion_to_policy_euler,
    policy_euler_to_mujoco_quaternion,
    rotation_body_to_world_ned,
)
from landing_mujoco.dynamics.mjcf_builder import (
    LEG_CONTACT_RADIUS_M,
    build_mjcf,
    landing_gear_sphere_centers_body_m,
)
from landing_mujoco.visualization.x500_shell import PREFIX, build_x500_visual_shell
from landing_rl.envs.landing_env import LandingConfig

MEASURED_YAML = REPO_ROOT / "landing_mujoco" / "configs" / "ugrp_vehicle_measured.yaml"
X500_YAML = REPO_ROOT / "landing_mujoco" / "configs" / "x500_visualization.yaml"

PHYSICS_DT = 0.002
MOTORS = ("FL", "FR", "RL", "RR")

# Canonical measured values (task spec sections 2, 4, 5), typed independently
# of the YAML so a hand-edit to the YAML cannot pass by agreeing with itself.
MASS_KG = 6.408
IXX, IYY, IZZ = 0.153184, 0.126285, 0.149050
RADIUS_M = 0.360
MOTOR_Z_FRD = -0.093  # motors ABOVE the CG; FRD +Z is down

# Landed-pose contact geometry (2026-09-19). Typed independently of the YAML.
# Three DIFFERENT quantities -- never interchange them:
GROUND_CLEARANCE_M = 0.241     # PHYSICAL: CG -> ground plane (stored in the YAML)
SPHERE_RADIUS_M = 0.020        # SIMULATION: contact-sphere radius (lives in the builder)
SPHERE_CENTRE_OFFSET_M = 0.221 # DERIVED: 0.241 - 0.020 (never stored)


def _params():
    return load_uav_params(MEASURED_YAML, validate=False)


def _compile(visual: bool = False):
    params = _params()
    shell = None
    if visual:
        shell = build_x500_visual_shell(load_visualization_config(X500_YAML), params)
        assert shell is not None, "x500 visual shell unexpectedly disabled"
    xml = build_mjcf(params, physics_dt=PHYSICS_DT, visual_shell=shell)
    model = mujoco.MjModel.from_xml_string(xml)
    return params, xml, model


def _geom_pos(model, name):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"geom {name!r} not in compiled model"
    return np.array(model.geom_pos[gid], dtype=np.float64)


class CompiledInertialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params, cls.xml, cls.model = _compile()
        cls.body_id = mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")

    def test_body_mass_equals_measured_total_mass(self):
        self.assertEqual(float(self.model.body_mass[self.body_id]), MASS_KG)
        # ... and equals what the parser produced from the YAML.
        self.assertEqual(
            float(self.model.body_mass[self.body_id]), self.params.mass_properties.mass_kg
        )

    def test_mass_is_counted_exactly_once(self):
        # World body carries no mass; the vehicle body carries all of it; the
        # subtree and total agree. A double-count (component or geom-derived
        # mass on top of the measured total) would push these above 6.408.
        self.assertEqual(float(self.model.body_mass[0]), 0.0)
        self.assertEqual(int(np.count_nonzero(self.model.body_mass)), 1)
        self.assertAlmostEqual(float(np.sum(self.model.body_mass)), MASS_KG, places=12)
        self.assertAlmostEqual(float(self.model.body_subtreemass[0]), MASS_KG, places=12)
        self.assertAlmostEqual(float(mujoco.mj_getTotalmass(self.model)), MASS_KG, places=12)

    def test_principal_inertia_equals_measured_values_exactly(self):
        got = np.array(self.model.body_inertia[self.body_id], dtype=np.float64)
        np.testing.assert_array_equal(got, [IXX, IYY, IZZ])

    def test_compiled_values_equal_the_source_yaml_not_just_the_parser(self):
        # Read the YAML directly, bypassing param_schema, so a parser/builder
        # regression cannot hide by agreeing with itself.
        raw = yaml.safe_load(MEASURED_YAML.read_text())["mass_properties"]
        inertia = raw["inertia_kgm2"]
        got = self.model.body_inertia[self.body_id]
        np.testing.assert_array_equal(got, [inertia["ixx"], inertia["iyy"], inertia["izz"]])
        self.assertEqual(float(self.model.body_mass[self.body_id]), raw["mass_kg"])

    def test_inertial_frame_is_the_cg_and_unrotated(self):
        # "MuJoCo did not modify the inertia" for a diagonal tensor means the
        # principal axes are still the body axes (identity quaternion) and the
        # COM is still the body origin. If the compiler ever re-diagonalised
        # or reordered axes, Ixx would silently stop being the roll inertia.
        np.testing.assert_array_equal(self.model.body_ipos[self.body_id], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(self.model.body_iquat[self.body_id], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(
            np.asarray(self.params.mass_properties.cg_body_m), [0.0, 0.0, 0.0]
        )

    def test_compiler_applies_no_inertia_rewriting(self):
        # Defaults are relied on, so assert them rather than trusting them:
        # balanceinertia / boundmass / boundinertia would silently alter the
        # measured tensor, and inertiafromgeom would recompute it from geoms.
        compiler = ET.fromstring(self.xml).find("compiler")
        self.assertIsNotNone(compiler)
        self.assertEqual(set(compiler.attrib), {"angle"})

    def test_exactly_one_explicit_inertial_element(self):
        root = ET.fromstring(self.xml)
        inertials = list(root.iter("inertial"))
        self.assertEqual(len(inertials), 1)
        self.assertEqual(set(inertials[0].attrib), {"pos", "mass", "diaginertia"})

    def test_diagonal_tensor_is_invariant_under_frd_to_flu(self):
        # The builder writes Ixx/Iyy/Izz with NO frame conversion. That is
        # correct only because the tensor is diagonal: R I R^T == I for
        # R = diag(1, -1, -1). (Products of inertia would flip sign.)
        i_frd = np.diag([IXX, IYY, IZZ])
        np.testing.assert_array_equal(REFLECT @ i_frd @ REFLECT.T, i_frd)
        i_with_products = np.array([[IXX, 0.01, 0.02], [0.01, IYY, 0.03], [0.02, 0.03, IZZ]])
        converted = REFLECT @ i_with_products @ REFLECT.T
        self.assertEqual(converted[0, 1], -0.01)
        self.assertEqual(converted[0, 2], -0.02)
        self.assertEqual(converted[1, 2], +0.03)

    def test_visual_shell_adds_no_mass_or_inertia(self):
        _, _, shell_model = _compile(visual=True)
        # The shell must add geoms (otherwise this test proves nothing) ...
        n_shell = sum(
            (mujoco.mj_id2name(shell_model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(PREFIX)
            for i in range(shell_model.ngeom)
        )
        self.assertGreater(n_shell, 0)
        # ... and yet leave every inertial quantity bit-identical.
        for attr in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_subtreemass"):
            np.testing.assert_array_equal(
                getattr(self.model, attr), getattr(shell_model, attr), err_msg=attr
            )
        self.assertEqual(float(shell_model.body_mass[self.body_id]), MASS_KG)


class CompiledMotorGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params, cls.xml, cls.model = _compile()

    def _expected_frd(self):
        a = RADIUS_M / math.sqrt(2.0)
        return {
            "FL": np.array([+a, -a, MOTOR_Z_FRD]),
            "FR": np.array([+a, +a, MOTOR_Z_FRD]),
            "RL": np.array([-a, -a, MOTOR_Z_FRD]),
            "RR": np.array([-a, +a, MOTOR_Z_FRD]),
        }

    def test_yaml_positions_match_the_task_spec_values(self):
        # a = 0.360 / sqrt(2) = 0.25455844...; typed here from the task spec.
        for name, expected in self._expected_frd().items():
            np.testing.assert_allclose(
                self.params.motors.positions_body_m[name], expected, atol=1e-12, err_msg=name
            )
        self.assertAlmostEqual(RADIUS_M / math.sqrt(2.0), 0.25455844, places=8)

    def test_compiled_motor_geoms_reproduce_the_frd_positions_exactly(self):
        for name in MOTORS:
            mj = _geom_pos(self.model, f"motor_{name}")
            # Bit-exact against the schema's FRD->FLU conversion ...
            np.testing.assert_array_equal(
                mj, frd_vector_to_mujoco_body(self.params.motors.positions_body_m[name]), name
            )
            # ... and round-tripping back to FRD recovers the YAML literals.
            np.testing.assert_array_equal(
                mujoco_body_vector_to_frd(mj), self.params.motors.positions_body_m[name], name
            )

    def test_compiled_motor_radial_distance_is_360mm_in_both_frames(self):
        for name in MOTORS:
            mj = _geom_pos(self.model, f"motor_{name}")
            self.assertAlmostEqual(math.hypot(mj[0], mj[1]), RADIUS_M, places=12, msg=name)
            frd = mujoco_body_vector_to_frd(mj)
            self.assertAlmostEqual(math.hypot(frd[0], frd[1]), RADIUS_M, places=12, msg=name)

    def test_motor_plane_is_above_the_cg(self):
        for name in MOTORS:
            mj = _geom_pos(self.model, f"motor_{name}")
            # MuJoCo local frame is FLU (+z up): above the CG is positive.
            self.assertAlmostEqual(mj[2], +0.093, places=12, msg=name)
            self.assertAlmostEqual(mujoco_body_vector_to_frd(mj)[2], MOTOR_Z_FRD, places=12)

    def test_frame_is_an_exact_symmetric_45_degree_x(self):
        pos = {n: np.asarray(self.params.motors.positions_body_m[n]) for n in MOTORS}
        for name, p in pos.items():
            self.assertAlmostEqual(abs(p[0]), abs(p[1]), places=15, msg=name)
            self.assertAlmostEqual(
                math.degrees(math.atan2(abs(p[1]), abs(p[0]))), 45.0, places=12, msg=name
            )
        # Point symmetry through the vertical axis through the CG.
        np.testing.assert_allclose(pos["FL"][:2], -pos["RR"][:2], atol=1e-15)
        np.testing.assert_allclose(pos["FR"][:2], -pos["RL"][:2], atol=1e-15)
        # FRD sign layout: +x forward, +y right => left is -y.
        self.assertGreater(pos["FL"][0], 0)
        self.assertLess(pos["FL"][1], 0)
        self.assertGreater(pos["RR"][1], 0)
        self.assertLess(pos["RR"][0], 0)

    def test_four_unique_motor_names_and_positions_in_the_model(self):
        names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            for i in range(self.model.ngeom)
        ]
        motor_names = [n for n in names if n and n.startswith("motor_")]
        self.assertEqual(sorted(motor_names), [f"motor_{n}" for n in sorted(MOTORS)])
        self.assertEqual(len(set(motor_names)), 4)
        positions = {tuple(_geom_pos(self.model, n)) for n in motor_names}
        self.assertEqual(len(positions), 4)

    def test_arm_markers_run_from_the_cg_to_each_motor(self):
        # The capsule's geom_pos is the midpoint of its fromto segment, so it
        # must be exactly half of the motor position.
        for name in MOTORS:
            np.testing.assert_allclose(
                _geom_pos(self.model, f"arm_{name}"),
                _geom_pos(self.model, f"motor_{name}") / 2.0,
                atol=1e-15,
                err_msg=name,
            )

    def test_motor_and_arm_markers_carry_no_mass_and_no_collision(self):
        for name in MOTORS:
            for prefix in ("arm_", "motor_"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}{name}")
                self.assertEqual(int(self.model.geom_contype[gid]), 0)
                self.assertEqual(int(self.model.geom_conaffinity[gid]), 0)


class MarkerCoordinateTextTest(unittest.TestCase):
    """``_coord`` must keep legacy MJCFs byte-identical (the reference-config
    hash pin in test_x500_visual_shell depends on it) yet never round a
    measured coordinate to 0.1 mm."""

    def test_lossless_short_form_is_kept_verbatim(self):
        from landing_mujoco.dynamics.mjcf_builder import _coord

        self.assertEqual(_coord(0.2606), "0.2606")
        self.assertEqual(_coord(-0.2606), "-0.2606")
        self.assertEqual(_coord(0.0), "0.0000")

    def test_lossy_short_form_falls_back_to_exact_repr(self):
        from landing_mujoco.dynamics.mjcf_builder import _coord

        a = RADIUS_M / math.sqrt(2.0)
        self.assertNotEqual(_coord(a), f"{a:.4f}")
        self.assertEqual(float(_coord(a)), a)
        self.assertEqual(float(_coord(-0.093)), -0.093)


# Skid-type landing gear, canonical footprint (2026-09-19). Typed independently
# of the YAML so a hand-edit cannot pass by agreeing with itself. FRD, +Y right
# (left is negative y). Order: left-front, left-rear, right-front, right-rear ==
# geoms leg_0 .. leg_3. PHYSICAL contact points DERIVED from the confirmed skid
# length 0.300 m / spacing 0.310 m under a centred-symmetry ASSUMPTION.
SKID_POINTS_FRD = np.array(
    [
        [+0.150, -0.155, +0.241],
        [-0.150, -0.155, +0.241],
        [+0.150, +0.155, +0.241],
        [-0.150, +0.155, +0.241],
    ]
)

# SYNTHETIC 3-leg layout -- TEST DATA ONLY (not a vehicle parameter). Used solely
# to characterise the solver's contact sinking against a different leg count.
SYNTHETIC_TRIANGLE_XY = [
    (0.25 * math.cos(a), 0.25 * math.sin(a))
    for a in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
]

# Ground friction exactly as a landing run applies it: MuJoCoDynamics passes
# control_cfg.ground_friction_xy to build_mjcf. The builder's own default (0.9)
# is NOT what an environment run uses, and the settled contact sinking depends
# on it (pyramidal-cone regularisation), so tests use the environment's value.
ENV_GROUND_FRICTION = LandingConfig().ground_friction_xy


def _params_with_points(points, semantics="physical_contact"):
    """Measured params with ``points`` (FRD, physical-contact reading by
    default) substituted for the YAML's gear points."""
    params = _params()
    return replace(
        params,
        geometry=replace(
            params.geometry,
            landing_gear_points_body_m=np.asarray(points, dtype=np.float64),
            landing_gear_points_semantics=semantics,
        ),
    )


def _params_without_gear():
    params = _params()
    return replace(params, geometry=replace(params.geometry, landing_gear_points_body_m=None))


def _compile_params(params, physics_dt=PHYSICS_DT, ground_friction_xy=ENV_GROUND_FRICTION):
    return mujoco.MjModel.from_xml_string(
        build_mjcf(params, physics_dt=physics_dt, ground_friction_xy=ground_friction_xy)
    )


def _leg_ids(model):
    return [
        i for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("leg_")
    ]


class LandingGearCoordinatePrecisionTest(unittest.TestCase):
    """The collision geoms must carry the gear coordinates at full precision --
    the touchdown reference reads the Python array, not the MJCF. The point
    below is SYNTHETIC test data (not a vehicle parameter): it only needs to be
    non-representable at 4 decimals."""

    SYNTHETIC = np.array([[0.123456789, -0.234567891, 0.345678912]])

    def test_geom_center_points_are_not_rounded_to_0p1_mm(self):
        model = _compile_params(_params_with_points(self.SYNTHETIC, semantics="geom_center"))
        np.testing.assert_array_equal(
            _geom_pos(model, "leg_0"), frd_vector_to_mujoco_body(self.SYNTHETIC[0])
        )

    def test_derived_sphere_centres_are_not_rounded_to_0p1_mm(self):
        model = _compile_params(_params_with_points(self.SYNTHETIC, semantics="physical_contact"))
        expected_frd = self.SYNTHETIC[0] - np.array([0.0, 0.0, LEG_CONTACT_RADIUS_M])
        np.testing.assert_array_equal(
            _geom_pos(model, "leg_0"), frd_vector_to_mujoco_body(expected_frd)
        )

    def test_legacy_reference_leg_text_is_unchanged(self):
        # The reference config's gear points are 4-dp exact, so the legacy
        # text (and the reference-MJCF hash pin) must be preserved.
        from landing_mujoco.tests._helpers import load_reference_params

        xml = build_mjcf(load_reference_params(), physics_dt=PHYSICS_DT)
        self.assertIn('pos="0.2606 -0.2606 -0.1200"', xml)


class MeasuredSkidCompiledGeometryTest(unittest.TestCase):
    """The measured skid footprint in the COMPILED mjModel.

    Physical gear: two continuous parallel skid bars. MuJoCo v0: four
    representative contact spheres at the skid endpoints. FRD physical contact
    z = +0.241 -> FRD sphere centre z = +0.221 -> MuJoCo (FLU, z up) centre
    z = -0.221. All frame conversions use the repository helper.
    """

    @classmethod
    def setUpClass(cls):
        cls.params = _params()
        cls.model = _compile_params(cls.params)
        cls.model_no_gear = _compile_params(_params_without_gear())
        cls.legs = _leg_ids(cls.model)

    def test_four_contact_geoms_named_leg_0_to_3(self):
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in self.legs]
        self.assertEqual(names, ["leg_0", "leg_1", "leg_2", "leg_3"])
        self.assertEqual(self.model.ngeom - self.model_no_gear.ngeom, 4)

    def test_geoms_are_collision_enabled_spheres_of_radius_0p020(self):
        for gid in self.legs:
            self.assertEqual(int(self.model.geom_type[gid]), int(mujoco.mjtGeom.mjGEOM_SPHERE))
            self.assertEqual(float(self.model.geom_size[gid][0]), SPHERE_RADIUS_M)
            self.assertEqual(int(self.model.geom_contype[gid]), 1)
            self.assertEqual(int(self.model.geom_conaffinity[gid]), 1)

    def test_compiled_centres_are_the_helper_converted_physical_points_minus_the_radius(self):
        for i, point in enumerate(SKID_POINTS_FRD):
            centre_frd = point - np.array([0.0, 0.0, SPHERE_RADIUS_M])
            np.testing.assert_array_equal(
                _geom_pos(self.model, f"leg_{i}"), frd_vector_to_mujoco_body(centre_frd), f"leg_{i}"
            )

    def test_frd_to_flu_reading_of_the_centres(self):
        # FLU (+x fwd, +y LEFT, +z up): the left skid is +y, everything is below
        # the CG (z < 0), and z = -(0.241 - 0.020) = -0.221.
        expected_flu = {
            "leg_0": (+0.150, +0.155, -0.221),  # left-front
            "leg_1": (-0.150, +0.155, -0.221),  # left-rear
            "leg_2": (+0.150, -0.155, -0.221),  # right-front
            "leg_3": (-0.150, -0.155, -0.221),  # right-rear
        }
        for name, want in expected_flu.items():
            np.testing.assert_allclose(_geom_pos(self.model, name), want, atol=1e-12, err_msg=name)

    def test_frd_round_trip_recovers_the_derived_centre_not_the_physical_point(self):
        for i, point in enumerate(SKID_POINTS_FRD):
            centre = mujoco_body_vector_to_frd(_geom_pos(self.model, f"leg_{i}"))
            np.testing.assert_allclose(centre[:2], point[:2], atol=1e-15)
            self.assertAlmostEqual(float(centre[2]), SPHERE_CENTRE_OFFSET_M, places=12)  # 0.221
            self.assertAlmostEqual(float(centre[2]) + SPHERE_RADIUS_M, GROUND_CLEARANCE_M, places=12)

    def test_sphere_bottoms_lie_on_the_physical_ground_plane_in_the_body_frame(self):
        for gid in self.legs:
            bottom_flu_z = float(self.model.geom_pos[gid][2]) - float(self.model.geom_size[gid][0])
            self.assertAlmostEqual(bottom_flu_z, -GROUND_CLEARANCE_M, places=12)

    def test_compiled_footprint_matches_the_skid_dimensions(self):
        pos = np.array([_geom_pos(self.model, f"leg_{i}") for i in range(4)])
        self.assertAlmostEqual(float(pos[:, 0].max() - pos[:, 0].min()), 0.300, places=12)  # skid length
        self.assertAlmostEqual(float(pos[:, 1].max() - pos[:, 1].min()), 0.310, places=12)  # spacing
        self.assertEqual(float(pos[:, 0].mean()), 0.0)
        self.assertEqual(float(pos[:, 1].mean()), 0.0)

    def test_only_the_legs_and_the_ground_collide(self):
        contact = sorted(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            for i in range(self.model.ngeom)
            if self.model.geom_contype[i] != 0
        )
        self.assertEqual(contact, ["ground", "leg_0", "leg_1", "leg_2", "leg_3"])

    def test_adding_the_gear_changes_no_inertial_or_motor_geometry(self):
        for attr in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_subtreemass"):
            np.testing.assert_array_equal(
                getattr(self.model, attr), getattr(self.model_no_gear, attr), err_msg=attr
            )
        for name in [f"{p}{m}" for p in ("arm_", "motor_") for m in MOTORS]:
            np.testing.assert_array_equal(
                _geom_pos(self.model, name), _geom_pos(self.model_no_gear, name), err_msg=name
            )
        # the gear geoms are massless: total mass is still the measured 6.408 kg
        self.assertAlmostEqual(float(mujoco.mj_getTotalmass(self.model)), MASS_KG, places=12)


class PhysicalVsSimulationGearTest(unittest.TestCase):
    """Physical contact geometry and its simulation representation are kept
    apart: the config stores the PHYSICAL 0.241 m; the builder owns the 0.020 m
    sphere radius and derives the 0.221 m sphere centre."""

    def test_builder_owns_the_sphere_radius(self):
        self.assertEqual(LEG_CONTACT_RADIUS_M, SPHERE_RADIUS_M)

    def test_measured_config_stores_the_physical_value_not_the_sphere_centre(self):
        g = _params().geometry
        self.assertEqual(g.ground_clearance_m, GROUND_CLEARANCE_M)
        self.assertNotEqual(g.ground_clearance_m, SPHERE_CENTRE_OFFSET_M)
        self.assertEqual(g.landing_gear_points_semantics, "physical_contact")
        np.testing.assert_array_equal(g.landing_gear_points_body_m[:, 2], np.full(4, GROUND_CLEARANCE_M))

    def test_building_the_mjcf_does_not_mutate_the_physical_points(self):
        params = _params()
        before = params.geometry.landing_gear_points_body_m.copy()
        _compile_params(params)
        landing_gear_sphere_centers_body_m(params.geometry)
        np.testing.assert_array_equal(params.geometry.landing_gear_points_body_m, before)
        self.assertTrue(np.all(params.geometry.landing_gear_points_body_m[:, 2] == GROUND_CLEARANCE_M))

    def test_legacy_geom_center_reading_is_unchanged(self):
        params = _params_with_points(SKID_POINTS_FRD, semantics="geom_center")
        centers = landing_gear_sphere_centers_body_m(params.geometry)
        np.testing.assert_array_equal(centers, params.geometry.landing_gear_points_body_m)

    def test_reference_config_legs_are_unchanged(self):
        from landing_mujoco.tests._helpers import load_reference_params

        ref = load_reference_params()
        self.assertEqual(ref.geometry.landing_gear_points_semantics, "geom_center")
        model = _compile_params(ref)
        for i, point in enumerate(ref.geometry.landing_gear_points_body_m):
            np.testing.assert_array_equal(
                _geom_pos(model, f"leg_{i}"), frd_vector_to_mujoco_body(point)
            )

    def test_no_contact_geoms_are_invented_when_there_are_no_points(self):
        # Without per-leg points the builder must still create NO vehicle
        # contact geom, even though ground_clearance_m = 0.241 is known.
        params = _params_without_gear()
        self.assertIsNone(landing_gear_sphere_centers_body_m(params.geometry))
        model = _compile_params(params)
        contact = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            for i in range(model.ngeom)
            if model.geom_contype[i] != 0
        ]
        self.assertEqual(contact, ["ground"])

    def test_reference_point_is_the_physical_contact_point_not_the_sphere_centre(self):
        r = landing_reference_point_body_m(_params().geometry)
        np.testing.assert_allclose(r, [0.0, 0.0, GROUND_CLEARANCE_M], atol=1e-12)


class LandedPoseSettleTest(unittest.TestCase):
    """Dynamic proof, with the CANONICAL measured skid footprint, that the
    landed pose is physically right:

        ground plane at world z = 0, gravity on, measured mass / inertia,
        vehicle released level from rest above the ground, settle, then

            CG world z            ~= 0.241 m   (physical ground clearance)
            sphere centre world z ~= 0.020 m   (radius above the ground)
            lowest sphere bottom  ~= 0

    TOLERANCE BASIS (measured, not assumed). MuJoCo contacts are soft: the
    settled sphere sinks slightly into the plane. Measured on this model (RK4,
    default solref/solimp, environment ground friction 0.55): 0.0215 mm with the
    4 canonical legs (15.7 N per contact), independent of drop height and
    physics_dt. The sinking is linear in load per contact at fixed friction
    (n_legs x penetration constant) but grows with the friction coefficient
    (0.0054 / 0.0215 / 0.0795 / 0.184 mm at mu = 0.3 / 0.55 / 0.9 / 1.2, from
    MuJoCo's pyramidal-cone regularisation). Even the worst conceivable case
    here -- one contact carrying the whole 62.8 N weight at mu = 1.2 -- sinks
    ~0.74 mm. TOL_M = 2 mm is >2x that, and ~10x below the 20 mm error this
    test exists to catch (0.241 wrongly read as a sphere centre rests the CG at
    ~0.261 m). No contact parameter is tuned.
    """

    TOL_M = 2.0e-3
    DROP_HEIGHT_M = 0.5
    SETTLE_SECONDS = 3.0  # measured settling time is ~0.5 s

    @staticmethod
    def _settle(params, *, physics_dt=PHYSICS_DT, z0=DROP_HEIGHT_M, seconds=SETTLE_SECONDS,
                ground_friction_xy=ENV_GROUND_FRICTION):
        model = _compile_params(params, physics_dt, ground_friction_xy)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, z0]      # released above the ground ...
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # ... level ...
        data.qvel[:] = 0.0                    # ... from rest, no thrust
        mujoco.mj_forward(model, data)
        for _ in range(int(round(seconds / physics_dt))):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        return model, data

    @staticmethod
    def _bottoms(model, data):
        return [
            float(data.geom_xpos[g][2]) - float(model.geom_size[g][0]) for g in _leg_ids(model)
        ]

    def _assert_landed(self, model, data, *, label):
        # settled: at rest, level, no drift, all legs in contact
        np.testing.assert_allclose(data.qvel, np.zeros(6), atol=1e-6, err_msg=label)
        np.testing.assert_allclose(data.qpos[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6, err_msg=label)
        np.testing.assert_allclose(data.qpos[0:2], [0.0, 0.0], atol=1e-6, err_msg=label)
        legs = _leg_ids(model)
        self.assertEqual(int(data.ncon), len(legs), label)

        cg_z = float(data.qpos[2])
        self.assertAlmostEqual(cg_z, GROUND_CLEARANCE_M, delta=self.TOL_M, msg=f"{label}: CG world z")
        for gid, bottom_z in zip(legs, self._bottoms(model, data)):
            centre_z = float(data.geom_xpos[gid][2])
            self.assertAlmostEqual(centre_z, SPHERE_RADIUS_M, delta=self.TOL_M, msg=f"{label}: sphere centre z")
            self.assertAlmostEqual(bottom_z, 0.0, delta=self.TOL_M, msg=f"{label}: sphere bottom z")
            # a soft contact can only sink INTO the plane, and only slightly
            self.assertLessEqual(bottom_z, 1e-9, f"{label}: sphere bottom above the ground")
            self.assertGreater(bottom_z, -5e-4, f"{label}: penetration larger than expected")
            self.assertAlmostEqual(cg_z - centre_z, SPHERE_CENTRE_OFFSET_M, delta=self.TOL_M, msg=label)

    def test_scene_setup_ground_plane_at_world_zero_with_gravity(self):
        model = _compile_params(_params())
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.assertEqual(int(model.geom_type[gid]), int(mujoco.mjtGeom.mjGEOM_PLANE))
        self.assertEqual(float(model.geom_pos[gid][2]), 0.0)
        self.assertEqual(int(model.geom_contype[gid]), 1)
        self.assertAlmostEqual(float(model.opt.gravity[2]), -9.80665, places=12)
        self.assertGreater(self.DROP_HEIGHT_M, GROUND_CLEARANCE_M)  # genuinely dropped

    def test_landed_cg_height_equals_the_physical_ground_clearance(self):
        model, data = self._settle(_params())
        self._assert_landed(model, data, label="canonical skid footprint")

    def test_landing_reference_point_reaches_the_ground_at_rest(self):
        # r_landing^B = mean physical contact point = [0, 0, 0.241]. Its world
        # height (level, FLU up) is cg_z - r_z, which is exactly what
        # MujocoLandingEnv._altitude_agl measures: it must be ~0 when landed.
        params = _params()
        _, data = self._settle(params)
        r = landing_reference_point_body_m(params.geometry)
        np.testing.assert_allclose(r[2], GROUND_CLEARANCE_M, atol=1e-12)
        self.assertAlmostEqual(float(data.qpos[2]) - float(r[2]), 0.0, delta=self.TOL_M)

    def test_landed_height_is_independent_of_drop_height_and_physics_dt(self):
        reference = None
        for physics_dt, z0 in ((0.002, 0.5), (0.002, 0.3), (0.001, 0.5), (0.005, 0.5)):
            with self.subTest(physics_dt=physics_dt, drop=z0):
                model, data = self._settle(_params(), physics_dt=physics_dt, z0=z0)
                self._assert_landed(model, data, label=f"dt={physics_dt}, z0={z0}")
                reference = float(data.qpos[2]) if reference is None else reference
                self.assertAlmostEqual(float(data.qpos[2]), reference, delta=1e-5)

    def test_measured_mass_and_inertia_are_the_ones_landing(self):
        # The settle test must run on the measured rigid body, not a default one.
        model = _compile_params(_params())
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")
        self.assertEqual(float(model.body_mass[body]), MASS_KG)
        np.testing.assert_array_equal(model.body_inertia[body], [IXX, IYY, IZZ])

    def test_negative_control_sphere_centre_at_the_physical_depth_rests_too_high(self):
        # The trap the physical/simulation separation removes: reading 0.241 as
        # the sphere CENTRE (legacy "geom_center", same canonical points) puts
        # the contact surface one radius lower, so the CG rests ~0.020 m high.
        params = _params_with_points(SKID_POINTS_FRD, semantics="geom_center")
        _, data = self._settle(params)
        cg_z = float(data.qpos[2])
        self.assertAlmostEqual(cg_z, GROUND_CLEARANCE_M + SPHERE_RADIUS_M, delta=self.TOL_M)
        self.assertGreater(cg_z - GROUND_CLEARANCE_M, 0.015)

    def test_tolerance_basis_penetration_is_linear_in_contact_load(self):
        pen = {}
        cases = {"canonical": _params(),
                 "triangle": _params_with_points([[x, y, GROUND_CLEARANCE_M] for x, y in SYNTHETIC_TRIANGLE_XY])}
        for name, params in cases.items():
            model, data = self._settle(params)
            bottoms = self._bottoms(model, data)
            pen[len(bottoms)] = -float(np.mean(bottoms))
        # n_legs x penetration ~ constant (linear soft contact at fixed mu) ...
        c4, c3 = 4 * pen[4], 3 * pen[3]
        self.assertAlmostEqual(c4 / c3, 1.0, delta=0.05, msg=str(pen))
        # ... so one contact carrying the whole weight sinks ~c mm, far below TOL_M.
        self.assertLess(max(c4, c3), self.TOL_M / 10.0, msg=str(pen))

    def test_tolerance_basis_penetration_grows_with_friction_but_stays_far_below_tolerance(self):
        worst = 0.0
        for mu in (0.3, ENV_GROUND_FRICTION, 0.9, 1.2):
            model, data = self._settle(_params(), ground_friction_xy=mu)
            pen = -float(np.mean(self._bottoms(model, data)))
            worst = max(worst, 4 * pen)       # single-contact-carries-all extrapolation
            self.assertGreater(pen, 0.0, f"mu={mu}")
            self.assertAlmostEqual(float(data.qpos[2]), GROUND_CLEARANCE_M, delta=self.TOL_M, msg=f"mu={mu}")
        self.assertLess(worst, self.TOL_M / 2.0)


class TiltedContactTest(unittest.TestCase):
    """Small roll / pitch perturbations on the canonical skid footprint.

    The vehicle is released at rest, tilted 5 degrees, with its lowest sphere
    bottom LIFT_M above the ground, and settles under gravity. Purpose: the
    CURRENT contact model must stay sane on the new geometry -- no NaN, no
    solver blow-up, the low-side skids touch first, the contact torque is
    restoring, nothing tunnels through the ground, and the vehicle settles level
    at the physical clearance. NO contact stiffness / friction is tuned.

    Attitude and convention come from the repository helpers
    (``policy_euler_to_mujoco_quaternion`` / ``rotation_body_to_world_ned``):
    FRD, positive roll = right side down, positive pitch = nose up.
    """

    TILT_DEG = 5.0
    LIFT_M = 0.02          # lowest sphere bottom above the ground at release
    SIM_SECONDS = 3.0
    TOL_M = 2.0e-3
    # Transient penetration is dynamic (impact ~0.63 m/s from LIFT_M), not the
    # ~0.02 mm resting sinking. Level control from the same lift: ~4.3 mm;
    # combined roll+pitch loads a single rear-right skid: ~7.8 mm (observed).
    # 12 mm keeps >= 8 mm margin below the 20 mm sphere radius (i.e. no tunnelling).
    PEAK_PENETRATION_MAX_M = 0.012
    CASES = ((+5.0, 0.0), (-5.0, 0.0), (0.0, +5.0), (0.0, -5.0), (+5.0, +5.0))

    @classmethod
    def setUpClass(cls):
        cls.params = _params()
        cls.model = _compile_params(cls.params)
        cls.legs = _leg_ids(cls.model)
        cls.body = mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")
        cls.centres_frd = landing_gear_sphere_centers_body_m(cls.params.geometry)

    # -- helpers ------------------------------------------------------------
    def _contact_torque_frd(self, data):
        """Net contact wrench ON the vehicle, about the CG: (torque in the FRD
        body frame, force in the world frame)."""
        torque_w = np.zeros(3)
        force_w = np.zeros(3)
        for i in range(data.ncon):
            c = data.contact[i]
            f6 = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, f6)
            f_world = np.asarray(c.frame).reshape(3, 3).T @ f6[:3]  # contact frame -> world
            if int(c.geom1) in self.legs:      # force acts on geom2; flip to get the force ON the leg
                f_world = -f_world
            torque_w += np.cross(np.asarray(c.pos) - data.qpos[0:3], f_world)
            force_w += f_world
        r_body_to_world = np.asarray(data.xmat[self.body]).reshape(3, 3)
        return mujoco_body_vector_to_frd(r_body_to_world.T @ torque_w), force_w

    def _predicted_lowest_legs(self, roll_deg, pitch_deg):
        """Legs that should touch first, from the repository's NED convention
        (an independent route from the MuJoCo quaternion)."""
        r = rotation_body_to_world_ned(math.radians(roll_deg), math.radians(pitch_deg), 0.0)
        z_ned = (r @ self.centres_frd.T)[2]        # NED z is DOWN: larger == lower
        return {i for i in range(len(z_ned)) if z_ned[i] > z_ned.max() - 1e-9}

    def _drop(self, roll_deg, pitch_deg):
        model = self.model
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = policy_euler_to_mujoco_quaternion(
            math.radians(roll_deg), math.radians(pitch_deg), 0.0
        )
        data.qpos[2] = 1.0
        mujoco.mj_forward(model, data)
        lowest_below_cg = min(data.geom_xpos[g][2] - 0.02 for g in self.legs) - 1.0
        data.qpos[2] = -lowest_below_cg + self.LIFT_M   # lowest bottom LIFT_M above ground
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        trace = {"finite": True, "min_bottom": np.inf, "max_rate": 0.0, "first": None}
        for step in range(int(round(self.SIM_SECONDS / PHYSICS_DT))):
            mujoco.mj_step(model, data)
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                trace["finite"] = False
                break
            trace["min_bottom"] = min(
                trace["min_bottom"], min(float(data.geom_xpos[g][2]) - 0.02 for g in self.legs)
            )
            trace["max_rate"] = max(trace["max_rate"], float(np.abs(data.qvel[3:6]).max()))
            if trace["first"] is None and data.ncon > 0:
                touching = set()
                for i in range(data.ncon):
                    c = data.contact[i]
                    geom = int(c.geom1) if int(c.geom1) in self.legs else int(c.geom2)
                    touching.add(self.legs.index(geom))
                tau, force = self._contact_torque_frd(data)
                trace["first"] = {"step": step, "legs": touching, "tau_frd": tau, "force_world": force}
        mujoco.mj_forward(model, data)
        return data, trace

    # -- tests --------------------------------------------------------------
    def test_no_nan_and_no_solver_blow_up(self):
        for roll, pitch in self.CASES:
            with self.subTest(roll=roll, pitch=pitch):
                data, trace = self._drop(roll, pitch)
                self.assertTrue(trace["finite"])
                self.assertLess(trace["max_rate"], 5.0)         # observed ~1.9 rad/s
                self.assertTrue(np.all(np.asarray([w.number for w in data.warning]) == 0))

    def test_the_low_side_skid_touches_first(self):
        for roll, pitch in self.CASES:
            with self.subTest(roll=roll, pitch=pitch):
                _, trace = self._drop(roll, pitch)
                self.assertIsNotNone(trace["first"], "the vehicle never touched the ground")
                self.assertEqual(trace["first"]["legs"], self._predicted_lowest_legs(roll, pitch))

    def test_first_contacts_match_the_frd_sign_convention(self):
        # positive roll = right side down -> right skid (FRD y > 0) first;
        # positive pitch = nose up -> rear (FRD x < 0) first.
        expect = {
            (+5.0, 0.0): {2, 3},   # right-front, right-rear
            (-5.0, 0.0): {0, 1},   # left-front, left-rear
            (0.0, +5.0): {1, 3},   # left-rear, right-rear
            (0.0, -5.0): {0, 2},   # left-front, right-front
            (+5.0, +5.0): {3},     # right-rear only
        }
        for (roll, pitch), legs in expect.items():
            with self.subTest(roll=roll, pitch=pitch):
                _, trace = self._drop(roll, pitch)
                self.assertEqual(trace["first"]["legs"], legs)

    def test_contact_torque_is_restoring_and_the_contact_supports_the_weight(self):
        for roll, pitch in self.CASES:
            with self.subTest(roll=roll, pitch=pitch):
                _, trace = self._drop(roll, pitch)
                tau = trace["first"]["tau_frd"]
                force = trace["first"]["force_world"]
                # sign convention of the force accounting: contact pushes UP (world +z)
                self.assertGreater(force[2], 0.0)
                if roll != 0.0:
                    self.assertLess(tau[0] * roll, 0.0, f"roll torque {tau[0]:+.3f} must oppose roll {roll:+.0f}")
                else:
                    self.assertLess(abs(tau[0]), 1e-6)
                if pitch != 0.0:
                    self.assertLess(tau[1] * pitch, 0.0, f"pitch torque {tau[1]:+.3f} must oppose pitch {pitch:+.0f}")
                else:
                    self.assertLess(abs(tau[1]), 1e-6)

    def test_the_vehicle_does_not_tunnel_at_this_release_height(self):
        # SCENARIO-SPECIFIC, not a physics invariant. The 12 mm bound is
        # calibrated to the 20 mm-lift release (impact ~0.63 m/s; observed peak
        # 4.3 mm single-axis, 7.8 mm roll+pitch). A faster touchdown legitimately
        # sinks more in the default soft contact (level drop, mu = 0.55: 9.4 mm at
        # 1.4 m/s, 16 mm at 2.2 m/s), so do not reuse this bound for other impact
        # speeds. The invariant that DOES hold generally is the second assertion:
        # the sphere centre must not cross the plane (peak < radius).
        for roll, pitch in self.CASES:
            with self.subTest(roll=roll, pitch=pitch):
                _, trace = self._drop(roll, pitch)
                peak = -trace["min_bottom"]
                self.assertLess(peak, self.PEAK_PENETRATION_MAX_M)
                self.assertLess(peak, SPHERE_RADIUS_M)   # sphere centre never crosses the plane

    def test_single_axis_tilt_impacts_like_the_level_control(self):
        # Same impact speed, same skid load on the first-contact side: the peak
        # sinking must be close to the level drop from the same lift.
        _, level = self._drop(0.0, 0.0)
        peak_level = -level["min_bottom"]
        for roll, pitch in self.CASES[:4]:
            with self.subTest(roll=roll, pitch=pitch):
                _, trace = self._drop(roll, pitch)
                self.assertAlmostEqual(-trace["min_bottom"], peak_level, delta=0.25 * peak_level)

    def test_the_vehicle_settles_level_at_the_physical_clearance(self):
        for roll, pitch in self.CASES:
            with self.subTest(roll=roll, pitch=pitch):
                data, _ = self._drop(roll, pitch)
                np.testing.assert_allclose(data.qvel, np.zeros(6), atol=1e-6)
                r, p, y = (math.degrees(a) for a in mujoco_quaternion_to_policy_euler(data.qpos[3:7]))
                self.assertLess(abs(r), 0.05)
                self.assertLess(abs(p), 0.05)
                self.assertLess(abs(y), 2.0)                       # observed <= 0.5 deg
                self.assertAlmostEqual(float(data.qpos[2]), GROUND_CLEARANCE_M, delta=self.TOL_M)
                self.assertEqual(int(data.ncon), 4)
                for g in self.legs:
                    bottom = float(data.geom_xpos[g][2]) - float(self.model.geom_size[g][0])
                    self.assertAlmostEqual(bottom, 0.0, delta=self.TOL_M)
                    self.assertLessEqual(bottom, 1e-9)
                # friction-driven slide during the righting transient stays small
                self.assertLess(float(np.hypot(*data.qpos[0:2])), 0.05)   # observed ~2 cm


class MeasuredModelForceLevelSanityTest(unittest.TestCase):
    """Force/torque-level checks on the compiled measured model. No
    propulsion / actuation model is involved: wrenches go straight into
    ``xfrc_applied``. The vehicle has no contact geoms in this config (the
    landing gear is unmeasured), so every scenario stays well above the
    ground plane -- ground contact is NOT exercised here."""

    @classmethod
    def setUpClass(cls):
        cls.params, cls.xml, cls.model = _compile()
        cls.body_id = mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_BODY, "vehicle")
        cls.g = float(-cls.model.opt.gravity[2])

    def _fresh(self, z=100.0):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        data.qpos[0:3] = [0.0, 0.0, z]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, data)
        return data

    def _assert_healthy(self, data):
        self.assertTrue(np.all(np.isfinite(data.qpos)))
        self.assertTrue(np.all(np.isfinite(data.qvel)))
        self.assertAlmostEqual(float(np.linalg.norm(data.qpos[3:7])), 1.0, places=12)
        self.assertTrue(np.all(np.asarray([w.number for w in data.warning]) == 0))

    def test_default_pose_is_finite_and_stable(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        self._assert_healthy(data)

    def test_free_fall_and_zero_actuation(self):
        data = self._fresh()
        mujoco.mj_forward(self.model, data)
        np.testing.assert_allclose(data.qacc[0:3], [0.0, 0.0, -self.g], atol=1e-9)
        n = 250
        for _ in range(n):
            mujoco.mj_step(self.model, data)
        t = n * PHYSICS_DT
        # Gravity only: vz = -g t, no lateral drift, no spin, attitude intact.
        np.testing.assert_allclose(data.qvel[0:3], [0.0, 0.0, -self.g * t], atol=1e-9)
        np.testing.assert_allclose(data.qpos[0:2], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(data.qvel[3:6], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(data.qpos[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        self.assertGreater(data.qpos[2], 0.0, "test must stay above the ground plane")
        self._assert_healthy(data)

    def test_hover_equivalent_force_balances_gravity(self):
        # Force-level only: this is m*g applied at the CG, NOT a claim that
        # the propulsion system can produce it (max thrust is UNKNOWN).
        hover_n = self.params.hover_thrust_n(self.g)
        self.assertAlmostEqual(hover_n, MASS_KG * self.g, places=12)
        self.assertAlmostEqual(hover_n, 62.841, places=2)
        self.assertAlmostEqual(hover_n / 4.0, 15.710, places=2)  # equal-sharing assumption
        self.assertAlmostEqual(hover_n / 4.0 / 9.80665, 1.602, places=3)  # kgf per motor

        data = self._fresh()
        data.xfrc_applied[self.body_id, 2] = hover_n
        mujoco.mj_forward(self.model, data)
        np.testing.assert_allclose(data.qacc[0:3], [0.0, 0.0, 0.0], atol=1e-9)
        for _ in range(250):
            mujoco.mj_step(self.model, data)
        np.testing.assert_allclose(data.qvel, np.zeros(6), atol=1e-9)
        np.testing.assert_allclose(data.qpos[0:3], [0.0, 0.0, 100.0], atol=1e-9)
        self._assert_healthy(data)

    def test_raw_torque_gives_alpha_equal_torque_over_measured_inertia(self):
        j = np.array([IXX, IYY, IZZ])
        tau_mag = 0.5
        for axis in range(3):
            data = self._fresh()
            data.xfrc_applied[self.body_id, 3 + axis] = tau_mag
            mujoco.mj_forward(self.model, data)
            expected = np.zeros(3)
            expected[axis] = tau_mag / j[axis]
            np.testing.assert_allclose(
                data.qacc[3:6], expected, rtol=1e-9, atol=1e-12, err_msg=f"axis {axis}"
            )
            np.testing.assert_allclose(data.qacc[0:3], [0.0, 0.0, -self.g], atol=1e-9)

    def test_inertia_is_attached_to_body_axes_not_world_axes(self):
        # Roll the vehicle +90 deg about its body x axis. World +z then lies
        # along body +y (R_x(90): y_body -> z_world). A world-z torque must
        # therefore accelerate the BODY y rate by tau/Iyy -- proving Iyy is
        # the pitch-axis inertia and that ordering is body-x/y/z, not
        # world-fixed.
        half = math.pi / 4.0
        data = self._fresh()
        data.qpos[3:7] = [math.cos(half), math.sin(half), 0.0, 0.0]
        data.xfrc_applied[self.body_id, 5] = 0.5  # world +z torque
        mujoco.mj_forward(self.model, data)
        np.testing.assert_allclose(
            data.qacc[3:6], [0.0, 0.5 / IYY, 0.0], rtol=1e-9, atol=1e-12
        )

    def test_frd_and_flu_body_rates_agree_on_the_roll_axis_sign(self):
        # FRD roll (about +x forward) and FLU roll (about +x forward) are the
        # same physical axis and direction; only y/z rates flip sign.
        data = self._fresh()
        data.xfrc_applied[self.body_id, 3] = 0.5
        mujoco.mj_forward(self.model, data)
        self.assertGreater(data.qacc[3], 0.0)
        self.assertAlmostEqual(
            float(mujoco_body_vector_to_frd(data.qacc[3:6])[0]), 0.5 / IXX, places=9
        )


if __name__ == "__main__":
    unittest.main()
