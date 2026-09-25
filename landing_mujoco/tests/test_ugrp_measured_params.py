"""Guards on the 2026-09-16 UGRP vehicle measurements.

Scope: this file tests PROVENANCE, not physics. Every assertion here is about
what ``ugrp_vehicle_measured.yaml`` records and, just as importantly, what it
must NOT silently record -- an invented inertia, a datum-converted z, or a
provisional estimate promoted into a field that dynamics code consumes.

No test here constructs an environment or steps any dynamics.
"""

import itertools
import math
import unittest
from pathlib import Path

import numpy as np

from landing_mujoco.configs.param_schema import (
    ParameterSet,
    load_uav_params,
    motor_xy_from_radial_distance,
)

# Raw measurements, in the datum resolved on 2026-09-16 (heights above the
# floor, +Z up). Body-frame z = CG_RAW_Z - raw, because FRD +Z points DOWN
# and the body origin is the CG.
CG_RAW_Z = 0.241
MOTOR_RAW_Z = 0.334
BATTERY_RAW_Z = 0.220
from landing_rl.envs.landing_env import LandingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASURED_YAML = REPO_ROOT / "landing_mujoco" / "configs" / "ugrp_vehicle_measured.yaml"

# The config is intentionally incomplete (max collective thrust and the
# identified closed-loop response are still unmeasured), so every test loads
# it unvalidated.
def _params():
    return load_uav_params(MEASURED_YAML, validate=False)


class MeasuredValuesTest(unittest.TestCase):
    def test_parameter_set_is_measured_vehicle(self):
        self.assertEqual(_params().parameter_set, ParameterSet.MEASURED_VEHICLE)

    def test_mass(self):
        self.assertEqual(_params().mass_properties.mass_kg, 6.408)

    def test_motor_radial_distance(self):
        p = _params()
        self.assertEqual(p.geometry.arm_length_m, 0.360)
        # The same quantity is duplicated into raw_measurements for locality;
        # the two must never drift apart.
        self.assertEqual(p.raw_measurements.motor_radial_distance_m, 0.360)

    def test_motor_spin_directions(self):
        spin = _params().motors.spin_directions
        self.assertEqual(
            spin, {"FL": "CW", "FR": "CCW", "RL": "CCW", "RR": "CW"}
        )

    def test_all_four_motors_present(self):
        self.assertEqual(
            set(_params().motors.spin_directions.keys()), {"FL", "FR", "RL", "RR"}
        )

    def test_diagonally_opposed_motors_share_direction(self):
        spin = _params().motors.spin_directions
        self.assertEqual(spin["FL"], spin["RR"])
        self.assertEqual(spin["FR"], spin["RL"])
        self.assertNotEqual(spin["FL"], spin["FR"])

    def test_provenance_tags_are_explicit_choices(self):
        # Tags are asserted, not just values: a future edit that promotes an
        # observation or a spec into "MEASURED" must break a test.
        p = _params()
        self.assertEqual(p.mass_properties.meta["mass_kg"]["source"], "MEASURED")
        self.assertEqual(p.geometry.meta["arm_length_m"]["source"], "MEASURED")
        # Spin layout is an observation of the build, NOT an instrument
        # reading -- deliberately weaker than MEASURED.
        self.assertEqual(
            p.motors.meta["spin_directions"]["source"], "USER_REPORTED_OBSERVATION"
        )
        self.assertEqual(p.motors.meta["model"]["source"], "USER_PROVIDED_SPEC")
        self.assertEqual(p.motors.meta["kv_rpm_per_v"]["source"], "USER_PROVIDED_SPEC")
        self.assertEqual(p.raw_measurements.meta["source"], "MEASURED")
        # Confirmed frame symmetry and the datum resolution are USER_CONFIRMED,
        # which is what licensed the DERIVED geometry below them.
        self.assertEqual(p.geometry.meta["arm_angle_deg"]["source"], "USER_CONFIRMED")
        self.assertEqual(p.motors.meta["positions_body_m"]["source"], "DERIVED")
        self.assertEqual(p.mass_properties.meta["cg_body_m"]["source"], "DERIVED")
        self.assertEqual(p.battery.meta["position_body_m"]["source"], "DERIVED")

    def test_unmeasured_fields_are_tagged_not_measured(self):
        p = _params()
        self.assertEqual(
            p.thrust.meta["max_collective_thrust_n"]["source"], "NOT_MEASURED"
        )
        # ground_clearance_m was promoted to MEASURED on 2026-09-19 (landed pose
        # confirmed) and the skid points were DERIVED from measured dimensions
        # -- see LandedPoseGroundClearanceTest / MeasuredSkidFootprintTest.
        self.assertEqual(p.geometry.meta["wheelbase_m"]["source"], "NOT_MEASURED")

    def test_arm_length_is_radial_not_diagonal(self):
        # Semantic guard. `geometry.arm_length_m` means CENTER-to-motor in
        # this repository (tarot680b_reference.yaml sets it to wheelbase/2).
        # If 0.360 were ever reinterpreted as an opposite-motor diagonal, the
        # derived XY would be wrong by a factor of 2. Pin the relationship:
        # the radius must equal arm_length_m, not half of it.
        p = _params()
        xy = p.motors.positions_body_m["FR"][:2]
        self.assertAlmostEqual(math.hypot(*xy), 0.360, places=12)
        self.assertNotAlmostEqual(math.hypot(*xy), 0.180, places=6)

    def test_component_specs_recorded(self):
        p = _params()
        self.assertEqual(p.motors.model, "T-MOTOR MN501-S IP45")
        self.assertEqual(p.motors.kv_rpm_per_v, 360)
        self.assertEqual(p.propellers.model, "T-MOTOR MS1704")
        self.assertEqual(p.esc.model, "HOBBYWING Skywalker V2 60A")
        self.assertEqual(p.esc.max_current_a, 60)
        self.assertIn("Poly-Tronics", p.battery.model)
        self.assertEqual(p.battery.nominal_voltage_v, 22.2)


class DerivedValuesTest(unittest.TestCase):
    def test_hover_thrust_matches_repository_gravity(self):
        p = _params()
        g = LandingConfig().gravity_mps2
        self.assertAlmostEqual(p.hover_thrust_n(g), 6.408 * g, places=9)
        # Sanity band around the ~62.84 N figure, independent of which of the
        # repository's two gravity constants is used.
        self.assertAlmostEqual(p.hover_thrust_n(g), 62.84, places=1)

    def test_hover_thrust_agrees_across_both_repository_gravity_constants(self):
        # landing_rl uses 9.8065; landing_mujoco hardcodes 9.80665. This is a
        # pre-existing, deliberately unreconciled discrepancy -- assert only
        # that it is immaterial at this precision, do NOT assert equality.
        p = _params()
        self.assertAlmostEqual(
            p.hover_thrust_n(9.8065), p.hover_thrust_n(9.80665), places=2
        )

    def test_motor_xy_preserves_measured_radius_exactly(self):
        p = _params()
        pos = p.motors.positions_body_m
        self.assertEqual(set(pos.keys()), {"FL", "FR", "RL", "RR"})
        for name, vec in pos.items():
            self.assertEqual(vec.shape, (3,), f"{name} must be a full 3-vector")
            self.assertAlmostEqual(
                math.hypot(vec[0], vec[1]), 0.360, places=12,
                msg=f"{name} does not lie at the measured 0.360 m radius",
            )

    def test_motor_xy_recomputed_not_trusted(self):
        # Recompute from arm_length_m + arm_angle_deg instead of trusting the
        # YAML literals, so a hand-edit to either side cannot drift unnoticed.
        p = _params()
        expected = motor_xy_from_radial_distance(
            p.geometry.arm_length_m, p.geometry.arm_angle_deg
        )
        for name, vec in p.motors.positions_body_m.items():
            np.testing.assert_allclose(vec[:2], expected[name], atol=1e-12)

    def test_motor_xy_signs_follow_frd(self):
        # FRD: +x forward, +y right => front is +x, left is -y.
        pos = _params().motors.positions_body_m
        self.assertGreater(pos["FL"][0], 0.0)
        self.assertLess(pos["FL"][1], 0.0)
        self.assertGreater(pos["FR"][0], 0.0)
        self.assertGreater(pos["FR"][1], 0.0)
        self.assertLess(pos["RL"][0], 0.0)
        self.assertLess(pos["RL"][1], 0.0)
        self.assertLess(pos["RR"][0], 0.0)
        self.assertGreater(pos["RR"][1], 0.0)

    def test_opposite_motor_span_derives_to_720mm(self):
        # geometry.wheelbase_m is deliberately null; x500_shell derives the
        # opposite-motor span as the max pairwise horizontal distance. Pin
        # that derivation so the ambiguity documented in handover 0.11.3a
        # cannot silently produce a different number.
        pos = _params().motors.positions_body_m
        xy = [np.asarray(v[:2], dtype=np.float64) for v in pos.values()]
        span = max(
            float(np.linalg.norm(a - b))
            for a, b in itertools.combinations(xy, 2)
        )
        self.assertAlmostEqual(span, 0.720, places=12)


class MeasuredInertiaTest(unittest.TestCase):
    """The bifilar-measured diagonal inertia (recorded 2026-09-19).

    The canonical Ixx/Iyy/Izz are the user's literals and are pinned as such.
    Everything else here is provenance: the raw periods are re-reduced with
    the bifilar formula so the stored means/stds can never drift from the
    data they came from -- and so the "reproduced, not merely quoted" claim
    in the YAML is itself under test.
    """

    IXX, IYY, IZZ = 0.153184, 0.126285, 0.149050
    # INFERRED (the user did not state g): reproduces every reported figure.
    G_REDUCTION = 9.8065

    @staticmethod
    def _axes():
        return _params().mass_properties.meta["inertia_kgm2"]["axes"]

    @classmethod
    def _per_trial_inertia(cls, axis: str) -> np.ndarray:
        a = cls._axes()[axis]
        m = _params().mass_properties.mass_kg
        period = np.asarray(a["total_times_s"], dtype=np.float64) / a["cycles_per_timing"]
        return (
            m * cls.G_REDUCTION * a["bifilar_D_m"] ** 2 * period**2
            / (16.0 * math.pi**2 * a["bifilar_L_m"])
        )

    # -- canonical values ---------------------------------------------------
    def test_canonical_values_pinned(self):
        mp = _params().mass_properties
        self.assertEqual(mp.ixx, self.IXX)
        self.assertEqual(mp.iyy, self.IYY)
        self.assertEqual(mp.izz, self.IZZ)

    def test_inertia_estimation_mode_stays_manual(self):
        # This flag is the actual guard: "auto" would let load_uav_params
        # synthesize a geometric inertia for the REAL vehicle config.
        self.assertEqual(_params().mass_properties.inertia_estimation_mode, "manual")

    def test_positive_and_triangle_inequalities(self):
        mp = _params().mass_properties
        ixx, iyy, izz = mp.ixx, mp.iyy, mp.izz
        self.assertGreater(ixx, 0.0)
        self.assertGreater(iyy, 0.0)
        self.assertGreater(izz, 0.0)
        self.assertLessEqual(ixx, iyy + izz)
        self.assertLessEqual(iyy, ixx + izz)
        self.assertLessEqual(izz, ixx + iyy)

    # -- provenance tags ----------------------------------------------------
    def test_inertia_is_tagged_measured_with_method(self):
        meta = _params().mass_properties.meta["inertia_kgm2"]
        self.assertEqual(meta["source"], "MEASURED")
        self.assertEqual(meta["method"], "bifilar_suspension")
        self.assertEqual(meta["period_timing"], "manual_stopwatch")
        self.assertEqual(meta["mass_used_kg"], 6.408)
        self.assertEqual(meta["reduction"], "mean_of_per_trial_inertia")

    def test_r2_is_not_available_and_not_fabricated(self):
        meta = _params().mass_properties.meta["inertia_kgm2"]
        self.assertEqual(meta["r2"], "N/A")
        # A manual stopwatch period has no regression, hence no R^2 anywhere.
        for axis, a in meta["axes"].items():
            self.assertNotIn("r2", a, f"{axis}: R^2 must not be invented")

    def test_off_diagonals_are_an_assumption_not_a_measurement(self):
        prod = _params().mass_properties.meta["inertia_products_kgm2"]
        self.assertEqual(prod["source"], "ASSUMED_ZERO_FOR_V0")
        self.assertNotEqual(prod["source"], "MEASURED")
        for k in ("ixy", "ixz", "iyz"):
            self.assertEqual(prod[k], 0.0)
        # MuJoCo receives diaginertia only; the products are documentation.
        self.assertFalse(prod["applied_by_mujoco"])

    def test_uncertainty_recorded_but_not_wired(self):
        meta = _params().mass_properties.meta["inertia_kgm2"]
        self.assertEqual(
            meta["uncertainty_use"], "RECORDED_ONLY_NOT_WIRED_INTO_RANDOMIZATION"
        )

    # -- per-axis statistics as reported ------------------------------------
    def test_reported_statistics_pinned(self):
        ax = self._axes()
        expected = {
            "ixx": dict(D=0.090, L=0.427, T=4.50433, n=10, mean=0.153184,
                        std=0.004362, var=1.903e-5),
            "iyy": dict(D=0.185, L=0.424, T=1.98267, n=10, mean=0.126285,
                        std=0.003041, var=9.246e-6),
            "izz": dict(D=0.163, L=0.505, T=2.66815, n=9, mean=0.149050,
                        std=0.001698, var=2.884e-6),
        }
        for axis, e in expected.items():
            a = ax[axis]
            self.assertEqual(a["bifilar_D_m"], e["D"], axis)
            self.assertEqual(a["bifilar_L_m"], e["L"], axis)
            self.assertEqual(a["cycles_per_timing"], 3, axis)
            self.assertEqual(a["T_mean_s"], e["T"], axis)
            self.assertEqual(a["n"], e["n"], axis)
            self.assertEqual(a["mean_kgm2"], e["mean"], axis)
            self.assertEqual(a["sample_std_kgm2"], e["std"], axis)
            self.assertEqual(a["sample_variance_kgm2_sq"], e["var"], axis)
            self.assertEqual(len(a["total_times_s"]), e["n"], axis)

    def test_stored_means_match_canonical_values(self):
        ax = self._axes()
        mp = _params().mass_properties
        self.assertEqual(ax["ixx"]["mean_kgm2"], mp.ixx)
        self.assertEqual(ax["iyy"]["mean_kgm2"], mp.iyy)
        self.assertEqual(ax["izz"]["mean_kgm2"], mp.izz)

    # -- reduction re-derived from the raw periods --------------------------
    def test_raw_periods_reproduce_reported_statistics(self):
        for axis in ("ixx", "iyy", "izz"):
            a = self._axes()[axis]
            inertia = self._per_trial_inertia(axis)
            period = np.asarray(a["total_times_s"]) / a["cycles_per_timing"]
            # 5 dp: the user's T_mean is rounded to 5 dp.
            self.assertAlmostEqual(float(period.mean()), a["T_mean_s"], places=5, msg=axis)
            self.assertAlmostEqual(float(inertia.mean()), a["mean_kgm2"], places=6, msg=axis)
            self.assertAlmostEqual(
                float(inertia.std(ddof=1)), a["sample_std_kgm2"], places=6, msg=axis
            )
            self.assertAlmostEqual(
                float(inertia.var(ddof=1)), a["sample_variance_kgm2_sq"], places=8, msg=axis
            )

    def test_canonical_values_are_the_mean_of_trials_not_of_mean_period(self):
        # Distinguishes the two reductions: I(mean T) would be 0.153156 /
        # 0.126268 / 0.149046. The gap is smallest on yaw (~3.9e-6), so the
        # threshold sits below that (1e-6) while staying well above the
        # 6-dp rounding of the canonical values (5e-7).
        for axis, canonical in (("ixx", self.IXX), ("iyy", self.IYY), ("izz", self.IZZ)):
            a = self._axes()[axis]
            m = _params().mass_properties.mass_kg
            period_mean = float(np.mean(a["total_times_s"])) / a["cycles_per_timing"]
            i_of_mean = (
                m * self.G_REDUCTION * a["bifilar_D_m"] ** 2 * period_mean**2
                / (16.0 * math.pi**2 * a["bifilar_L_m"])
            )
            self.assertGreater(abs(i_of_mean - canonical), 1e-6, axis)
            self.assertAlmostEqual(float(self._per_trial_inertia(axis).mean()), canonical, places=6)

    # -- yaw outlier policy -------------------------------------------------
    def test_yaw_outlier_excluded_from_canonical_but_preserved(self):
        izz = self._axes()["izz"]
        self.assertEqual(izz["excluded_total_times_s"], [8.48])
        self.assertNotIn(8.48, izz["total_times_s"])
        self.assertEqual(izz["n"], 9)
        self.assertEqual(len(izz["total_times_s"]), 9)
        with_outlier = izz["with_excluded_trial_included"]
        self.assertEqual(with_outlier["n"], 10)
        self.assertEqual(with_outlier["mean_kgm2"], 0.150874)
        self.assertEqual(with_outlier["sample_std_kgm2"], 0.005984)
        # The alternative statistic must also be reproducible from the raw
        # data, so the exclusion decision stays auditable.
        all_times = np.asarray(izz["total_times_s"] + izz["excluded_total_times_s"])
        m = _params().mass_properties.mass_kg
        period = all_times / izz["cycles_per_timing"]
        inertia = (
            m * self.G_REDUCTION * izz["bifilar_D_m"] ** 2 * period**2
            / (16.0 * math.pi**2 * izz["bifilar_L_m"])
        )
        self.assertAlmostEqual(float(inertia.mean()), 0.150874, places=6)
        self.assertAlmostEqual(float(inertia.std(ddof=1)), 0.005984, places=6)
        # Canonical Izz is the outlier-EXCLUDED value, not the alternative.
        self.assertEqual(_params().mass_properties.izz, 0.149050)
        self.assertNotEqual(_params().mass_properties.izz, 0.150874)

    def test_other_axes_have_no_exclusions(self):
        for axis in ("ixx", "iyy"):
            self.assertNotIn("excluded_total_times_s", self._axes()[axis])
            self.assertEqual(self._axes()[axis]["outlier_policy"], "none")

    # -- ownership: mass/inertia counted exactly once -----------------------
    def test_component_masses_stay_null_so_nothing_can_be_double_counted(self):
        p = _params()
        self.assertIsNone(p.motors.mass_kg_each)
        self.assertIsNone(p.battery.mass_kg)

    def test_included_hardware_contributes_nothing_to_mujoco(self):
        hw = _params().mass_properties.meta["included_hardware"]
        self.assertFalse(hw["contributes_mass_to_mujoco"])
        self.assertFalse(hw["contributes_inertia_to_mujoco"])
        self.assertEqual(hw["ballast_brick"]["mass_kg_approx"], 1.8)
        self.assertEqual(hw["ballast_brick"]["center_below_cg_m_approx"], 0.08)
        # 0.021 is the same battery datum conversion as battery.position_body_m.
        self.assertAlmostEqual(
            hw["battery"]["center_below_cg_m"],
            float(_params().battery.position_body_m[2]),
            places=12,
        )


class UnknownsStayUnknownTest(unittest.TestCase):
    def test_unmeasured_propulsion_fields_stay_null(self):
        p = _params()
        self.assertIsNone(p.thrust.max_collective_thrust_n)
        self.assertIsNone(p.thrust.thrust_to_weight_max)
        self.assertIsNone(p.motors.mass_kg_each)
        self.assertIsNone(p.propellers.diameter_m)
        self.assertIsNone(p.propellers.pitch_m)
        self.assertIsNone(p.propellers.blade_count)

    def test_identified_response_stays_unidentified(self):
        r = _params().identified_response
        for attr in ("tau_roll_s", "tau_pitch_s", "tau_thrust_s", "tau_yaw_s", "delay_s"):
            self.assertIsNone(getattr(r, attr), f"{attr} must stay unidentified")

    def test_config_still_fails_fast_overall(self):
        from landing_mujoco.configs.param_schema import missing_required_fields

        missing = missing_required_fields(_params())
        # Geometry (2026-09-16) and inertia (2026-09-19) are resolved; what
        # remains is exactly the propulsion / closed-loop response set. None
        # of it may be back-filled just to make the config load.
        for resolved in ("total_mass_kg", "cg_body_m", "Ixx", "Iyy", "Izz",
                         "motor_positions_body_m"):
            self.assertNotIn(resolved, missing)
        self.assertEqual(
            set(missing),
            {"max_collective_thrust_n", "tau_roll_s", "tau_pitch_s",
             "tau_thrust_s", "actuator_delay_s"},
        )
        self.assertTrue(missing, "measured config must remain incomplete")


class DatumConversionTest(unittest.TestCase):
    """The datum was resolved on 2026-09-16: raw heights are measured from
    the floor, +Z up; the body frame is FRD (+Z down) with its origin at the
    CG. Every converted body-frame z must therefore equal
    ``cg_raw_z - raw_z``, recomputed here from the raw pair rather than
    trusted as a literal."""

    def test_datum_is_resolved_and_documented(self):
        rm = _params().raw_measurements
        self.assertEqual(rm.datum_status, "RESOLVED")
        self.assertTrue(rm.datum_resolved)
        self.assertEqual(rm.datum_origin, "floor")
        self.assertEqual(rm.datum_up_axis, "+z_up")

    def test_raw_values_preserved_verbatim(self):
        # Provenance must survive the conversion: the originals stay put.
        rm = _params().raw_measurements
        np.testing.assert_allclose(rm.cg_raw_m, [0.0, 0.0, CG_RAW_Z])
        self.assertEqual(rm.motor_plane_raw_z_m, MOTOR_RAW_Z)
        np.testing.assert_allclose(rm.battery_center_raw_m, [0.0, 0.0, BATTERY_RAW_Z])
        self.assertEqual(rm.motor_radial_distance_m, 0.360)

    def test_unsigned_separations_still_match_the_raw_pair(self):
        # Kept as an INDEPENDENT check: computed straight from the raw
        # numbers, so they cross-check the stored body-frame magnitudes.
        rm = _params().raw_measurements
        self.assertAlmostEqual(rm.abs_motor_plane_to_cg_m(), 0.093, places=9)
        self.assertAlmostEqual(rm.abs_cg_to_battery_m(), 0.021, places=9)

    def test_motor_z_is_negative_93mm(self):
        # Motors sit ABOVE the CG, and FRD +Z is DOWN => negative.
        for name, vec in _params().motors.positions_body_m.items():
            self.assertAlmostEqual(
                float(vec[2]), -0.093, places=9, msg=f"{name} z"
            )

    def test_battery_z_is_positive_21mm(self):
        # Battery sits BELOW the CG, and FRD +Z is DOWN => positive.
        pos = _params().battery.position_body_m
        np.testing.assert_allclose(pos, [0.0, 0.0, 0.021], atol=1e-12)

    def test_cg_is_body_origin(self):
        np.testing.assert_allclose(
            _params().mass_properties.cg_body_m, [0.0, 0.0, 0.0], atol=1e-12
        )

    def test_conversions_recomputed_from_raw_not_trusted(self):
        # The drift guard: derive each body z from the raw pair via the
        # schema's own conversion and compare to what is stored.
        p = _params()
        rm = p.raw_measurements
        self.assertAlmostEqual(
            rm.body_z_from_raw_height(MOTOR_RAW_Z),
            float(p.motors.positions_body_m["FR"][2]), places=12,
        )
        self.assertAlmostEqual(
            rm.body_z_from_raw_height(BATTERY_RAW_Z),
            float(p.battery.position_body_m[2]), places=12,
        )
        # The CG converts to the body origin by construction.
        self.assertAlmostEqual(rm.body_z_from_raw_height(CG_RAW_Z), 0.0, places=12)

    def test_conversion_sign_convention_is_not_accidentally_flipped(self):
        # If someone "fixes" the conversion to raw - cg, both signs invert
        # and this fails. Motors are physically above the CG; battery below.
        rm = _params().raw_measurements
        self.assertLess(rm.body_z_from_raw_height(MOTOR_RAW_Z), 0.0)
        self.assertGreater(rm.body_z_from_raw_height(BATTERY_RAW_Z), 0.0)

    def test_conversion_refuses_an_unresolved_datum(self):
        from landing_mujoco.configs.param_schema import RawMeasurements

        pending = RawMeasurements(cg_raw_m=np.array([0.0, 0.0, 0.241]))
        with self.assertRaises(ValueError) as ctx:
            pending.body_z_from_raw_height(0.334)
        self.assertIn("RESOLVED", str(ctx.exception))

    def test_arm_angle_is_confirmed(self):
        # Exact symmetric 45-degree X-frame, confirmed 2026-09-16. This is
        # what unblocked the motor XY derivation.
        self.assertEqual(_params().geometry.arm_angle_deg, 45.0)

    def test_fields_the_datum_does_not_determine_stay_null(self):
        # Resolving the datum did NOT reveal these. (ground_clearance_m used
        # to be in this list; the user confirmed the landed-pose datum on
        # 2026-09-19, so it is now pinned by LandedPoseGroundClearanceTest.)
        # (The skid contact points were later DERIVED from measured skid
        # dimensions -- see MeasuredSkidFootprintTest; the datum alone never
        # determined them.)
        p = _params()
        for label, value in (
            ("geometry.frame_height_m", p.geometry.frame_height_m),
        ):
            self.assertIsNone(value, f"{label} is not implied by the datum")

    def test_no_body_frame_field_equals_a_raw_datum_number(self):
        # (geometry.ground_clearance_m == 0.241 == the raw CG height is NOT a
        # copy error: the CG-to-ground distance in the landed pose IS the CG's
        # height above the ground. It is recomputed from the raw block in
        # LandedPoseGroundClearanceTest instead.)
        # Guard against copying 0.241 / 0.334 / 0.220 into a body-frame
        # field instead of converting it.
        p = _params()
        raw_z = {CG_RAW_Z, MOTOR_RAW_Z, BATTERY_RAW_Z}
        self.assertNotIn(round(float(p.battery.position_body_m[2]), 6), raw_z)
        for vec in p.motors.positions_body_m.values():
            self.assertNotIn(round(float(vec[2]), 6), raw_z)

    def test_motor_keysets_join(self):
        # Both dicts must use the same motor naming, or a rotor-level model
        # would silently fail to pair a position with its spin direction.
        p = _params()
        self.assertIsNotNone(p.motors.positions_body_m)
        self.assertEqual(
            set(p.motors.positions_body_m.keys()),
            set(p.motors.spin_directions.keys()),
        )


class LandedPoseGroundClearanceTest(unittest.TestCase):
    """Landed-pose ground plane (USER_CONFIRMED 2026-09-19).

    Three different quantities must never be conflated:

        physical ground clearance   0.241 m  (CG -> ground; stored, PHYSICAL)
        gear sphere radius          0.020 m  (SIMULATION; lives in the builder)
        derived sphere-centre       0.221 m  (= 0.241 - 0.020; DERIVED, never
                                              stored)
    """

    CLEARANCE = 0.241
    RADIUS = 0.020
    CENTRE_OFFSET = 0.221

    def test_ground_clearance_is_the_physical_cg_to_ground_distance(self):
        self.assertEqual(_params().geometry.ground_clearance_m, self.CLEARANCE)

    def test_clearance_recomputed_from_the_raw_block_not_trusted(self):
        # Ground = 0 in the raw datum, so the ground plane converts to
        # body z = cg_raw_z - 0 = +0.241 (BELOW the CG in FRD). The stored
        # clearance must agree with that conversion and with the raw CG height.
        p = _params()
        rm = p.raw_measurements
        self.assertAlmostEqual(rm.body_z_from_raw_height(0.0), p.geometry.ground_clearance_m, places=12)
        self.assertAlmostEqual(float(rm.cg_raw_m[2]), p.geometry.ground_clearance_m, places=12)

    def test_cg_is_the_origin_and_the_ground_is_below_it(self):
        # Wording guard: the CG is NOT at body z = +0.241; the GROUND is.
        p = _params()
        np.testing.assert_array_equal(np.asarray(p.mass_properties.cg_body_m), [0.0, 0.0, 0.0])
        self.assertGreater(p.geometry.ground_clearance_m, 0.0)

    def test_provenance_is_measured_and_dated(self):
        meta = _params().geometry.meta["ground_clearance_m"]
        self.assertEqual(meta["source"], "MEASURED")
        self.assertEqual(str(meta["date"]), "2026-09-19")

    def test_simulation_offset_is_never_stored_as_a_physical_parameter(self):
        p = _params()
        self.assertNotEqual(p.geometry.ground_clearance_m, self.CENTRE_OFFSET)

        # No numeric leaf anywhere in the parsed YAML (comments and strings
        # excluded) may equal 0.221 or the 0.020 sphere radius. BOTH are
        # intended: 0.221 is a DERIVED simulation value and 0.020 is a
        # SIMULATION constant owned by mjcf_builder.py (parameter ownership).
        # If provenance ever needs to mention them, write them in prose, not as
        # a YAML number; relaxing this must be a deliberate decision.
        import yaml

        def leaves(node):
            if isinstance(node, dict):
                for v in node.values():
                    yield from leaves(v)
            elif isinstance(node, (list, tuple)):
                for v in node:
                    yield from leaves(v)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                yield float(node)

        with open(MEASURED_YAML) as f:
            numbers = list(leaves(yaml.safe_load(f)))
        for forbidden in (self.CENTRE_OFFSET, self.RADIUS):
            self.assertNotIn(forbidden, numbers, f"{forbidden} must not be stored in the YAML")

    def test_radius_and_derived_offset_live_in_the_builder(self):
        from landing_mujoco.dynamics.mjcf_builder import (
            LEG_CONTACT_RADIUS_M,
            landing_gear_sphere_centers_body_m,
        )
        from dataclasses import replace

        self.assertEqual(LEG_CONTACT_RADIUS_M, self.RADIUS)
        g = _params().geometry
        # Synthetic x/y (unknown for the real vehicle); only z is examined.
        pts = np.array([[0.2, 0.3, g.ground_clearance_m]])
        centers = landing_gear_sphere_centers_body_m(replace(g, landing_gear_points_body_m=pts))
        self.assertAlmostEqual(float(centers[0, 2]), g.ground_clearance_m - LEG_CONTACT_RADIUS_M, places=12)
        self.assertAlmostEqual(float(centers[0, 2]), self.CENTRE_OFFSET, places=12)
        np.testing.assert_array_equal(centers[0, :2], pts[0, :2])  # x/y untouched

    def test_points_are_declared_physical_contact_points(self):
        p = _params()
        self.assertEqual(p.geometry.landing_gear_points_semantics, "physical_contact")
        self.assertEqual(p.geometry.meta["landing_gear_points_semantics"]["source"], "USER_CONFIRMED")

    def test_reference_point_falls_back_to_the_physical_clearance(self):
        from landing_mujoco.configs.param_schema import landing_reference_point_body_m

        # With no per-leg points, r_landing^B = [0, 0, ground_clearance_m]
        # (x/y ASSUMED centred). It is the PHYSICAL 0.241, not the 0.221
        # sphere centre. (The measured config now HAS points, so drop them
        # explicitly to exercise the fallback branch.)
        from dataclasses import replace

        geometry = replace(_params().geometry, landing_gear_points_body_m=None)
        r = landing_reference_point_body_m(geometry)
        np.testing.assert_array_equal(r, [0.0, 0.0, self.CLEARANCE])

    def test_future_contact_points_must_agree_with_the_ground_clearance(self):
        # Forward guard, VACUOUS today (per-leg x/y are unknown, so the points
        # are null). Once they are measured as PHYSICAL contact points, a level
        # landing means every point's z must equal ground_clearance_m -- the two
        # fields describe the same fact and must not drift apart.
        g = _params().geometry
        pts = g.landing_gear_points_body_m
        if pts is None:
            return
        if g.landing_gear_points_semantics == "physical_contact":
            np.testing.assert_allclose(pts[:, 2], g.ground_clearance_m, atol=1e-6)

    def test_invalid_semantics_is_rejected(self):
        from dataclasses import replace

        with self.assertRaises(ValueError):
            replace(_params().geometry, landing_gear_points_semantics="sphere_centre")

    def test_reference_config_keeps_the_legacy_reading(self):
        # tarot680b_reference.yaml predates the field; its gear points ARE
        # sphere centres and must keep behaving exactly as before.
        ref = load_uav_params(REPO_ROOT / "landing_mujoco" / "configs" / "tarot680b_reference.yaml")
        self.assertEqual(ref.geometry.landing_gear_points_semantics, "geom_center")
        self.assertEqual(ref.geometry.ground_clearance_m, 0.12)


class MeasuredSkidFootprintTest(unittest.TestCase):
    """Skid-type landing gear (2026-09-19).

    PHYSICAL gear: two continuous, parallel skid bars along body X. MuJoCo v0
    APPROXIMATION: four representative contact points at the skid endpoints.

    Provenance is deliberately NOT "measured point coordinates": the confirmed
    inputs are skid length 0.300 m, centre-line spacing 0.310 m, ground
    clearance 0.241 m and the skid direction; the points are DERIVED from them
    under a CENTERED_SYMMETRY_ASSUMPTION (footprint centred on the CG).
    """

    LENGTH = 0.300
    SPACING = 0.310
    CLEARANCE = 0.241
    # FRD, +Y right (so left is negative y): left-front, left-rear,
    # right-front, right-rear. Typed independently of the YAML.
    EXPECTED = np.array(
        [
            [+0.150, -0.155, +0.241],
            [-0.150, -0.155, +0.241],
            [+0.150, +0.155, +0.241],
            [-0.150, +0.155, +0.241],
        ]
    )

    @staticmethod
    def _pts():
        return np.asarray(_params().geometry.landing_gear_points_body_m)

    # A. count
    def test_exactly_four_contact_points(self):
        self.assertEqual(self._pts().shape, (4, 3))
        self.assertEqual(len({tuple(row) for row in self._pts()}), 4, "points must be distinct")

    def test_canonical_points_and_order(self):
        np.testing.assert_array_equal(self._pts(), self.EXPECTED)

    # B. contact z
    def test_all_contact_points_are_at_the_physical_ground_plane(self):
        pts = self._pts()
        np.testing.assert_array_equal(pts[:, 2], np.full(4, self.CLEARANCE))
        self.assertTrue(np.all(pts[:, 2] == _params().geometry.ground_clearance_m))

    # C. x extent
    def test_x_extent_is_the_skid_length(self):
        x = self._pts()[:, 0]
        self.assertEqual(set(np.round(x, 12)), {-0.150, +0.150})
        self.assertAlmostEqual(float(x.max() - x.min()), self.LENGTH, places=12)

    # D. y extent
    def test_y_extent_is_the_skid_spacing(self):
        y = self._pts()[:, 1]
        self.assertEqual(set(np.round(y, 12)), {-0.155, +0.155})
        self.assertAlmostEqual(float(y.max() - y.min()), self.SPACING, places=12)

    # E. symmetry
    def test_footprint_is_centred_on_the_cg(self):
        pts = self._pts()
        self.assertEqual(float(pts[:, 0].mean()), 0.0)
        self.assertEqual(float(pts[:, 1].mean()), 0.0)

    # F. semantics
    def test_points_are_physical_contact_points(self):
        self.assertEqual(_params().geometry.landing_gear_points_semantics, "physical_contact")

    # G. sphere-centre conversion
    def test_derived_sphere_centre_z_is_clearance_minus_radius(self):
        from landing_mujoco.dynamics.mjcf_builder import (
            LEG_CONTACT_RADIUS_M,
            landing_gear_sphere_centers_body_m,
        )

        self.assertEqual(LEG_CONTACT_RADIUS_M, 0.020)
        centres = landing_gear_sphere_centers_body_m(_params().geometry)
        np.testing.assert_allclose(centres[:, 2], 0.221, atol=1e-12)
        np.testing.assert_allclose(centres[:, 2], self.CLEARANCE - LEG_CONTACT_RADIUS_M, atol=1e-15)
        np.testing.assert_array_equal(centres[:, :2], self._pts()[:, :2])  # x/y untouched

    # H. no contamination
    def test_derived_sphere_centre_is_not_stored_as_a_physical_coordinate(self):
        pts = self._pts()
        self.assertNotIn(0.221, [round(float(v), 12) for v in pts.ravel()])
        self.assertNotIn(0.020, [round(float(v), 12) for v in pts.ravel()])
        self.assertNotEqual(_params().geometry.ground_clearance_m, 0.221)
        # ... and the stored points were not mutated by deriving the centres.
        np.testing.assert_array_equal(pts[:, 2], np.full(4, self.CLEARANCE))

    # skid geometry: two parallel bars along body X
    def test_skids_are_two_parallel_bars_along_body_x(self):
        pts = self._pts()
        left, right = pts[[0, 1]], pts[[2, 3]]
        self.assertTrue(np.all(left[:, 1] < 0), "FRD +Y is right: the left skid is negative y")
        self.assertTrue(np.all(right[:, 1] > 0))
        # Each skid: constant y (its long direction is X) and x spanning the length.
        self.assertEqual(left[0, 1], left[1, 1])
        self.assertEqual(right[0, 1], right[1, 1])
        self.assertAlmostEqual(float(left[0, 0] - left[1, 0]), self.LENGTH, places=12)
        self.assertAlmostEqual(float(right[0, 0] - right[1, 0]), self.LENGTH, places=12)
        self.assertAlmostEqual(float(right[0, 1] - left[0, 1]), self.SPACING, places=12)

    # provenance
    def test_provenance_is_derived_from_dimensions_plus_a_symmetry_assumption(self):
        meta = _params().geometry.meta["landing_gear_points_body_m"]
        self.assertEqual(meta["source"], "DERIVED_FROM_MEASURED_DIMENSIONS")
        self.assertIn("CENTERED_SYMMETRY_ASSUMPTION", meta["assumptions"])
        # never promoted to a measurement of the point coordinates
        self.assertNotIn(meta["source"], ("MEASURED", "USER_CONFIRMED"))
        skid = meta["skid_geometry"]
        self.assertEqual(skid["centering"]["source"], "CENTERED_SYMMETRY_ASSUMPTION")
        self.assertEqual(skid["skid_long_axis"], "body_x (vehicle roll axis)")
        self.assertIn("four", skid["v0_approximation"])
        self.assertIn("endpoints", skid["v0_approximation"])
        self.assertIn("two continuous", skid["physical_gear"])

    def test_dimension_provenance(self):
        skid = _params().geometry.meta["landing_gear_points_body_m"]["skid_geometry"]
        self.assertEqual(skid["skid_effective_length_m"]["value"], self.LENGTH)
        self.assertEqual(skid["skid_effective_length_m"]["source"], "USER_CONFIRMED")
        self.assertEqual(skid["skid_centerline_spacing_m"]["value"], self.SPACING)
        self.assertEqual(skid["skid_centerline_spacing_m"]["source"], "USER_CONFIRMED")
        self.assertEqual(skid["half_length_m"]["source"], "DERIVED")
        self.assertEqual(skid["half_spacing_m"]["source"], "DERIVED")
        self.assertEqual(_params().geometry.meta["ground_clearance_m"]["source"], "MEASURED")

    def test_points_recomputed_from_the_recorded_dimensions_not_trusted(self):
        # The stored points must equal what the recorded dimensions imply, so
        # a hand-edit to either side cannot drift unnoticed.
        g = _params().geometry
        skid = g.meta["landing_gear_points_body_m"]["skid_geometry"]
        half_l = skid["skid_effective_length_m"]["value"] / 2.0
        half_s = skid["skid_centerline_spacing_m"]["value"] / 2.0
        self.assertEqual(skid["half_length_m"]["value"], half_l)
        self.assertEqual(skid["half_spacing_m"]["value"], half_s)
        expected = np.array(
            [
                [+half_l, -half_s, g.ground_clearance_m],
                [-half_l, -half_s, g.ground_clearance_m],
                [+half_l, +half_s, g.ground_clearance_m],
                [-half_l, +half_s, g.ground_clearance_m],
            ]
        )
        np.testing.assert_array_equal(self._pts(), expected)

    def test_landing_reference_point_is_the_physical_touchdown_point(self):
        from dataclasses import replace

        from landing_mujoco.configs.param_schema import landing_reference_point_body_m

        g = _params().geometry
        via_points = landing_reference_point_body_m(g)
        via_fallback = landing_reference_point_body_m(replace(g, landing_gear_points_body_m=None))
        np.testing.assert_array_equal(via_points, [0.0, 0.0, self.CLEARANCE])
        # centred symmetry => the points path and the ground_clearance fallback agree
        np.testing.assert_array_equal(via_points, via_fallback)

    # nothing else moved
    def test_other_physical_parameters_are_unchanged(self):
        p = _params()
        mp = p.mass_properties
        self.assertEqual(mp.mass_kg, 6.408)
        np.testing.assert_array_equal(np.asarray(mp.cg_body_m), [0.0, 0.0, 0.0])
        self.assertEqual((mp.ixx, mp.iyy, mp.izz), (0.153184, 0.126285, 0.149050))
        self.assertEqual(p.geometry.arm_length_m, 0.360)
        self.assertEqual(p.geometry.ground_clearance_m, 0.241)
        for vec in p.motors.positions_body_m.values():
            self.assertEqual(float(vec[2]), -0.093)
        self.assertEqual(p.motors.spin_directions, {"FL": "CW", "FR": "CCW", "RL": "CCW", "RR": "CW"})
        np.testing.assert_allclose(p.battery.position_body_m, [0.0, 0.0, 0.021], atol=1e-12)

    def test_propulsion_stays_unknown(self):
        p = _params()
        self.assertIsNone(p.thrust.max_collective_thrust_n)
        for attr in ("tau_roll_s", "tau_pitch_s", "tau_thrust_s", "tau_yaw_s", "delay_s"):
            self.assertIsNone(getattr(p.identified_response, attr), attr)


class YamlIntegrityTest(unittest.TestCase):
    """PyYAML silently keeps the LAST value of a duplicated mapping key, so a
    duplicated motor name (or a second ``ixx:``) would parse cleanly and pass
    every value test. Load with a strict loader instead."""

    def test_no_duplicate_mapping_keys_anywhere(self):
        import yaml

        class StrictLoader(yaml.SafeLoader):
            pass

        def construct_mapping(loader, node, deep=False):
            seen = set()
            for key_node, _ in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in seen:
                    raise AssertionError(
                        f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
                    )
                seen.add(key)
            return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

        StrictLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
        )
        with open(MEASURED_YAML) as f:
            data = yaml.load(f, Loader=StrictLoader)
        self.assertEqual(data["parameter_set"], "MEASURED_VEHICLE")

    def test_motor_names_are_unique_in_both_motor_maps(self):
        p = _params()
        self.assertEqual(len(p.motors.positions_body_m), 4)
        self.assertEqual(len(p.motors.spin_directions), 4)
        # Four distinct positions, four distinct names.
        unique = {tuple(np.round(v, 12)) for v in p.motors.positions_body_m.values()}
        self.assertEqual(len(unique), 4)


class ReferenceConfigUnaffectedTest(unittest.TestCase):
    """The provisional reference config must parse exactly as before -- the
    raw_measurements block is optional and absent there."""

    def test_reference_config_gets_empty_provenance_block(self):
        ref = REPO_ROOT / "landing_mujoco" / "configs" / "tarot680b_reference.yaml"
        p = load_uav_params(ref)
        self.assertEqual(p.parameter_set, ParameterSet.PROVISIONAL_REFERENCE)
        # No datum was ever defined for the reference airframe.
        self.assertEqual(p.raw_measurements.datum_status, "PENDING_DEFINITION")
        self.assertFalse(p.raw_measurements.datum_resolved)
        self.assertIsNone(p.raw_measurements.cg_raw_m)
        # And it carries no confirmed arm angle -- its motor XY remains an
        # explicitly-tagged assumption about a different vehicle.
        self.assertIsNone(p.geometry.arm_angle_deg)


if __name__ == "__main__":
    unittest.main()
