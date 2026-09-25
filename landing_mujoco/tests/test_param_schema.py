import unittest
from pathlib import Path

from landing_mujoco.configs.param_schema import (
    MissingMeasurementError,
    ParameterSet,
    load_uav_params,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_YAML = REPO_ROOT / "landing_mujoco" / "configs" / "tarot680b_reference.yaml"
MEASURED_YAML = REPO_ROOT / "landing_mujoco" / "configs" / "ugrp_vehicle_measured.yaml"


class ParamSchemaTest(unittest.TestCase):
    def test_reference_loads_and_reports_provisional(self):
        params = load_uav_params(REFERENCE_YAML)
        self.assertEqual(params.parameter_set, ParameterSet.PROVISIONAL_REFERENCE)
        self.assertEqual(params.summary()["parameter_set"], "PROVISIONAL_REFERENCE")

    def test_reference_inertia_auto_estimated_not_measured(self):
        params = load_uav_params(REFERENCE_YAML)
        mp = params.mass_properties
        self.assertEqual(mp.inertia_estimation_mode, "auto")
        self.assertIsNotNone(mp.ixx)
        self.assertIsNotNone(mp.iyy)
        self.assertIsNotNone(mp.izz)
        self.assertGreater(mp.ixx, 0.0)
        self.assertGreater(mp.iyy, 0.0)
        self.assertGreater(mp.izz, 0.0)

    def test_measured_raises_with_expected_missing_fields(self):
        with self.assertRaises(MissingMeasurementError) as ctx:
            load_uav_params(MEASURED_YAML)
        # This list has shrunk as real data arrived: total_mass_kg
        # (measured 2026-09-16), cg_body_m and motor_positions_body_m (once
        # the measurement datum and the exact 45-degree X symmetry were
        # confirmed, 2026-09-16), then Ixx/Iyy/Izz (bifilar, 2026-09-19).
        # Everything below is a propulsion / closed-loop-response quantity
        # that is still genuinely unmeasured -- the config must keep failing
        # fast rather than have any of these invented to make it load.
        expected = {
            "max_collective_thrust_n",
            "tau_roll_s", "tau_pitch_s", "tau_thrust_s", "actuator_delay_s",
        }
        self.assertEqual(set(ctx.exception.missing), expected)
        for resolved in (
            "total_mass_kg", "cg_body_m", "motor_positions_body_m",
            "Ixx", "Iyy", "Izz",
        ):
            self.assertNotIn(resolved, ctx.exception.missing)
        self.assertEqual(ctx.exception.parameter_set, ParameterSet.MEASURED_VEHICLE)
        self.assertIn("ERROR: measured vehicle model incomplete", str(ctx.exception))

    def test_measured_can_be_loaded_unvalidated_for_inspection(self):
        params = load_uav_params(MEASURED_YAML, validate=False)
        self.assertEqual(params.parameter_set, ParameterSet.MEASURED_VEHICLE)
        # Mass and inertia are measured; max collective thrust is the field
        # that stands in for "still incomplete" in this inspection path.
        self.assertEqual(params.mass_properties.mass_kg, 6.408)
        self.assertEqual(params.mass_properties.ixx, 0.153184)
        self.assertIsNone(params.thrust.max_collective_thrust_n)

    def test_unity_gain_default_when_k_missing(self):
        params = load_uav_params(MEASURED_YAML, validate=False)
        self.assertEqual(params.identified_response.gain("roll"), 1.0)

    def test_thrust_margin_and_hover_thrust(self):
        params = load_uav_params(REFERENCE_YAML)
        g = 9.80665
        hover = params.hover_thrust_n(g)
        self.assertAlmostEqual(hover, params.mass_properties.mass_kg * g, places=6)
        margin = params.thrust_margin(g)
        self.assertGreater(margin, 1.0, "T_max must exceed hover thrust")
        self.assertAlmostEqual(margin, params.thrust.thrust_to_weight_max, places=2)


if __name__ == "__main__":
    unittest.main()
