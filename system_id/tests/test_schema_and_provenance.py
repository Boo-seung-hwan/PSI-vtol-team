import unittest

from system_id.preprocessing import schema
from system_id.preprocessing.provenance import (
    DATASET_ROLE_VALIDATION,
    SOURCE_PROJECT_SAMPLE,
    build_provenance,
)
from system_id.tests._synth import build_synthetic_loaded


class RatesSetpointFieldTest(unittest.TestCase):
    def test_scalar_form(self):
        rf = schema.rates_setpoint_fields(["roll", "pitch", "yaw", "thrust_body[2]", "timestamp"])
        self.assertEqual((rf.roll, rf.pitch, rf.yaw), ("roll", "pitch", "yaw"))
        self.assertEqual(rf.thrust_z, "thrust_body[2]")

    def test_array_form(self):
        rf = schema.rates_setpoint_fields(["xyz[0]", "xyz[1]", "xyz[2]", "timestamp"])
        self.assertEqual((rf.roll, rf.pitch, rf.yaw), ("xyz[0]", "xyz[1]", "xyz[2]"))
        self.assertIsNone(rf.thrust_z)

    def test_unknown_form_raises(self):
        with self.assertRaises(schema.SchemaError):
            schema.rates_setpoint_fields(["timestamp", "reset_integral"])


class SchemaColumnsTest(unittest.TestCase):
    def test_every_loop_has_columns_with_full_meta(self):
        for loop in schema.ALL_LOOPS:
            cols = schema.LOOP_COLUMNS[loop]
            self.assertTrue(cols)
            names = [c.name for c in cols]
            self.assertEqual(names[0], "t")
            self.assertEqual(len(names), len(set(names)), f"dup column in {loop}")
            for c in cols:
                self.assertTrue(c.unit)
                self.assertTrue(c.frame)
                self.assertTrue(c.source)


class ProvenanceTest(unittest.TestCase):
    def test_sample_labels_and_param_subset(self):
        loaded = build_synthetic_loaded()
        p = build_provenance(loaded, SOURCE_PROJECT_SAMPLE, DATASET_ROLE_VALIDATION).to_dict()
        self.assertEqual(p["source_project"], "OTHER_PROJECT_SAMPLE")
        self.assertEqual(p["dataset_role"], "PIPELINE_VALIDATION_ONLY")
        self.assertEqual(p["sha256"], "0" * 64)
        self.assertIn("MPC_XY_VEL_P_ACC", p["mpc_params"])
        self.assertIn("SYS_AUTOSTART", p["airframe"])
        self.assertIn("python", p["tool_versions"])
        self.assertIn("pyulog", p["tool_versions"])
        self.assertIn("NOT UGRP", p["notes"])
        self.assertIn("esc_status", p["missing_optional_topics"])

    def test_schema_and_preprocessing_versions_recorded(self):
        loaded = build_synthetic_loaded()
        p = build_provenance(loaded, SOURCE_PROJECT_SAMPLE, DATASET_ROLE_VALIDATION).to_dict()
        self.assertEqual(p["schema_version"], schema.SCHEMA_VERSION)
        self.assertEqual(p["preprocessing_version"], schema.PREPROCESSING_VERSION)


if __name__ == "__main__":
    unittest.main()
