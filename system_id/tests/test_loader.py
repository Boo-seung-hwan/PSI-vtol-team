import os
import unittest

from system_id.preprocessing.ulog_loader import ULogLoadError, load_ulog, sha256_file

_SAMPLE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "landing_rl", "flight_log_ulg"
)
_SAMPLE = os.path.join(_SAMPLE_DIR, "08_53_49.ulg")
_HAVE_SAMPLE = os.path.exists(_SAMPLE)


class LoaderErrorTest(unittest.TestCase):
    def test_missing_file_raises_cleanly(self):
        with self.assertRaises(ULogLoadError):
            load_ulog("/no/such/file.ulg")

    def test_non_ulog_file_raises_cleanly(self):
        here = os.path.abspath(__file__)
        with self.assertRaises(ULogLoadError):
            load_ulog(here)


@unittest.skipUnless(_HAVE_SAMPLE, "sample ULog not present")
class LoaderSampleTest(unittest.TestCase):
    def test_loads_and_reports_missing_optional(self):
        loaded = load_ulog(_SAMPLE)
        # esc_status is absent in the sample logs -> recorded, not fatal
        self.assertIn("esc_status", loaded.missing_optional)
        self.assertEqual(loaded.missing_required, ())
        self.assertTrue(loaded.sha256 and len(loaded.sha256) == 64)
        self.assertEqual(loaded.sha256, sha256_file(_SAMPLE))
        self.assertIn("vehicle_local_position_setpoint", loaded.topics)
        self.assertGreater(loaded.last_us, loaded.start_us)

    def test_required_topic_filter_can_force_failure(self):
        with self.assertRaises(ULogLoadError):
            load_ulog(_SAMPLE, required_topics=("this_topic_does_not_exist",))


if __name__ == "__main__":
    unittest.main()
