from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROBE = PROJECT_ROOT / "tools/v3/isaac/probe_official_lidar_writer_runtime.py"
RUNNER = PROJECT_ROOT / "tools/v3/run_official_lidar_writer_control.py"


class OfficialLidarWriterControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.probe = PROBE.read_text(encoding="utf-8")
        self.runner = RUNNER.read_text(encoding="utf-8")

    def test_control_uses_only_builtin_example_rotary(self) -> None:
        self.assertIn('config="Example_Rotary"', self.probe)
        self.assertNotIn("usd_path=", self.probe)
        self.assertNotIn("mtare_vlp16", self.probe.lower())

    def test_control_matches_official_four_cube_layout(self) -> None:
        for name in ("cube_front", "cube_left", "cube_right", "cube_above"):
            self.assertIn(name, self.probe)
        self.assertIn("MAX_RENDER_FRAMES = 300", self.probe)

    def test_control_uses_writer_and_records_direct_health_counts(self) -> None:
        self.assertIn("class OfficialGmoRuntimeControlWriter(Writer):", self.probe)
        self.assertIn('self.data_structure = "renderProduct"', self.probe)
        self.assertIn('rep.annotators.get("GenericModelOutput")', self.probe)
        for metric in (
            "callback_count",
            "valid_header_count",
            "positive_element_count",
            "zero_element_count",
            "complete_scan_count",
        ):
            self.assertIn(metric, self.probe)

    def test_control_never_saves_point_samples_or_visualizations(self) -> None:
        self.assertNotIn("np.save", self.probe)
        self.assertNotIn("matplotlib", self.probe)
        self.assertNotIn("range_image", self.probe)
        self.assertIn('"saved_point_samples": 0', self.probe)

    def test_runner_freezes_official_reference_and_classifies_ab(self) -> None:
        self.assertIn(
            "fb6ab0cb1599d3a84bdc47f92e0db8cee7481dca240880cb20b440c23da89d46",
            self.runner,
        )
        self.assertIn("CUSTOM_SENSOR_OR_SCENE_COUPLING_REMAINS", self.runner)
        self.assertIn("writer_drain_timeout_in_log", self.runner)


if __name__ == "__main__":
    unittest.main()
