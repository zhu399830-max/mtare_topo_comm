from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROBE = PROJECT_ROOT / "tools/v3/isaac/probe_official_local_usda_writer_runtime.py"
RUNNER = PROJECT_ROOT / "tools/v3/run_official_local_usda_writer_control.py"


class OfficialLocalUsdaWriterControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.probe = PROBE.read_text(encoding="utf-8")
        self.runner = RUNNER.read_text(encoding="utf-8")

    def test_uses_exact_frozen_official_asset_only(self) -> None:
        self.assertIn("0812faf5c310f40316d5a11ab0c6786e19e18ac12cea10edc2e7ee44fc56c8c6", self.runner)
        self.assertIn("EXPECTED_ASSET_BYTES = 15137", self.runner)
        self.assertIn("usd_path=str(official_usda)", self.probe)
        self.assertNotIn('config="Example_Rotary"', self.probe)
        self.assertNotIn("mtare_vlp16", self.probe.lower())

    def test_matches_official_control_scene_and_writer_sequence(self) -> None:
        for name in ("cube_front", "cube_left", "cube_right", "cube_above"):
            self.assertIn(name, self.probe)
        self.assertIn("MAX_RENDER_FRAMES = 300", self.probe)
        self.assertIn("class OfficialLocalUsdaGmoControlWriter(Writer):", self.probe)
        self.assertIn('self.data_structure = "renderProduct"', self.probe)
        self.assertIn('rep.annotators.get("GenericModelOutput")', self.probe)

    def test_records_physical_creation_attach_frame_and_callback_counts(self) -> None:
        for token in (
            "sensor_creation_success",
            "writer_attach_success",
            "render_frames_updated",
            "callback_count",
            "valid_header_count",
            "positive_element_count",
            "complete_scan_count",
        ):
            self.assertIn(token, self.probe + self.runner)

    def test_run_is_network_disabled_and_never_saves_samples_or_visuals(self) -> None:
        self.assertIn('"--network",\n        "none"', self.runner)
        self.assertNotIn("np.save", self.probe)
        self.assertNotIn("matplotlib", self.probe)
        self.assertNotIn("range_image", self.probe)
        self.assertIn('"saved_point_samples": 0', self.probe)
        self.assertIn('"formal_dataset_samples": 0', self.runner)
        self.assertIn('"training_samples": 0', self.runner)
        self.assertIn('"models": 0', self.runner)

    def test_ab_classification_and_no_automatic_fallback(self) -> None:
        self.assertIn("HOST_OR_HEADLESS_REPLICATOR_RUNTIME_BLOCKED", self.probe)
        self.assertIn("CUSTOM_SENSOR_OR_SCENE_COUPLING_REMAINS", self.runner)
        self.assertIn("LOCAL_USDA_CONTROL_IMPLEMENTATION_OR_RUNTIME_EXCEPTION", self.runner)
        self.assertIn("FAIL_STOPPED_NO_FALLBACK", self.probe)
        self.assertNotIn('config="Example_Rotary"', self.probe)


if __name__ == "__main__":
    unittest.main()
