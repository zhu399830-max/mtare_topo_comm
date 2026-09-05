from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_JSON = PROJECT_ROOT / "configs/v3/gate0/sensors/mtare_vlp16_720_50m_v1.json"
SENSOR_USDA = PROJECT_ROOT / "configs/v3/gate0/sensors/mtare_vlp16_720_50m_v1.usda"
ISAAC_PROBE = PROJECT_ROOT / "tools/v3/isaac/probe_exact_rtx_lidar_profile.py"
RUNNER = PROJECT_ROOT / "tools/v3/run_exact_rtx_lidar_profile_probe.py"


class ExactRtxProfileProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))["profile"]
        self.usda = SENSOR_USDA.read_text(encoding="utf-8")
        self.isaac_probe = ISAAC_PROBE.read_text(encoding="utf-8")
        self.runner = RUNNER.read_text(encoding="utf-8")

    def test_legacy_profile_freezes_exact_nominal_grid(self) -> None:
        self.assertEqual(self.profile["numberOfEmitters"], 16)
        self.assertEqual(self.profile["scanRateBaseHz"], 10.0)
        self.assertEqual(self.profile["patternFiringRateHz"], 7200)
        self.assertEqual(
            self.profile["patternFiringRateHz"] / self.profile["scanRateBaseHz"], 720
        )
        self.assertEqual(self.profile["farRangeM"], 50.0)
        self.assertEqual(
            self.profile["emitterStates"][0]["elevationDeg"],
            list(range(-15, 16, 2)),
        )

    def test_usda_uses_omnilidar_core_and_no_builtin_reference(self) -> None:
        self.assertIn('def OmniLidar "MTARE_VLP16_720_50M_V1"', self.usda)
        self.assertIn('prepend apiSchemas = ["OmniSensorGenericLidarCoreAPI"]', self.usda)
        self.assertNotIn("references =", self.usda)
        self.assertNotIn("/Isaac/Sensors/", self.usda)

    def test_usda_explicitly_overrides_channel_defaults(self) -> None:
        self.assertRegex(
            self.usda,
            r"uint omni:sensor:Core:numberOfChannels = 16(?:\s|$)",
        )
        self.assertRegex(
            self.usda,
            r"uint omni:sensor:Core:numberOfEmitters = 16(?:\s|$)",
        )
        channel_match = re.search(
            r"emitterState:s001:channelId = \[([^\]]+)\]", self.usda
        )
        self.assertIsNotNone(channel_match)
        assert channel_match is not None
        channels = [int(value.strip()) for value in channel_match.group(1).split(",")]
        self.assertEqual(channels, list(range(1, 17)))

    def test_usda_preserves_frozen_ranges_rates_and_directions(self) -> None:
        required = (
            "float omni:sensor:Core:nearRangeM = 0.3",
            "float omni:sensor:Core:farRangeM = 50",
            "uint omni:sensor:Core:scanRateBaseHz = 10",
            "uint omni:sensor:Core:patternFiringRateHz = 7200",
            'token omni:sensor:Core:rotationDirection = "CCW"',
            'token omni:sensor:Core:rayType = "IDEALIZED"',
            "uint omni:sensor:Core:maxReturns = 1",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.usda)

    def test_runtime_uses_official_writer_callback_not_direct_polling(self) -> None:
        self.assertIn("class _ExactProfileGmoCaptureWriter(Writer):", self.isaac_probe)
        self.assertIn('self.data_structure = "renderProduct"', self.isaac_probe)
        self.assertIn('rep.annotators.get("GenericModelOutput")', self.isaac_probe)
        self.assertIn('sensor.attach_writer("_ExactProfileGmoCaptureWriter")', self.isaac_probe)
        self.assertNotIn('sensor.get_data("generic-model-output")', self.isaac_probe)

    def test_runner_counts_scan_only_from_complete_evidence(self) -> None:
        self.assertIn('complete_scan_evidence = (', self.runner)
        self.assertIn('== "PASS_EXACT_PROFILE_COMPLETE_SCAN"', self.runner)
        self.assertIn(
            '"complete_diagnostic_scans": 1 if complete_scan_evidence else 0',
            self.runner,
        )


if __name__ == "__main__":
    unittest.main()
