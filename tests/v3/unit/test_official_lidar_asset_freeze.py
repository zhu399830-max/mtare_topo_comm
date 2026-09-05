from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "tools/v3/run_official_lidar_asset_freeze.py"
AUDITOR = PROJECT_ROOT / "tools/v3/isaac/audit_frozen_usda.py"


class OfficialLidarAssetFreezeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = RUNNER.read_text(encoding="utf-8")
        self.auditor = AUDITOR.read_text(encoding="utf-8")

    def test_exact_official_asset_url_is_frozen(self) -> None:
        self.assertIn("omniverse-content-production.s3-us-west-2.amazonaws.com", self.runner)
        self.assertIn("Isaac/Sensors/NVIDIA/Example_Rotary.usda", self.runner)
        self.assertIn('parsed.scheme != "https"', self.runner)

    def test_download_is_single_bounded_and_does_not_follow_redirects(self) -> None:
        self.assertIn('"request_count": 1', self.runner)
        self.assertIn('"redirect_following": False', self.runner)
        self.assertIn('"--max-redirs"', self.runner)
        self.assertIn('"0"', self.runner)
        self.assertIn("MAX_BYTES = 10 * 1024 * 1024", self.runner)
        self.assertNotIn('"--location"', self.runner)
        self.assertNotIn('"--retry"', self.runner)

    def test_auditor_records_all_usd_dependency_surfaces(self) -> None:
        for token in (
            "subLayerPaths",
            "GetExternalReferences",
            "GetCompositionAssetDependencies",
            "GetExternalAssetDependencies",
            "Usd.Stage.LoadNone",
        ):
            self.assertIn(token, self.auditor)

    def test_audit_container_is_network_disabled_and_has_no_gpu(self) -> None:
        self.assertIn('"--network",\n            "none"', self.runner)
        self.assertNotIn('"--gpus"', self.runner)
        self.assertIn('"--output -"', self.runner)
        self.assertNotIn("SimulationApp", self.runner)
        self.assertNotIn("Lidar", self.auditor)
        self.assertNotIn("Writer", self.auditor)

    def test_counts_forbid_project_data_and_training(self) -> None:
        for token in (
            '"gpu_runs": 0',
            '"isaac_simulation_runs": 0',
            '"writer_runs": 0',
            '"cano_worlds": 0',
            '"saved_point_samples": 0',
            '"formal_dataset_samples": 0',
            '"labels": 0',
            '"training_samples": 0',
            '"models": 0',
        ):
            self.assertIn(token, self.runner)


if __name__ == "__main__":
    unittest.main()
