import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from execute_cano_c09_route_conditioned_geometry_qualification_v1 import materialize_trajectories
import run_cano_c09_route_conditioned_geometry_qualification_v1 as c09_runner
import run_cano_c09_route_conditioned_geometry_qualification_v1r2 as c09_v1r2_runner


class C09GeometryQualificationTests(unittest.TestCase):
    def test_v1r_frozen_input_paths_exist_and_match_hashes(self):
        root = Path(__file__).resolve().parents[3]
        spec_path = root / "configs/v3/gate4/cano_c09_route_conditioned_geometry_qualification_v1r.json"
        if not spec_path.is_file():
            self.skipTest("V1R run spec has not been frozen yet")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        for relative, expected in spec["frozen_inputs"].items():
            path = root / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, relative)

    def test_v1r2_is_a_full_new_run_with_shutdown_inhibition(self):
        root = Path(__file__).resolve().parents[3]
        spec_path = root / "configs/v3/gate4/cano_c09_route_conditioned_geometry_qualification_v1r2.json"
        if not spec_path.is_file():
            self.skipTest("V1R2 run spec has not been frozen yet")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        command = spec["command"]
        self.assertEqual(command[0], "/usr/bin/systemd-inhibit")
        self.assertIn("--what=sleep:shutdown", command)
        self.assertIn("--mode=block", command)
        self.assertIn("/usr/bin/timeout", command)
        self.assertIn("90000s", command)
        self.assertIn("tools/v3/run_cano_c09_route_conditioned_geometry_qualification_v1r2.py", command)
        self.assertTrue(command[-1].endswith("gate4_20260819_cano_c09_route_conditioned_geometry_qualification_v1r2_seed0"))
        frozen_paths = set(spec["frozen_inputs"])
        self.assertFalse(any("geometry_qualification_v1r_seed0" in path for path in frozen_paths))

    def test_v1r2_frozen_input_paths_exist_and_match_hashes(self):
        root = Path(__file__).resolve().parents[3]
        spec_path = root / "configs/v3/gate4/cano_c09_route_conditioned_geometry_qualification_v1r2.json"
        if not spec_path.is_file():
            self.skipTest("V1R2 run spec has not been frozen yet")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        for relative, expected in spec["frozen_inputs"].items():
            path = root / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, relative)

    def test_v1r2_runner_identity_and_resource_profile(self):
        # The wrapper assignments are intentionally simple and are also frozen by hash.
        source = Path(c09_v1r2_runner.__file__).read_text(encoding="utf-8")
        self.assertIn("gate4_20260819_cano_c09_route_conditioned_geometry_qualification_v1r2_seed0", source)
        self.assertIn("execute_cano_c09_route_conditioned_geometry_qualification_v1r2.py", source)
        c09_runner.configure_core()
        self.assertEqual(c09_runner.core.TIME_LIMIT_SECONDS, 24 * 60 * 60)
        self.assertEqual(c09_runner.core.RAM_LIMIT_BYTES, 4 * 1024**3)
        self.assertEqual(c09_runner.core.DISK_LIMIT_BYTES, 8 * 1024**3)

    def test_materializer_matches_readonly_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "artifacts").mkdir()
            (run / "metrics").mkdir()
            suite = materialize_trajectories(run)
            summary = json.loads((run / "metrics/trajectory_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(len(suite), 10)
        self.assertEqual(sum(frames for _, frames in suite), 15833)
        self.assertEqual(summary["edges"], 1027)
        self.assertEqual(summary["directed_traversals"], 2054)
        self.assertAlmostEqual(summary["route_length_m"], 31654.75249119315)

    def test_runner_profile_freezes_counts_and_24_hour_cap(self):
        c09_runner.configure_core()
        self.assertEqual(c09_runner.core.TIME_LIMIT_SECONDS, 24 * 60 * 60)
        self.assertEqual(c09_runner.core.RAM_LIMIT_BYTES, 4 * 1024**3)
        self.assertEqual(c09_runner.core.DISK_LIMIT_BYTES, 8 * 1024**3)
        self.assertEqual(c09_runner.core.EXPECTED_SUMMARY["frames"], 15833)
        self.assertEqual(c09_runner.core.EXPECTED_SUMMARY["visual_patch_count"], 639)
        self.assertEqual(c09_runner.core.EXPECTED_SUMMARY["c09_worlds_read"], 10)


if __name__ == "__main__":
    unittest.main()
