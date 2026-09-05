from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import torch

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.deployment.m1d_checkpoint import DEPLOYMENT_SCHEMA, export_m1d_ros_checkpoint
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


class M1DCheckpointExportTests(unittest.TestCase):
    def test_export_removes_path_metadata_and_is_bit_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.pt", root / "deployment.pt"
            torch.save({"epoch": 4, "seed": 2, "mode": "M1D", "config": {"path": root}, "model": StructuralSemanticNet().state_dict()}, source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            audit = export_m1d_ros_checkpoint(source=source, destination=output, expected_source_sha256=source_hash, expected_seed=2)
            payload = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema_version"], DEPLOYMENT_SCHEMA)
            self.assertNotIn("config", payload)
            self.assertTrue(audit["parity"]["bit_exact_same_runtime"])
            self.assertEqual(audit["training_steps"], 0)

    def test_refuses_hash_drift_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.pt"; output = root / "deployment.pt"
            torch.save({"epoch": 1, "seed": 0, "mode": "M1D", "model": StructuralSemanticNet().state_dict()}, source)
            with self.assertRaisesRegex(ValueError, "hash drift"):
                export_m1d_ros_checkpoint(source=source, destination=output, expected_source_sha256="0" * 64, expected_seed=0)
            output.write_bytes(b"occupied")
            with self.assertRaises(FileExistsError):
                export_m1d_ros_checkpoint(source=source, destination=output, expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), expected_seed=0)

    def test_aee_head_adapted_mode_is_preserved_without_tensor_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "adapted.pt", root / "adapted_ros.pt"
            torch.save(
                {"epoch": 3, "seed": 1, "mode": "M1D_AEE_HEAD_ADAPTED_V1", "model": StructuralSemanticNet().state_dict()},
                source,
            )
            report = export_m1d_ros_checkpoint(
                source=source,
                destination=output,
                expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                expected_seed=1,
                expected_mode="M1D_AEE_HEAD_ADAPTED_V1",
            )
            payload = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(payload["mode"], "M1D_AEE_HEAD_ADAPTED_V1")
            self.assertEqual(report["mode"], "M1D_AEE_HEAD_ADAPTED_V1")
            self.assertTrue(report["parity"]["bit_exact_same_runtime"])

    def test_v9_composite_mode_and_runtime_contract_are_preserved(self):
        from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE, COMPOSITE_V9_RUNTIME_CONTRACT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "v9.pt", root / "v9_ros.pt"
            torch.save({"epoch": 3, "seed": 0, "mode": COMPOSITE_V9_MODE,
                        "runtime_contract": COMPOSITE_V9_RUNTIME_CONTRACT,
                        "model": StructuralSemanticNet().state_dict()}, source)
            report = export_m1d_ros_checkpoint(
                source=source, destination=output,
                expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                expected_seed=0, expected_mode=COMPOSITE_V9_MODE,
            )
            payload = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(payload["runtime_contract"], COMPOSITE_V9_RUNTIME_CONTRACT)
            self.assertEqual(report["runtime_contract"], COMPOSITE_V9_RUNTIME_CONTRACT)

    def test_v9_refuses_missing_runtime_contract(self):
        from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "v9.pt"
            torch.save({"epoch": 1, "seed": 0, "mode": COMPOSITE_V9_MODE,
                        "model": StructuralSemanticNet().state_dict()}, source)
            with self.assertRaisesRegex(ValueError, "runtime contract"):
                export_m1d_ros_checkpoint(
                    source=source, destination=root / "out.pt",
                    expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    expected_seed=0, expected_mode=COMPOSITE_V9_MODE,
                )


if __name__ == "__main__":
    unittest.main()
