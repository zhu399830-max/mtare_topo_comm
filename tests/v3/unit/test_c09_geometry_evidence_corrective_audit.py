import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.v3.execute_cano_c09_geometry_evidence_corrective_audit_v1 import physical_patch_contract


class C09GeometryEvidenceCorrectiveAuditTest(unittest.TestCase):
    def test_physical_layer_patch_count_is_not_directed_support_arc_count(self):
        windows = [{"world":"S","node_id":"n","incident_tunnel_ids":[3,7]}]
        patches = []
        for tunnel in (3, 7):
            for resolution in (0.1, 0.05, 0.025):
                patches.append({
                    "world":"S","node_id":"n","layer_tunnel_id":tunnel,"resolution_m":resolution,
                    "artifact":"RUN_STATE.json","degenerate_triangle_count":0,"nonmanifold_edge_count":0,
                    "seam":{"passed":True},"transition_semantics":{"passed":True},
                })
        result = physical_patch_contract(windows, patches)
        self.assertTrue(result["passed"])
        self.assertEqual(result["physical_layer_count"], 2)
        self.assertEqual(result["expected_patch_count"], 6)

    def test_real_v1r2_manifests_prove_426(self):
        root = Path(__file__).resolve().parents[3]
        run = root / "results/gate4_topology/gate4_20260819_cano_c09_route_conditioned_geometry_qualification_v1r2_seed0"
        windows = json.loads((run / "artifacts/window_manifest.json").read_text())
        patches = json.loads((run / "artifacts/patch_provenance_manifest.json").read_text())
        result = physical_patch_contract(windows, patches)
        self.assertTrue(result["passed"])
        self.assertEqual(result["window_count"], 71)
        self.assertEqual(result["physical_layer_count"], 142)
        self.assertEqual(result["observed_patch_count"], 426)
        self.assertEqual(result["physical_layers_per_window"], {1: 3, 2: 65, 3: 3})

    def test_v1r_real_main_path_accepts_list_root_manifests(self):
        root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root / "tools/v3")))
        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            for directory in ("artifacts", "metrics"):
                (run / directory).mkdir()
            completed = subprocess.run(
                [
                    "/home/zeng-workstation/anaconda3/bin/python",
                    str(root / "tools/v3/execute_cano_c09_geometry_evidence_corrective_audit_v1r.py"),
                    "--run-dir", str(run),
                ],
                cwd=root, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=30, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            summary = json.loads((run / "metrics/summary.json").read_text())
            self.assertEqual(summary["overall_status"], "PASS_CANO_C09_GEOMETRY_EVIDENCE_CORRECTIVE_AUDIT_V1R")
            self.assertEqual(summary["visual_patch_count"], 426)
            self.assertEqual(summary["frames"], 15833)


if __name__ == "__main__":
    unittest.main()
