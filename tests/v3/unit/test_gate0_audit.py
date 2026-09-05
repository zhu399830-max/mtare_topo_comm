from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


TOOLS_ROOT = Path(__file__).resolve().parents[3] / "tools" / "v3"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from validate_gate0_audit import validate_documents


def fixtures() -> tuple[dict, dict, dict, dict]:
    interface = {
        "schema_version": "gate0_interface_contract_v1",
        "decision_status": "DRAFT_FOR_USER_REVIEW",
        "system_boundary": {
            "retained_components": ["local_planner"],
            "replaced_components": ["tare_planner_node"],
        },
        "interfaces": [{
            "id": "global_waypoint",
            "direction": "OUTPUT",
            "topic": "/way_point",
            "message_type": "geometry_msgs/PointStamped",
            "frame": "map",
            "producer": "replacement_global_planner",
            "consumer": "localPlanner",
            "required_for_replacement": True,
            "rate_hz": 1.0,
            "status": "VERIFIED",
            "evidence": ["source:line"],
        }],
    }
    worlds = {
        "schema_version": "gate0_world_inventory_v1",
        "worlds": [{
            "id": name,
            "underground_relevance": "HIGH",
            "asset": {"path": f"worlds/{name}.world"},
            "historical_exposure": ["old_run"] if name.startswith("dev") else [],
            "strict_test_clean": name.startswith("test"),
            "eligibility": "DEVELOPMENT" if name.startswith("dev") else "STRICT_TEST",
            "evidence": ["inventory"],
        } for name in ("dev_a", "dev_b", "dev_c", "test_a", "test_b")],
    }
    benchmark = {
        "schema_version": "gate0_benchmark_proposal_v1",
        "development_worlds": ["dev_a", "dev_b", "dev_c"],
        "strict_test_worlds": [],
        "strict_test_status": "BLOCKED_NO_CLEAN_WORLDS",
        "required_new_strict_test_worlds": 2,
        "protocol": {"seed_control": {"status": "UNVERIFIED"}},
    }
    matrix = {
        "schema_version": "gate0_baseline_rerun_matrix_v1",
        "historical_runs": [{
            "run_id": "old_dev_a",
            "classification": "INSUFFICIENT_EVIDENCE",
            "seed_status": "UNVERIFIED",
            "final_benchmark_eligible": False,
            "evidence": ["old/status.json"],
        }],
        "required_reruns": [{
            "world": "dev_a",
            "authorization_status": "NOT_AUTHORIZED",
        }],
    }
    return interface, worlds, benchmark, matrix


class Gate0AuditValidationTests(unittest.TestCase):
    def test_evidence_complete_blocked_audit_is_valid(self) -> None:
        report = validate_documents(*fixtures())
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["gate_conclusion"], "GATE_MIXED")

    def test_interface_without_evidence_is_rejected(self) -> None:
        interface, worlds, benchmark, matrix = fixtures()
        interface = deepcopy(interface)
        interface["interfaces"][0]["evidence"] = []
        report = validate_documents(interface, worlds, benchmark, matrix)
        self.assertFalse(report["passed"])
        self.assertTrue(any("evidence" in error for error in report["errors"]))

    def test_ready_claim_requires_two_clean_test_worlds(self) -> None:
        interface, worlds, benchmark, matrix = fixtures()
        benchmark = deepcopy(benchmark)
        benchmark["strict_test_status"] = "READY"
        benchmark["strict_test_worlds"] = ["test_a"]
        report = validate_documents(interface, worlds, benchmark, matrix)
        self.assertFalse(report["passed"])
        self.assertTrue(any("at least two" in error for error in report["errors"]))

    def test_exposed_world_cannot_claim_clean(self) -> None:
        interface, worlds, benchmark, matrix = fixtures()
        worlds = deepcopy(worlds)
        worlds["worlds"][0]["strict_test_clean"] = True
        report = validate_documents(interface, worlds, benchmark, matrix)
        self.assertFalse(report["passed"])
        self.assertTrue(any("historical exposure" in error for error in report["errors"]))

    def test_final_baseline_requires_verified_seed(self) -> None:
        interface, worlds, benchmark, matrix = fixtures()
        matrix = deepcopy(matrix)
        matrix["historical_runs"][0]["final_benchmark_eligible"] = True
        report = validate_documents(interface, worlds, benchmark, matrix)
        self.assertFalse(report["passed"])
        self.assertTrue(any("verified seed" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
