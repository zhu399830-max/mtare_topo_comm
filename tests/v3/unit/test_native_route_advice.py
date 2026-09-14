from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest

from mtare_topo.integration.native_route_advice import (
    decode_advice, make_route_advice, validate_route_advice, validate_snapshot,
)


def fixture():
    return dict(schema_version="native_route_snapshot_v1", epoch="session:7",
        stamp_ns=1000, max_age_ns=100, max_extra_cost_units=0,
        source_frame_keys=["registered_scan:900", "registered_scan:1000"],
        candidate_ids=[1, 2], original_route=[0, 1, 2, 0],
        native_edges=[[a, b, 0 if a == b else 10] for a in range(3) for b in range(3)],
        original_cost_units=30)


class NativeRouteAdviceTest(unittest.TestCase):
    def setUp(self):
        self.s = fixture()
        self.a = make_route_advice(self.s, [0, 2, 1, 0], evidence_refs=["verification:1"])

    def reject(self, key, value, reason=None):
        a = deepcopy(self.a)
        a[key] = value
        before = deepcopy(self.s)
        result = validate_route_advice(self.s, a, now_ns=1001)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["route"], self.s["original_route"])
        self.assertEqual(self.s, before)
        if reason:
            self.assertEqual(result["reason"], reason)

    def test_valid_reorder_and_input_immutability(self):
        before = deepcopy(self.s)
        result = validate_route_advice(self.s, self.a, now_ns=1001)
        self.assertTrue(result["changed"])
        self.assertEqual(result["route"], [0, 2, 1, 0])
        self.assertEqual(before, self.s)
        result["route"][1] = 999
        self.assertEqual(self.a["route"], [0, 2, 1, 0])

    def test_missing_preserves_native_route(self):
        result = validate_route_advice(self.s, None, now_ns=1001)
        self.assertEqual(result["reason"], "NO_ADVICE")
        self.assertEqual(result["route"], self.s["original_route"])

    def test_identity_needs_no_intervention_evidence(self):
        a = make_route_advice(self.s, self.s["original_route"], evidence_refs=[])
        result = validate_route_advice(self.s, a, now_ns=1001)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["changed"])

    def test_old_epoch(self):
        self.reject("epoch", "session:6", "EPOCH_MISMATCH")

    def test_epoch_reuse_across_session(self):
        self.reject("epoch", "another_session:7", "EPOCH_MISMATCH")

    def test_wrong_snapshot_stamp(self):
        self.reject("snapshot_stamp_ns", 999, "STAMP_MISMATCH")

    def test_future_and_expired_advice(self):
        for now in [999, 1101]:
            result = validate_route_advice(self.s, self.a, now_ns=now)
            self.assertEqual(result["reason"], "ADVICE_NOT_FRESH")
        self.assertTrue(validate_route_advice(self.s, self.a, now_ns=1100)["accepted"])

    def test_changed_source_frame(self):
        self.reject("source_frame_keys", ["registered_scan:800", "registered_scan:1000"], "SOURCE_MISMATCH")

    def test_source_order_is_binding(self):
        self.reject("source_frame_keys", list(reversed(self.s["source_frame_keys"])), "SOURCE_MISMATCH")

    def test_removed_candidate(self):
        self.reject("candidate_ids", [1], "CANDIDATE_SET_MISMATCH")

    def test_duplicate_candidate(self):
        self.reject("candidate_ids", [1, 1, 2])

    def test_changed_endpoint(self):
        self.reject("route", [1, 0, 2, 0], "ENDPOINT_MISMATCH")

    def test_duplicate_or_deleted_route_member(self):
        self.reject("route", [0, 1, 1, 0], "ROUTE_MULTISET_MISMATCH")
        self.reject("route", [0, 1, 0], "ROUTE_MULTISET_MISMATCH")

    def test_unknown_native_edge_rejected(self):
        self.s["native_edges"] = [e for e in self.s["native_edges"] if e[:2] != [0, 2]]
        self.assertEqual(validate_route_advice(self.s, self.a, now_ns=1001)["reason"], "UNREACHABLE_NATIVE_EDGE")

    def test_claimed_cost_not_trusted(self):
        self.reject("claimed_cost_units", 29, "ADVISED_COST_MISMATCH")

    def test_native_cost_budget_not_advice_controlled(self):
        for e in self.s["native_edges"]:
            if e[:2] == [0, 2]:
                e[2] = 11
        self.a["claimed_cost_units"] = 31
        self.assertEqual(validate_route_advice(self.s, self.a, now_ns=1001)["reason"], "COST_BUDGET_EXCEEDED")
        self.s["max_extra_cost_units"] = 1
        self.assertTrue(validate_route_advice(self.s, self.a, now_ns=1001)["accepted"])

    def test_missing_changed_route_evidence(self):
        self.reject("evidence_refs", [], "ADVICE_EVIDENCE")

    def test_advice_cannot_supply_new_edges(self):
        self.reject("native_edges", [], "ADVICE_FIELDS")

    def test_bool_is_not_candidate_id(self):
        self.reject("candidate_ids", [True, 2])

    def test_invalid_native_snapshot_is_reported_not_repaired(self):
        self.s["original_cost_units"] = 29
        with self.assertRaisesRegex(ValueError, "ORIGINAL_COST_MISMATCH"):
            validate_snapshot(self.s)

    def test_duplicate_json_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
            decode_advice('{"epoch":"one","epoch":"two"}')

    def test_nonfinite_json_rejected(self):
        with self.assertRaisesRegex(ValueError, "NONFINITE_JSON"):
            decode_advice('{"cost":NaN}')

    def test_cpp_core_compiles_and_runs_synthetic_only(self):
        root = Path(__file__).resolve().parents[3]
        source = root / "integration/native_structure_bridge/core_contract_test.cpp"
        with tempfile.TemporaryDirectory(prefix="native-route-core-test-") as output:
            binary = Path(output) / "core_test"
            subprocess.run(["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror",
                str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
