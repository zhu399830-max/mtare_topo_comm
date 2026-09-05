import unittest

import numpy as np

from mtare_topo.data.cano_phase2_evidence_audit import (
    compare_scan_replay,
    event_coverage,
    frame_failure_reasons,
    selection_identity_audit,
)


class Phase2EvidenceAuditTest(unittest.TestCase):
    def test_scan_replay_requires_masks_and_ranges(self):
        ranges = np.ones((16, 720), dtype=np.float32)
        valid = np.ones((16, 720), dtype=np.uint8)
        self.assertTrue(compare_scan_replay(ranges, valid, ranges.copy(), valid.copy(), ranges.copy(), valid.copy())["passed"])
        changed = ranges.copy(); changed[0, 0] += 1e-3
        self.assertFalse(compare_scan_replay(ranges, valid, changed, valid, ranges, valid)["passed"])
        invalid = valid.copy(); invalid[0, 0] = 0
        self.assertFalse(compare_scan_replay(ranges, valid, ranges, invalid, ranges, valid)["passed"])

    def test_failure_reasons_are_explicit(self):
        evaluation = {"branch_count": 2, "minimum_horizontal_clearance_m": 0.7, "branch_los": [True, False]}
        reasons = frame_failure_reasons(evaluation, {"passed": False})
        self.assertEqual(reasons, ["minimum_horizontal_clearance_below_0p8m", "representative_branch_los_failed", "independent_scene_replay_failed"])

    def test_selection_identity_is_order_sensitive(self):
        sealed = [{"cluster_id":"a","primary_role":"interior"},{"cluster_id":"b","primary_role":"junction"}]
        self.assertTrue(selection_identity_audit(sealed, sealed)["passed"])
        result = selection_identity_audit(list(reversed(sealed)), sealed)
        self.assertFalse(result["passed"]); self.assertEqual(result["first_mismatch_index"], 0)

    def test_event_coverage_respects_primary_role(self):
        rows = [
            {"primary_role":"junction","junction_event_ids":["j1"],"terminal_event_ids":[]},
            {"primary_role":"terminal","junction_event_ids":["j2"],"terminal_event_ids":["t1"]},
        ]
        result = event_coverage(rows)
        self.assertEqual(result["junction_event_ids"], ["j1"])
        self.assertEqual(result["terminal_event_ids"], ["t1"])


if __name__ == "__main__":
    unittest.main()
