import unittest

from mtare_topo.evaluation.topology_frontier_attempt_evidence import (
    audit_frontier_attempt_evidence,
)


def _node(node_id, xyz, headings):
    return {
        "id": node_id,
        "xyz_m": xyz,
        "exit_stubs": [
            {"heading_world_deg": heading, "state": "observed", "confidence": 1.0}
            for heading in headings
        ],
    }


def _row(frame, arc, node, reason, *, stub=0, mode="frontier_exit"):
    target = {"mode": mode}
    if mode == "frontier_exit":
        target["frontier"] = {"node_id": node, "stub_index": stub}
    return {
        "frame_index": frame,
        "route_arc_m": arc,
        "graph_update": {"node_id": node, "reason": reason},
        "target": target,
    }


class FrontierAttemptEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.graph = {
            "nodes": [
                _node(0, [0.0, 0.0, 0.0], [0.0, 90.0]),
                _node(1, [10.0, 0.0, 0.0], [180.0]),
            ]
        }

    def test_classifies_matching_divergent_and_same_node_attempts(self):
        rows = [
            _row(0, 0.0, 0, "route_start", stub=0),
            _row(1, 8.0, 1, "distance_anchor", stub=0),
            _row(2, 16.0, 0, "distance_anchor", stub=1),
            _row(3, 24.0, 1, "distance_anchor", stub=0),
            _row(4, 32.0, 0, "loop_merge", stub=1),
            _row(5, 40.0, 0, "loop_merge", stub=1),
        ]
        audit = audit_frontier_attempt_evidence(rows, self.graph)
        self.assertEqual(audit["classified_attempt_event_count"], 5)
        self.assertEqual(audit["outcome_counts"]["matched_verified_departure"], 3)
        self.assertEqual(audit["outcome_counts"]["divergent_verified_departure"], 1)
        self.assertEqual(audit["outcome_counts"]["same_node_loop_merge"], 1)
        self.assertEqual(audit["nonmatching_attempt_event_count"], 2)
        self.assertEqual(audit["events"][0]["target_run_start_frame"], 0)
        self.assertEqual(audit["events"][0]["target_run_frames_before_event"], 1)
        self.assertEqual(audit["events"][0]["target_run_route_arc_m_before_event"], 8.0)
        self.assertFalse(audit["uses_evaluator_gt"])

    def test_ignores_no_event_and_graph_backtrack(self):
        rows = [
            _row(0, 0.0, 0, "route_start", stub=0),
            _row(1, 1.0, 0, "no_event", stub=0),
            _row(2, 2.0, 0, "no_event", mode="graph_backtrack"),
        ]
        audit = audit_frontier_attempt_evidence(rows, self.graph)
        self.assertEqual(audit["classified_attempt_event_count"], 0)

    def test_rejects_nonmonotonic_trace(self):
        rows = [
            _row(0, 1.0, 0, "route_start", stub=0),
            _row(1, 0.0, 0, "no_event", stub=0),
        ]
        with self.assertRaisesRegex(ValueError, "route arc"):
            audit_frontier_attempt_evidence(rows, self.graph)


if __name__ == "__main__":
    unittest.main()
