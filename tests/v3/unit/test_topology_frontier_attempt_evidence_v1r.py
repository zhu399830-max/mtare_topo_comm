from __future__ import annotations

import copy
import unittest

from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r import (
    audit_frontier_attempt_evidence_v1r,
)
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig, CausalTopometricGraphV2


def _target(mode, node, stub, path):
    return {
        "mode": mode,
        "frontier": {"node_id": node, "stub_index": stub},
        "frontier_node_id": node,
        "next_hop_node_id": path[1] if len(path) > 1 else node,
        "graph_path_node_ids": path,
    }


def _row(frame, arc, node, reason, target):
    return {
        "frame_index": frame,
        "route_arc_m": arc,
        "graph_update": {
            "frame_index": frame,
            "route_arc_m": arc,
            "node_id": node,
            "reason": reason,
        },
        "target": target,
    }


def _fixture():
    nodes = [
        {
            "id": 0,
            "xyz_m": [0.0, 0.0, 0.0],
            "exit_stubs": [
                {"heading_world_deg": 0.0, "state": "traversed"},
                {"heading_world_deg": 90.0, "state": "observed"},
            ],
        },
        {
            "id": 1,
            "xyz_m": [10.0, 0.0, 0.0],
            "exit_stubs": [
                {"heading_world_deg": 180.0, "state": "traversed"},
            ],
        },
    ]
    graph = {
        "schema_version": "causal_topometric_graph_v2",
        "nodes": nodes,
        "edges": [{
            "id": 0,
            "from": 0,
            "to": 1,
            "kind": "verified_traversed",
            "traversals": [
                {
                    "from": 0,
                    "to": 1,
                    "start_frame": 0,
                    "end_frame": 1,
                    "start_route_arc_m": 0.0,
                    "end_route_arc_m": 8.0,
                },
                {
                    "from": 1,
                    "to": 0,
                    "start_frame": 1,
                    "end_frame": 2,
                    "start_route_arc_m": 8.0,
                    "end_route_arc_m": 16.0,
                },
            ],
        }],
    }
    rows = [
        _row(0, 0.0, 0, "route_start", _target("frontier_exit", 0, 0, [0])),
        _row(1, 8.0, 1, "distance_anchor", _target("graph_backtrack", 0, 1, [1, 0])),
        _row(2, 16.0, 0, "distance_anchor", _target("frontier_exit", 0, 1, [0])),
        _row(3, 24.0, 0, "loop_merge", _target("frontier_exit", 0, 1, [0])),
    ]
    return rows, graph


class FrontierAttemptEvidenceV1RTest(unittest.TestCase):
    def test_binds_attempt_departure_to_exact_verified_traversal(self):
        rows, graph = _fixture()
        value = audit_frontier_attempt_evidence_v1r(rows, graph)
        self.assertEqual(value["schema_version"], "topology_frontier_attempt_evidence_v1r")
        self.assertEqual(value["verified_graph_transition_count"], 2)
        self.assertEqual(value["verified_attempt_departure_count"], 1)
        self.assertEqual(value["events"][0]["verified_edge_id"], 0)
        self.assertEqual(value["events"][1]["outcome"], "same_node_loop_merge")

    def test_rejects_invalid_stub_identity(self):
        rows, graph = _fixture()
        rows[0]["target"]["frontier"]["stub_index"] = 9
        with self.assertRaisesRegex(ValueError, "stub identity"):
            audit_frontier_attempt_evidence_v1r(rows, graph)

    def test_rejects_graph_update_frame_or_arc_drift(self):
        rows, graph = _fixture()
        rows[1]["graph_update"]["frame_index"] = 7
        with self.assertRaisesRegex(ValueError, "graph update identity"):
            audit_frontier_attempt_evidence_v1r(rows, graph)
        rows, graph = _fixture()
        rows[1]["graph_update"]["route_arc_m"] = 8.5
        with self.assertRaisesRegex(ValueError, "graph update identity"):
            audit_frontier_attempt_evidence_v1r(rows, graph)

    def test_rejects_unverified_transition_or_backtrack_path(self):
        rows, graph = _fixture()
        graph["edges"][0]["traversals"][0]["end_frame"] = 99
        with self.assertRaisesRegex(ValueError, "one exact verified traversal"):
            audit_frontier_attempt_evidence_v1r(rows, graph)
        rows, graph = _fixture()
        rows[1]["target"]["graph_path_node_ids"] = [1, 1, 0]
        with self.assertRaisesRegex(ValueError, "graph_backtrack"):
            audit_frontier_attempt_evidence_v1r(rows, graph)

    def test_frozen_graph_update_is_append_only_for_stub_indices(self):
        graph = CausalTopometricGraphV2(CausalGraphConfig())
        graph.nodes = [{
            "id": 0,
            "exit_stubs": [
                {"heading_world_deg": 0.0, "state": "observed", "confidence": 1.0},
                {"heading_world_deg": 180.0, "state": "observed", "confidence": 1.0},
            ],
            "role_probabilities_mean": [1.0, 0.0, 0.0],
            "confidence_mean": 1.0,
            "z_role_mean": None,
            "observation_count": 1,
        }]
        before = copy.deepcopy(graph.nodes[0]["exit_stubs"])
        graph._update_node(0, [1.0, 0.0, 0.0], [0.0, 90.0, 180.0], 1.0, None, 1)
        self.assertEqual(graph.nodes[0]["exit_stubs"][:2], before)
        self.assertEqual(graph.nodes[0]["exit_stubs"][2]["heading_world_deg"], 90.0)


if __name__ == "__main__":
    unittest.main()
