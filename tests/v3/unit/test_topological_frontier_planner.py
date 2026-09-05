from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.integration.mtare_handoff import MTAReGlobalHandoff
from mtare_topo.planning.topological_frontier import (
    FrontierIdentity,
    RuleBasedTopologicalFrontierPlanner,
    TopologicalPlannerConfig,
)


def node(node_id, xyz, stubs):
    return {"id": node_id, "xyz_m": xyz, "exit_stubs": stubs}


def edge(first, second, length):
    return {
        "id": first,
        "from": first,
        "to": second,
        "kind": "verified_traversed",
        "minimum_traversed_length_m": length,
        "traversals": [{"start_frame": 0, "end_frame": 1}],
    }


class TopologicalFrontierPlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = RuleBasedTopologicalFrontierPlanner(
            TopologicalPlannerConfig(4.0, 20.0, 0.25, 0.05)
        )

    def test_current_frontier_emits_heading_lookahead(self):
        graph = {"current_node": 0, "nodes": [node(0, [0, 0, 0], [
            {"state": "observed", "confidence": 0.8, "heading_world_deg": 90}
        ])], "edges": []}
        target = self.planner.select_target(graph, robot_xyz_m=[1, 2, 3])
        self.assertEqual(target.status, "TARGET")
        self.assertEqual(target.mode, "frontier_exit")
        self.assertAlmostEqual(target.waypoint_xyz_m[0], 1.0)
        self.assertAlmostEqual(target.waypoint_xyz_m[1], 6.0)
        self.assertEqual(target.waypoint_xyz_m[2], 3.0)

    def test_remote_frontier_routes_over_verified_next_hop(self):
        graph = {
            "current_node": 0,
            "nodes": [
                node(0, [0, 0, 0], []),
                node(1, [10, 0, 0], []),
                node(2, [20, 0, 0], [{"state": "observed", "confidence": 1.0, "heading_world_deg": 0}]),
            ],
            "edges": [edge(0, 1, 10), edge(1, 2, 10)],
        }
        target = self.planner.select_target(graph, robot_xyz_m=[0, 0, 0])
        self.assertEqual(target.mode, "graph_backtrack")
        self.assertEqual(target.graph_path_node_ids, (0, 1, 2))
        self.assertEqual(target.next_hop_node_id, 1)
        self.assertEqual(target.waypoint_xyz_m, (4.0, 0.0, 0.0))

    def test_utility_and_retry_penalty_choose_deterministically(self):
        graph = {"current_node": 0, "nodes": [node(0, [0, 0, 0], [
            {"state": "observed", "confidence": 0.8, "heading_world_deg": 0},
            {"state": "observed", "confidence": 0.8, "heading_world_deg": 180},
        ])], "edges": []}
        first = self.planner.select_target(graph, robot_xyz_m=[0, 0, 0])
        self.assertEqual(first.frontier, FrontierIdentity(0, 0))
        penalized = self.planner.select_target(
            graph,
            robot_xyz_m=[0, 0, 0],
            retry_counts={FrontierIdentity(0, 0): 1},
        )
        self.assertEqual(penalized.frontier, FrontierIdentity(0, 1))

    def test_only_trace_verified_edges_may_be_routed(self):
        graph = {
            "current_node": 0,
            "nodes": [node(0, [0, 0, 0], []), node(1, [1, 0, 0], [])],
            "edges": [{"from": 0, "to": 1, "kind": "predicted", "traversals": []}],
        }
        with self.assertRaises(ValueError):
            self.planner.select_target(graph, robot_xyz_m=[0, 0, 0])

    def test_complete_is_distinct_from_unreachable(self):
        complete = {"current_node": 0, "nodes": [node(0, [0, 0, 0], [])], "edges": []}
        self.assertEqual(self.planner.select_target(complete, robot_xyz_m=[0, 0, 0]).status, "CANDIDATE_COMPLETE")
        unreachable = {
            "current_node": 0,
            "nodes": [node(0, [0, 0, 0], []), node(1, [2, 0, 0], [
                {"state": "observed", "confidence": 1.0, "heading_world_deg": 0}
            ])],
            "edges": [],
        }
        self.assertEqual(self.planner.select_target(unreachable, robot_xyz_m=[0, 0, 0]).status, "NO_REACHABLE_FRONTIER")

    def test_mtare_handoff_contract(self):
        graph = {"current_node": 0, "nodes": [node(0, [0, 0, 0], [
            {"state": "observed", "confidence": 1.0, "heading_world_deg": 0}
        ])], "edges": []}
        target = self.planner.select_target(graph, robot_xyz_m=[0, 0, 0])
        handoff = MTAReGlobalHandoff()
        self.assertIsNone(handoff.observe_free_path_count(0))
        self.assertEqual(handoff.observe_free_path_count(0), 8.0)
        output = handoff.package(target, stamp_sec=2.0, cycle_started_monotonic=time.monotonic())
        self.assertEqual(output.waypoint["frame_id"], "map")
        self.assertFalse(output.exploration_finish)
        self.assertGreaterEqual(output.runtime_sec, 0.0)

    def test_completion_requires_explicit_stable_confirmation(self):
        graph = {"current_node": 0, "nodes": [node(0, [0, 0, 0], [])], "edges": []}
        target = self.planner.select_target(graph, robot_xyz_m=[0, 0, 0])
        handoff = MTAReGlobalHandoff()
        unconfirmed = handoff.package(
            target, stamp_sec=1.0, cycle_started_monotonic=time.monotonic()
        )
        self.assertFalse(unconfirmed.exploration_finish)
        confirmed = handoff.package(
            target,
            stamp_sec=1.0,
            cycle_started_monotonic=time.monotonic(),
            completion_confirmed=True,
        )
        self.assertTrue(confirmed.exploration_finish)


if __name__ == "__main__":
    unittest.main()
