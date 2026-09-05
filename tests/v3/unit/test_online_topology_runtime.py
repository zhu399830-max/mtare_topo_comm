from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.integration.mtare_handoff_v2 import MTAReGlobalHandoffV2
from mtare_topo.integration.online_topology_runtime import OnlineTopologyPlannerRuntime, SemanticPrediction
from mtare_topo.planning.topological_frontier import TopologicalPlannerConfig
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig


def prediction(headings=(0, 360), role=0, count=2):
    logits = np.full(720, -20.0, dtype=np.float32)
    for heading in headings:
        logits[int(round(heading * 2)) % 720] = 20.0
    counts = np.zeros(6, dtype=np.float32); counts[count - 1] = 1.0
    roles = np.zeros(3, dtype=np.float32); roles[role] = 1.0
    return SemanticPrediction(logits, counts, roles, np.ones(128, dtype=np.float32) / np.sqrt(128.0))


class OnlineTopologyRuntimeTests(unittest.TestCase):
    def runtime(self):
        return OnlineTopologyPlannerRuntime(
            CausalGraphConfig(stable_frames=2, minimum_event_travel_m=8.0, loop_merge_radius_m=6.0,
                              branch_heading_merge_deg=20.0, turn_event_deg=45.0,
                              distance_anchor_interval_m=20.0),
            TopologicalPlannerConfig(4.0, 20.0, 0.25, 0.05),
        )

    def test_handoff_v2_resets_after_each_empty_pair(self):
        handoff = MTAReGlobalHandoffV2()
        self.assertIsNone(handoff.observe_free_path_count(0))
        self.assertEqual(handoff.observe_free_path_count(0), 8.0)
        self.assertIsNone(handoff.observe_free_path_count(0))
        self.assertEqual(handoff.observe_free_path_count(0), 8.0)
        self.assertIsNone(handoff.observe_free_path_count(3))

    def test_candidate_complete_packages_current_position_as_hold(self):
        runtime = self.runtime()
        empty = prediction(headings=(), role=2, count=1)
        cycle = runtime.update(stamp_sec=1.0, sensor_xyz_m=(2, 3, 4),
                               sensor_orientation_xyzw=(0, 0, 0, 1), prediction=empty,
                               cycle_started_monotonic=time.monotonic())
        self.assertEqual(cycle.target.status, "CANDIDATE_COMPLETE")
        self.assertEqual(cycle.handoff.waypoint["xyz_m"], [2.0, 3.0, 4.0])
        self.assertTrue(cycle.handoff.waypoint["hold"])
        self.assertFalse(cycle.handoff.exploration_finish)

    def test_route_arc_graph_and_handoff_are_causal(self):
        runtime = self.runtime()
        first = runtime.update(stamp_sec=1.0, sensor_xyz_m=(0, 0, 1),
                               sensor_orientation_xyzw=(0, 0, 0, 1), prediction=prediction(),
                               cycle_started_monotonic=time.monotonic())
        second = runtime.update(stamp_sec=2.0, sensor_xyz_m=(3, 4, 1),
                                sensor_orientation_xyzw=(0, 0, 0, 1), prediction=prediction(),
                                cycle_started_monotonic=time.monotonic())
        self.assertEqual(first.route_arc_m, 0.0)
        self.assertEqual(second.route_arc_m, 5.0)
        self.assertEqual(first.target.status, "TARGET")
        self.assertEqual(first.handoff.waypoint["frame_id"], "map")
        self.assertNotIn("evaluator_gt_edge_id", str(runtime.snapshot()))

    def test_one_rejection_episode_penalizes_active_frontier(self):
        runtime = self.runtime()
        cycle = runtime.update(stamp_sec=1.0, sensor_xyz_m=(0, 0, 0),
                               sensor_orientation_xyzw=(0, 0, 0, 1),
                               prediction=prediction(headings=(0, 180)),
                               cycle_started_monotonic=time.monotonic())
        first = cycle.target.frontier
        self.assertIsNone(runtime.observe_free_path_count(0))
        self.assertEqual(runtime.observe_free_path_count(0), 8.0)
        later = runtime.update(stamp_sec=2.0, sensor_xyz_m=(0.1, 0, 0),
                               sensor_orientation_xyzw=(0, 0, 0, 1),
                               prediction=prediction(headings=(0, 180)),
                               cycle_started_monotonic=time.monotonic())
        self.assertEqual(runtime.retry_ledger.counts()[first], 1)
        self.assertNotEqual(later.target.frontier, first)

    def test_rejects_duplicate_timestamp_and_bad_probabilities(self):
        runtime = self.runtime()
        runtime.update(stamp_sec=1.0, sensor_xyz_m=(0, 0, 0),
                       sensor_orientation_xyzw=(0, 0, 0, 1), prediction=prediction())
        with self.assertRaises(ValueError):
            runtime.update(stamp_sec=1.0, sensor_xyz_m=(0, 0, 0),
                           sensor_orientation_xyzw=(0, 0, 0, 1), prediction=prediction())
        bad = prediction(); bad.count_probabilities[0] = 0.5
        fresh = self.runtime()
        with self.assertRaises(ValueError):
            fresh.update(stamp_sec=1.0, sensor_xyz_m=(0, 0, 0),
                         sensor_orientation_xyzw=(0, 0, 0, 1), prediction=bad)

    def test_explicit_b0_count_is_not_truncated_to_neural_classes(self):
        runtime = self.runtime()
        base = prediction(headings=tuple(range(0, 360, 45)), role=1, count=6)
        explicit = SemanticPrediction(
            base.direction_logits, base.count_probabilities, base.role_probabilities,
            base.z_role, branch_count_override=8,
        )
        runtime.update(stamp_sec=1.0, sensor_xyz_m=(0, 0, 0),
                       sensor_orientation_xyzw=(0, 0, 0, 1), prediction=explicit)
        self.assertEqual(runtime.snapshot()["graph"]["nodes"][0]["branch_count"], 8)

    def test_explicit_b0_zero_count_is_preserved(self):
        runtime = self.runtime()
        base = prediction(headings=(0,), role=2, count=1)
        explicit = SemanticPrediction(
            base.direction_logits, base.count_probabilities, base.role_probabilities,
            base.z_role, branch_count_override=0,
        )
        runtime.update(stamp_sec=1.0, sensor_xyz_m=(0, 0, 0),
                       sensor_orientation_xyzw=(0, 0, 0, 1), prediction=explicit)
        self.assertEqual(runtime.snapshot()["graph"]["nodes"][0]["branch_count"], 0)


if __name__ == "__main__":
    unittest.main()
