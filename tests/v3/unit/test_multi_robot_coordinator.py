"""CPU-only contracts for the ROS-free multi-robot coordinator runtime."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.planning.multi_robot_coordinator import (  # noqa: E402
    MultiRobotCoordinatorRuntime,
)


def _snapshot(*, headings: tuple[float, ...] = (0.0, 180.0)) -> dict:
    """Return a minimal valid local causal graph with observed exits."""
    return {
        "schema_version": "causal_topometric_graph_v2",
        "nodes": [
            {
                "id": 0,
                "node_kind": "anchor",
                "xyz_m": [0.0, 0.0, 0.0],
                "role": "interior",
                "role_probabilities_mean": [1.0, 0.0, 0.0],
                "exit_headings_world_deg": list(headings),
                "exit_stubs": [
                    {"heading_world_deg": h, "state": "observed", "confidence": 0.9}
                    for h in headings
                ],
                "confidence_mean": 0.9,
                "observation_count": 1,
            }
        ],
        "edges": [],
        "current_node": 0,
    }


def _runtime(*, active_r1: bool = True) -> MultiRobotCoordinatorRuntime:
    runtime = MultiRobotCoordinatorRuntime()
    runtime.ingest_robot_snapshot(
        robot_id="r0", revision=1, stamp_sec=1.0, snapshot=_snapshot()
    )
    runtime.ingest_robot_snapshot(
        robot_id="r1", revision=1, stamp_sec=1.0, snapshot=_snapshot(), active=active_r1
    )
    return runtime


class MultiRobotCoordinatorRuntimeTests(unittest.TestCase):
    def test_fused_graph_has_unique_one_to_one_assignments(self) -> None:
        runtime = _runtime()

        decision = runtime.allocate(now_sec=2.0)
        assigned = [item.frontier_id for item in decision.assignments if item.frontier_id is not None]

        self.assertEqual(runtime.snapshot()["graph"]["node_count"], 1)
        self.assertEqual(runtime.snapshot()["graph"]["frontier_count"], 2)
        self.assertEqual(len(assigned), 2)
        self.assertEqual(len(set(assigned)), 2)
        self.assertEqual(decision.active_robot_count, 2)

    def test_valid_leases_are_preserved_across_repeated_allocate(self) -> None:
        runtime = _runtime()

        first = runtime.allocate(now_sec=2.0)
        second = runtime.allocate(now_sec=3.0)

        self.assertEqual(first.assignments, second.assignments)
        self.assertEqual(second.preserved_lease_count, 2)
        self.assertEqual(second.new_assignment_count, 0)
        self.assertEqual(runtime.snapshot()["decision_trace"][-1]["preserved_lease_count"], 2)

    def test_failed_execution_increments_retry_and_changes_next_choice(self) -> None:
        runtime = MultiRobotCoordinatorRuntime()
        runtime.ingest_robot_snapshot(
            robot_id="r0", revision=1, stamp_sec=1.0, snapshot=_snapshot()
        )

        first = runtime.allocate(now_sec=2.0)
        first_id = next(item.frontier_id for item in first.assignments if item.frontier_id)
        report = runtime.report_execution(
            robot_id="r0", frontier_id=first_id, success=False, stamp_sec=2.5
        )
        second = runtime.allocate(now_sec=3.0)
        second_id = next(item.frontier_id for item in second.assignments if item.frontier_id)

        self.assertEqual(report["retry_count"], 1)
        self.assertNotEqual(second_id, first_id)
        self.assertEqual(runtime.snapshot()["retry_counts"], [
            {"robot_id": "r0", "frontier_id": first_id, "count": 1}
        ])

    def test_success_clears_assignment_and_marks_frontier_complete(self) -> None:
        runtime = _runtime()
        decision = runtime.allocate(now_sec=2.0)
        assignment = next(item for item in decision.assignments if item.robot_id == "r0")

        runtime.report_execution(robot_id="r0", frontier_id=assignment.frontier_id, success=True)
        state = runtime.snapshot()

        self.assertNotIn("r0", state["active_leases"])
        self.assertIn(assignment.frontier_id, state["completed_frontiers"])
        self.assertEqual(state["execution_trace"][-1]["success"], True)

    def test_inactive_robot_is_not_assigned(self) -> None:
        runtime = _runtime(active_r1=False)

        decision = runtime.allocate(now_sec=2.0)
        by_robot = {item.robot_id: item for item in decision.assignments}

        self.assertIsNone(by_robot["r1"].frontier_id)
        self.assertEqual(by_robot["r1"].reason, "robot_inactive")
        self.assertEqual(decision.active_robot_count, 1)

    def test_same_inputs_produce_same_decision_and_audit_trace(self) -> None:
        first = _runtime()
        second = _runtime()

        first_decision = first.allocate(now_sec=2.0)
        second_decision = second.allocate(now_sec=2.0)

        self.assertEqual(first_decision.to_dict(), second_decision.to_dict())
        self.assertEqual(first.snapshot(), second.snapshot())
        state = first.snapshot()
        self.assertGreater(state["graph"]["received_bytes_total"], 0)
        self.assertGreater(state["decision_trace"][0]["allocation_payload_bytes"], 0)
        self.assertEqual(state["schema_version"], "multi_robot_coordinator_runtime_v1")


if __name__ == "__main__":
    unittest.main()
