"""CPU-only contracts for the shared-graph frontier allocator.

These tests deliberately exercise only the allocator's pure data interface.  No
ROS, simulator, sensor, or trajectory data is involved.  The nearest-frontier
comparison is retained as a diagnostic baseline: it is allowed to select the
same frontier for more than one robot, whereas the central allocator must not.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.planning.multi_robot_allocation import (  # noqa: E402
    MultiRobotAllocationConfig,
    RobotAllocationState,
    SharedFrontierTask,
    allocate_shared_frontiers,
    independent_nearest_assignments,
)


def _robots(count: int) -> list[RobotAllocationState]:
    return [RobotAllocationState(f"r{index}") for index in range(count)]


def _frontiers(count: int) -> list[SharedFrontierTask]:
    return [
        SharedFrontierTask(f"f{index}", index, exploration_potential=1.0, confidence=0.9)
        for index in range(count)
    ]


def _costs(robots, frontiers, *, diagonal: float = 1.0, off_diagonal: float = 8.0):
    return {
        (robot.robot_id, frontier.frontier_id): (
            diagonal if robot.robot_id[1:] == frontier.frontier_id[1:] else off_diagonal
        )
        for robot in robots
        for frontier in frontiers
    }


class MultiRobotAllocationTests(unittest.TestCase):
    def test_one_through_four_robots_have_one_to_one_assignments(self) -> None:
        """The central assignment scales while preserving task uniqueness."""
        for count in range(1, 5):
            with self.subTest(robot_count=count):
                robots = _robots(count)
                frontiers = _frontiers(count)
                result = allocate_shared_frontiers(
                    robots, frontiers, _costs(robots, frontiers), now_sec=10.0
                )
                assigned = [item.frontier_id for item in result.assignments if item.frontier_id]
                self.assertEqual(result.active_robot_count, count)
                self.assertEqual(result.assigned_count, count)
                self.assertEqual(len(assigned), len(set(assigned)))
                self.assertEqual(set(assigned), {item.frontier_id for item in frontiers})

    def test_assignment_is_deterministic_under_exact_utility_ties(self) -> None:
        robots = _robots(2)
        frontiers = _frontiers(2)
        equal_costs = {(robot.robot_id, task.frontier_id): 2.0 for robot in robots for task in frontiers}
        first = allocate_shared_frontiers(robots, frontiers, equal_costs, now_sec=3.0)
        second = allocate_shared_frontiers(robots, frontiers, equal_costs, now_sec=3.0)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            [(item.robot_id, item.frontier_id) for item in first.assignments],
            [("r0", "f0"), ("r1", "f1")],
        )

    def test_inactive_robot_is_not_assigned(self) -> None:
        robots = [RobotAllocationState("r0"), RobotAllocationState("r1", active=False)]
        frontiers = _frontiers(2)
        result = allocate_shared_frontiers(robots, frontiers, _costs(robots, frontiers), now_sec=0.0)
        self.assertEqual(result.active_robot_count, 1)
        self.assertEqual(result.assigned_count, 1)
        by_robot = {item.robot_id: item for item in result.assignments}
        self.assertEqual(set(by_robot), {"r0", "r1"})
        self.assertIsNotNone(by_robot["r0"].frontier_id)
        self.assertIsNone(by_robot["r1"].frontier_id)
        self.assertEqual(by_robot["r1"].reason, "robot_inactive")

    def test_unexpired_lease_is_isolated_and_holder_lease_is_renewed(self) -> None:
        robots = _robots(2)
        frontiers = [
            SharedFrontierTask("leased", 1, 1.0, 1.0, "r0", 20.0),
            SharedFrontierTask("free", 2, 1.0, 1.0),
        ]
        costs = {(robot.robot_id, task.frontier_id): 1.0 for robot in robots for task in frontiers}
        result = allocate_shared_frontiers(robots, frontiers, costs, now_sec=10.0)
        by_robot = {item.robot_id: item for item in result.assignments}
        self.assertEqual(by_robot["r0"].frontier_id, "leased")
        self.assertEqual(by_robot["r1"].frontier_id, "free")
        self.assertEqual(by_robot["r0"].lease_expires_sec, 25.0)

        # Once the original lease expires, another robot may acquire it.
        expired = [
            SharedFrontierTask("leased", 1, 1.0, 1.0, "r0", 10.0),
            SharedFrontierTask("free", 2, 1.0, 1.0),
        ]
        later = allocate_shared_frontiers(robots, expired, costs, now_sec=10.0)
        self.assertEqual(later.assigned_count, 2)
        self.assertEqual(len({item.frontier_id for item in later.assignments}), 2)

    def test_low_utility_frontier_is_left_unassigned(self) -> None:
        robots = [RobotAllocationState("r0")]
        frontiers = [SharedFrontierTask("f0", 0, 0.01, 0.1)]
        costs = {("r0", "f0"): 100.0}
        result = allocate_shared_frontiers(
            robots,
            frontiers,
            costs,
            now_sec=0.0,
            config=MultiRobotAllocationConfig(minimum_utility=0.0),
        )
        self.assertEqual(result.assigned_count, 0)
        self.assertIsNone(result.assignments[0].frontier_id)

    def test_independent_nearest_baseline_exposes_conflict(self) -> None:
        robots = _robots(2)
        frontiers = [
            SharedFrontierTask("near", 0, 1.0, 1.0),
            SharedFrontierTask("far", 1, 1.0, 1.0),
        ]
        costs = {
            ("r0", "near"): 1.0,
            ("r0", "far"): 5.0,
            ("r1", "near"): 1.1,
            ("r1", "far"): 10.0,
        }
        nearest = independent_nearest_assignments(robots, frontiers, costs, now_sec=0.0)
        self.assertEqual(nearest, {"r0": "near", "r1": "near"})
        result = allocate_shared_frontiers(robots, frontiers, costs, now_sec=0.0)
        self.assertEqual(result.independent_nearest_conflict_count, 1)
        self.assertEqual(result.assigned_count, 2)
        self.assertEqual(len({item.frontier_id for item in result.assignments}), 2)

    def test_shared_state_communication_bytes_are_audited_and_deterministic(self) -> None:
        robots = _robots(2)
        frontiers = _frontiers(2)
        costs = _costs(robots, frontiers)
        first = allocate_shared_frontiers(robots, frontiers, costs, now_sec=4.0)
        second = allocate_shared_frontiers(robots, frontiers, costs, now_sec=4.0)
        self.assertIsInstance(first.serialized_shared_state_bytes, int)
        self.assertGreater(first.serialized_shared_state_bytes, 0)
        self.assertEqual(first.serialized_shared_state_bytes, second.serialized_shared_state_bytes)
        larger = allocate_shared_frontiers(
            _robots(4), _frontiers(4), _costs(_robots(4), _frontiers(4)), now_sec=4.0
        )
        self.assertGreater(larger.serialized_shared_state_bytes, first.serialized_shared_state_bytes)


if __name__ == "__main__":
    unittest.main()
