"""Stateful coordinator for shared causal-graph frontier allocation.

This module is deliberately ROS-free.  It turns immutable per-robot causal
graph snapshots into auditable, leased one-to-one frontier decisions; a thin
transport adapter can publish those decisions in simulation later.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from mtare_topo.planning.multi_robot_allocation import (
    FrontierAssignment,
    MultiRobotAllocationConfig,
    RobotAllocationState,
    SharedFrontierTask,
    allocate_shared_frontiers,
)
from mtare_topo.topology.shared_causal_graph import (
    SharedCausalGraphConfig,
    SharedCausalTopometricGraph,
)


@dataclass(frozen=True)
class CoordinatorDecision:
    epoch: int
    stamp_sec: float
    assignments: Tuple[FrontierAssignment, ...]
    preserved_lease_count: int
    new_assignment_count: int
    active_robot_count: int
    available_frontier_count: int
    shared_graph_received_bytes_total: int
    allocation_payload_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiRobotCoordinatorRuntime:
    """Fuse robot graphs and maintain deterministic assignment leases/retries."""

    def __init__(
        self,
        *,
        graph_config: Optional[SharedCausalGraphConfig] = None,
        allocation_config: Optional[MultiRobotAllocationConfig] = None,
    ) -> None:
        self.graph = SharedCausalTopometricGraph(graph_config)
        self.allocation_config = allocation_config or MultiRobotAllocationConfig()
        self._active: Dict[str, bool] = {}
        self._leases: Dict[str, FrontierAssignment] = {}
        self._retry_counts: Dict[Tuple[str, str], int] = {}
        self._completed_frontiers: Set[str] = set()
        self._decisions: List[CoordinatorDecision] = []
        self._execution_trace: List[Dict[str, Any]] = []

    def ingest_robot_snapshot(
        self,
        *,
        robot_id: str,
        revision: int,
        stamp_sec: float,
        snapshot: Mapping[str, Any],
        active: bool = True,
    ) -> Dict[str, Any]:
        result = self.graph.ingest(
            robot_id=robot_id,
            revision=revision,
            stamp_sec=stamp_sec,
            snapshot=snapshot,
        )
        self._active[robot_id] = bool(active)
        return result

    def set_robot_active(self, robot_id: str, active: bool) -> None:
        if robot_id not in self._active:
            raise ValueError(f"unknown robot: {robot_id}")
        self._active[robot_id] = bool(active)
        if not active:
            self._leases.pop(robot_id, None)

    @staticmethod
    def _with_lease(
        task: SharedFrontierTask,
        assignment: Optional[FrontierAssignment],
    ) -> SharedFrontierTask:
        if assignment is None:
            return task
        return SharedFrontierTask(
            frontier_id=task.frontier_id,
            node_id=task.node_id,
            exploration_potential=task.exploration_potential,
            confidence=task.confidence,
            lease_holder=assignment.robot_id,
            lease_expires_sec=assignment.lease_expires_sec,
        )

    def allocate(self, *, now_sec: float) -> CoordinatorDecision:
        if not math.isfinite(now_sec):
            raise ValueError("coordinator allocation time must be finite")
        all_tasks = {
            task.frontier_id: task
            for task in self.graph.frontier_tasks()
            if task.frontier_id not in self._completed_frontiers
        }
        graph_costs = self.graph.graph_costs()

        # Preserve a still-valid lease exactly.  This prevents target thrashing
        # and makes a repeated call without new evidence idempotent.
        preserved: Dict[str, FrontierAssignment] = {}
        occupied: Set[str] = set()
        for robot_id, assignment in sorted(self._leases.items()):
            valid = (
                self._active.get(robot_id, False)
                and assignment.frontier_id in all_tasks
                and assignment.lease_expires_sec is not None
                and assignment.lease_expires_sec > now_sec
                and math.isfinite(graph_costs.get((robot_id, str(assignment.frontier_id)), math.inf))
            )
            if valid and str(assignment.frontier_id) not in occupied:
                preserved[robot_id] = assignment
                occupied.add(str(assignment.frontier_id))

        free_robots = [
            RobotAllocationState(robot_id, active=True)
            for robot_id, active in sorted(self._active.items())
            if active and robot_id not in preserved and robot_id in self.graph.current_node_by_robot
        ]
        free_tasks = [task for key, task in sorted(all_tasks.items()) if key not in occupied]
        free_costs = {
            (robot.robot_id, task.frontier_id): graph_costs.get((robot.robot_id, task.frontier_id), math.inf)
            for robot in free_robots
            for task in free_tasks
        }
        result = allocate_shared_frontiers(
            free_robots,
            free_tasks,
            free_costs,
            now_sec=now_sec,
            retry_counts=self._retry_counts,
            config=self.allocation_config,
        )
        combined = list(preserved.values()) + list(result.assignments)
        assigned_robot_ids = {item.robot_id for item in combined}
        combined.extend(
            FrontierAssignment(robot_id, None, None, None, None, None, "robot_graph_not_ready")
            for robot_id, active in sorted(self._active.items())
            if active
            and robot_id not in self.graph.current_node_by_robot
            and robot_id not in assigned_robot_ids
        )
        assigned_robot_ids = {item.robot_id for item in combined}
        combined.extend(
            FrontierAssignment(robot_id, None, None, None, None, None, "robot_inactive")
            for robot_id, active in sorted(self._active.items())
            if not active and robot_id not in assigned_robot_ids
        )
        combined.sort(key=lambda item: item.robot_id)
        self._leases = {
            item.robot_id: item
            for item in combined
            if item.frontier_id is not None
        }
        graph_snapshot = self.graph.snapshot()
        decision = CoordinatorDecision(
            epoch=len(self._decisions),
            stamp_sec=float(now_sec),
            assignments=tuple(combined),
            preserved_lease_count=len(preserved),
            new_assignment_count=sum(
                item.frontier_id is not None and item.robot_id not in preserved
                for item in combined
            ),
            active_robot_count=sum(self._active.values()),
            available_frontier_count=len(all_tasks),
            shared_graph_received_bytes_total=int(graph_snapshot["received_bytes_total"]),
            allocation_payload_bytes=int(result.serialized_shared_state_bytes),
        )
        self._decisions.append(decision)
        return decision

    def report_execution(
        self,
        *,
        robot_id: str,
        frontier_id: str,
        success: bool,
        stamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        assignment = self._leases.get(robot_id)
        if assignment is None or assignment.frontier_id != frontier_id:
            raise ValueError("execution report does not match the robot's active lease")
        if stamp_sec is not None and not math.isfinite(stamp_sec):
            raise ValueError("execution report timestamp must be finite")
        key = (robot_id, frontier_id)
        if success:
            self._completed_frontiers.add(frontier_id)
            self._retry_counts.pop(key, None)
        else:
            self._retry_counts[key] = self._retry_counts.get(key, 0) + 1
        self._leases.pop(robot_id)
        record = {
            "robot_id": robot_id,
            "frontier_id": frontier_id,
            "success": bool(success),
            "stamp_sec": None if stamp_sec is None else float(stamp_sec),
            "retry_count": int(self._retry_counts.get(key, 0)),
        }
        self._execution_trace.append(record)
        return dict(record)

    def snapshot(self) -> Dict[str, Any]:
        value = {
            "schema_version": "multi_robot_coordinator_runtime_v1",
            "graph": self.graph.snapshot(),
            "allocation_config": self.allocation_config.to_dict(),
            "robot_active": dict(sorted(self._active.items())),
            "active_leases": {
                robot_id: assignment.to_dict()
                for robot_id, assignment in sorted(self._leases.items())
            },
            "retry_counts": [
                {"robot_id": key[0], "frontier_id": key[1], "count": count}
                for key, count in sorted(self._retry_counts.items())
            ],
            "completed_frontiers": sorted(self._completed_frontiers),
            "decision_trace": [item.to_dict() for item in self._decisions],
            "execution_trace": list(self._execution_trace),
        }
        # Canonical round trip both proves JSON serializability and prevents
        # callers from mutating coordinator state through returned containers.
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


__all__ = ["CoordinatorDecision", "MultiRobotCoordinatorRuntime"]
