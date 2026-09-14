"""Deterministic one-to-one allocation of shared-graph exit frontiers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MultiRobotAllocationConfig:
    graph_cost_weight: float = 1.0
    exploration_potential_weight: float = 8.0
    confidence_weight: float = 2.0
    retry_penalty_weight: float = 4.0
    minimum_utility: float = 0.0
    lease_duration_sec: float = 15.0

    def __post_init__(self) -> None:
        values = (
            self.graph_cost_weight,
            self.exploration_potential_weight,
            self.confidence_weight,
            self.retry_penalty_weight,
            self.lease_duration_sec,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("allocation weights and lease duration must be finite/nonnegative")
        if not math.isfinite(self.minimum_utility):
            raise ValueError("minimum utility must be finite")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RobotAllocationState:
    robot_id: str
    active: bool = True

    def __post_init__(self) -> None:
        if not self.robot_id:
            raise ValueError("robot ID cannot be empty")


@dataclass(frozen=True)
class SharedFrontierTask:
    frontier_id: str
    node_id: int
    exploration_potential: float
    confidence: float
    lease_holder: Optional[str] = None
    lease_expires_sec: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.frontier_id or self.node_id < 0:
            raise ValueError("frontier identity/node is invalid")
        if not math.isfinite(self.exploration_potential) or self.exploration_potential < 0:
            raise ValueError("frontier potential must be finite/nonnegative")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("frontier confidence must lie in [0,1]")
        if (self.lease_holder is None) != (self.lease_expires_sec is None):
            raise ValueError("lease holder and expiry must be present together")
        if self.lease_expires_sec is not None and not math.isfinite(self.lease_expires_sec):
            raise ValueError("lease expiry must be finite")


@dataclass(frozen=True)
class FrontierAssignment:
    robot_id: str
    frontier_id: Optional[str]
    node_id: Optional[int]
    utility: Optional[float]
    graph_cost_m: Optional[float]
    lease_expires_sec: Optional[float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiRobotAllocationResult:
    assignments: Tuple[FrontierAssignment, ...]
    active_robot_count: int
    frontier_count: int
    assigned_count: int
    total_utility: float
    total_graph_cost_m: float
    independent_nearest_conflict_count: int
    serialized_shared_state_bytes: int
    utility_matrix: Tuple[Tuple[Optional[float], ...], ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_inputs(
    robots: Sequence[RobotAllocationState],
    frontiers: Sequence[SharedFrontierTask],
    graph_cost_m: Mapping[Tuple[str, str], float],
    retry_counts: Mapping[Tuple[str, str], int],
) -> Tuple[List[RobotAllocationState], List[SharedFrontierTask]]:
    robot_values = sorted(robots, key=lambda item: item.robot_id)
    frontier_values = sorted(frontiers, key=lambda item: item.frontier_id)
    if len({item.robot_id for item in robot_values}) != len(robot_values):
        raise ValueError("robot IDs must be unique")
    if len({item.frontier_id for item in frontier_values}) != len(frontier_values):
        raise ValueError("frontier IDs must be unique")
    for robot in robot_values:
        for frontier in frontier_values:
            key = (robot.robot_id, frontier.frontier_id)
            if key not in graph_cost_m:
                raise ValueError(f"missing graph cost: {key}")
            cost = float(graph_cost_m[key])
            if math.isnan(cost) or cost < 0:
                raise ValueError(f"invalid graph cost: {key}")
            retry = int(retry_counts.get(key, 0))
            if retry < 0:
                raise ValueError(f"negative retry count: {key}")
    return robot_values, frontier_values


def independent_nearest_assignments(
    robots: Sequence[RobotAllocationState],
    frontiers: Sequence[SharedFrontierTask],
    graph_cost_m: Mapping[Tuple[str, str], float],
    *,
    now_sec: float,
) -> Dict[str, Optional[str]]:
    if not math.isfinite(now_sec):
        raise ValueError("allocation time must be finite")
    result: Dict[str, Optional[str]] = {}
    for robot in sorted(robots, key=lambda item: item.robot_id):
        if not robot.active:
            result[robot.robot_id] = None
            continue
        eligible = [
            frontier for frontier in frontiers
            if math.isfinite(float(graph_cost_m[(robot.robot_id, frontier.frontier_id)]))
            and (
                frontier.lease_holder in (None, robot.robot_id)
                or float(frontier.lease_expires_sec) <= now_sec
            )
        ]
        if not eligible:
            result[robot.robot_id] = None
            continue
        chosen = min(
            eligible,
            key=lambda item: (float(graph_cost_m[(robot.robot_id, item.frontier_id)]), item.frontier_id),
        )
        result[robot.robot_id] = chosen.frontier_id
    return result


def allocate_shared_frontiers(
    robots: Sequence[RobotAllocationState],
    frontiers: Sequence[SharedFrontierTask],
    graph_cost_m: Mapping[Tuple[str, str], float],
    *,
    now_sec: float,
    retry_counts: Optional[Mapping[Tuple[str, str], int]] = None,
    config: Optional[MultiRobotAllocationConfig] = None,
) -> MultiRobotAllocationResult:
    if not math.isfinite(now_sec):
        raise ValueError("allocation time must be finite")
    policy = config or MultiRobotAllocationConfig()
    retries = retry_counts or {}
    all_robots, tasks = _validate_inputs(robots, frontiers, graph_cost_m, retries)
    active = [robot for robot in all_robots if robot.active]
    payload = {
        "robots": [asdict(item) for item in all_robots],
        "frontiers": [asdict(item) for item in tasks],
        "graph_cost_m": [
            {
                "robot_id": robot.robot_id,
                "frontier_id": task.frontier_id,
                "cost_m": (
                    float(graph_cost_m[(robot.robot_id, task.frontier_id)])
                    if math.isfinite(float(graph_cost_m[(robot.robot_id, task.frontier_id)]))
                    else None
                ),
            }
            for robot in all_robots for task in tasks
        ],
        "retry_counts": [
            {"robot_id": robot.robot_id, "frontier_id": task.frontier_id, "count": int(retries.get((robot.robot_id, task.frontier_id), 0))}
            for robot in all_robots for task in tasks
        ],
    }
    serialized_bytes = len(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if not active:
        inactive = tuple(
            FrontierAssignment(robot.robot_id, None, None, None, None, None, "robot_inactive")
            for robot in all_robots
        )
        return MultiRobotAllocationResult(inactive, 0, len(tasks), 0, 0.0, 0.0, 0, serialized_bytes, ())

    utility = np.full((len(active), len(tasks)), -np.inf, dtype=np.float64)
    for row, robot in enumerate(active):
        for column, task in enumerate(tasks):
            leased_elsewhere = (
                task.lease_holder not in (None, robot.robot_id)
                and float(task.lease_expires_sec) > now_sec
            )
            if leased_elsewhere:
                continue
            cost = float(graph_cost_m[(robot.robot_id, task.frontier_id)])
            if not math.isfinite(cost):
                continue
            retry = int(retries.get((robot.robot_id, task.frontier_id), 0))
            utility[row, column] = (
                policy.exploration_potential_weight * task.exploration_potential
                + policy.confidence_weight * task.confidence
                - policy.graph_cost_weight * cost
                - policy.retry_penalty_weight * retry
            )

    # One private dummy column per robot permits deterministic unassignment.
    columns = len(tasks) + len(active)
    objective = np.full((len(active), columns), -1e15, dtype=np.float64)
    if tasks:
        objective[:, : len(tasks)] = utility
    for row in range(len(active)):
        objective[row, len(tasks) + row] = policy.minimum_utility
    # Stable lexicographic perturbation is far below reported numerical precision.
    rank = np.arange(objective.size, dtype=np.float64).reshape(objective.shape)
    cost_matrix = -objective + rank * 1e-12
    rows, selected_columns = linear_sum_assignment(cost_matrix)
    selected = {int(row): int(column) for row, column in zip(rows, selected_columns)}
    assignments = []
    total_utility = total_cost = 0.0
    for row, robot in enumerate(active):
        column = selected[row]
        if column >= len(tasks) or not math.isfinite(float(utility[row, column])) or float(utility[row, column]) < policy.minimum_utility:
            assignments.append(FrontierAssignment(robot.robot_id, None, None, None, None, None, "no_eligible_positive_utility_frontier"))
            continue
        task = tasks[column]
        value = float(utility[row, column])
        cost = float(graph_cost_m[(robot.robot_id, task.frontier_id)])
        assignments.append(FrontierAssignment(robot.robot_id, task.frontier_id, task.node_id, value, cost, now_sec + policy.lease_duration_sec, "hungarian_shared_graph_assignment"))
        total_utility += value
        total_cost += cost
    nearest = independent_nearest_assignments(active, tasks, graph_cost_m, now_sec=now_sec)
    chosen = [value for value in nearest.values() if value is not None]
    conflicts = len(chosen) - len(set(chosen))
    matrix = tuple(
        tuple(None if not math.isfinite(float(value)) else float(value) for value in row)
        for row in utility
    )
    assignments.extend(
        FrontierAssignment(robot.robot_id, None, None, None, None, None, "robot_inactive")
        for robot in all_robots if not robot.active
    )
    assignments.sort(key=lambda item: item.robot_id)
    return MultiRobotAllocationResult(
        assignments=tuple(assignments),
        active_robot_count=len(active),
        frontier_count=len(tasks),
        assigned_count=sum(item.frontier_id is not None for item in assignments),
        total_utility=total_utility,
        total_graph_cost_m=total_cost,
        independent_nearest_conflict_count=conflicts,
        serialized_shared_state_bytes=serialized_bytes,
        utility_matrix=matrix,
    )


__all__ = [
    "FrontierAssignment",
    "MultiRobotAllocationConfig",
    "MultiRobotAllocationResult",
    "RobotAllocationState",
    "SharedFrontierTask",
    "allocate_shared_frontiers",
    "independent_nearest_assignments",
]
