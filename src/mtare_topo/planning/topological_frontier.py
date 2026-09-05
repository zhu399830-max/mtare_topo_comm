"""Deterministic global target selection on a causal topometric graph.

The planner owns only global selection.  Collision avoidance and path tracking
remain the responsibility of M-TARE's standalone localPlanner/pathFollower.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TopologicalPlannerConfig:
    """All quantities that must be frozen by a formal Gate-5 run spec."""

    waypoint_lookahead_m: float
    graph_cost_scale_m: float
    retry_penalty: float
    minimum_frontier_confidence: float

    def __post_init__(self) -> None:
        if self.waypoint_lookahead_m <= 0 or self.graph_cost_scale_m <= 0:
            raise ValueError("distance scales must be positive")
        if self.retry_penalty < 0:
            raise ValueError("retry penalty must be non-negative")
        if not 0 <= self.minimum_frontier_confidence <= 1:
            raise ValueError("frontier confidence must lie in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FrontierIdentity:
    node_id: int
    stub_index: int


@dataclass(frozen=True)
class GlobalTarget:
    status: str
    mode: str
    waypoint_xyz_m: tuple[float, float, float] | None
    frontier: FrontierIdentity | None
    frontier_node_id: int | None
    next_hop_node_id: int | None
    graph_path_node_ids: tuple[int, ...]
    graph_cost_m: float | None
    exploration_potential: float | None
    utility: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.frontier is not None:
            value["frontier"] = asdict(self.frontier)
        return value


def _finite_xyz(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must be three finite map-frame coordinates")
    xyz = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in xyz):
        raise ValueError(f"{name} must be three finite map-frame coordinates")
    return xyz


def _verified_adjacency(snapshot: Mapping[str, Any]) -> dict[int, list[tuple[int, float]]]:
    nodes = snapshot.get("nodes", [])
    node_ids = {int(node["id"]) for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("graph node IDs must be unique")
    adjacency = {node_id: [] for node_id in node_ids}
    for edge in snapshot.get("edges", []):
        if edge.get("kind") != "verified_traversed" or not edge.get("traversals"):
            raise ValueError("planner may route only over trace-verified traversed edges")
        first, second = int(edge["from"]), int(edge["to"])
        if first not in node_ids or second not in node_ids or first == second:
            raise ValueError("edge endpoints must reference two distinct graph nodes")
        length = float(edge.get("minimum_traversed_length_m", math.nan))
        if not math.isfinite(length) or length <= 0:
            raise ValueError("verified graph edge length must be finite and positive")
        adjacency[first].append((second, length))
        adjacency[second].append((first, length))
    for neighbours in adjacency.values():
        neighbours.sort()
    return adjacency


def _shortest_paths(
    adjacency: Mapping[int, Sequence[tuple[int, float]]], source: int
) -> tuple[dict[int, float], dict[int, int]]:
    distance = {source: 0.0}
    predecessor: dict[int, int] = {}
    pending: list[tuple[float, int]] = [(0.0, source)]
    while pending:
        cost, node = heapq.heappop(pending)
        if cost != distance.get(node):
            continue
        for neighbour, edge_cost in adjacency[node]:
            candidate = cost + edge_cost
            previous = distance.get(neighbour, math.inf)
            if candidate < previous or (
                math.isclose(candidate, previous, rel_tol=0.0, abs_tol=1e-12)
                and node < predecessor.get(neighbour, node + 1)
            ):
                distance[neighbour] = candidate
                predecessor[neighbour] = node
                heapq.heappush(pending, (candidate, neighbour))
    return distance, predecessor


def _path(predecessor: Mapping[int, int], source: int, target: int) -> tuple[int, ...]:
    result = [target]
    while result[-1] != source:
        if result[-1] not in predecessor:
            raise ValueError("target is not reachable in the verified graph")
        result.append(predecessor[result[-1]])
    result.reverse()
    return tuple(result)


class RuleBasedTopologicalFrontierPlanner:
    """Select an observed, untraversed exit and emit one safe handoff target.

    Utility follows the frozen project decomposition: exploration potential,
    minus verified graph-travel cost, minus repeated local-rejection penalty.
    No hidden geometry or GT edge identity is read.
    """

    def __init__(self, config: TopologicalPlannerConfig) -> None:
        self.config = config

    def select_target(
        self,
        snapshot: Mapping[str, Any],
        *,
        robot_xyz_m: Sequence[float],
        retry_counts: Mapping[FrontierIdentity, int] | None = None,
        blocked_frontiers: set[FrontierIdentity] | None = None,
    ) -> GlobalTarget:
        robot = _finite_xyz(robot_xyz_m, name="robot_xyz_m")
        nodes = {int(node["id"]): node for node in snapshot.get("nodes", [])}
        current_raw = snapshot.get("current_node")
        if current_raw is None or not nodes:
            return GlobalTarget(
                "WAITING", "hold", tuple(robot), None, None, None, (), None,
                None, None, "graph_has_no_current_node",
            )
        current = int(current_raw)
        if current not in nodes:
            raise ValueError("current_node does not exist in graph")
        adjacency = _verified_adjacency(snapshot)
        distance, predecessor = _shortest_paths(adjacency, current)
        retry_counts = retry_counts or {}
        blocked_frontiers = blocked_frontiers or set()

        candidates: list[tuple[tuple[float, float, int, int], dict[str, Any]]] = []
        observed_any = False
        for node_id in sorted(nodes):
            node = nodes[node_id]
            _finite_xyz(node["xyz_m"], name=f"node[{node_id}].xyz_m")
            for stub_index, stub in enumerate(node.get("exit_stubs", [])):
                if stub.get("state") != "observed":
                    continue
                observed_any = True
                identity = FrontierIdentity(node_id, stub_index)
                if identity in blocked_frontiers or node_id not in distance:
                    continue
                confidence = float(stub.get("confidence", math.nan))
                heading = float(stub.get("heading_world_deg", math.nan))
                if not math.isfinite(confidence) or not math.isfinite(heading):
                    raise ValueError("frontier confidence and heading must be finite")
                if confidence < self.config.minimum_frontier_confidence:
                    continue
                structural_potential = float(stub.get("structural_potential", confidence))
                if not math.isfinite(structural_potential) or not 0 <= structural_potential <= 1:
                    raise ValueError("structural potential must lie in [0, 1]")
                retries = int(retry_counts.get(identity, 0))
                if retries < 0:
                    raise ValueError("retry count must be non-negative")
                graph_cost = distance[node_id]
                utility = (
                    structural_potential
                    - graph_cost / self.config.graph_cost_scale_m
                    - self.config.retry_penalty * retries
                )
                # Descending utility, then shortest verified path, then stable IDs.
                key = (-utility, graph_cost, node_id, stub_index)
                candidates.append((key, {
                    "identity": identity,
                    "node_id": node_id,
                    "stub_index": stub_index,
                    "heading_world_deg": heading,
                    "potential": structural_potential,
                    "graph_cost_m": graph_cost,
                    "utility": utility,
                }))

        if not candidates:
            # One empty semantic frame cannot end exploration: a later scan or
            # graph event may expose another branch.  Completion is a separate
            # integration-level confirmation using stable graph/coverage state.
            status = "NO_REACHABLE_FRONTIER" if observed_any else "CANDIDATE_COMPLETE"
            reason = "observed_frontiers_are_blocked_or_unreachable" if observed_any else "no_observed_exit_stub_remains"
            return GlobalTarget(status, "hold", tuple(robot), None, None, None, (), None, None, None, reason)

        candidate = min(candidates, key=lambda item: item[0])[1]
        frontier_node = int(candidate["node_id"])
        path = _path(predecessor, current, frontier_node)
        if frontier_node == current:
            angle = math.radians(float(candidate["heading_world_deg"]))
            waypoint = (
                robot[0] + self.config.waypoint_lookahead_m * math.cos(angle),
                robot[1] + self.config.waypoint_lookahead_m * math.sin(angle),
                robot[2],
            )
            next_hop = current
            mode = "frontier_exit"
        else:
            next_hop = path[1]
            hop = _finite_xyz(nodes[next_hop]["xyz_m"], name=f"node[{next_hop}].xyz_m")
            delta = tuple(hop[index] - robot[index] for index in range(3))
            norm = math.sqrt(sum(value * value for value in delta))
            waypoint = hop if norm <= self.config.waypoint_lookahead_m else tuple(
                robot[index] + delta[index] * (self.config.waypoint_lookahead_m / norm)
                for index in range(3)
            )
            mode = "graph_backtrack"

        return GlobalTarget(
            "TARGET", mode, tuple(map(float, waypoint)), candidate["identity"],
            frontier_node, next_hop, path, float(candidate["graph_cost_m"]),
            float(candidate["potential"]), float(candidate["utility"]),
            "best_deterministic_rule_based_frontier",
        )
