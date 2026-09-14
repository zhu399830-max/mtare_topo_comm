"""Causal graph-event feedback for an active frontier identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import sys
from typing import Any, Mapping

from mtare_topo.planning.topological_frontier import FrontierIdentity, GlobalTarget
from mtare_topo.topology.causal_graph_v2 import circular_distance_deg, wrap_deg


@dataclass(frozen=True)
class FrontierExecutionFeedback:
    classified: bool
    outcome: str
    target_frontier: FrontierIdentity | None
    actual_departure_stub_index: int | None
    record_rejection: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.target_frontier is not None:
            value["target_frontier"] = asdict(self.target_frontier)
        return value


def _departure_stub_index(source: Mapping[str, Any], destination: Mapping[str, Any]) -> int:
    origin = tuple(float(value) for value in source["xyz_m"])
    target = tuple(float(value) for value in destination["xyz_m"])
    if len(origin) != 3 or len(target) != 3 or not all(math.isfinite(value) for value in (*origin, *target)):
        raise ValueError("frontier feedback nodes require finite xyz")
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    # ``math.ulp`` is unavailable in the frozen ROS Python 3.8 runtime.
    # ``sys.float_info.epsilon`` is exactly the IEEE-754 binary64 ULP at 1.0.
    if math.hypot(dx, dy) <= sys.float_info.epsilon:
        raise ValueError("frontier feedback departure nodes must be spatially distinct")
    stubs = source.get("exit_stubs")
    if not isinstance(stubs, list) or not stubs:
        raise ValueError("frontier feedback source node requires exit stubs")
    heading = wrap_deg(math.degrees(math.atan2(dy, dx)))
    candidates = []
    for index, stub in enumerate(stubs):
        value = float(stub.get("heading_world_deg", math.nan))
        if not math.isfinite(value):
            raise ValueError("frontier feedback stub heading must be finite")
        candidates.append((circular_distance_deg(heading, value), index))
    return min(candidates)[1]


def evaluate_frontier_execution_feedback(
    previous_target: GlobalTarget | None,
    *,
    previous_node_id: int | None,
    graph_update: Mapping[str, Any],
    graph_snapshot: Mapping[str, Any],
) -> FrontierExecutionFeedback:
    """Classify only a graph event following a current-node frontier target."""

    if previous_target is None or previous_target.mode != "frontier_exit":
        return FrontierExecutionFeedback(False, "not_frontier_exit", None, None, False)
    frontier = previous_target.frontier
    if frontier is None or previous_node_id is None or frontier.node_id != int(previous_node_id):
        raise ValueError("frontier execution target is inconsistent with previous current node")
    current_node = int(graph_update.get("node_id", -1))
    reason = str(graph_update.get("reason"))
    if current_node == int(previous_node_id):
        if reason != "loop_merge":
            return FrontierExecutionFeedback(False, "no_graph_event", frontier, None, False)
        return FrontierExecutionFeedback(True, "same_node_loop_merge", frontier, None, True)
    nodes = {int(node["id"]): node for node in graph_snapshot.get("nodes", [])}
    if int(previous_node_id) not in nodes or current_node not in nodes:
        raise ValueError("frontier execution graph event references an unknown node")
    actual = _departure_stub_index(nodes[int(previous_node_id)], nodes[current_node])
    matched = actual == frontier.stub_index
    return FrontierExecutionFeedback(
        True,
        "matched_verified_departure" if matched else "divergent_verified_departure",
        frontier,
        actual,
        not matched,
    )


__all__ = ["FrontierExecutionFeedback", "evaluate_frontier_execution_feedback"]
