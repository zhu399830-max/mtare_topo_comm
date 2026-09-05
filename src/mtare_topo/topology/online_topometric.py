"""Small causal topometric graph used by the vertical-slice prototype."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


def _wrap_deg(value: float) -> float:
    return float(value % 360.0)


def _circular_distance_deg(first: float, second: float) -> float:
    return abs((_wrap_deg(first) - _wrap_deg(second) + 180.0) % 360.0 - 180.0)


def _role(branch_count: int) -> str:
    if branch_count <= 1:
        return "terminal"
    if branch_count == 2:
        return "corridor"
    return "junction"


@dataclass(frozen=True)
class OnlineTopometricConfig:
    minimum_event_travel_m: float = 6.0
    turn_event_deg: float = 35.0
    loop_merge_radius_m: float = 4.0
    branch_heading_merge_deg: float = 20.0
    stable_frames: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OnlineTopometricGraph:
    """Create event nodes and traversed edges from causal branch observations.

    The graph never sees the oracle world graph.  A node is created for a
    stable terminal/junction observation or a sufficiently large accumulated
    route turn.  Edges are added only after physical movement between nodes.
    """

    def __init__(self, config: OnlineTopometricConfig | None = None) -> None:
        self.config = config or OnlineTopometricConfig()
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.current_node: int | None = None
        self._candidate_role: str | None = None
        self._candidate_frames = 0

    def update(
        self,
        xyz_m: Sequence[float],
        yaw_deg: float,
        headings_robot_deg: Sequence[float],
        frame_index: int,
    ) -> dict[str, Any]:
        xyz = np.asarray(xyz_m, dtype=np.float64)
        headings_robot = sorted(_wrap_deg(value) for value in headings_robot_deg)
        headings_world = sorted(_wrap_deg(value + yaw_deg) for value in headings_robot)
        role = _role(len(headings_world))
        if role == self._candidate_role:
            self._candidate_frames += 1
        else:
            self._candidate_role = role
            self._candidate_frames = 1

        if self.current_node is None:
            self.current_node = self._append_node(
                xyz, yaw_deg, headings_world, role, frame_index, "route_start"
            )
            return {"node_id": self.current_node, "created": True, "reason": "route_start"}

        current = self.nodes[self.current_node]
        travel = float(np.linalg.norm(xyz[:2] - np.asarray(current["xyz_m"][:2])))
        yaw_change = _circular_distance_deg(yaw_deg, float(current["yaw_deg"]))
        stable_structural = (
            self._candidate_frames >= self.config.stable_frames and role in {"terminal", "junction"}
        )
        role_changed = role != current["role"]
        turn_event = role == "corridor" and yaw_change >= self.config.turn_event_deg
        world_heading_changed = self._heading_set_distance(
            headings_world, current["exit_headings_world_deg"]
        ) >= self.config.branch_heading_merge_deg
        event = travel >= self.config.minimum_event_travel_m and (
            (stable_structural and (role_changed or world_heading_changed)) or turn_event
        )
        if not event:
            return {"node_id": self.current_node, "created": False, "reason": "no_stable_event"}

        matched = self._nearby_match(xyz, role, headings_world)
        previous = self.current_node
        if matched is None:
            reason = "turn_event" if turn_event else f"{role}_event"
            self.current_node = self._append_node(
                xyz, yaw_deg, headings_world, role, frame_index, reason
            )
            created = True
        else:
            self.current_node = matched
            reason = "loop_merge"
            created = False
        if previous != self.current_node and not self._edge_exists(previous, self.current_node):
            length = float(
                np.linalg.norm(
                    np.asarray(self.nodes[previous]["xyz_m"][:2])
                    - np.asarray(self.nodes[self.current_node]["xyz_m"][:2])
                )
            )
            self.edges.append(
                {
                    "id": len(self.edges),
                    "from": previous,
                    "to": self.current_node,
                    "length_m": length,
                    "created_at_frame": int(frame_index),
                    "kind": "loop_closure" if reason == "loop_merge" else "traversed_transition",
                }
            )
            self._consume_stub(previous, self.nodes[self.current_node]["xyz_m"])
            self._consume_stub(self.current_node, self.nodes[previous]["xyz_m"])
        return {"node_id": self.current_node, "created": created, "reason": reason}

    def _append_node(
        self,
        xyz: np.ndarray,
        yaw_deg: float,
        headings_world: list[float],
        role: str,
        frame_index: int,
        reason: str,
    ) -> int:
        node_id = len(self.nodes)
        self.nodes.append(
            {
                "id": node_id,
                "xyz_m": xyz.astype(float).tolist(),
                "yaw_deg": _wrap_deg(yaw_deg),
                "role": role,
                "branch_count": len(headings_world),
                "exit_headings_world_deg": headings_world,
                "exit_stub_state": ["unexplored" for _ in headings_world],
                "created_at_frame": int(frame_index),
                "creation_reason": reason,
            }
        )
        return node_id

    def _nearby_match(
        self, xyz: np.ndarray, role: str, headings_world: Sequence[float]
    ) -> int | None:
        candidates = []
        for node in self.nodes:
            distance = float(np.linalg.norm(xyz[:2] - np.asarray(node["xyz_m"][:2])))
            if distance > self.config.loop_merge_radius_m or node["role"] != role:
                continue
            heading_distance = self._heading_set_distance(
                headings_world, node["exit_headings_world_deg"]
            )
            if heading_distance <= self.config.branch_heading_merge_deg:
                candidates.append((distance, heading_distance, int(node["id"])))
        return min(candidates)[2] if candidates else None

    @staticmethod
    def _heading_set_distance(first: Sequence[float], second: Sequence[float]) -> float:
        if len(first) != len(second):
            return 180.0
        if not first:
            return 0.0
        forward = [min(_circular_distance_deg(value, other) for other in second) for value in first]
        backward = [min(_circular_distance_deg(value, other) for other in first) for value in second]
        return float(max(max(forward), max(backward)))

    def _edge_exists(self, first: int, second: int) -> bool:
        target = {first, second}
        return any({int(edge["from"]), int(edge["to"])} == target for edge in self.edges)

    def _consume_stub(self, node_id: int, destination_xyz: Sequence[float]) -> None:
        node = self.nodes[node_id]
        if not node["exit_headings_world_deg"]:
            return
        origin = np.asarray(node["xyz_m"], dtype=np.float64)
        destination = np.asarray(destination_xyz, dtype=np.float64)
        heading = _wrap_deg(math.degrees(math.atan2(destination[1] - origin[1], destination[0] - origin[0])))
        index = min(
            range(len(node["exit_headings_world_deg"])),
            key=lambda item: _circular_distance_deg(
                heading, node["exit_headings_world_deg"][item]
            ),
        )
        node["exit_stub_state"][index] = "traversed"

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "online_topometric_graph_prototype_v1",
            "config": self.config.to_dict(),
            "current_node": self.current_node,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "unexplored_exit_stub_count": sum(
                state == "unexplored"
                for node in self.nodes
                for state in node["exit_stub_state"]
            ),
            "nodes": self.nodes,
            "edges": self.edges,
        }
