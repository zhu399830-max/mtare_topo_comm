"""Causal structural topometric graph with trace-verified traversed edges."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


ROLE_NAMES = ("interior", "junction", "terminal")


def wrap_deg(value: float) -> float:
    return float(value % 360.0)


def circular_distance_deg(first: float, second: float) -> float:
    return abs((wrap_deg(first) - wrap_deg(second) + 180.0) % 360.0 - 180.0)


def heading_set_distance(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        return 180.0
    if not first:
        return 0.0
    forward = [min(circular_distance_deg(value, other) for other in second) for value in first]
    backward = [min(circular_distance_deg(value, other) for other in first) for value in second]
    return float(max(max(forward), max(backward)))


def role_from_branch_count(branch_count: int) -> str:
    if branch_count <= 1:
        return "terminal"
    if branch_count == 2:
        return "interior"
    return "junction"


@dataclass(frozen=True)
class CausalGraphConfig:
    stable_frames: int = 2
    minimum_event_travel_m: float = 6.0
    loop_merge_radius_m: float = 4.0
    branch_heading_merge_deg: float = 25.0
    turn_event_deg: float = 35.0
    distance_anchor_interval_m: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalTopometricGraphV2:
    """Incremental graph that consumes current semantics, pose and past state only.

    GT edge IDs may be attached to updates as evaluator-only trace metadata. They
    are copied into edge evidence but never participate in event or association
    decisions.
    """

    def __init__(self, config: CausalGraphConfig) -> None:
        self.config = config
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.current_node: int | None = None
        self.current_node_route_arc_m = 0.0
        self._candidate_signature: tuple[str, tuple[int, ...]] | None = None
        self._candidate_frames = 0
        self._last_yaw_deg: float | None = None
        self._accumulated_turn_deg = 0.0
        self._trace_frames: list[int] = []
        self._trace_route_arc_m: list[float] = []
        self._trace_gt_edge_ids: list[str] = []
        self.decision_trace: list[dict[str, Any]] = []

    @staticmethod
    def _heading_signature(headings_world_deg: Sequence[float]) -> tuple[int, ...]:
        return tuple(sorted(int(round(wrap_deg(value) / 5.0)) % 72 for value in headings_world_deg))

    def update(
        self,
        *,
        frame_index: int,
        route_arc_m: float,
        xyz_m: Sequence[float],
        yaw_deg: float,
        headings_robot_deg: Sequence[float],
        role_probabilities: Sequence[float] | None = None,
        branch_count: int | None = None,
        confidence: float = 1.0,
        z_role: Sequence[float] | None = None,
        evaluator_gt_edge_id: str | None = None,
    ) -> dict[str, Any]:
        xyz = np.asarray(xyz_m, dtype=np.float64)
        headings_robot = sorted(wrap_deg(value) for value in headings_robot_deg)
        headings_world = sorted(wrap_deg(value + yaw_deg) for value in headings_robot)
        count = int(branch_count if branch_count is not None else len(headings_world))
        if role_probabilities is None:
            role = role_from_branch_count(count)
            role_probs = [0.0, 0.0, 0.0]
            role_probs[ROLE_NAMES.index(role)] = 1.0
        else:
            role_probs = np.asarray(role_probabilities, dtype=np.float64).tolist()
            if len(role_probs) != 3 or not np.all(np.isfinite(role_probs)):
                raise ValueError("role probabilities must contain three finite values")
            role = ROLE_NAMES[int(np.argmax(role_probs))]

        if self._last_yaw_deg is not None:
            self._accumulated_turn_deg += circular_distance_deg(yaw_deg, self._last_yaw_deg)
        self._last_yaw_deg = float(yaw_deg)
        self._trace_frames.append(int(frame_index))
        self._trace_route_arc_m.append(float(route_arc_m))
        if evaluator_gt_edge_id is not None:
            self._trace_gt_edge_ids.append(str(evaluator_gt_edge_id))

        signature = (role, self._heading_signature(headings_world))
        if signature == self._candidate_signature:
            self._candidate_frames += 1
        else:
            self._candidate_signature = signature
            self._candidate_frames = 1

        if self.current_node is None:
            node_id = self._append_node(
                xyz, route_arc_m, yaw_deg, headings_world, role, role_probs,
                count, confidence, z_role, frame_index, "route_start", "anchor"
            )
            self.current_node = node_id
            self.current_node_route_arc_m = float(route_arc_m)
            return self._record(frame_index, route_arc_m, node_id, True, "route_start")

        travel = float(route_arc_m - self.current_node_route_arc_m)
        stable_structural = (
            role in {"junction", "terminal"}
            and self._candidate_frames >= self.config.stable_frames
            and travel >= self.config.minimum_event_travel_m
        )
        turn_event = (
            role == "interior"
            and travel >= self.config.minimum_event_travel_m
            and self._accumulated_turn_deg >= self.config.turn_event_deg
        )
        distance_anchor = travel >= self.config.distance_anchor_interval_m
        if not (stable_structural or turn_event or distance_anchor):
            return self._record(frame_index, route_arc_m, self.current_node, False, "no_event")

        if stable_structural:
            reason, node_kind = f"stable_{role}", "structural"
        elif turn_event:
            reason, node_kind = "turn_event", "anchor"
        else:
            reason, node_kind = "distance_anchor", "anchor"

        matched = self._nearby_match(xyz, role, headings_world, node_kind)
        previous = int(self.current_node)
        if matched is None:
            target = self._append_node(
                xyz, route_arc_m, yaw_deg, headings_world, role, role_probs,
                count, confidence, z_role, frame_index, reason, node_kind
            )
            created = True
        else:
            target = matched
            reason = "loop_merge"
            created = False
            self._update_node(target, role_probs, headings_world, confidence, z_role, frame_index)

        if target != previous:
            self._append_verified_traversal(previous, target, frame_index, route_arc_m)
            self._consume_stub(previous, self.nodes[target]["xyz_m"])
            self._consume_stub(target, self.nodes[previous]["xyz_m"])
        self.current_node = target
        self.current_node_route_arc_m = float(route_arc_m)
        self._accumulated_turn_deg = 0.0
        self._trace_frames = [int(frame_index)]
        self._trace_route_arc_m = [float(route_arc_m)]
        self._trace_gt_edge_ids = [str(evaluator_gt_edge_id)] if evaluator_gt_edge_id is not None else []
        return self._record(frame_index, route_arc_m, target, created, reason)

    def _append_node(
        self, xyz: np.ndarray, route_arc_m: float, yaw_deg: float,
        headings_world: list[float], role: str, role_probs: list[float],
        branch_count: int, confidence: float, z_role: Sequence[float] | None,
        frame_index: int, reason: str, node_kind: str,
    ) -> int:
        node_id = len(self.nodes)
        self.nodes.append({
            "id": node_id,
            "node_kind": node_kind,
            "xyz_m": xyz.astype(float).tolist(),
            "route_arc_m_first": float(route_arc_m),
            "yaw_deg": wrap_deg(yaw_deg),
            "role": role,
            "role_probabilities_mean": [float(value) for value in role_probs],
            "branch_count": int(branch_count),
            "exit_headings_world_deg": [float(value) for value in headings_world],
            "exit_stubs": [
                {"heading_world_deg": float(value), "state": "observed", "confidence": float(confidence)}
                for value in headings_world
            ],
            "confidence_mean": float(confidence),
            "z_role_mean": list(map(float, z_role)) if z_role is not None else None,
            "observation_count": 1,
            "created_at_frame": int(frame_index),
            "last_observed_frame": int(frame_index),
            "creation_reason": reason,
        })
        return node_id

    def _nearby_match(self, xyz: np.ndarray, role: str, headings: Sequence[float], node_kind: str) -> int | None:
        candidates=[]
        for node in self.nodes:
            if node["node_kind"] != node_kind:
                continue
            distance=float(np.linalg.norm(xyz-np.asarray(node["xyz_m"],dtype=np.float64)))
            if distance > self.config.loop_merge_radius_m:
                continue
            if node_kind == "structural" and node["role"] != role:
                continue
            heading_error=heading_set_distance(headings,node["exit_headings_world_deg"])
            if node_kind == "structural" and heading_error > self.config.branch_heading_merge_deg:
                continue
            candidates.append((distance,heading_error,int(node["id"])))
        return min(candidates)[2] if candidates else None

    def _update_node(self, node_id: int, role_probs: Sequence[float], headings: Sequence[float], confidence: float, z_role: Sequence[float] | None, frame_index: int) -> None:
        node=self.nodes[node_id];count=int(node["observation_count"]);new_count=count+1
        node["role_probabilities_mean"]=[(count*a+float(b))/new_count for a,b in zip(node["role_probabilities_mean"],role_probs)]
        node["confidence_mean"]=(count*float(node["confidence_mean"])+float(confidence))/new_count
        if z_role is not None:
            old=node["z_role_mean"] or [0.0]*len(z_role)
            node["z_role_mean"]=[(count*a+float(b))/new_count for a,b in zip(old,z_role)]
        node["observation_count"]=new_count;node["last_observed_frame"]=int(frame_index)
        for heading in headings:
            if not node["exit_stubs"] or min(circular_distance_deg(heading,item["heading_world_deg"]) for item in node["exit_stubs"]) > self.config.branch_heading_merge_deg:
                node["exit_stubs"].append({"heading_world_deg":float(heading),"state":"observed","confidence":float(confidence)})

    def _append_verified_traversal(self, first: int, second: int, frame_index: int, route_arc_m: float) -> None:
        if len(self._trace_frames) < 2 or len(self._trace_route_arc_m) < 2:
            raise RuntimeError("verified edge requires at least two physical trace frames")
        length=float(route_arc_m-self.current_node_route_arc_m)
        if length <= 0:
            raise RuntimeError("verified edge requires positive accumulated route length")
        target=frozenset((first,second));edge=next((item for item in self.edges if frozenset((item["from"],item["to"]))==target),None)
        traversal={
            "from":first,"to":second,"start_frame":int(self._trace_frames[0]),"end_frame":int(frame_index),
            "start_route_arc_m":float(self.current_node_route_arc_m),"end_route_arc_m":float(route_arc_m),
            "length_m":length,"trace_frame_count":len(self._trace_frames),
            "evaluator_gt_edge_ids":sorted(set(self._trace_gt_edge_ids)),
        }
        if edge is None:
            edge={"id":len(self.edges),"from":first,"to":second,"kind":"verified_traversed","traversals":[]}
            self.edges.append(edge)
        edge["traversals"].append(traversal)
        edge["verified_traversal_count"]=len(edge["traversals"])
        edge["minimum_traversed_length_m"]=min(item["length_m"] for item in edge["traversals"])

    def _consume_stub(self, node_id: int, destination_xyz: Sequence[float]) -> None:
        node=self.nodes[node_id]
        if not node["exit_stubs"]:
            return
        origin=np.asarray(node["xyz_m"],dtype=np.float64);destination=np.asarray(destination_xyz,dtype=np.float64)
        heading=wrap_deg(math.degrees(math.atan2(destination[1]-origin[1],destination[0]-origin[0])))
        index=min(range(len(node["exit_stubs"])),key=lambda item:circular_distance_deg(heading,node["exit_stubs"][item]["heading_world_deg"]))
        node["exit_stubs"][index]["state"]="traversed"

    def _record(self, frame_index: int, route_arc_m: float, node_id: int, created: bool, reason: str) -> dict[str, Any]:
        result={"frame_index":int(frame_index),"route_arc_m":float(route_arc_m),"node_id":int(node_id),"created":bool(created),"reason":reason}
        self.decision_trace.append(result)
        return result

    def snapshot(self) -> dict[str, Any]:
        if any(not edge.get("traversals") for edge in self.edges):
            raise RuntimeError("all graph edges must preserve physical traversal evidence")
        return {
            "schema_version":"causal_topometric_graph_v2",
            "config":self.config.to_dict(),
            "current_node":self.current_node,
            "node_count":len(self.nodes),
            "structural_node_count":sum(node["node_kind"]=="structural" for node in self.nodes),
            "anchor_node_count":sum(node["node_kind"]=="anchor" for node in self.nodes),
            "edge_count":len(self.edges),
            "untraversed_exit_stub_count":sum(stub["state"]=="observed" for node in self.nodes for stub in node["exit_stubs"]),
            "nodes":self.nodes,"edges":self.edges,
        }
