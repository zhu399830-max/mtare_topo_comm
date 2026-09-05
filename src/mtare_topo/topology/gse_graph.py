"""Geometry-semantic event graph with rejection-aware learned association."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any, Sequence

import numpy as np

from mtare_topo.semantics.geometric_semantics import (
    EVENT_NAMES,
    GeometricSemanticObservation,
    StructuralEvent,
)
from mtare_topo.topology.causal_graph_v2 import heading_set_distance, wrap_deg


@dataclass(frozen=True)
class GSEGraphConfig:
    stable_event_frames: int
    minimum_event_travel_m: float
    metric_anchor_interval_m: float
    event_probability_threshold: float
    maximum_uncertainty: float
    association_radius_m: float
    descriptor_minimum_similarity: float
    exit_heading_tolerance_deg: float
    exit_descriptor_minimum_similarity: float
    exit_width_log_tolerance: float
    exit_vertical_profile_tolerance: float
    maximum_exit_count_difference: int
    ambiguity_similarity_margin: float

    def __post_init__(self) -> None:
        if self.stable_event_frames < 1:
            raise ValueError("stable_event_frames must be positive")
        for name in ("minimum_event_travel_m", "metric_anchor_interval_m", "association_radius_m", "exit_heading_tolerance_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("event_probability_threshold", "maximum_uncertainty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if not -1.0 <= self.descriptor_minimum_similarity <= 1.0:
            raise ValueError("descriptor_minimum_similarity must be in [-1,1]")
        if not -1.0 <= self.exit_descriptor_minimum_similarity <= 1.0:
            raise ValueError("exit_descriptor_minimum_similarity must be in [-1,1]")
        if not math.isfinite(self.exit_width_log_tolerance) or self.exit_width_log_tolerance <= 0.0:
            raise ValueError("exit_width_log_tolerance must be positive and finite")
        if not math.isfinite(self.exit_vertical_profile_tolerance) or self.exit_vertical_profile_tolerance <= 0.0:
            raise ValueError("exit_vertical_profile_tolerance must be positive and finite")
        if self.maximum_exit_count_difference < 0:
            raise ValueError("maximum_exit_count_difference must be nonnegative")
        if not 0.0 <= self.ambiguity_similarity_margin <= 2.0:
            raise ValueError("ambiguity_similarity_margin must be in [0,2]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unit(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if array.ndim != 1 or norm <= 1e-12 or not np.all(np.isfinite(array)):
        raise ValueError("descriptor must be a finite non-zero vector")
    return array / norm


def _minimum_assignment(cost: np.ndarray) -> tuple[int, ...]:
    """Assign every column to one unique row for token sets of size at most six."""

    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] > values.shape[0]:
        raise ValueError("assignment requires rows >= columns")

    @lru_cache(maxsize=None)
    def solve(column: int, used: int) -> tuple[float, tuple[int, ...]]:
        if column == values.shape[1]:
            return 0.0, ()
        best: tuple[float, tuple[int, ...]] | None = None
        for row in range(values.shape[0]):
            if used & (1 << row):
                continue
            tail_cost, tail = solve(column + 1, used | (1 << row))
            candidate = (float(values[row, column]) + tail_cost, (row,) + tail)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("no complete exit-token assignment")
        return best

    return solve(0, 0)[1]


def _exit_token_match(
    query_tokens: Sequence[Any],
    query_yaw_deg: float,
    node: dict[str, Any],
    config: GSEGraphConfig,
) -> dict[str, float] | None:
    stored_tokens = tuple(node["exit_tokens"])
    if abs(len(query_tokens) - len(stored_tokens)) > config.maximum_exit_count_difference:
        return None
    if not query_tokens and not stored_tokens:
        return {
            "mean_cost": 0.0,
            "maximum_heading_error_deg": 0.0,
            "maximum_width_log_error": 0.0,
            "minimum_descriptor_similarity": 1.0,
            "maximum_vertical_profile_error": 0.0,
        }
    if not query_tokens or not stored_tokens:
        return None
    query = [
        {
            "heading": wrap_deg(float(token.heading_robot_deg) + query_yaw_deg),
            "width": float(token.opening_width_m),
            "descriptor": _unit(token.descriptor),
            "vertical_profile": np.asarray(token.vertical_profile, dtype=np.float64),
        }
        for token in query_tokens
    ]
    stored = [
        {
            "heading": wrap_deg(float(token["heading_robot_deg"]) + float(node["yaw_deg"])),
            "width": float(token["opening_width_m"]),
            "descriptor": _unit(token["descriptor"]),
            "vertical_profile": np.asarray(token["vertical_profile"], dtype=np.float64),
        }
        for token in stored_tokens
    ]
    rows, columns = (query, stored) if len(query) >= len(stored) else (stored, query)
    heading = np.empty((len(rows), len(columns)), dtype=np.float64)
    width = np.empty_like(heading)
    similarity = np.empty_like(heading)
    vertical = np.empty_like(heading)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            heading[row_index, column_index] = abs(
                (float(row["heading"]) - float(column["heading"]) + 180.0) % 360.0 - 180.0
            )
            width[row_index, column_index] = abs(
                math.log(float(row["width"]) / float(column["width"]))
            )
            similarity[row_index, column_index] = float(row["descriptor"] @ column["descriptor"])
            if row["vertical_profile"].shape != column["vertical_profile"].shape:
                return None
            vertical[row_index, column_index] = float(
                np.mean(np.abs(row["vertical_profile"] - column["vertical_profile"]))
            )
    cost = (
        heading / config.exit_heading_tolerance_deg
        + width / config.exit_width_log_tolerance
        + (1.0 - similarity) / 2.0
        + vertical / config.exit_vertical_profile_tolerance
    )
    assignment = _minimum_assignment(cost)
    pairs = tuple((row, column) for column, row in enumerate(assignment))
    matched_heading = [float(heading[row, column]) for row, column in pairs]
    matched_width = [float(width[row, column]) for row, column in pairs]
    matched_similarity = [float(similarity[row, column]) for row, column in pairs]
    matched_vertical = [float(vertical[row, column]) for row, column in pairs]
    if (
        max(matched_heading) > config.exit_heading_tolerance_deg
        or max(matched_width) > config.exit_width_log_tolerance
        or min(matched_similarity) < config.exit_descriptor_minimum_similarity
        or max(matched_vertical) > config.exit_vertical_profile_tolerance
    ):
        return None
    return {
        "mean_cost": float(np.mean([cost[row, column] for row, column in pairs])),
        "maximum_heading_error_deg": max(matched_heading),
        "maximum_width_log_error": max(matched_width),
        "minimum_descriptor_similarity": min(matched_similarity),
        "maximum_vertical_profile_error": max(matched_vertical),
    }


class GeometrySemanticEventGraph:
    """Online graph consuming only typed learned observations and past state."""

    def __init__(
        self,
        config: GSEGraphConfig,
        *,
        association_reason: str = "learned_association",
        association_backend: Any | None = None,
        node_generation_backend: Any | None = None,
        structural_events: frozenset[str] | None = None,
        reject_multiple_backend_accepts: bool = False,
    ) -> None:
        if association_reason not in {
            "learned_association",
            "rule_association",
            "frozen_exit_token_ensemble",
            "factorized_consensus_metric",
        }:
            raise ValueError("association_reason must identify learned or rule association")
        backend_reason = association_reason in {
            "frozen_exit_token_ensemble", "factorized_consensus_metric",
        }
        if backend_reason != (association_backend is not None):
            raise ValueError("frozen ensemble reason and backend must be provided together")
        structural = (
            frozenset(EVENT_NAMES) - {StructuralEvent.CORRIDOR.value}
            if structural_events is None else frozenset(str(value) for value in structural_events)
        )
        if (
            not structural
            or StructuralEvent.CORRIDOR.value in structural
            or not structural.issubset(set(EVENT_NAMES))
        ):
            raise ValueError("structural_events must be non-corridor event names")
        self.config = config
        self.association_reason = association_reason
        self.association_backend = association_backend
        self.node_generation_backend = node_generation_backend
        self.structural_events = structural
        self.reject_multiple_backend_accepts = bool(reject_multiple_backend_accepts)
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.current_node: int | None = None
        self.current_node_route_arc_m = 0.0
        self._candidate_is_structural: bool | None = None
        self._candidate_frames = 0
        self._emitted_structural_episode = False
        self._last_frame_index: int | None = None
        self._last_route_arc_m: float | None = None
        self._trace: list[dict[str, Any]] = []
        self.decision_trace: list[dict[str, Any]] = []
        self._node_generation_decision: dict[str, Any] | None = None
        self._spatial_cells: dict[tuple[int, int, int], list[int]] = {}

    def _spatial_cell(self, xyz: Sequence[float]) -> tuple[int, int, int]:
        value = np.asarray(xyz, dtype=np.float64) / self.config.association_radius_m
        return tuple(int(item) for item in np.floor(value))

    def _nearby_nodes(self, xyz: np.ndarray) -> list[dict[str, Any]]:
        cell = self._spatial_cell(xyz)
        node_ids = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    node_ids.extend(
                        self._spatial_cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ())
                    )
        return [self.nodes[node_id] for node_id in sorted(node_ids)]

    def update(
        self,
        *,
        frame_index: int,
        route_arc_m: float,
        xyz_m: Sequence[float],
        yaw_deg: float,
        observation: GeometricSemanticObservation,
        association_key: int | None = None,
    ) -> dict[str, Any]:
        xyz = np.asarray(xyz_m, dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("xyz_m must contain three finite values")
        if self._last_frame_index is not None and int(frame_index) <= self._last_frame_index:
            raise ValueError("frame_index must increase strictly")
        if self._last_route_arc_m is not None and float(route_arc_m) + 1e-12 < self._last_route_arc_m:
            raise ValueError("route_arc_m must be monotonic")
        self._last_frame_index = int(frame_index)
        self._last_route_arc_m = float(route_arc_m)
        if (self.association_backend is not None or self.node_generation_backend is not None) and association_key is None:
            raise ValueError("frozen deployment backend requires an observation key")
        self._trace.append(
            {
                "frame_index": int(frame_index),
                "route_arc_m": float(route_arc_m),
                "width_m": observation.width_m,
                "height_m": observation.height_m,
                "slope_deg": observation.slope_deg,
                "curvature_per_m": observation.curvature_per_m,
                "local_axis": list(observation.local_axis),
                "uncertainty": observation.uncertainty,
            }
        )

        self._node_generation_decision = (
            self.node_generation_backend.evaluate_node(int(association_key))
            if self.node_generation_backend is not None else None
        )
        fused_event = (
            None
            if self._node_generation_decision is None
            else self._node_generation_decision.get("mean_event_probability")
        )
        if fused_event is None:
            event = observation.event.value
            event_probability = float(observation.event_probabilities[event])
            event_confident = (
                event_probability >= self.config.event_probability_threshold
                and observation.uncertainty <= self.config.maximum_uncertainty
            )
        else:
            probabilities = np.asarray(fused_event, dtype=np.float64)
            if (
                probabilities.shape != (len(EVENT_NAMES),)
                or not np.all(np.isfinite(probabilities))
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
            ):
                raise RuntimeError("frozen event-node backend returned invalid event probabilities")
            event = EVENT_NAMES[int(np.argmax(probabilities))]
            event_probability = float(np.max(probabilities))
            # The product score is the separately calibrated confidence gate.
            # Mean event probabilities provide only the causal episode state
            # and class, so the legacy per-seed uncertainty gate is not applied
            # a second time.
            event_confident = True
        is_structural_event = event in self.structural_events
        node_accepted = (
            self._node_generation_decision is None
            or bool(self._node_generation_decision["accepted"])
        )
        if is_structural_event == self._candidate_is_structural and event_confident:
            self._candidate_frames += 1
        else:
            self._candidate_is_structural = is_structural_event if event_confident else None
            self._candidate_frames = 1 if event_confident else 0
        if (
            not is_structural_event
            and event_confident
            and self._candidate_frames >= self.config.stable_event_frames
        ):
            self._emitted_structural_episode = False

        headings_world = tuple(
            sorted(wrap_deg(token.heading_robot_deg + yaw_deg) for token in observation.exit_tokens)
        )
        if self.current_node is None:
            initial_kind = (
                "structural"
                if is_structural_event and event_confident and node_accepted
                else "anchor"
            )
            node_id = self._append_node(
                xyz=xyz,
                route_arc_m=route_arc_m,
                yaw_deg=yaw_deg,
                observation=observation,
                headings_world=headings_world,
                node_kind=initial_kind,
                association_status="persistent",
                frame_index=frame_index,
                reason="route_start",
                association_key=association_key,
            )
            self.current_node = node_id
            self.current_node_route_arc_m = float(route_arc_m)
            if initial_kind == "structural":
                self._emitted_structural_episode = True
            return self._record(frame_index, route_arc_m, node_id, True, "route_start", "persistent")

        travel = float(route_arc_m - self.current_node_route_arc_m)
        structural = (
            is_structural_event
            and event_confident
            and node_accepted
            and self._candidate_frames >= self.config.stable_event_frames
            and travel >= self.config.minimum_event_travel_m
            and not self._emitted_structural_episode
        )
        metric_anchor = (
            not is_structural_event
            and travel >= self.config.metric_anchor_interval_m
        )
        if not structural and not metric_anchor:
            return self._record(frame_index, route_arc_m, self.current_node, False, "no_event", None)

        node_kind = "structural" if structural else "anchor"
        match, ambiguous, candidates = self._associate(
            xyz=xyz,
            event=event,
            headings_world=headings_world,
            descriptor=observation.place_descriptor,
            exit_tokens=observation.exit_tokens,
            yaw_deg=yaw_deg,
            node_kind=node_kind,
            association_key=association_key,
        )
        previous = int(self.current_node)
        if ambiguous:
            target = self._append_node(
                xyz=xyz,
                route_arc_m=route_arc_m,
                yaw_deg=yaw_deg,
                observation=observation,
                headings_world=headings_world,
                node_kind=node_kind,
                association_status="provisional",
                frame_index=frame_index,
                reason="ambiguous_association",
                association_key=association_key,
            )
            created = True
            reason = "ambiguous_association"
            association_status = "provisional"
        elif match is None:
            target = self._append_node(
                xyz=xyz,
                route_arc_m=route_arc_m,
                yaw_deg=yaw_deg,
                observation=observation,
                headings_world=headings_world,
                node_kind=node_kind,
                association_status="persistent",
                frame_index=frame_index,
                reason=f"new_{event}" if structural else "metric_anchor",
                association_key=association_key,
            )
            created = True
            reason = f"new_{event}" if structural else "metric_anchor"
            association_status = "persistent"
        else:
            target = match
            created = False
            reason = self.association_reason
            association_status = self.nodes[target]["association_status"]
            self._update_node(target, observation, headings_world, frame_index, association_key)
            if association_status == "provisional":
                self.nodes[target]["association_status"] = "persistent"
                association_status = "persistent"

        if target != previous:
            self._append_trace_verified_edge(previous, target, frame_index, route_arc_m)
        if structural:
            self._emitted_structural_episode = True
        self.current_node = target
        self.current_node_route_arc_m = float(route_arc_m)
        self._trace = [self._trace[-1]]
        record = self._record(frame_index, route_arc_m, target, created, reason, association_status)
        record["association_candidate_count"] = len(candidates)
        if self.association_backend is not None:
            record["association_accepted_candidate_count"] = sum(
                bool(candidate.get("accepted")) for candidate in candidates
            )
            record["association_candidates"] = candidates
        return record

    def _associate(
        self,
        *,
        xyz: np.ndarray,
        event: str,
        headings_world: Sequence[float],
        descriptor: Sequence[float],
        exit_tokens: Sequence[Any],
        yaw_deg: float,
        node_kind: str,
        association_key: int | None = None,
    ) -> tuple[int | None, bool, list[dict[str, float | int]]]:
        if self.association_backend is not None:
            if association_key is None:
                raise ValueError("frozen ensemble association key is missing")
            candidates: list[dict[str, Any]] = []
            accepted: list[dict[str, Any]] = []
            for node in self._nearby_nodes(xyz):
                if node["node_kind"] != node_kind:
                    continue
                distance = float(np.linalg.norm(xyz - np.asarray(node["xyz_m"], dtype=np.float64)))
                if distance > self.config.association_radius_m:
                    continue
                decision = self.association_backend.evaluate_pair(
                    int(association_key), int(node["association_key"]), distance
                )
                candidate = {
                    "node_id": int(node["id"]),
                    "event_compatible": bool(node["event"] == event),
                    **decision,
                }
                candidates.append(candidate)
                if bool(decision["accepted"]):
                    accepted.append(candidate)
            candidates.sort(
                key=lambda item: (
                    -float(item["ensemble_score"]),
                    float(item["distance_m"]),
                    int(item["node_id"]),
                )
            )
            accepted.sort(
                key=lambda item: (
                    -float(item["ensemble_score"]),
                    float(item["distance_m"]),
                    int(item["node_id"]),
                )
            )
            if not accepted:
                return None, False, candidates
            if len(accepted) > 1:
                if self.reject_multiple_backend_accepts:
                    return None, True, candidates
                score_gap = float(accepted[0]["ensemble_score"]) - float(accepted[1]["ensemble_score"])
                if score_gap < self.config.ambiguity_similarity_margin:
                    return None, True, candidates
            return int(accepted[0]["node_id"]), False, candidates
        query = _unit(descriptor)
        candidates: list[dict[str, float | int]] = []
        for node in self._nearby_nodes(xyz):
            if node["node_kind"] != node_kind:
                continue
            if node_kind == "structural" and node["event"] != event:
                continue
            distance = float(np.linalg.norm(xyz - np.asarray(node["xyz_m"], dtype=np.float64)))
            if distance > self.config.association_radius_m:
                continue
            similarity = float(query @ _unit(node["place_descriptor"]))
            if similarity < self.config.descriptor_minimum_similarity:
                continue
            exit_distance = heading_set_distance(headings_world, node["exit_headings_world_deg"])
            if exit_distance > self.config.exit_heading_tolerance_deg:
                continue
            exit_match = _exit_token_match(exit_tokens, yaw_deg, node, self.config)
            if exit_match is None:
                continue
            candidates.append(
                {
                    "node_id": int(node["id"]),
                    "similarity": similarity,
                    "exit_distance_deg": exit_distance,
                    "exit_match_cost": exit_match["mean_cost"],
                    "exit_descriptor_similarity": exit_match["minimum_descriptor_similarity"],
                    "exit_width_log_error": exit_match["maximum_width_log_error"],
                    "exit_vertical_profile_error": exit_match["maximum_vertical_profile_error"],
                    "distance_m": distance,
                }
            )
        candidates.sort(key=lambda item: (-float(item["similarity"]), float(item["exit_match_cost"]), float(item["distance_m"]), int(item["node_id"])))
        if not candidates:
            return None, False, candidates
        if len(candidates) > 1:
            similarity_gap = float(candidates[0]["similarity"]) - float(candidates[1]["similarity"])
            if similarity_gap < self.config.ambiguity_similarity_margin:
                return None, True, candidates
        return int(candidates[0]["node_id"]), False, candidates

    def _append_node(
        self,
        *,
        xyz: np.ndarray,
        route_arc_m: float,
        yaw_deg: float,
        observation: GeometricSemanticObservation,
        headings_world: Sequence[float],
        node_kind: str,
        association_status: str,
        frame_index: int,
        reason: str,
        association_key: int | None = None,
    ) -> int:
        node_id = len(self.nodes)
        self.nodes.append(
            {
                "id": node_id,
                "xyz_m": xyz.astype(float).tolist(),
                "route_arc_m": float(route_arc_m),
                "yaw_deg": float(yaw_deg),
                "node_kind": node_kind,
                "association_status": association_status,
                "event": observation.event.value,
                "event_probabilities": dict(observation.event_probabilities),
                "uncertainty": observation.uncertainty,
                "place_descriptor": list(observation.place_descriptor),
                "exit_headings_world_deg": list(headings_world),
                "exit_tokens": [token.to_dict() for token in observation.exit_tokens],
                "geometry": {
                    "width_m": observation.width_m,
                    "height_m": observation.height_m,
                    "slope_deg": observation.slope_deg,
                    "curvature_per_m": observation.curvature_per_m,
                },
                "observation_frames": [int(frame_index)],
                "creation_reason": reason,
                "association_key": None if association_key is None else int(association_key),
                "association_backend": self.association_reason,
            }
        )
        self._spatial_cells.setdefault(self._spatial_cell(xyz), []).append(node_id)
        return node_id

    def _update_node(
        self,
        node_id: int,
        observation: GeometricSemanticObservation,
        headings_world: Sequence[float],
        frame_index: int,
        association_key: int | None = None,
    ) -> None:
        node = self.nodes[node_id]
        node["observation_frames"].append(int(frame_index))
        node["uncertainty"] = min(float(node["uncertainty"]), observation.uncertainty)
        node["place_descriptor"] = list(observation.place_descriptor)
        node["exit_headings_world_deg"] = list(headings_world)
        node["exit_tokens"] = [token.to_dict() for token in observation.exit_tokens]
        if self.association_backend is not None:
            if association_key is None:
                raise ValueError("frozen ensemble node update lacks an association key")
            # The deployment representative remains the node-creation
            # observation because its frozen payload and node xyz are one
            # auditable pair.  Revisit evidence is retained in
            # observation_frames but cannot silently replace that pair.
            if node["association_key"] is None:
                raise RuntimeError("frozen ensemble node lacks its creation representative")

    def _append_trace_verified_edge(self, previous: int, target: int, frame_index: int, route_arc_m: float) -> None:
        if len(self._trace) < 2:
            raise RuntimeError("edge creation requires a multi-frame physical trace")
        start_arc = float(self._trace[0]["route_arc_m"])
        length = float(route_arc_m - start_arc)
        if length <= 0.0:
            raise RuntimeError("edge creation requires positive traversed distance")
        widths = [float(item["width_m"]) for item in self._trace]
        heights = [float(item["height_m"]) for item in self._trace]
        slopes = [float(item["slope_deg"]) for item in self._trace]
        curvatures = [float(item["curvature_per_m"]) for item in self._trace]
        traversal = {
            "start_frame": int(self._trace[0]["frame_index"]),
            "end_frame": int(frame_index),
            "length_m": length,
        }
        geometry = {
            "minimum_width_m": min(widths),
            "minimum_height_m": min(heights),
            "maximum_abs_slope_deg": max(abs(value) for value in slopes),
            "maximum_curvature_per_m": max(curvatures),
        }
        existing = next(
            (
                edge
                for edge in self.edges
                if {int(edge["from"]), int(edge["to"])} == {int(previous), int(target)}
            ),
            None,
        )
        if existing is not None:
            count = int(existing["traversal_count"])
            existing["traversal_count"] = count + 1
            existing["length_m"] = (float(existing["length_m"]) * count + length) / (count + 1)
            existing["minimum_length_m"] = min(float(existing["minimum_length_m"]), length)
            existing["maximum_length_m"] = max(float(existing["maximum_length_m"]), length)
            existing["geometry"]["minimum_width_m"] = min(
                float(existing["geometry"]["minimum_width_m"]), geometry["minimum_width_m"]
            )
            existing["geometry"]["minimum_height_m"] = min(
                float(existing["geometry"]["minimum_height_m"]), geometry["minimum_height_m"]
            )
            existing["geometry"]["maximum_abs_slope_deg"] = max(
                float(existing["geometry"]["maximum_abs_slope_deg"]), geometry["maximum_abs_slope_deg"]
            )
            existing["geometry"]["maximum_curvature_per_m"] = max(
                float(existing["geometry"]["maximum_curvature_per_m"]), geometry["maximum_curvature_per_m"]
            )
            existing["traversals"].append(traversal)
            return
        self.edges.append(
            {
                "id": len(self.edges),
                "from": int(previous),
                "to": int(target),
                "kind": "trace_verified",
                "start_frame": traversal["start_frame"],
                "end_frame": traversal["end_frame"],
                "length_m": length,
                "minimum_length_m": length,
                "maximum_length_m": length,
                "traversal_count": 1,
                "traversals": [traversal],
                "geometry": geometry,
            }
        )

    def _record(self, frame_index: int, route_arc_m: float, node_id: int, created: bool, reason: str, association_status: str | None) -> dict[str, Any]:
        record = {
            "frame_index": int(frame_index),
            "route_arc_m": float(route_arc_m),
            "node_id": int(node_id),
            "created": bool(created),
            "reason": reason,
            "association_status": association_status,
        }
        if self._node_generation_decision is not None:
            record["node_generation_decision"] = self._node_generation_decision
        self.decision_trace.append(record)
        return record


__all__ = ["GSEGraphConfig", "GeometrySemanticEventGraph"]
