"""Known-pose, class-free, segmented multi-anchor software adapter.

Reuses the explicit SegmentCoordinate and existing proper-SE3 validation. The
legacy graph kernel cannot represent multiple independent anchor positions, so
it is intentionally not fed synthetic junction labels or fabricated robot poses.
This is not deployment binding, calibrated tracking, ICP or cross-visit fusion.
Only uninterrupted short-term 3D tracks may retain a node identity. Opening
observations are evidence records, NOT already matched persistent graph ports.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import re

import numpy as np

from .gse_graph_frames import _rotation
from .gse_segmented_graph_v1 import SegmentCoordinate
from .gse_surface_observation_v1 import SurfaceObservationV1, _index, _number, _probability, _vector


@dataclass(frozen=True)
class SurfaceGraphConfigV1:
    stable_observations: int
    anchor_presence_threshold: float
    maximum_anchor_uncertainty_m: float
    short_track_radius_m: float
    arrival_radius_m: float
    opening_presence_threshold: float
    relation_threshold: float
    calibration_status: str = "SOFTWARE_FIXTURE_NOT_CALIBRATED"
    anchor_uncertainty_policy: str = "require_supplied_scale"

    def __post_init__(self):
        if self.anchor_uncertainty_policy not in ('require_supplied_scale','known_pose_partial_diagnostic'):
            raise ValueError('explicit anchor uncertainty policy required')
        _index(self.stable_observations)
        if self.stable_observations < 1:
            raise ValueError("positive stability count required")
        for p in (self.anchor_presence_threshold, self.opening_presence_threshold, self.relation_threshold):
            _probability(p)
        for length in (self.maximum_anchor_uncertainty_m, self.short_track_radius_m, self.arrival_radius_m):
            _number(length)
            if length <= 0:
                raise ValueError("explicit positive metric software parameters required")
        if self.calibration_status != "SOFTWARE_FIXTURE_NOT_CALIBRATED":
            raise ValueError("this adapter has no production calibration or safety qualification")


@dataclass(frozen=True)
class KnownSurfacePoseV1:
    coordinate: SegmentCoordinate
    reference_frame: str
    rotation_reference_from_robot: tuple[tuple[float, float, float], ...]
    translation_reference_from_robot_m: tuple[float, float, float]

    def __post_init__(self):
        if type(self.coordinate) is not SegmentCoordinate:
            raise ValueError("explicit seconds or order-index coordinate required")
        if type(self.reference_frame) is not str or not self.reference_frame.strip():
            raise ValueError("known common reference frame required")
        _vector(self.translation_reference_from_robot_m, 3)
        if type(self.rotation_reference_from_robot) is not tuple or len(self.rotation_reference_from_robot) != 3:
            raise ValueError("immutable rotation tuple required")
        for row in self.rotation_reference_from_robot:
            _vector(row, 3)
        _rotation(self.rotation_reference_from_robot)
        # No invented timestamp, covariance or claimed localization accuracy.


@dataclass(frozen=True)
class SurfaceExecutionStartV1:
    """External execution-start receipt; not model belief or safety approval.

    This software DTO does not authenticate an actual controller. Recorded
    poses are still necessary, and no collision qualification is implied.
    """
    execution_key: str
    runtime_stream_key: str
    decision_index: int
    opening_query_index: int
    input_binding_sha256: str

    def __post_init__(self):
        for value in (self.execution_key, self.runtime_stream_key):
            if type(value) is not str or not value.strip():
                raise ValueError("explicit external execution/source key required")
        _index(self.decision_index); _index(self.opening_query_index, 64)
        if type(self.input_binding_sha256) is not str or re.fullmatch(r"[a-f0-9]{64}", self.input_binding_sha256) is None:
            raise ValueError("execution must bind exact current observation digest")


class SurfaceSegmentedGraphV1:
    def __init__(self, config: SurfaceGraphConfigV1):
        if type(config) is not SurfaceGraphConfigV1:
            raise ValueError("explicit software graph configuration required")
        self.config = config
        self._nodes, self._edges, self._trace = [], [], []
        self._seen_segments = set()
        self._segment = self._stream = self._kind = self._reference = None
        self._last_pose = self._last_observation = None
        self._stream_progress, self._key_orders, self._order_keys = {}, {}, {}
        self._tracks, self._next_track = {}, 0
        self._query_nodes, self._current_node = {}, None
        self._pending = None
        self._next_departure = 0
        self._departure_states = {}
        self._execution_keys = set()
        self._unresolved_traces = []

    def _record(self, item):
        self._trace.append(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return deepcopy(item)

    @property
    def immutable_trace(self):
        return tuple(self._trace)

    def begin_segment(self, segment_id, *, runtime_stream_key, coordinate_kind):
        if self._segment is not None:
            raise ValueError("end current segment first")
        if type(segment_id) is not str or not segment_id.strip() or segment_id in self._seen_segments:
            raise ValueError("unique nonempty segment ID required")
        if type(runtime_stream_key) is not str or not runtime_stream_key.strip():
            raise ValueError("explicit runtime stream required")
        if coordinate_kind not in ("seconds", "order_index"):
            raise ValueError("explicit coordinate kind required")
        self._segment, self._stream, self._kind = segment_id, runtime_stream_key, coordinate_kind
        self._seen_segments.add(segment_id)
        return self._record({"action": "begin_segment", "segment_id": segment_id,
                             "runtime_stream_key": runtime_stream_key, "coordinate_kind": coordinate_kind})

    def _require_active(self):
        if self._segment is None:
            raise ValueError("begin_segment required")

    def _validate_update(self, observation, pose, continuous):
        self._require_active()
        if type(observation) is not SurfaceObservationV1 or type(pose) is not KnownSurfacePoseV1 or type(continuous) is not bool:
            raise ValueError("typed observation/known pose and boolean continuity required")
        if observation.runtime_stream_key != self._stream or pose.coordinate.kind != self._kind:
            raise ValueError("segment/stream/coordinate binding drift")
        if self._reference is not None and pose.reference_frame != self._reference:
            raise ValueError("common reference frame changed")
        if self._last_pose is not None and pose.coordinate.value <= self._last_pose.coordinate.value:
            raise ValueError("segment coordinate must strictly increase")
        if pose.coordinate.kind == "order_index" and pose.coordinate.value != observation.decision_index:
            raise ValueError("order-index pose must bind this observation decision")
        if pose.coordinate.kind == "seconds" and pose.coordinate.value != observation.timestamp_s:
            raise ValueError("seconds pose must match explicit actual observation timestamp")
        previous = self._stream_progress.get(self._stream)
        if previous is not None and (observation.decision_index <= previous[0] or observation.current_source_index <= previous[1]):
            raise ValueError("source current frame and decision must both strictly increase per stream")
        for frame in observation.source_frames:
            old_order = self._key_orders.get((self._stream, frame.frame_key))
            old_key = self._order_keys.get((self._stream, frame.order_index))
            if (old_order is not None and old_order != frame.order_index) or (old_key is not None and old_key != frame.frame_key):
                raise ValueError("source identity/order remapping across observations")

    def update(self, observation: SurfaceObservationV1, pose: KnownSurfacePoseV1, *, continuous=True):
        self._validate_update(observation, pose, continuous)
        # Validation/computation are transactional: rejected input cannot partly
        # mutate source ledgers, tracks, execution states or the immutable trace.
        trial = deepcopy(self)
        result = trial._apply(observation, pose, continuous)
        self.__dict__.update(trial.__dict__)
        return result

    def _apply(self, observation, pose, continuous):
        self._reference = pose.reference_frame
        self._stream_progress[self._stream] = (observation.decision_index, observation.current_source_index)
        for f in observation.source_frames:
            self._key_orders[self._stream, f.frame_key] = f.order_index
            self._order_keys[self._stream, f.order_index] = f.frame_key
        coordinate = asdict(pose.coordinate)
        pose_record = {"segment_id": self._segment, "coordinate": coordinate,
                       "xyz_m": pose.translation_reference_from_robot_m,
                       "rotation_reference_from_robot": pose.rotation_reference_from_robot,
                       "input_binding_sha256": observation.input_binding_sha256}
        if self._pending is not None:
            self._pending["poses"].append(pose_record)
            self._pending["continuous"] &= continuous
        rotation = np.asarray(pose.rotation_reference_from_robot)
        origin = np.asarray(pose.translation_reference_from_robot_m)
        qualified = []
        unknown = []
        for a in observation.anchors:
            require_scale = self.config.anchor_uncertainty_policy == 'require_supplied_scale'
            if (not a.numerically_supported or a.position_robot_m is None
                    or a.existence_probability < self.config.anchor_presence_threshold
                    or (require_scale and (a.uncertainty_m is None
                        or max(a.uncertainty_m) > self.config.maximum_anchor_uncertainty_m))):
                unknown.append(a.query_index)
                continue
            xyz = tuple(float(v) for v in rotation @ np.asarray(a.position_robot_m) + origin)
            if not all(math.isfinite(v) for v in xyz):
                raise ValueError("anchor transform overflow")
            qualified.append((a.query_index, xyz))
        qualified.sort(key=lambda value: (value[1], value[0]))
        old = self._tracks if continuous else {}
        matches = {q: tuple(t for t, track in old.items() if math.dist(xyz, track["xyz_m"]) <= self.config.short_track_radius_m)
                   for q, xyz in qualified}
        claims = Counter(t for candidates in matches.values() for t in candidates)
        close_observations = {q for q, xyz in qualified if any(
            q != other and math.dist(xyz, other_xyz) <= self.config.short_track_radius_m for other, other_xyz in qualified)}
        tracks, decisions, self._query_nodes = {}, [], {}
        for query, xyz in qualified:
            candidates = matches[query]
            if query in close_observations or len(candidates) > 1 or (candidates and claims[candidates[0]] != 1):
                decisions.append({"query_index": query, "action": "pending_ambiguous_short_track", "candidate_tracks": candidates})
                continue
            if candidates:
                track_id = candidates[0]; track = deepcopy(old[track_id]); track["count"] += 1
            else:
                track_id = self._next_track; self._next_track += 1
                track = {"xyz_m": xyz, "count": 1, "node_id": None}
            action = "short_track_only"
            if track["node_id"] is None and track["count"] >= self.config.stable_observations:
                node_id = len(self._nodes)
                self._nodes.append({"id": node_id, "xyz_m": track["xyz_m"], "reference_frame": self._reference,
                    "created_segment": self._segment, "created_coordinate": coordinate,
                    "creation_reason": "stable_independent_3d_anchor", "opening_observations": []})
                track["node_id"] = node_id; action = "new_local_node_not_loop_merge"
            elif track["node_id"] is not None:
                action = "continued_short_track_not_revisit_merge"
            tracks[track_id] = track
            if track["node_id"] is not None:
                self._query_nodes[query] = track["node_id"]
            decisions.append({"query_index": query, "action": action, "track_id": track_id, "node_id": track["node_id"]})
        self._tracks = tracks  # Missing/unknown/ambiguous tracks expire; no hidden long-range identity reuse.
        for opening in observation.openings:
            for query, node in sorted(self._query_nodes.items()):
                if opening.anchor_relation_valid[query]:
                    self._nodes[node]["opening_observations"].append({"decision_index": observation.decision_index,
                        "source_query_index": opening.query_index, "anchor_query_index": query,
                        "relation_probability": opening.anchor_relation_probability[query],
                        "opening_robot_frame": asdict(opening), "pose": pose_record})
        arrivals = sorted({node for node in self._query_nodes.values()
                           if math.dist(self._nodes[node]["xyz_m"], pose.translation_reference_from_robot_m) <= self.config.arrival_radius_m})
        arrival_action, edge_id = "no_arrival", None
        if len(arrivals) > 1:
            arrival_action = "arrival_ambiguous_no_edge"
        elif arrivals:
            node = arrivals[0]
            arrival_action = "known_pose_arrival"
            if self._pending is not None and node != self._pending["source_node"]:
                pending = self._pending
                length = sum(math.dist(a["xyz_m"], b["xyz_m"]) for a, b in zip(pending["poses"], pending["poses"][1:]))
                if not math.isfinite(length):
                    raise ValueError("recorded trace length overflow")
                # A segment passing close to a known intermediate anchor is
                # NOT a sampled arrival. Preserve this unresolved polyline,
                # never interpolate arrival or assert an adjacent A->C edge.
                intermediate = []
                for candidate in self._nodes:
                    if candidate["id"] in (pending["source_node"], node):
                        continue
                    p = np.asarray(candidate["xyz_m"])
                    for a, b in zip(pending["poses"], pending["poses"][1:]):
                        start, finish = np.asarray(a["xyz_m"]), np.asarray(b["xyz_m"])
                        delta = finish - start
                        scale = float(delta @ delta)
                        fraction = float(np.clip((p - start) @ delta / scale, 0., 1.)) if scale else 0.
                        if float(np.linalg.norm(p - (start + fraction * delta))) <= self.config.arrival_radius_m:
                            intermediate.append(candidate["id"])
                            break
                if intermediate and pending["continuous"] and length > 0:
                    self._unresolved_traces.append({**deepcopy(pending), "destination_node": node,
                        "intermediate_candidate_nodes": intermediate,
                        "reason": "polyline_near_intermediate_anchor_without_recorded_arrival"})
                    self._departure_states[pending["departure_id"]] = "unresolved_intermediate_anchor"
                    arrival_action = "unresolved_trace_no_adjacent_edge"
                elif pending["continuous"] and length > 0:
                    edge_id = len(self._edges)
                    self._edges.append({"id": edge_id, "from": pending["source_node"], "to": node,
                        "segment_id": self._segment, "kind": "recorded_traversed", "length_m": length,
                        "poses": deepcopy(pending["poses"]), "departure_id": pending["departure_id"],
                        "departure_opening": deepcopy(pending["opening"]),
                        "external_execution_start": deepcopy(pending["external_execution_start"]),
                        "qualification": "known_pose_trace_not_collision_or_deployment_certificate"})
                    self._departure_states[pending["departure_id"]] = "traversed"
                    arrival_action = "completed_adjacent_anchor_trace"
                else:
                    self._departure_states[pending["departure_id"]] = "failed_discontinuous_or_zero_length"
                    arrival_action = "trace_rejected"
                self._pending = None
            self._current_node = node
        self._last_pose, self._last_observation = pose, observation
        return self._record({"action": "observe", "segment_id": self._segment, "coordinate": coordinate,
            "decision_index": observation.decision_index, "current_source_index": observation.current_source_index,
            "source_frames": [asdict(f) for f in observation.source_frames],
            "input_binding_sha256": observation.input_binding_sha256,
            "anchor_decisions": decisions, "unqualified_anchor_queries": unknown,
            "opening_count": len(observation.openings), "arrival_action": arrival_action,
            "arrival_candidates": arrivals, "edge_id": edge_id})

    def depart(self, opening_query_index, *, execution_start=None):
        self._require_active(); _index(opening_query_index, 64)
        if type(execution_start) is not SurfaceExecutionStartV1:
            raise ValueError("explicit external execution-start evidence required; model reachability cannot authorize execution")
        if (self._last_observation is None or execution_start.runtime_stream_key != self._stream
                or execution_start.decision_index != self._last_observation.decision_index
                or execution_start.opening_query_index != opening_query_index
                or execution_start.input_binding_sha256 != self._last_observation.input_binding_sha256
                or execution_start.execution_key in self._execution_keys):
            raise ValueError("execution-start receipt source mismatch or duplicate key")
        if self._pending is not None or self._current_node is None or self._last_pose is None:
            raise ValueError("departure requires a current arrived node and no active trace")
        if math.dist(self._nodes[self._current_node]["xyz_m"], self._last_pose.translation_reference_from_robot_m) > self.config.arrival_radius_m:
            raise ValueError("cannot depart from a stale node after moving away")
        arrivals = {node for node in self._query_nodes.values() if math.dist(
            self._nodes[node]["xyz_m"], self._last_pose.translation_reference_from_robot_m) <= self.config.arrival_radius_m}
        if arrivals != {self._current_node}:
            raise ValueError("departure requires unique currently observed arrival")
        opening = next((o for o in self._last_observation.openings if o.query_index == opening_query_index), None)
        queries = [q for q, node in self._query_nodes.items() if node == self._current_node]
        if (opening is None or not opening.numerically_supported or opening.direction_robot is None
                or opening.existence_probability < self.config.opening_presence_threshold
                or not any(
                    opening.anchor_relation_valid[q] and opening.anchor_relation_probability[q] >= self.config.relation_threshold for q in queries)):
            raise ValueError("departure needs current supported opening relation; model reachability is not execution evidence")
        departure = self._next_departure; self._next_departure += 1
        self._execution_keys.add(execution_start.execution_key)
        self._departure_states[departure] = "attempted"
        pose = self._last_pose
        self._pending = {"departure_id": departure, "source_node": self._current_node,
            "opening": asdict(opening), "continuous": True, "external_execution_start": asdict(execution_start),
            "poses": [{"segment_id": self._segment, "coordinate": asdict(pose.coordinate),
                "xyz_m": pose.translation_reference_from_robot_m,
                "rotation_reference_from_robot": pose.rotation_reference_from_robot,
                "input_binding_sha256": self._last_observation.input_binding_sha256}]}
        return self._record({"action": "depart", "segment_id": self._segment,
            "coordinate": asdict(pose.coordinate), "departure_id": departure,
            "source_node": self._current_node, "opening_query_index": opening_query_index,
            "external_execution_start": asdict(execution_start)})

    def end_segment(self):
        self._require_active()
        pending = deepcopy(self._pending)
        if pending is not None:
            self._departure_states[pending["departure_id"]] = "abandoned_segment_end"
        record = self._record({"action": "end_segment", "segment_id": self._segment,
                               "abandoned_departure": pending})
        self._segment = self._stream = self._kind = self._last_pose = self._last_observation = None
        self._tracks, self._query_nodes, self._current_node, self._pending = {}, {}, None, None
        return record

    def snapshot(self):
        return deepcopy({"schema_version": "gse_surface_segmented_graph_v1", "config": asdict(self.config),
            "nodes": self._nodes, "edges": self._edges, "current_node": self._current_node,
            "active_segment": self._segment, "departure_states": self._departure_states,
            "unresolved_traces": self._unresolved_traces,
            "decision_trace": [json.loads(s) for s in self._trace],
            "cross_visit_merge_implemented": False, "persistent_port_matching_implemented": False,
            "deployment_source_binding_verified": False, "scientific_gate_pass": False,
            "time_metrics_available": False})
