"""Explicit segment boundaries around the unchanged causal graph kernel.

Only software/known-pose diagnostics: not LocalStructureObservationV2 binding,
6DoF registration, port-frame conversion or verified deployment association.
External association evidence is never synthesized by this wrapper.

The legacy timestamp_s slot receives a private monotonically increasing call
ordinal, NOT seconds. Public snapshots restore the explicit segment-local
coordinate and never expose that ordinal as a timestamp or duration metric.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math

from .gse_causal_graph import (
    AssociationEvidence, CausalStructuralGraph, PoseEstimate,
    StructuralObservation, StructuralPort,
)


@dataclass(frozen=True)
class SegmentCoordinate:
    kind: str
    value: int | float

    def __post_init__(self):
        if self.kind == "order_index":
            if type(self.value) is not int or self.value < 0:
                raise ValueError("order_index must be a nonnegative integer, not seconds")
        elif self.kind == "seconds":
            if type(self.value) not in (int, float) or not math.isfinite(self.value):
                raise ValueError("seconds require a finite explicit source timestamp")
        else:
            raise ValueError("coordinate kind must be seconds or order_index")


@dataclass(frozen=True)
class SegmentPose:
    xyz_m: tuple[float, float, float]
    covariance_diagonal_m2: tuple[float, float, float]

    def __post_init__(self):
        PoseEstimate(0., self.xyz_m, self.covariance_diagonal_m2)


@dataclass(frozen=True)
class SegmentObservation:
    event: str
    confidence: float
    uncertainty: float
    ports: tuple[StructuralPort, ...] = ()

    def __post_init__(self):
        StructuralObservation(0., self.event, self.confidence, self.uncertainty, self.ports)


@dataclass(frozen=True)
class SegmentAssociationEvidence:
    """Caller evidence for a graph-local candidate, not an ICP implementation."""
    candidate_node: int
    segment_id: str
    coordinate: SegmentCoordinate
    registration_passed: bool
    ports_passed: bool
    trace_context_passed: bool
    unambiguous: bool

    def __post_init__(self):
        if not isinstance(self.coordinate, SegmentCoordinate) or not isinstance(self.segment_id, str) or not self.segment_id:
            raise ValueError("association requires explicit segment/coordinate binding")
        AssociationEvidence(self.candidate_node, 0., self.registration_passed, self.ports_passed,
                            self.trace_context_passed, self.unambiguous)


class SegmentedCausalStructuralGraph:
    """Segment-local monotone input coordinates and persistent confirmed graph.

    Distinct segments may restart their order indices/timestamps. That does not
    establish temporal or traversal continuity between them. Settings are all
    caller-explicit, not newly calibrated scientific defaults.
    """
    def __init__(self, *, stable_frames, event_confidence, maximum_uncertainty):
        if type(stable_frames) is not int:
            raise ValueError("stable_frames must be a non-bool integer")
        self._kernel = CausalStructuralGraph(stable_frames=stable_frames,
            event_confidence=event_confidence, maximum_uncertainty=maximum_uncertainty)
        self._active = None
        self._kind = None
        self._last_coordinate = None
        self._seen = set()
        self._ordinal = 0
        self._coordinates = {}
        self._trace = []  # immutable JSON records internally; copies on export
        self._verified_ports = set()

    def _record(self, record):
        self._trace.append(json.dumps(record, sort_keys=True, allow_nan=False))
        return deepcopy(record)

    @property
    def immutable_trace(self):
        return tuple(self._trace)

    def begin_segment(self, segment_id: str, *, coordinate_kind: str):
        if self._active is not None:
            raise ValueError("end the active segment before beginning another")
        if type(segment_id) is not str or not segment_id or segment_id in self._seen:
            raise ValueError("segment IDs must be nonempty and never reused")
        if coordinate_kind not in ("order_index", "seconds"):
            raise ValueError("explicit coordinate kind required")
        self._active, self._kind, self._last_coordinate = segment_id, coordinate_kind, None
        self._seen.add(segment_id)
        return self._record({"action": "begin_segment", "segment_id": segment_id,
                             "coordinate_kind": coordinate_kind})

    def _require_active(self):
        if self._active is None:
            raise ValueError("begin_segment is required")

    def _public_pose(self, pose):
        return {"coordinate": deepcopy(self._coordinates[pose["timestamp_s"]]),
                "xyz_m": deepcopy(pose["xyz_m"]),
                "covariance_diagonal_m2": deepcopy(pose["covariance_diagonal_m2"])}

    def _restore_verified_ports(self):
        for node, port in self._verified_ports:
            self._kernel.nodes[node]["port_states"][port] = "traversed"

    def depart(self, port_id: int):
        self._require_active()
        if type(port_id) is not int or port_id < 0:
            raise ValueError("graph-local port ID must be a nonnegative integer")
        self._kernel.depart(port_id)
        self._restore_verified_ports()
        return self._record({"action": "depart", "segment_id": self._active,
            "coordinate": deepcopy(self._coordinates[self._kernel._last_pose.timestamp_s]),
            "source_node": self._kernel.current_node, "departure_port": port_id,
            "trace_id": self._kernel._pending["id"]})

    def update(self, observation: SegmentObservation, pose: SegmentPose, *,
               coordinate: SegmentCoordinate, candidates: tuple[SegmentAssociationEvidence, ...] = (),
               continuous=True):
        self._require_active()
        if not isinstance(observation, SegmentObservation) or not isinstance(pose, SegmentPose):
            raise ValueError("explicit segment observation/pose DTO required; no fabricated timestamps")
        if not isinstance(coordinate, SegmentCoordinate) or coordinate.kind != self._kind:
            raise ValueError("coordinate kind differs from active segment")
        if self._last_coordinate is not None and coordinate.value <= self._last_coordinate.value:
            raise ValueError("segment coordinates must increase strictly")
        if type(continuous) is not bool:
            raise ValueError("continuity evidence must be boolean")
        if not isinstance(candidates, tuple) or any(not isinstance(c, SegmentAssociationEvidence) for c in candidates):
            raise ValueError("explicit association evidence tuple required")
        for c in candidates:
            if c.segment_id != self._active or c.coordinate != coordinate:
                raise ValueError("association evidence has stale/wrong segment coordinate")
        # Increment only after success. The private ordinal never labels seconds.
        ordinal = self._ordinal
        old_observation = StructuralObservation(ordinal, observation.event, observation.confidence,
                                                observation.uncertainty, deepcopy(observation.ports))
        old_pose = PoseEstimate(ordinal, deepcopy(pose.xyz_m), deepcopy(pose.covariance_diagonal_m2))
        old_candidates = tuple(AssociationEvidence(c.candidate_node, ordinal, c.registration_passed,
            c.ports_passed, c.trace_context_passed, c.unambiguous) for c in candidates)
        result = self._kernel.update(old_observation, old_pose, old_candidates, continuous=continuous)
        self._coordinates[ordinal] = {"segment_id": self._active, "kind": coordinate.kind, "value": coordinate.value}
        self._ordinal += 1
        self._last_coordinate = coordinate
        if result.edge_id is not None:
            edge = self._kernel.edges[result.edge_id]
            self._verified_ports.add((edge.source_node, edge.departure_port))
        self._restore_verified_ports()
        return self._record({"action": result.action, "coordinate": deepcopy(self._coordinates[ordinal]),
            "segment_id": self._active, "node_id": result.node_id, "edge_id": result.edge_id,
            "candidate_nodes": result.candidate_nodes})

    def end_segment(self):
        self._require_active()
        pending = self._kernel._pending
        interrupted = None
        if pending is not None:
            interrupted = {"trace_id": pending["id"], "source_node": pending["source"],
                "departure_port": pending["port"], "reason": "SEGMENT_ENDED_BEFORE_ARRIVAL",
                "poses": [self._public_pose({"timestamp_s": p.timestamp_s, "xyz_m": p.xyz_m,
                    "covariance_diagonal_m2": p.covariance_diagonal_m2}) for p in pending["poses"]]}
        self._kernel._pending = None
        self._kernel._last_pose = None
        self._kernel._event = None
        self._kernel._streak = 0
        self._kernel._episode_node = None
        self._kernel.current_node = None
        self._restore_verified_ports()
        record = self._record({"action": "end_segment", "segment_id": self._active,
                               "coordinate_kind": self._kind, "abandoned_departure": interrupted})
        self._active, self._kind, self._last_coordinate = None, None, None
        return record

    def snapshot(self):
        graph = self._kernel.snapshot()
        for node in graph["nodes"]:
            node["pose"] = self._public_pose(node["pose"])
        for edge in graph["edges"]:
            edge["poses"] = [self._public_pose(p) for p in edge["poses"]]
            edge["segment_id"] = edge["poses"][0]["coordinate"]["segment_id"]
            if any(p["coordinate"]["segment_id"] != edge["segment_id"] for p in edge["poses"]):
                raise RuntimeError("cross-segment traversal invariant violated")
        graph["decision_trace"] = [json.loads(record) for record in self._trace]
        graph["active_segment"] = self._active
        graph["coordinate_kind"] = self._kind
        graph["time_metrics_available"] = False
        graph["legacy_clock_adapter"] = "PRIVATE_CALL_ORDINAL_NOT_SECONDS"
        return graph
