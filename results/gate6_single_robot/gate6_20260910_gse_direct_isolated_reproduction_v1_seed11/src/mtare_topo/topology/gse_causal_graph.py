"""Causal structural graph kernel, independent of teacher and map identities.

This is an interface implementation, not a scan-registration implementation.
Callers supply verifier evidence; similarity alone can never merge a node.
Every edge needs an explicit departure and uninterrupted measured pose trace.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math


def _finite_vector(value, size):
    return len(value) == size and all(math.isfinite(x) for x in value)


@dataclass(frozen=True)
class PoseEstimate:
    timestamp_s: float
    xyz_m: tuple[float, float, float]
    covariance_diagonal_m2: tuple[float, float, float]

    def __post_init__(self):
        if (not math.isfinite(self.timestamp_s) or not _finite_vector(self.xyz_m, 3)
                or not _finite_vector(self.covariance_diagonal_m2, 3)
                or min(self.covariance_diagonal_m2) < 0):
            raise ValueError("invalid deployment pose")


@dataclass(frozen=True)
class StructuralPort:
    local_id: int
    direction_robot: tuple[float, float, float]
    width_m: float
    height_m: float
    confidence: float

    def __post_init__(self):
        if (self.local_id < 0 or not _finite_vector(self.direction_robot, 3)
                or abs(math.hypot(*self.direction_robot) - 1) > 1e-5
                or not all(math.isfinite(x) for x in (self.width_m, self.height_m, self.confidence))
                or min(self.width_m, self.height_m) <= 0 or not 0 <= self.confidence <= 1):
            raise ValueError("invalid structural port")


@dataclass(frozen=True)
class StructuralObservation:
    timestamp_s: float
    event: str
    confidence: float
    uncertainty: float
    ports: tuple[StructuralPort, ...] = ()

    def __post_init__(self):
        if (self.event not in ("corridor", "junction", "terminal", "unknown")
                or not math.isfinite(self.timestamp_s)
                or not 0 <= self.confidence <= 1 or not 0 <= self.uncertainty <= 1
                or len({p.local_id for p in self.ports}) != len(self.ports)):
            raise ValueError("invalid structural observation")


@dataclass(frozen=True)
class AssociationEvidence:
    candidate_node: int            # graph-local ID, never teacher identity
    timestamp_s: float
    registration_passed: bool
    ports_passed: bool
    trace_context_passed: bool
    unambiguous: bool

    def __post_init__(self):
        if (type(self.candidate_node) is not int or self.candidate_node < 0
                or not math.isfinite(self.timestamp_s)
                or any(type(value) is not bool for value in (self.registration_passed, self.ports_passed,
                                                            self.trace_context_passed, self.unambiguous))):
            raise ValueError("invalid association evidence")

    @property
    def accepted(self):
        return all((self.registration_passed, self.ports_passed,
                    self.trace_context_passed, self.unambiguous))


@dataclass(frozen=True)
class GraphUpdate:
    timestamp_s: float
    action: str
    node_id: int | None = None
    edge_id: int | None = None
    candidate_nodes: tuple[int, ...] = ()


@dataclass(frozen=True)
class TraversalEvidence:
    trace_id: int                  # runtime-generated departure counter
    source_node: int
    departure_port: int
    target_node: int
    poses: tuple[PoseEstimate, ...]
    length_m: float
    execution_state: str = "verified"


class CausalStructuralGraph:
    def __init__(self, *, stable_frames=5, event_confidence=.8, maximum_uncertainty=.2):
        # Software fixtures use explicit settings; these are not calibrated
        # scientific thresholds or permission to change a formal run contract.
        if (not isinstance(stable_frames, int) or stable_frames < 1
                or not 0 <= event_confidence <= 1 or not 0 <= maximum_uncertainty <= 1):
            raise ValueError("invalid graph settings")
        self.stable_frames = stable_frames
        self.event_confidence = event_confidence
        self.maximum_uncertainty = maximum_uncertainty
        self.nodes = []
        self.edges = []
        self.decisions = []
        self.current_node = None
        self._last_pose = None
        self._event = None
        self._streak = 0
        self._episode_node = None
        self._pending = None
        self._trace_counter = 0

    def depart(self, port_id: int):
        if self.current_node is None or self._pending is not None or self._last_pose is None:
            raise ValueError("departure requires a confirmed node and no active trace")
        node = self.nodes[self.current_node]
        if port_id not in node["port_states"]:
            raise ValueError("departure port not observed at current node")
        node["port_states"][port_id] = "attempted"
        self._pending = {"id": self._trace_counter, "source": self.current_node,
                         "port": port_id, "poses": [self._last_pose], "continuous": True}
        self._trace_counter += 1
        # The departure episode stays latched until corridor/unknown; repeated
        # measurements while leaving cannot create a self-loop immediately.

    def update(self, observation: StructuralObservation, pose: PoseEstimate,
               candidates: tuple[AssociationEvidence, ...] = (), *, continuous=True):
        if type(continuous) is not bool:
            raise ValueError("continuity evidence must be boolean")
        if pose.timestamp_s != observation.timestamp_s:
            raise ValueError("sensor/graph pose timestamp mismatch")
        if self._last_pose is not None and pose.timestamp_s <= self._last_pose.timestamp_s:
            raise ValueError("observations must be strictly causal")
        if len({c.candidate_node for c in candidates}) != len(candidates):
            raise ValueError("duplicate association candidates")
        for c in candidates:
            if (c.timestamp_s != pose.timestamp_s or c.candidate_node < 0
                    or c.candidate_node >= len(self.nodes)):
                raise ValueError("stale/future/unknown association evidence")
        self._last_pose = pose
        if self._pending is not None:
            self._pending["poses"].append(pose)
            self._pending["continuous"] &= bool(continuous)
        stable = (observation.event in ("junction", "terminal")
                  and observation.confidence >= self.event_confidence
                  and observation.uncertainty <= self.maximum_uncertainty)
        if not stable:
            self._event, self._streak = None, 0
            # A confidence dip inside an event does not prove departure.
            if observation.event in ("corridor", "unknown"):
                self._episode_node = None
            return self._record(GraphUpdate(pose.timestamp_s, "observe"))
        self._streak = self._streak + 1 if self._event == observation.event else 1
        self._event = observation.event
        if self._streak < self.stable_frames or self._episode_node is not None:
            return self._record(GraphUpdate(pose.timestamp_s, "event_pending"))
        accepted = tuple(sorted(c.candidate_node for c in candidates if c.accepted
                                and self.nodes[c.candidate_node]["event"] == observation.event))
        if len(accepted) == 1:
            node_id, action = accepted[0], "verified_revisit"
        else:
            node_id, action = len(self.nodes), "local_confirmed"
            if len(accepted) > 1:
                action = "local_confirmed_merge_ambiguous"
            self.nodes.append({"id": node_id, "event": observation.event,
                               "pose": asdict(pose), "ports": [asdict(p) for p in observation.ports],
                               "port_states": {p.local_id: "observed" for p in observation.ports},
                               "alias_candidates": list(accepted)})
        edge_id = None
        if self._pending is not None:
            trace = self._pending
            length = sum(math.dist(a.xyz_m, b.xyz_m) for a, b in zip(trace["poses"], trace["poses"][1:]))
            if trace["continuous"] and length > 0:
                edge_id = len(self.edges)
                self.edges.append(TraversalEvidence(trace["id"], trace["source"], trace["port"],
                                                     node_id, tuple(trace["poses"]), length))
                self.nodes[trace["source"]]["port_states"][trace["port"]] = "traversed"
            else:
                action += "_trace_rejected"
                self.nodes[trace["source"]]["port_states"][trace["port"]] = "temporarily_failed"
            self._pending = None
        self.current_node, self._episode_node = node_id, node_id
        return self._record(GraphUpdate(pose.timestamp_s, action, node_id, edge_id, accepted))

    def _record(self, update):
        self.decisions.append(update)
        return update

    def snapshot(self):
        return deepcopy({"nodes": self.nodes, "edges": [asdict(e) for e in self.edges],
                         "decision_trace": [asdict(d) for d in self.decisions],
                         "current_node": self.current_node})
