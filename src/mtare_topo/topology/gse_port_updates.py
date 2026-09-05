"""Pure, causal node-local port updates; NOT a registration implementation.

The caller must transform observed geometry into the persistent node frame and
compute deployment-only candidate evidence. Observation-local IDs are never
compared with graph-local IDs to infer correspondence. No state is mutated.
"""
from collections import Counter
from dataclasses import dataclass, replace
import math


def _id(value):
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class PortGeometry:
    direction_node: tuple[float, float, float]
    width_m: float
    height_m: float
    confidence: float

    def __post_init__(self):
        if (type(self.direction_node) is not tuple or len(self.direction_node) != 3
                or not all(math.isfinite(v) for v in self.direction_node)
                or abs(math.hypot(*self.direction_node) - 1.) > 1e-5
                or not all(math.isfinite(v) for v in (self.width_m, self.height_m, self.confidence))
                or min(self.width_m, self.height_m) <= 0 or not 0 <= self.confidence <= 1):
            raise ValueError("invalid node-frame port geometry")


@dataclass(frozen=True)
class ObservedPort:
    observation_local_id: int
    geometry: PortGeometry

    def __post_init__(self):
        if not _id(self.observation_local_id) or not isinstance(self.geometry, PortGeometry):
            raise ValueError("invalid observed port")


@dataclass(frozen=True)
class GraphPort:
    graph_local_id: int
    geometry: PortGeometry
    execution_state: str
    last_observed_s: float

    def __post_init__(self):
        if (not _id(self.graph_local_id) or not isinstance(self.geometry, PortGeometry)
                or self.execution_state not in ("observed", "attempted", "traversed",
                                                "temporarily_failed", "failed")
                or not math.isfinite(self.last_observed_s)):
            raise ValueError("invalid graph port")


@dataclass(frozen=True)
class PortTable:
    node_id: int
    timestamp_s: float
    ports: tuple[GraphPort, ...] = ()

    def __post_init__(self):
        if (not _id(self.node_id) or not math.isfinite(self.timestamp_s)
                or type(self.ports) is not tuple or any(not isinstance(p, GraphPort) for p in self.ports)
                or len({p.graph_local_id for p in self.ports}) != len(self.ports)
                or any(p.last_observed_s > self.timestamp_s for p in self.ports)):
            raise ValueError("invalid port table")


@dataclass(frozen=True)
class PortMatchEvidence:
    node_id: int
    timestamp_s: float
    observation_local_id: int
    candidate_graph_ids: tuple[int, ...]
    geometry_verified: bool

    def __post_init__(self):
        if (not _id(self.node_id) or not _id(self.observation_local_id)
                or not math.isfinite(self.timestamp_s)
                or type(self.candidate_graph_ids) is not tuple
                or any(not _id(i) for i in self.candidate_graph_ids)
                or len(set(self.candidate_graph_ids)) != len(self.candidate_graph_ids)
                or type(self.geometry_verified) is not bool):
            raise ValueError("invalid port match evidence")


@dataclass(frozen=True)
class PortUpdateDecision:
    observation_local_id: int
    action: str
    graph_local_id: int | None
    candidate_graph_ids: tuple[int, ...]


@dataclass(frozen=True)
class PortUpdateResult:
    table: PortTable
    decisions: tuple[PortUpdateDecision, ...]
    pending_observations: tuple[ObservedPort, ...]


def update_ports(table: PortTable, *, timestamp_s: float,
                 observations: tuple[ObservedPort, ...],
                 evidence: tuple[PortMatchEvidence, ...]) -> PortUpdateResult:
    """Update only mutually unique, geometrically verified matches.

    Every observation requires explicit evidence. An empty candidate list means
    the caller completed candidate search and found no match: allocate a new
    graph-local ID. Missing evidence is an error, not implicit permission to add.
    Ambiguous or unverified candidates remain pending and create no duplicate
    confirmed port. A candidate claimed by *any* other observation is ambiguous,
    including a competing observation whose own geometric check failed.

    Matched geometry is replaced by the latest verified observation (no invented
    averaging/calibration). Existing execution states survive unchanged. This
    function does not mark a traversal, confirm nodes, or establish graph edges.
    """
    if not isinstance(table, PortTable) or not math.isfinite(timestamp_s) or timestamp_s <= table.timestamp_s:
        raise ValueError("port update must be strictly causal")
    if (type(observations) is not tuple or type(evidence) is not tuple
            or any(not isinstance(p, ObservedPort) for p in observations)
            or any(not isinstance(e, PortMatchEvidence) for e in evidence)):
        raise ValueError("typed immutable observations and evidence required")
    observed = {p.observation_local_id: p for p in observations}
    matches = {e.observation_local_id: e for e in evidence}
    if len(observed) != len(observations) or len(matches) != len(evidence) or set(observed) != set(matches):
        raise ValueError("exactly one evidence record per distinct observation required")
    ports = {p.graph_local_id: p for p in table.ports}
    for item in evidence:
        if item.node_id != table.node_id or item.timestamp_s != timestamp_s:
            raise ValueError("wrong node or stale/future evidence")
        if not set(item.candidate_graph_ids) <= ports.keys():
            raise ValueError("candidate must refer to an existing graph-local port")
    claims = Counter(i for item in evidence for i in item.candidate_graph_ids)
    next_id = max(ports, default=-1) + 1
    decisions, pending = [], []
    for local_id in sorted(observed):
        observation, item = observed[local_id], matches[local_id]
        candidates = tuple(sorted(item.candidate_graph_ids))
        graph_id = None
        if not candidates:
            graph_id, next_id = next_id, next_id + 1
            ports[graph_id] = GraphPort(graph_id, observation.geometry, "observed", timestamp_s)
            action = "new_unmatched"
        elif len(candidates) != 1 or claims[candidates[0]] != 1:
            action = "pending_ambiguous"
            pending.append(observation)
        elif not item.geometry_verified:
            action = "pending_unverified"
            pending.append(observation)
        else:
            graph_id = candidates[0]
            ports[graph_id] = replace(ports[graph_id], geometry=observation.geometry,
                                     last_observed_s=timestamp_s)
            action = "updated_verified_unique"
        decisions.append(PortUpdateDecision(local_id, action, graph_id, candidates))
    return PortUpdateResult(PortTable(table.node_id, timestamp_s,
                                     tuple(ports[i] for i in sorted(ports))),
                            tuple(decisions), tuple(pending))
