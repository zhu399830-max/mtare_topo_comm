"""Clean-room deterministic tunnel-network-graph generation.

This module creates topology parents only.  It deliberately does not create a
mesh, clutter, robot trajectories, sensor observations, or learning labels.
The graph-stage clearance check samples line segments deterministically; it is
a conservative early rejection test, not a replacement for mesh collision or
robot-footprint validation in later gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from statistics import mean
from typing import Iterable, Mapping, Sequence


class GenerationError(RuntimeError):
    """Raised when a graph satisfying the requested constraints cannot be made."""


class SplitLeakageError(ValueError):
    """Raised when one topology parent occurs in more than one dataset split."""


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def distance(self, other: "Vec3") -> float:
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))

    def is_finite(self) -> bool:
        return all(math.isfinite(value) for value in (self.x, self.y, self.z))


@dataclass(frozen=True)
class SeedBundle:
    """Independent seeds prevent downstream variation changing topology."""

    master: int
    topology: int
    geometry: int
    clutter: int
    trajectory: int
    sensor: int

    @classmethod
    def from_master(cls, master_seed: int) -> "SeedBundle":
        if not isinstance(master_seed, int):
            raise TypeError("master_seed must be an integer")

        def derive(namespace: str) -> int:
            digest = hashlib.sha256(
                f"mtare-topo-v1:{master_seed}:{namespace}".encode("utf-8")
            ).digest()
            return int.from_bytes(digest[:8], "big", signed=False)

        return cls(
            master=master_seed,
            topology=derive("topology"),
            geometry=derive("geometry"),
            clutter=derive("clutter"),
            trajectory=derive("trajectory"),
            sensor=derive("sensor"),
        )


@dataclass(frozen=True)
class TNGParameters:
    target_nodes: int = 24
    connector_count: int = 3
    base_segment_length: float = 12.0
    segment_length_noise: float = 2.0
    horizontal_noise_rad: float = 0.38
    vertical_noise_rad: float = 0.10
    max_abs_pitch_rad: float = 0.32
    branch_probability: float = 0.30
    max_degree: int = 4
    min_node_clearance: float = 3.0
    min_edge_clearance: float = 1.5
    connector_min_length: float = 7.0
    connector_max_length: float = 42.0
    collision_samples: int = 13
    candidate_retries: int = 256

    def validate(self) -> None:
        if self.target_nodes < 4:
            raise ValueError("target_nodes must be at least 4")
        if self.connector_count < 0:
            raise ValueError("connector_count cannot be negative")
        if self.max_degree < 2:
            raise ValueError("max_degree must be at least 2")
        if self.connector_count > self.target_nodes:
            raise ValueError("connector_count is implausibly large")
        positive = {
            "base_segment_length": self.base_segment_length,
            "min_node_clearance": self.min_node_clearance,
            "min_edge_clearance": self.min_edge_clearance,
            "connector_min_length": self.connector_min_length,
            "connector_max_length": self.connector_max_length,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.segment_length_noise < 0:
            raise ValueError("segment_length_noise cannot be negative")
        if not 0 <= self.branch_probability <= 1:
            raise ValueError("branch_probability must be in [0, 1]")
        if self.connector_min_length >= self.connector_max_length:
            raise ValueError("connector_min_length must be below connector_max_length")
        if self.collision_samples < 3:
            raise ValueError("collision_samples must be at least 3")
        if self.candidate_retries < 1:
            raise ValueError("candidate_retries must be positive")


@dataclass(frozen=True)
class TNGNode:
    id: str
    position: Vec3


@dataclass(frozen=True)
class TNGEdge:
    id: str
    u: str
    v: str
    kind: str
    tunnel_graph_id: int


@dataclass(frozen=True)
class TunnelNetworkGraph:
    nodes: tuple[TNGNode, ...]
    edges: tuple[TNGEdge, ...]
    topology_seed: int
    parameters: TNGParameters

    def node_map(self) -> dict[str, TNGNode]:
        return {node.id: node for node in self.nodes}

    def adjacency(self) -> dict[str, set[str]]:
        result = {node.id: set() for node in self.nodes}
        for edge in self.edges:
            result[edge.u].add(edge.v)
            result[edge.v].add(edge.u)
        return result

    def validate(self) -> None:
        self.parameters.validate()
        node_ids = [node.id for node in self.nodes]
        edge_ids = [edge.id for edge in self.edges]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs are not unique")
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge IDs are not unique")
        if len(self.nodes) != self.parameters.target_nodes:
            raise ValueError("node count does not match target_nodes")
        if any(not node.position.is_finite() for node in self.nodes):
            raise ValueError("all node coordinates must be finite")

        known = set(node_ids)
        undirected: set[tuple[str, str]] = set()
        node_map = self.node_map()
        for edge in self.edges:
            if edge.u not in known or edge.v not in known:
                raise ValueError(f"edge {edge.id} has an unknown endpoint")
            if edge.u == edge.v:
                raise ValueError(f"edge {edge.id} is a self-loop")
            pair = tuple(sorted((edge.u, edge.v)))
            if pair in undirected:
                raise ValueError(f"duplicate undirected edge {pair}")
            undirected.add(pair)
            if edge.kind not in {"RGTG", "CTG"}:
                raise ValueError(f"unsupported edge kind {edge.kind}")
            if node_map[edge.u].position.distance(node_map[edge.v].position) <= 0:
                raise ValueError(f"edge {edge.id} has zero length")

        adjacency = self.adjacency()
        if adjacency:
            pending = [next(iter(adjacency))]
            visited: set[str] = set()
            while pending:
                node_id = pending.pop()
                if node_id in visited:
                    continue
                visited.add(node_id)
                pending.extend(adjacency[node_id] - visited)
            if len(visited) != len(self.nodes):
                raise ValueError("graph is disconnected")
        if any(len(neighbors) > self.parameters.max_degree for neighbors in adjacency.values()):
            raise ValueError("maximum degree exceeded")
        rgtg_count = sum(edge.kind == "RGTG" for edge in self.edges)
        ctg_count = sum(edge.kind == "CTG" for edge in self.edges)
        if rgtg_count != len(self.nodes) - 1:
            raise ValueError("RGTG must form the spanning tree")
        if ctg_count != self.parameters.connector_count:
            raise ValueError("CTG count does not match connector_count")

    def stats(self) -> dict[str, object]:
        adjacency = self.adjacency()
        node_map = self.node_map()
        lengths = [
            node_map[edge.u].position.distance(node_map[edge.v].position)
            for edge in self.edges
        ]
        edge_pitches = []
        for edge in self.edges:
            start = node_map[edge.u].position
            end = node_map[edge.v].position
            horizontal = math.hypot(end.x - start.x, end.y - start.y)
            edge_pitches.append(math.atan2(abs(end.z - start.z), horizontal))
        degree_histogram: dict[str, int] = {}
        for neighbors in adjacency.values():
            key = str(len(neighbors))
            degree_histogram[key] = degree_histogram.get(key, 0) + 1
        components = 1 if self.nodes else 0
        z_values = [node.position.z for node in self.nodes]
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "rgtg_edge_count": sum(edge.kind == "RGTG" for edge in self.edges),
            "ctg_edge_count": sum(edge.kind == "CTG" for edge in self.edges),
            "component_count": components,
            "cycle_rank": len(self.edges) - len(self.nodes) + components,
            "dead_end_count": sum(len(n) == 1 for n in adjacency.values()),
            "branch_node_count": sum(len(n) >= 3 for n in adjacency.values()),
            "degree_histogram": dict(sorted(degree_histogram.items())),
            "total_edge_length": round(sum(lengths), 9),
            "mean_edge_length": round(mean(lengths), 9) if lengths else 0.0,
            "min_edge_length": round(min(lengths), 9) if lengths else 0.0,
            "max_edge_length": round(max(lengths), 9) if lengths else 0.0,
            "vertical_span": round(max(z_values) - min(z_values), 9) if z_values else 0.0,
            "maximum_abs_edge_pitch_rad": round(max(edge_pitches), 9) if edge_pitches else 0.0,
            "sampled_min_nonincident_edge_distance": round(
                minimum_nonincident_edge_sample_distance(
                    self, self.parameters.collision_samples
                ),
                9,
            ),
            "sampled_min_nonincident_node_edge_distance": round(
                minimum_nonincident_node_edge_sample_distance(
                    self, self.parameters.collision_samples
                ),
                9,
            ),
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "tng_parent_v1",
            "topology_seed": self.topology_seed,
            "parameters": asdict(self.parameters),
            "nodes": [
                {
                    "id": node.id,
                    "position": [
                        round(node.position.x, 9),
                        round(node.position.y, 9),
                        round(node.position.z, 9),
                    ],
                }
                for node in sorted(self.nodes, key=lambda item: item.id)
            ],
            "edges": [
                {
                    "id": edge.id,
                    "u": min(edge.u, edge.v),
                    "v": max(edge.u, edge.v),
                    "kind": edge.kind,
                    "tunnel_graph_id": edge.tunnel_graph_id,
                }
                for edge in sorted(self.edges, key=lambda item: item.id)
            ],
        }

    def canonical_hash(self) -> str:
        payload = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def topology_parent_id(self) -> str:
        return f"tng_{self.canonical_hash()[:16]}"

    def to_dict(self, seed_bundle: SeedBundle | None = None) -> dict[str, object]:
        result = self.canonical_payload()
        result["topology_parent_id"] = self.topology_parent_id
        result["canonical_hash"] = self.canonical_hash()
        result["stats"] = self.stats()
        if seed_bundle is not None:
            result["seed_bundle"] = asdict(seed_bundle)
        return result


@dataclass
class _Tip:
    node_id: str
    heading: float
    pitch: float
    tunnel_graph_id: int


def _sample_points(start: Vec3, end: Vec3, samples: int) -> list[Vec3]:
    return [
        Vec3(
            start.x + (end.x - start.x) * index / (samples - 1),
            start.y + (end.y - start.y) * index / (samples - 1),
            start.z + (end.z - start.z) * index / (samples - 1),
        )
        for index in range(samples)
    ]


def _sampled_segment_distance(
    first_start: Vec3,
    first_end: Vec3,
    second_start: Vec3,
    second_end: Vec3,
    samples: int,
) -> float:
    first = _sample_points(first_start, first_end, samples)
    second = _sample_points(second_start, second_end, samples)
    return min(a.distance(b) for a in first for b in second)


def _candidate_has_clearance(
    start_id: str,
    end_id: str | None,
    start: Vec3,
    end: Vec3,
    nodes: Mapping[str, TNGNode],
    edges: Sequence[TNGEdge],
    parameters: TNGParameters,
) -> bool:
    candidate_samples = _sample_points(start, end, parameters.collision_samples)
    for node_id, node in nodes.items():
        if node_id in {start_id, end_id}:
            continue
        if min(point.distance(node.position) for point in candidate_samples) < parameters.min_node_clearance:
            return False
    for edge in edges:
        edge_points = _sample_points(
            nodes[edge.u].position,
            nodes[edge.v].position,
            parameters.collision_samples,
        )
        if end_id is None and min(end.distance(point) for point in edge_points) < parameters.min_node_clearance:
            return False
        if start_id in {edge.u, edge.v} or end_id in {edge.u, edge.v}:
            continue
        distance = _sampled_segment_distance(
            start,
            end,
            nodes[edge.u].position,
            nodes[edge.v].position,
            parameters.collision_samples,
        )
        if distance < parameters.min_edge_clearance:
            return False
    return True


def _next_position(base: Vec3, length: float, heading: float, pitch: float) -> Vec3:
    horizontal = length * math.cos(pitch)
    return Vec3(
        base.x + horizontal * math.cos(heading),
        base.y + horizontal * math.sin(heading),
        base.z + length * math.sin(pitch),
    )


def generate_tng(
    seed_bundle: SeedBundle,
    parameters: TNGParameters | None = None,
) -> TunnelNetworkGraph:
    """Generate one deterministic 3-D topology parent from only the topology seed."""

    params = parameters or TNGParameters()
    params.validate()
    rng = random.Random(seed_bundle.topology)
    nodes: dict[str, TNGNode] = {"n000": TNGNode("n000", Vec3(0.0, 0.0, 0.0))}
    edges: list[TNGEdge] = []
    tips = [_Tip("n000", rng.uniform(-math.pi, math.pi), 0.0, 0)]
    next_tunnel_graph_id = 1

    for node_index in range(1, params.target_nodes):
        accepted: tuple[_Tip, float, float, Vec3, int] | None = None
        for _ in range(params.candidate_retries):
            adjacency_degree = {node_id: 0 for node_id in nodes}
            for edge in edges:
                adjacency_degree[edge.u] += 1
                adjacency_degree[edge.v] += 1
            branchable = [
                node_id
                for node_id, degree in adjacency_degree.items()
                if degree < params.max_degree
            ]
            use_branch = (
                len(nodes) > 2
                and branchable
                and rng.random() < params.branch_probability
            )
            if use_branch:
                base_id = rng.choice(branchable)
                base_tip = _Tip(
                    base_id,
                    rng.uniform(-math.pi, math.pi),
                    rng.uniform(-params.max_abs_pitch_rad / 2, params.max_abs_pitch_rad / 2),
                    next_tunnel_graph_id,
                )
                proposed_tunnel_graph_id = next_tunnel_graph_id
            else:
                viable_tips = [
                    tip
                    for tip in tips
                    if adjacency_degree.get(tip.node_id, params.max_degree) < params.max_degree
                ]
                if not viable_tips:
                    continue
                base_tip = rng.choice(viable_tips)
                proposed_tunnel_graph_id = base_tip.tunnel_graph_id

            heading = base_tip.heading + rng.gauss(0.0, params.horizontal_noise_rad)
            pitch = max(
                -params.max_abs_pitch_rad,
                min(
                    params.max_abs_pitch_rad,
                    base_tip.pitch + rng.gauss(0.0, params.vertical_noise_rad),
                ),
            )
            length = max(
                params.min_node_clearance * 1.25,
                rng.gauss(params.base_segment_length, params.segment_length_noise),
            )
            base_position = nodes[base_tip.node_id].position
            position = _next_position(base_position, length, heading, pitch)
            if _candidate_has_clearance(
                base_tip.node_id,
                None,
                base_position,
                position,
                nodes,
                edges,
                params,
            ):
                accepted = (base_tip, heading, pitch, position, proposed_tunnel_graph_id)
                break

        if accepted is None:
            raise GenerationError(
                f"failed to place RGTG node {node_index} after {params.candidate_retries} attempts"
            )
        base_tip, heading, pitch, position, tunnel_graph_id = accepted
        node_id = f"n{node_index:03d}"
        nodes[node_id] = TNGNode(node_id, position)
        edge_id = f"e{len(edges):03d}"
        edges.append(
            TNGEdge(edge_id, base_tip.node_id, node_id, "RGTG", tunnel_graph_id)
        )
        tips.append(_Tip(node_id, heading, pitch, tunnel_graph_id))
        if tunnel_graph_id == next_tunnel_graph_id:
            next_tunnel_graph_id += 1

    adjacency_degree = {node_id: 0 for node_id in nodes}
    existing_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        adjacency_degree[edge.u] += 1
        adjacency_degree[edge.v] += 1
        existing_pairs.add(tuple(sorted((edge.u, edge.v))))
    connector_candidates: list[tuple[str, str]] = []
    node_ids = sorted(nodes)
    for first_index, first_id in enumerate(node_ids):
        for second_id in node_ids[first_index + 1 :]:
            pair = (first_id, second_id)
            if pair in existing_pairs:
                continue
            distance = nodes[first_id].position.distance(nodes[second_id].position)
            first_position = nodes[first_id].position
            second_position = nodes[second_id].position
            connector_pitch = math.atan2(
                abs(second_position.z - first_position.z),
                math.hypot(
                    second_position.x - first_position.x,
                    second_position.y - first_position.y,
                ),
            )
            if (
                params.connector_min_length <= distance <= params.connector_max_length
                and connector_pitch <= params.max_abs_pitch_rad
            ):
                connector_candidates.append(pair)
    rng.shuffle(connector_candidates)

    connectors_added = 0
    for first_id, second_id in connector_candidates:
        if connectors_added >= params.connector_count:
            break
        if (
            adjacency_degree[first_id] >= params.max_degree
            or adjacency_degree[second_id] >= params.max_degree
        ):
            continue
        if not _candidate_has_clearance(
            first_id,
            second_id,
            nodes[first_id].position,
            nodes[second_id].position,
            nodes,
            edges,
            params,
        ):
            continue
        edge_id = f"e{len(edges):03d}"
        edges.append(
            TNGEdge(
                edge_id,
                first_id,
                second_id,
                "CTG",
                next_tunnel_graph_id + connectors_added,
            )
        )
        adjacency_degree[first_id] += 1
        adjacency_degree[second_id] += 1
        connectors_added += 1

    if connectors_added != params.connector_count:
        raise GenerationError(
            f"placed {connectors_added}/{params.connector_count} CTG connectors"
        )

    graph = TunnelNetworkGraph(
        nodes=tuple(nodes[node_id] for node_id in sorted(nodes)),
        edges=tuple(edges),
        topology_seed=seed_bundle.topology,
        parameters=params,
    )
    graph.validate()
    return graph


def minimum_nonincident_edge_sample_distance(
    graph: TunnelNetworkGraph,
    samples: int | None = None,
) -> float:
    """Return sampled clearance among graph edges that share no endpoint."""

    sample_count = samples or graph.parameters.collision_samples
    node_map = graph.node_map()
    distances: list[float] = []
    for first_index, first in enumerate(graph.edges):
        for second in graph.edges[first_index + 1 :]:
            if {first.u, first.v} & {second.u, second.v}:
                continue
            distances.append(
                _sampled_segment_distance(
                    node_map[first.u].position,
                    node_map[first.v].position,
                    node_map[second.u].position,
                    node_map[second.v].position,
                    sample_count,
                )
            )
    return min(distances) if distances else math.inf


def minimum_nonincident_node_edge_sample_distance(
    graph: TunnelNetworkGraph,
    samples: int | None = None,
) -> float:
    """Return sampled clearance between edges and their non-endpoint nodes."""

    sample_count = samples or graph.parameters.collision_samples
    node_map = graph.node_map()
    distances: list[float] = []
    for edge in graph.edges:
        points = _sample_points(
            node_map[edge.u].position,
            node_map[edge.v].position,
            sample_count,
        )
        for node in graph.nodes:
            if node.id in {edge.u, edge.v}:
                continue
            distances.append(min(point.distance(node.position) for point in points))
    return min(distances) if distances else math.inf


def validate_parent_split(records: Iterable[Mapping[str, object]]) -> dict[str, str]:
    """Validate that every topology parent belongs to exactly one data split."""

    assignments: dict[str, str] = {}
    valid_splits = {"train", "val", "test"}
    for index, record in enumerate(records):
        parent_id = record.get("topology_parent_id")
        split = record.get("split")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError(f"record {index} has no topology_parent_id")
        if split not in valid_splits:
            raise ValueError(f"record {index} has invalid split {split!r}")
        previous = assignments.get(parent_id)
        if previous is not None and previous != split:
            raise SplitLeakageError(
                f"topology parent {parent_id} occurs in both {previous} and {split}"
            )
        assignments[parent_id] = split
    return assignments


def tng_from_dict(payload: Mapping[str, object]) -> TunnelNetworkGraph:
    """Reconstruct and validate a topology parent from its JSON representation."""

    if payload.get("schema_version") != "tng_parent_v1":
        raise ValueError("unsupported topology schema_version")
    raw_parameters = payload.get("parameters")
    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")
    topology_seed = payload.get("topology_seed")
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("topology parameters are missing")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError("topology nodes/edges are missing")
    if not isinstance(topology_seed, int):
        raise ValueError("topology_seed is missing")
    parameters = TNGParameters(**raw_parameters)
    nodes: list[TNGNode] = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            raise ValueError("invalid node record")
        position = raw_node.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("invalid node position")
        nodes.append(
            TNGNode(
                id=str(raw_node["id"]),
                position=Vec3(*(float(value) for value in position)),
            )
        )
    edges: list[TNGEdge] = []
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, Mapping):
            raise ValueError("invalid edge record")
        edges.append(
            TNGEdge(
                id=str(raw_edge["id"]),
                u=str(raw_edge["u"]),
                v=str(raw_edge["v"]),
                kind=str(raw_edge["kind"]),
                tunnel_graph_id=int(raw_edge["tunnel_graph_id"]),
            )
        )
    graph = TunnelNetworkGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        topology_seed=topology_seed,
        parameters=parameters,
    )
    graph.validate()
    expected_hash = payload.get("canonical_hash")
    if expected_hash is not None and expected_hash != graph.canonical_hash():
        raise ValueError("topology canonical_hash does not match its content")
    return graph
