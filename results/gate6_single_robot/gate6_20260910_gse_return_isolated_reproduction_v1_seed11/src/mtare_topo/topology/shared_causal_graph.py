"""Deterministic fusion of latest per-robot causal topometric snapshots."""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from mtare_topo.planning.multi_robot_allocation import SharedFrontierTask
from mtare_topo.topology.causal_graph_v2 import circular_distance_deg, heading_set_distance


@dataclass(frozen=True)
class SharedCausalGraphConfig:
    node_merge_radius_m: float = 6.0
    structural_heading_merge_deg: float = 20.0
    embedding_minimum_cosine: float = 0.75

    def __post_init__(self) -> None:
        if not math.isfinite(self.node_merge_radius_m) or self.node_merge_radius_m <= 0:
            raise ValueError("shared node merge radius must be finite/positive")
        if not math.isfinite(self.structural_heading_merge_deg) or not 0 <= self.structural_heading_merge_deg <= 180:
            raise ValueError("shared heading threshold must lie in [0,180]")
        if not math.isfinite(self.embedding_minimum_cosine) or not -1 <= self.embedding_minimum_cosine <= 1:
            raise ValueError("shared embedding cosine threshold must lie in [-1,1]")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _finite_xyz(value: Sequence[float]) -> np.ndarray:
    xyz = np.asarray(value, dtype=np.float64)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("shared graph node xyz must be three finite values")
    return xyz


def _embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (128,) or not np.all(np.isfinite(vector)):
        raise ValueError("shared graph embedding must be finite shape [128]")
    return vector


def _cosine(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> float:
    if first is None or second is None:
        return 1.0
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 0.0 if norm <= np.finfo(np.float64).tiny else float(np.dot(first, second) / norm)


def _serialized_bytes(value: Mapping[str, Any]) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


class SharedCausalTopometricGraph:
    """Fuse latest robot snapshots; rebuild avoids repeated-evidence inflation."""

    def __init__(self, config: Optional[SharedCausalGraphConfig] = None) -> None:
        self.config = config or SharedCausalGraphConfig()
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._revision: Dict[str, int] = {}
        self._stamp: Dict[str, float] = {}
        self._received_bytes: Dict[str, int] = {}
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.current_node_by_robot: Dict[str, int] = {}
        self.local_to_shared: Dict[str, Dict[int, int]] = {}

    def ingest(
        self,
        *,
        robot_id: str,
        revision: int,
        stamp_sec: float,
        snapshot: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not robot_id:
            raise ValueError("shared graph robot ID cannot be empty")
        if revision < 0 or revision <= self._revision.get(robot_id, -1):
            raise ValueError("shared graph revision must increase per robot")
        if not math.isfinite(stamp_sec) or stamp_sec <= self._stamp.get(robot_id, -math.inf):
            raise ValueError("shared graph timestamp must increase per robot")
        payload = json.loads(json.dumps(snapshot, sort_keys=True, allow_nan=False))
        self._validate_local_snapshot(payload)
        received = _serialized_bytes(payload)
        self._latest[robot_id] = payload
        self._revision[robot_id] = int(revision)
        self._stamp[robot_id] = float(stamp_sec)
        self._received_bytes[robot_id] = self._received_bytes.get(robot_id, 0) + received
        self._rebuild()
        return {
            "robot_id": robot_id,
            "revision": revision,
            "stamp_sec": stamp_sec,
            "received_bytes": received,
            "shared_node_count": len(self.nodes),
            "shared_edge_count": len(self.edges),
            "mapping": {str(key): value for key, value in sorted(self.local_to_shared[robot_id].items())},
        }

    @staticmethod
    def _validate_local_snapshot(snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("schema_version") != "causal_topometric_graph_v2":
            raise ValueError("shared graph accepts causal_topometric_graph_v2 only")
        nodes = snapshot.get("nodes")
        edges = snapshot.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("local graph nodes/edges must be lists")
        ids = {int(node["id"]) for node in nodes}
        if len(ids) != len(nodes):
            raise ValueError("local graph node IDs must be unique")
        current = snapshot.get("current_node")
        if current is not None and int(current) not in ids:
            raise ValueError("local current node is missing")
        for node in nodes:
            _finite_xyz(node["xyz_m"])
            if node.get("node_kind") not in ("anchor", "structural"):
                raise ValueError("local node kind is invalid")
            if node.get("role") not in ("interior", "junction", "terminal"):
                raise ValueError("local node role is invalid")
            _embedding(node.get("z_role_mean"))
            for stub in node.get("exit_stubs", []):
                if stub.get("state") not in ("observed", "traversed"):
                    raise ValueError("local exit stub state is invalid")
                if not math.isfinite(float(stub["heading_world_deg"])) or not math.isfinite(float(stub["confidence"])):
                    raise ValueError("local exit stub values must be finite")
        for edge in edges:
            if edge.get("kind") != "verified_traversed" or not edge.get("traversals"):
                raise ValueError("shared graph accepts trace-verified edges only")
            if int(edge["from"]) not in ids or int(edge["to"]) not in ids or int(edge["from"]) == int(edge["to"]):
                raise ValueError("local edge endpoints are invalid")
            length = float(edge.get("minimum_traversed_length_m", math.nan))
            if not math.isfinite(length) or length <= 0:
                raise ValueError("local verified edge length must be finite/positive")

    def _candidate(self, node: Mapping[str, Any], used: Set[int]) -> Optional[int]:
        xyz = _finite_xyz(node["xyz_m"])
        embedding = _embedding(node.get("z_role_mean"))
        candidates = []
        for shared in self.nodes:
            shared_id = int(shared["id"])
            if shared_id in used or shared["node_kind"] != node["node_kind"]:
                continue
            distance = float(np.linalg.norm(xyz - np.asarray(shared["xyz_m"], dtype=np.float64)))
            if distance > self.config.node_merge_radius_m:
                continue
            if node["node_kind"] == "structural" and shared["role"] != node["role"]:
                continue
            heading_error = heading_set_distance(node.get("exit_headings_world_deg", []), shared["exit_headings_world_deg"])
            if node["node_kind"] == "structural" and heading_error > self.config.structural_heading_merge_deg:
                continue
            similarity = _cosine(embedding, _embedding(shared.get("z_role_mean")))
            if similarity < self.config.embedding_minimum_cosine:
                continue
            candidates.append((distance, heading_error, -similarity, shared_id))
        return None if not candidates else min(candidates)[3]

    def _new_node(self, robot_id: str, node: Mapping[str, Any]) -> int:
        identifier = len(self.nodes)
        observations = max(1, int(node.get("observation_count", 1)))
        copied = {
            "id": identifier,
            "node_kind": node["node_kind"],
            "xyz_m": _finite_xyz(node["xyz_m"]).tolist(),
            "role": node["role"],
            "role_probabilities_mean": list(map(float, node["role_probabilities_mean"])),
            "exit_headings_world_deg": sorted(float(value) % 360.0 for value in node.get("exit_headings_world_deg", [])),
            "exit_stubs": [
                {"heading_world_deg": float(stub["heading_world_deg"]) % 360.0, "state": stub["state"], "confidence": float(stub["confidence"]), "observed_by": [robot_id]}
                for stub in node.get("exit_stubs", [])
            ],
            "confidence_mean": float(node.get("confidence_mean", 0.0)),
            "z_role_mean": None if node.get("z_role_mean") is None else list(map(float, node["z_role_mean"])),
            "observation_count": observations,
            "source_nodes": [{"robot_id": robot_id, "local_node_id": int(node["id"])}],
        }
        self.nodes.append(copied)
        return identifier

    def _merge_node(self, shared_id: int, robot_id: str, node: Mapping[str, Any]) -> None:
        shared = self.nodes[shared_id]
        old_count = int(shared["observation_count"])
        new_count = max(1, int(node.get("observation_count", 1)))
        total = old_count + new_count
        shared["xyz_m"] = ((old_count * np.asarray(shared["xyz_m"]) + new_count * _finite_xyz(node["xyz_m"])) / total).tolist()
        shared["role_probabilities_mean"] = [
            (old_count * first + new_count * float(second)) / total
            for first, second in zip(shared["role_probabilities_mean"], node["role_probabilities_mean"])
        ]
        shared["confidence_mean"] = (old_count * shared["confidence_mean"] + new_count * float(node.get("confidence_mean", 0.0))) / total
        first_embedding = _embedding(shared.get("z_role_mean"))
        second_embedding = _embedding(node.get("z_role_mean"))
        if first_embedding is not None and second_embedding is not None:
            shared["z_role_mean"] = ((old_count * first_embedding + new_count * second_embedding) / total).tolist()
        elif second_embedding is not None:
            shared["z_role_mean"] = second_embedding.tolist()
        shared["observation_count"] = total
        shared["source_nodes"].append({"robot_id": robot_id, "local_node_id": int(node["id"])})
        for stub in node.get("exit_stubs", []):
            heading = float(stub["heading_world_deg"]) % 360.0
            candidates = [
                (circular_distance_deg(heading, value["heading_world_deg"]), index)
                for index, value in enumerate(shared["exit_stubs"])
            ]
            if not candidates or min(candidates)[0] > self.config.structural_heading_merge_deg:
                shared["exit_stubs"].append({"heading_world_deg": heading, "state": stub["state"], "confidence": float(stub["confidence"]), "observed_by": [robot_id]})
                continue
            _, index = min(candidates)
            target = shared["exit_stubs"][index]
            target["state"] = "traversed" if "traversed" in (target["state"], stub["state"]) else "observed"
            target["confidence"] = max(float(target["confidence"]), float(stub["confidence"]))
            target["observed_by"] = sorted(set(target["observed_by"]) | {robot_id})
        shared["exit_stubs"].sort(key=lambda item: item["heading_world_deg"])
        shared["exit_headings_world_deg"] = [item["heading_world_deg"] for item in shared["exit_stubs"]]

    def _rebuild(self) -> None:
        self.nodes = []
        self.edges = []
        self.current_node_by_robot = {}
        self.local_to_shared = {}
        for robot_id in sorted(self._latest):
            snapshot = self._latest[robot_id]
            mapping: Dict[int, int] = {}
            used: Set[int] = set()
            for node in sorted(snapshot["nodes"], key=lambda item: int(item["id"])):
                shared_id = self._candidate(node, used)
                if shared_id is None:
                    shared_id = self._new_node(robot_id, node)
                else:
                    self._merge_node(shared_id, robot_id, node)
                mapping[int(node["id"])] = shared_id
                used.add(shared_id)
            self.local_to_shared[robot_id] = mapping
            if snapshot.get("current_node") is not None:
                self.current_node_by_robot[robot_id] = mapping[int(snapshot["current_node"])]
            for edge in sorted(snapshot["edges"], key=lambda item: int(item["id"])):
                first, second = mapping[int(edge["from"])], mapping[int(edge["to"])]
                if first == second:
                    raise RuntimeError("shared node fusion collapsed a verified edge")
                pair = tuple(sorted((first, second)))
                target = next((item for item in self.edges if (item["from"], item["to"]) == pair), None)
                evidence = {
                    "robot_id": robot_id,
                    "local_edge_id": int(edge["id"]),
                    "verified_traversal_count": int(edge.get("verified_traversal_count", len(edge["traversals"]))),
                    "minimum_traversed_length_m": float(edge["minimum_traversed_length_m"]),
                }
                if target is None:
                    target = {"id": len(self.edges), "from": pair[0], "to": pair[1], "kind": "verified_traversed", "minimum_traversed_length_m": evidence["minimum_traversed_length_m"], "source_edges": []}
                    self.edges.append(target)
                target["minimum_traversed_length_m"] = min(target["minimum_traversed_length_m"], evidence["minimum_traversed_length_m"])
                target["source_edges"].append(evidence)
                target["verified_traversal_count"] = sum(item["verified_traversal_count"] for item in target["source_edges"])
        self.edges.sort(key=lambda item: (item["from"], item["to"]))
        for index, edge in enumerate(self.edges):
            edge["id"] = index

    def frontier_tasks(self) -> Tuple[SharedFrontierTask, ...]:
        tasks = []
        for node in self.nodes:
            for index, stub in enumerate(node["exit_stubs"]):
                if stub["state"] != "observed":
                    continue
                tasks.append(SharedFrontierTask(
                    frontier_id=f"n{node['id']}:s{index}",
                    node_id=int(node["id"]),
                    exploration_potential=float(stub["confidence"]),
                    confidence=float(stub["confidence"]),
                ))
        return tuple(tasks)

    def graph_costs(self) -> Dict[Tuple[str, str], float]:
        adjacency = {int(node["id"]): [] for node in self.nodes}
        for edge in self.edges:
            first, second = int(edge["from"]), int(edge["to"])
            cost = float(edge["minimum_traversed_length_m"])
            adjacency[first].append((second, cost))
            adjacency[second].append((first, cost))
        tasks = self.frontier_tasks()
        result = {}
        for robot_id, source in sorted(self.current_node_by_robot.items()):
            distance = {source: 0.0}
            pending = [(0.0, source)]
            while pending:
                value, node = heapq.heappop(pending)
                if value != distance.get(node):
                    continue
                for neighbour, cost in adjacency[node]:
                    candidate = value + cost
                    if candidate < distance.get(neighbour, math.inf):
                        distance[neighbour] = candidate
                        heapq.heappush(pending, (candidate, neighbour))
            for task in tasks:
                result[(robot_id, task.frontier_id)] = distance.get(task.node_id, math.inf)
        return result

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema_version": "shared_causal_topometric_graph_v1",
            "config": self.config.to_dict(),
            "robot_count": len(self._latest),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "frontier_count": len(self.frontier_tasks()),
            "nodes": self.nodes,
            "edges": self.edges,
            "current_node_by_robot": dict(sorted(self.current_node_by_robot.items())),
            "latest_revision_by_robot": dict(sorted(self._revision.items())),
            "latest_stamp_sec_by_robot": dict(sorted(self._stamp.items())),
            "received_bytes_by_robot": dict(sorted(self._received_bytes.items())),
            "received_bytes_total": sum(self._received_bytes.values()),
        }


__all__ = ["SharedCausalGraphConfig", "SharedCausalTopometricGraph"]
