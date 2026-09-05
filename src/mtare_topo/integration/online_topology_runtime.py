"""ROS-free causal runtime joining semantics, graph, planner, and handoff."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components
from mtare_topo.integration.mtare_handoff import MTAReHandoffOutput
from mtare_topo.integration.mtare_handoff_v2 import FrontierRetryLedger, MTAReGlobalHandoffV2
from mtare_topo.planning.topological_frontier import (
    GlobalTarget,
    RuleBasedTopologicalFrontierPlanner,
    TopologicalPlannerConfig,
)
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig, CausalTopometricGraphV2


@dataclass(frozen=True)
class SemanticPrediction:
    direction_logits: Sequence[float]
    count_probabilities: Sequence[float]
    role_probabilities: Sequence[float]
    z_role: Sequence[float]
    branch_count_override: int | None = None


@dataclass(frozen=True)
class OnlineTopologyCycle:
    frame_index: int
    stamp_sec: float
    route_arc_m: float
    target: GlobalTarget
    handoff: MTAReHandoffOutput
    graph_update: dict[str, Any]
    graph_node_count: int
    graph_edge_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target"] = self.target.to_dict()
        value["handoff"] = self.handoff.to_dict()
        return value


def quaternion_yaw_deg(orientation_xyzw: Sequence[float]) -> float:
    orientation = np.asarray(orientation_xyzw, dtype=np.float64)
    if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
        raise ValueError("orientation must contain four finite xyzw values")
    norm = float(np.linalg.norm(orientation))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("orientation quaternion must have nonzero norm")
    x, y, z, w = orientation / norm
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class OnlineTopologyPlannerRuntime:
    """Update one causal graph and emit one M-TARE-compatible target per scan."""

    def __init__(
        self,
        graph_config: CausalGraphConfig,
        planner_config: TopologicalPlannerConfig,
        *,
        direction_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= direction_threshold <= 1.0:
            raise ValueError("direction threshold must lie in [0, 1]")
        self.graph = CausalTopometricGraphV2(graph_config)
        self.planner = RuleBasedTopologicalFrontierPlanner(planner_config)
        self.handoff = MTAReGlobalHandoffV2()
        self.retry_ledger = FrontierRetryLedger()
        self.direction_threshold = float(direction_threshold)
        self.frame_index = 0
        self.route_arc_m = 0.0
        self.last_xyz_m: np.ndarray | None = None
        self.last_stamp_sec: float | None = None
        self.active_target: GlobalTarget | None = None
        self.map_clearing_events = 0

    @staticmethod
    def _validated_prediction(prediction: SemanticPrediction) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]:
        direction = np.asarray(prediction.direction_logits, dtype=np.float64)
        counts = np.asarray(prediction.count_probabilities, dtype=np.float64)
        roles = np.asarray(prediction.role_probabilities, dtype=np.float64)
        embedding = np.asarray(prediction.z_role, dtype=np.float64)
        if direction.shape != (720,) or counts.shape != (6,) or roles.shape != (3,) or embedding.shape != (128,):
            raise ValueError("semantic prediction shapes must be 720/6/3/128")
        if not all(np.all(np.isfinite(value)) for value in (direction, counts, roles, embedding)):
            raise ValueError("semantic prediction must be finite")
        if np.any(counts < 0) or np.any(roles < 0) or not math.isclose(float(counts.sum()), 1.0, abs_tol=1e-4) or not math.isclose(float(roles.sum()), 1.0, abs_tol=1e-4):
            raise ValueError("count and role probabilities must be normalized")
        override = prediction.branch_count_override
        if override is not None and (isinstance(override, bool) or int(override) != override or not 0 <= int(override) <= 8):
            raise ValueError("explicit branch count must be an integer in [0, 8]")
        return direction, counts, roles, embedding, None if override is None else int(override)

    def update(
        self,
        *,
        stamp_sec: float,
        sensor_xyz_m: Sequence[float],
        sensor_orientation_xyzw: Sequence[float],
        prediction: SemanticPrediction,
        cycle_started_monotonic: float | None = None,
    ) -> OnlineTopologyCycle:
        if not math.isfinite(stamp_sec):
            raise ValueError("stamp must be finite")
        if self.last_stamp_sec is not None and stamp_sec <= self.last_stamp_sec:
            raise ValueError("scan timestamps must be strictly increasing")
        xyz = np.asarray(sensor_xyz_m, dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("sensor xyz must contain three finite values")
        direction, counts, roles, embedding, explicit_count = self._validated_prediction(prediction)
        if self.last_xyz_m is not None:
            self.route_arc_m += float(np.linalg.norm(xyz - self.last_xyz_m))
        yaw_deg = quaternion_yaw_deg(sensor_orientation_xyzw)
        headings = decode_direction_components(direction, self.direction_threshold)
        branch_count = explicit_count if explicit_count is not None else int(np.argmax(counts)) + 1
        confidence = float(np.max(roles))
        graph_update = self.graph.update(
            frame_index=self.frame_index,
            route_arc_m=self.route_arc_m,
            xyz_m=xyz,
            yaw_deg=yaw_deg,
            headings_robot_deg=headings,
            role_probabilities=roles,
            branch_count=branch_count,
            confidence=confidence,
            z_role=embedding,
        )
        snapshot = self.graph.snapshot()
        target = self.planner.select_target(
            snapshot,
            robot_xyz_m=xyz,
            retry_counts=self.retry_ledger.counts(),
        )
        started = time.monotonic() if cycle_started_monotonic is None else cycle_started_monotonic
        packaged = self.handoff.package(
            target,
            stamp_sec=stamp_sec,
            cycle_started_monotonic=started,
            completion_confirmed=False,
        )
        cycle = OnlineTopologyCycle(
            frame_index=self.frame_index,
            stamp_sec=float(stamp_sec),
            route_arc_m=float(self.route_arc_m),
            target=target,
            handoff=packaged,
            graph_update=graph_update,
            graph_node_count=int(snapshot["node_count"]),
            graph_edge_count=int(snapshot["edge_count"]),
        )
        self.active_target = target
        self.last_xyz_m = xyz.copy()
        self.last_stamp_sec = float(stamp_sec)
        self.frame_index += 1
        return cycle

    def observe_free_path_count(self, count: int) -> float | None:
        clearing = self.handoff.observe_free_path_count(count)
        if clearing is not None:
            frontier = None if self.active_target is None else self.active_target.frontier
            self.retry_ledger.record_rejection(frontier)
            self.map_clearing_events += 1
        return clearing

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "online_topology_planner_runtime_v1",
            "frames": self.frame_index,
            "route_arc_m": self.route_arc_m,
            "last_stamp_sec": self.last_stamp_sec,
            "map_clearing_events": self.map_clearing_events,
            "retry_counts": self.retry_ledger.to_dict(),
            "active_target": None if self.active_target is None else self.active_target.to_dict(),
            "graph": self.graph.snapshot(),
        }
