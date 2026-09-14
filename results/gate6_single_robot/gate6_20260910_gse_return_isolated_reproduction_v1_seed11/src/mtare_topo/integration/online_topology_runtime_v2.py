"""Online runtime V2 adding causal verified-backtrack node re-anchoring."""

from __future__ import annotations

import time
from typing import Sequence

from mtare_topo.integration.online_topology_runtime import (
    OnlineTopologyCycle,
    OnlineTopologyPlannerRuntime,
    SemanticPrediction,
)
from mtare_topo.planning.topological_frontier import TopologicalPlannerConfig
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig
from mtare_topo.topology.causal_graph_v3 import CausalTopometricGraphV3
from mtare_topo.topology.verified_reanchor import evaluate_verified_reanchor


class OnlineTopologyPlannerRuntimeV2(OnlineTopologyPlannerRuntime):
    """Preserve V1 behavior except for one explicit verified arrival transition."""

    def __init__(
        self,
        graph_config: CausalGraphConfig,
        planner_config: TopologicalPlannerConfig,
        *,
        direction_threshold: float = 0.5,
    ) -> None:
        super().__init__(graph_config, planner_config, direction_threshold=direction_threshold)
        self.graph = CausalTopometricGraphV3(graph_config)
        self.verified_reanchor_count = 0

    def update(
        self,
        *,
        stamp_sec: float,
        sensor_xyz_m: Sequence[float],
        sensor_orientation_xyzw: Sequence[float],
        prediction: SemanticPrediction,
        cycle_started_monotonic: float | None = None,
    ) -> OnlineTopologyCycle:
        previous_target = self.active_target
        started = time.monotonic() if cycle_started_monotonic is None else cycle_started_monotonic
        cycle = super().update(
            stamp_sec=stamp_sec,
            sensor_xyz_m=sensor_xyz_m,
            sensor_orientation_xyzw=sensor_orientation_xyzw,
            prediction=prediction,
            cycle_started_monotonic=started,
        )
        if cycle.graph_update.get("reason") != "no_event" or previous_target is None:
            return cycle

        snapshot = self.graph.snapshot()
        eligibility = evaluate_verified_reanchor(
            snapshot,
            previous_target.to_dict(),
            sensor_xyz_m=sensor_xyz_m,
            route_arc_m=cycle.route_arc_m,
            current_node_route_arc_m=self.graph.current_node_route_arc_m,
            trace_frame_count=self.graph.trace_frame_count,
        )
        if not eligibility.eligible:
            return cycle

        graph_update = self.graph.apply_verified_backtrack_reanchor(
            frame_index=cycle.frame_index,
            route_arc_m=cycle.route_arc_m,
            next_hop_node_id=int(eligibility.next_hop_node_id),
        )
        graph_update["eligibility"] = eligibility.to_dict()
        snapshot = self.graph.snapshot()
        target = self.planner.select_target(
            snapshot,
            robot_xyz_m=sensor_xyz_m,
            retry_counts=self.retry_ledger.counts(),
        )
        packaged = self.handoff.package(
            target,
            stamp_sec=stamp_sec,
            cycle_started_monotonic=started,
            completion_confirmed=False,
        )
        corrected = OnlineTopologyCycle(
            frame_index=cycle.frame_index,
            stamp_sec=cycle.stamp_sec,
            route_arc_m=cycle.route_arc_m,
            target=target,
            handoff=packaged,
            graph_update=graph_update,
            graph_node_count=int(snapshot["node_count"]),
            graph_edge_count=int(snapshot["edge_count"]),
        )
        self.active_target = target
        self.verified_reanchor_count += 1
        return corrected

    def snapshot(self):
        value = super().snapshot()
        value["schema_version"] = "online_topology_planner_runtime_v2"
        value["verified_reanchor_count"] = self.verified_reanchor_count
        return value


__all__ = ["OnlineTopologyPlannerRuntimeV2"]
