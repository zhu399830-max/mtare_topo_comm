"""Online runtime V3 combining verified re-anchor and graph-event feedback."""

from __future__ import annotations

from collections import Counter
import time
from typing import Sequence

from mtare_topo.integration.online_topology_runtime import (
    OnlineTopologyCycle,
    SemanticPrediction,
)
from mtare_topo.integration.online_topology_runtime_v2 import OnlineTopologyPlannerRuntimeV2
from mtare_topo.planning.topological_frontier import TopologicalPlannerConfig
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig
from mtare_topo.topology.frontier_execution_feedback import (
    evaluate_frontier_execution_feedback,
)


class OnlineTopologyPlannerRuntimeV3(OnlineTopologyPlannerRuntimeV2):
    """Feed verified nonmatching departures into the existing retry penalty."""

    def __init__(
        self,
        graph_config: CausalGraphConfig,
        planner_config: TopologicalPlannerConfig,
        *,
        direction_threshold: float = 0.5,
    ) -> None:
        super().__init__(graph_config, planner_config, direction_threshold=direction_threshold)
        self.frontier_execution_outcomes: Counter[str] = Counter()
        self.frontier_execution_rejection_count = 0

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
        previous_node = self.graph.current_node
        started = time.monotonic() if cycle_started_monotonic is None else cycle_started_monotonic
        cycle = super().update(
            stamp_sec=stamp_sec,
            sensor_xyz_m=sensor_xyz_m,
            sensor_orientation_xyzw=sensor_orientation_xyzw,
            prediction=prediction,
            cycle_started_monotonic=started,
        )
        snapshot = self.graph.snapshot()
        feedback = evaluate_frontier_execution_feedback(
            previous_target,
            previous_node_id=previous_node,
            graph_update=cycle.graph_update,
            graph_snapshot=snapshot,
        )
        if not feedback.classified:
            return cycle

        self.frontier_execution_outcomes[feedback.outcome] += 1
        graph_update = dict(cycle.graph_update)
        graph_update["frontier_execution_feedback"] = feedback.to_dict()
        target, packaged = cycle.target, cycle.handoff
        if feedback.record_rejection:
            self.retry_ledger.record_rejection(feedback.target_frontier)
            self.frontier_execution_rejection_count += 1
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
            self.active_target = target
        return OnlineTopologyCycle(
            frame_index=cycle.frame_index,
            stamp_sec=cycle.stamp_sec,
            route_arc_m=cycle.route_arc_m,
            target=target,
            handoff=packaged,
            graph_update=graph_update,
            graph_node_count=cycle.graph_node_count,
            graph_edge_count=cycle.graph_edge_count,
        )

    def snapshot(self):
        value = super().snapshot()
        value["schema_version"] = "online_topology_planner_runtime_v3"
        value["frontier_execution_outcomes"] = dict(sorted(self.frontier_execution_outcomes.items()))
        value["frontier_execution_rejection_count"] = self.frontier_execution_rejection_count
        return value


__all__ = ["OnlineTopologyPlannerRuntimeV3"]
