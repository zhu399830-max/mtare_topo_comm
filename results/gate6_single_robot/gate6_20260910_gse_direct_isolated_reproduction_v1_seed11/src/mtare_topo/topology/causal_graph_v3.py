"""Causal graph V3 with an explicit verified-backtrack re-anchor transition."""

from __future__ import annotations

import math
from typing import Any

from mtare_topo.topology.causal_graph_v2 import CausalTopometricGraphV2


class CausalTopometricGraphV3(CausalTopometricGraphV2):
    """Extend V2 without changing its event, association, or edge contracts."""

    @property
    def trace_frame_count(self) -> int:
        return len(self._trace_frames)

    def apply_verified_backtrack_reanchor(
        self,
        *,
        frame_index: int,
        route_arc_m: float,
        next_hop_node_id: int,
    ) -> dict[str, Any]:
        if self.current_node is None:
            raise RuntimeError("re-anchor requires a current node")
        previous = int(self.current_node)
        next_hop = int(next_hop_node_id)
        if previous == next_hop:
            raise RuntimeError("re-anchor next hop is already current")
        if not 0 <= next_hop < len(self.nodes):
            raise RuntimeError("re-anchor next hop does not exist")
        if not self.decision_trace or self.decision_trace[-1].get("reason") != "no_event":
            raise RuntimeError("re-anchor may replace only a no-event transition")
        if not self._trace_frames or int(self._trace_frames[-1]) != int(frame_index):
            raise RuntimeError("re-anchor frame must be the latest physical trace frame")
        if not self._trace_route_arc_m or not math.isclose(
            float(self._trace_route_arc_m[-1]), float(route_arc_m), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("re-anchor route arc must match the latest physical trace")
        edge = next(
            (
                item for item in self.edges
                if item.get("kind") == "verified_traversed"
                and frozenset((int(item["from"]), int(item["to"]))) == frozenset((previous, next_hop))
            ),
            None,
        )
        if edge is None or not edge.get("traversals"):
            raise RuntimeError("re-anchor requires an existing trace-verified edge")

        self._append_verified_traversal(previous, next_hop, frame_index, route_arc_m)
        self._consume_stub(previous, self.nodes[next_hop]["xyz_m"])
        self._consume_stub(next_hop, self.nodes[previous]["xyz_m"])
        self.current_node = next_hop
        self.current_node_route_arc_m = float(route_arc_m)
        self._candidate_signature = None
        self._candidate_frames = 0
        self._accumulated_turn_deg = 0.0
        self._trace_frames = [int(frame_index)]
        self._trace_route_arc_m = [float(route_arc_m)]
        self._trace_gt_edge_ids = self._trace_gt_edge_ids[-1:]
        transition = {
            "frame_index": int(frame_index),
            "route_arc_m": float(route_arc_m),
            "node_id": next_hop,
            "previous_node_id": previous,
            "created": False,
            "reason": "verified_backtrack_reanchor",
        }
        self.decision_trace[-1] = transition
        return transition


__all__ = ["CausalTopometricGraphV3"]
