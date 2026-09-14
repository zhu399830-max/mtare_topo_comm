"""Distance-and-exit-angle association ablation for geometry-event graphs."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mtare_topo.topology.causal_graph_v2 import heading_set_distance
from mtare_topo.topology.gse_graph import GSEGraphConfig, GeometrySemanticEventGraph


class RuleAssociatedGeometryEventGraph(GeometrySemanticEventGraph):
    """Use no learned descriptor when associating graph nodes.

    Candidate generation retains the same event type, spatial radius, exit-count
    and heading-set contracts as GSE.  The nearest candidate is accepted unless
    the normalized distance gap to the runner-up is below the predeclared
    ambiguity margin, in which case the observation remains provisional.
    """

    def __init__(self, config: GSEGraphConfig) -> None:
        super().__init__(config, association_reason="rule_association")

    def _associate(
        self,
        *,
        xyz: np.ndarray,
        event: str,
        headings_world: Sequence[float],
        descriptor: Sequence[float],
        exit_tokens: Sequence[Any],
        yaw_deg: float,
        node_kind: str,
        association_key: int | None = None,
    ) -> tuple[int | None, bool, list[dict[str, float | int]]]:
        del descriptor, yaw_deg, association_key
        candidates: list[dict[str, float | int]] = []
        for node in self._nearby_nodes(xyz):
            if node["node_kind"] != node_kind:
                continue
            if node_kind == "structural" and node["event"] != event:
                continue
            distance = float(np.linalg.norm(xyz - np.asarray(node["xyz_m"], dtype=np.float64)))
            if distance > self.config.association_radius_m:
                continue
            stored_headings = node["exit_headings_world_deg"]
            if abs(len(exit_tokens) - len(node["exit_tokens"])) > self.config.maximum_exit_count_difference:
                continue
            exit_distance = heading_set_distance(headings_world, stored_headings)
            if exit_distance > self.config.exit_heading_tolerance_deg:
                continue
            candidates.append(
                {
                    "node_id": int(node["id"]),
                    "distance_m": distance,
                    "normalized_distance": distance / self.config.association_radius_m,
                    "exit_distance_deg": float(exit_distance),
                }
            )
        candidates.sort(
            key=lambda item: (
                float(item["distance_m"]),
                float(item["exit_distance_deg"]),
                int(item["node_id"]),
            )
        )
        if not candidates:
            return None, False, candidates
        if len(candidates) > 1:
            normalized_gap = (
                float(candidates[1]["distance_m"]) - float(candidates[0]["distance_m"])
            ) / self.config.association_radius_m
            if normalized_gap < self.config.ambiguity_similarity_margin:
                return None, True, candidates
        return int(candidates[0]["node_id"]), False, candidates


__all__ = ["RuleAssociatedGeometryEventGraph"]
