"""Objective post-replay spatial scoring for GSE trace-commit graphs.

Runtime graph construction must not consume Teacher identities or objective
centres.  This module is deliberately post-hoc: it one-to-one matches frozen
committed node locations to objective TNG decision-node centres, then scores
verified edges through that mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class ObjectiveGraphNode:
    identity: str
    world: str
    event: str
    xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_m, dtype=np.float64)
        if (
            not self.identity or not self.world
            or self.event not in ("junction", "terminal")
            or xyz.shape != (3,) or not np.all(np.isfinite(xyz))
        ):
            raise ValueError("objective graph node contract drift")


@dataclass(frozen=True)
class PredictedGraphNode:
    hypothesis_id: int
    world: str
    event: str
    xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_m, dtype=np.float64)
        if (
            self.hypothesis_id < 0 or not self.world
            or self.event not in ("junction", "terminal")
            or xyz.shape != (3,) or not np.all(np.isfinite(xyz))
        ):
            raise ValueError("predicted graph node contract drift")


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _group_assignment(
    predicted: Sequence[PredictedGraphNode],
    objective: Sequence[ObjectiveGraphNode],
    *,
    distance_cap_m: float,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, then minimum-distance, bipartite assignment.

    The augmented Hungarian matrix makes every valid match cheaper than
    leaving both endpoints unmatched, while invalid matches remain forbidden.
    This removes greedy-order dependence and naturally penalizes duplicates.
    """

    n_predicted, n_objective = len(predicted), len(objective)
    if not n_predicted or not n_objective:
        return []
    unmatched_cost = distance_cap_m + 1.0
    forbidden_cost = 1_000_000.0
    size = n_predicted + n_objective
    cost = np.full((size, size), forbidden_cost, dtype=np.float64)
    distances = np.empty((n_predicted, n_objective), dtype=np.float64)
    for row, first in enumerate(predicted):
        first_xyz = np.asarray(first.xyz_m, dtype=np.float64)
        for column, second in enumerate(objective):
            distance = float(np.linalg.norm(first_xyz - np.asarray(second.xyz_m, dtype=np.float64)))
            distances[row, column] = distance
            if distance <= distance_cap_m + 1e-12:
                cost[row, column] = distance
    # Each real prediction/target may choose its own dummy endpoint.
    for row in range(n_predicted):
        cost[row, n_objective + row] = unmatched_cost
    for column in range(n_objective):
        cost[n_predicted + column, column] = unmatched_cost
    # Dummy-to-dummy assignments complete the square at zero cost.
    cost[n_predicted:, n_objective:] = 0.0
    rows, columns = linear_sum_assignment(cost)
    matches = []
    for row, column in zip(rows, columns, strict=True):
        if row < n_predicted and column < n_objective:
            distance = float(distances[row, column])
            if distance <= distance_cap_m + 1e-12:
                matches.append((row, column, distance))
    return matches


def match_objective_graph_nodes(
    predicted_nodes: Sequence[PredictedGraphNode],
    objective_nodes: Sequence[ObjectiveGraphNode],
    *,
    distance_cap_m: float = 4.0,
) -> dict[str, object]:
    """Match only within identical world and structural-event groups."""

    if not math.isfinite(distance_cap_m) or distance_cap_m <= 0.0:
        raise ValueError("objective graph distance cap must be positive and finite")
    predicted_ids = [value.hypothesis_id for value in predicted_nodes]
    objective_ids = [value.identity for value in objective_nodes]
    if len(predicted_ids) != len(set(predicted_ids)):
        raise ValueError("predicted graph hypothesis IDs must be unique")
    if len(objective_ids) != len(set(objective_ids)):
        raise ValueError("objective graph identities must be unique")

    groups = sorted(
        {(value.world, value.event) for value in predicted_nodes}
        | {(value.world, value.event) for value in objective_nodes}
    )
    records: list[dict[str, object]] = []
    mapping: dict[int, str] = {}
    for world, event in groups:
        predicted = [value for value in predicted_nodes if value.world == world and value.event == event]
        objective = [value for value in objective_nodes if value.world == world and value.event == event]
        for row, column, distance in _group_assignment(
            predicted, objective, distance_cap_m=distance_cap_m,
        ):
            first, second = predicted[row], objective[column]
            mapping[first.hypothesis_id] = second.identity
            records.append({
                "hypothesis_id": first.hypothesis_id,
                "objective_identity": second.identity,
                "world": world,
                "event": event,
                "distance_m": distance,
                "predicted_xyz_m": list(first.xyz_m),
                "objective_xyz_m": list(second.xyz_m),
            })
    records.sort(key=lambda value: int(value["hypothesis_id"]))
    distances = [float(value["distance_m"]) for value in records]
    return {
        "distance_cap_m": float(distance_cap_m),
        "mapping": mapping,
        "matches": records,
        "matched_nodes": len(records),
        "unmatched_predicted_nodes": len(predicted_nodes) - len(records),
        "unmatched_objective_nodes": len(objective_nodes) - len(records),
        "mean_match_distance_m": float(np.mean(distances)) if distances else None,
        "maximum_match_distance_m": max(distances, default=None),
    }


def score_objective_spatial_graph(
    *,
    predicted_nodes: Sequence[PredictedGraphNode],
    predicted_edges: Sequence[Mapping[str, int]],
    objective_nodes: Sequence[ObjectiveGraphNode],
    objective_relations: set[tuple[str, str]],
    legacy_false_loop_merges: int = 0,
    distance_cap_m: float = 4.0,
) -> dict[str, object]:
    """Score frozen nodes and verified edge instances after spatial matching."""

    match = match_objective_graph_nodes(
        predicted_nodes, objective_nodes, distance_cap_m=distance_cap_m,
    )
    mapping = match["mapping"]
    assert isinstance(mapping, dict)
    matched = int(match["matched_nodes"])
    node_precision = matched / len(predicted_nodes) if predicted_nodes else 0.0
    node_recall = matched / len(objective_nodes) if objective_nodes else 0.0
    node_f1 = _f1(node_precision, node_recall)

    normalized_truth = {tuple(sorted((str(left), str(right)))) for left, right in objective_relations}
    edge_instances: list[tuple[str, str] | None] = []
    for edge in predicted_edges:
        left_id = int(edge["from_hypothesis"])
        right_id = int(edge["to_hypothesis"])
        left = mapping.get(left_id)
        right = mapping.get(right_id)
        edge_instances.append(
            tuple(sorted((left, right))) if left is not None and right is not None and left != right else None
        )
    correct_unique_edges = {
        value for value in edge_instances if value is not None and value in normalized_truth
    }
    edge_precision = len(correct_unique_edges) / len(edge_instances) if edge_instances else 0.0
    edge_recall = len(correct_unique_edges) / len(normalized_truth) if normalized_truth else 0.0
    edge_f1 = _f1(edge_precision, edge_recall)
    false_loop_fraction = legacy_false_loop_merges / len(predicted_nodes) if predicted_nodes else 0.0
    return {
        "scoring_contract": "post_replay_world_event_3d_one_to_one_v1",
        "distance_cap_m": float(distance_cap_m),
        "true_nodes": len(objective_nodes),
        "committed_nodes": len(predicted_nodes),
        "correct_unique_nodes": matched,
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "false_loop_merges": int(legacy_false_loop_merges),
        "false_loop_merge_fraction": false_loop_fraction,
        "true_trace_relations": len(normalized_truth),
        "committed_edges": len(edge_instances),
        "correct_unique_edges": len(correct_unique_edges),
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "node_edge_macro_f1": (node_f1 + edge_f1) / 2.0,
        "unmatched_predicted_edges": sum(value is None for value in edge_instances),
        "matching": match,
    }


__all__ = [
    "ObjectiveGraphNode", "PredictedGraphNode", "match_objective_graph_nodes",
    "score_objective_spatial_graph",
]
