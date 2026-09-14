"""Evidence metrics for GSE perception, association, and graph structure."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _prf(true_positive: int, predicted: int, actual: int) -> dict[str, float]:
    precision = float(true_positive / predicted) if predicted else 0.0
    recall = float(true_positive / actual) if actual else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(y_true: Sequence[str], y_predicted: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    if len(y_true) != len(y_predicted) or not y_true:
        raise ValueError("classification arrays must be non-empty and equal length")
    label_set = tuple(str(label) for label in labels)
    if len(set(label_set)) != len(label_set):
        raise ValueError("labels must be unique")
    per_class: dict[str, dict[str, float]] = {}
    for label in label_set:
        true_positive = sum(truth == label and prediction == label for truth, prediction in zip(y_true, y_predicted))
        predicted = sum(prediction == label for prediction in y_predicted)
        actual = sum(truth == label for truth in y_true)
        per_class[label] = _prf(true_positive, predicted, actual)
        per_class[label]["support"] = float(actual)
    return {"macro_f1": float(np.mean([values["f1"] for values in per_class.values()])), "per_class": per_class}


def continuous_geometry_mae(target: Mapping[str, Sequence[float]], predicted: Mapping[str, Sequence[float]]) -> dict[str, float]:
    if set(target) != set(predicted) or not target:
        raise ValueError("target and predicted geometry fields must match")
    result: dict[str, float] = {}
    lengths: set[int] = set()
    for name in sorted(target):
        truth = np.asarray(target[name], dtype=np.float64)
        estimate = np.asarray(predicted[name], dtype=np.float64)
        if truth.shape != estimate.shape or truth.ndim != 1 or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
            raise ValueError(f"invalid geometry arrays for {name}")
        lengths.add(len(truth))
        result[f"{name}_mae"] = float(np.mean(np.abs(truth - estimate)))
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("geometry arrays must share one non-zero length")
    return result


def relative_geometry_improvement(
    learned_mae: Mapping[str, float],
    baseline_mae: Mapping[str, float],
    *,
    minimum_macro_improvement: float = 0.10,
    maximum_single_field_regression: float = 0.05,
) -> dict[str, Any]:
    """Unitless paper gate across heterogeneous continuous geometry fields."""

    if set(learned_mae) != set(baseline_mae) or not learned_mae:
        raise ValueError("learned and baseline MAE fields must match and be non-empty")
    if not 0.0 <= minimum_macro_improvement < 1.0:
        raise ValueError("minimum macro improvement must be in [0,1)")
    if not 0.0 <= maximum_single_field_regression < 1.0:
        raise ValueError("maximum field regression must be in [0,1)")
    improvements: dict[str, float] = {}
    for name in sorted(learned_mae):
        learned = float(learned_mae[name])
        baseline = float(baseline_mae[name])
        if not math.isfinite(learned) or learned < 0.0 or not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError(f"invalid MAE pair for {name}")
        improvements[name] = float(1.0 - learned / baseline)
    macro = float(np.mean(list(improvements.values())))
    worst = float(min(improvements.values()))
    return {
        "per_field_relative_improvement": improvements,
        "macro_relative_improvement": macro,
        "worst_field_relative_improvement": worst,
        "minimum_macro_improvement": float(minimum_macro_improvement),
        "maximum_single_field_regression": float(maximum_single_field_regression),
        "macro_gate_passed": bool(macro >= minimum_macro_improvement),
        "no_material_field_regression": bool(worst >= -maximum_single_field_regression),
        "passed": bool(
            macro >= minimum_macro_improvement
            and worst >= -maximum_single_field_regression
        ),
    }


def association_metrics(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate only attempted persistent merges; provisional rejects are separate."""

    merge_attempts = 0
    correct_merges = 0
    false_merges = 0
    provisional = 0
    matchable_revisits = 0
    accepted_revisits = 0
    for decision in decisions:
        truth = decision.get("evaluator_gt_identity")
        existing_truth = decision.get("evaluator_matched_node_gt_identity")
        matchable = bool(decision.get("evaluator_has_existing_true_node", False))
        opportunity = bool(decision.get("evaluator_association_opportunity", True))
        if matchable and opportunity:
            matchable_revisits += 1
        status = decision.get("association_status")
        reason = decision.get("reason")
        if not opportunity:
            continue
        if status == "provisional" or reason == "ambiguous_association":
            provisional += 1
            continue
        if reason not in {
            "learned_association",
            "rule_association",
            "frozen_exit_token_ensemble",
            "factorized_consensus_metric",
        }:
            continue
        merge_attempts += 1
        if truth is not None and existing_truth == truth:
            correct_merges += 1
            if matchable:
                accepted_revisits += 1
        else:
            false_merges += 1
    precision = float(correct_merges / merge_attempts) if merge_attempts else 0.0
    recall = float(accepted_revisits / matchable_revisits) if matchable_revisits else 0.0
    return {
        "merge_attempts": merge_attempts,
        "correct_merges": correct_merges,
        "false_merges": false_merges,
        "provisional_rejections": provisional,
        "association_precision": precision,
        "association_recall": recall,
        "false_loop_merge_rate": float(false_merges / merge_attempts) if merge_attempts else 0.0,
    }


def graph_invariants(node_ids: Sequence[Any], edges: Sequence[Sequence[Any]]) -> dict[str, int]:
    unique_nodes = tuple(dict.fromkeys(node_ids))
    if len(unique_nodes) != len(node_ids):
        raise ValueError("graph node IDs must be unique")
    parent = {node: node for node in unique_nodes}

    def find(node: Any) -> Any:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: Any, second: Any) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    edge_count = 0
    for edge in edges:
        if len(edge) != 2 or edge[0] not in parent or edge[1] not in parent or edge[0] == edge[1]:
            raise ValueError("graph edge must connect two distinct existing nodes")
        union(edge[0], edge[1])
        edge_count += 1
    components = len({find(node) for node in unique_nodes}) if unique_nodes else 0
    cycle_rank = edge_count - len(unique_nodes) + components
    if cycle_rank < 0:
        raise RuntimeError("graph cycle rank cannot be negative")
    return {"node_count": len(unique_nodes), "edge_count": edge_count, "connected_components": components, "cycle_rank": cycle_rank}


def mapped_graph_f1(
    *,
    predicted_node_ids: Sequence[Any],
    predicted_edges: Sequence[Sequence[Any]],
    predicted_to_teacher: Mapping[Any, str],
    teacher_node_ids: Sequence[str],
    teacher_edges: Sequence[Sequence[str]],
) -> dict[str, Any]:
    teacher_nodes = {str(value) for value in teacher_node_ids}
    predicted_nodes = {str(predicted_to_teacher[value]) for value in predicted_node_ids if value in predicted_to_teacher}
    node_scores = _prf(len(predicted_nodes & teacher_nodes), len(predicted_nodes), len(teacher_nodes))

    teacher_edge_set = {frozenset((str(edge[0]), str(edge[1]))) for edge in teacher_edges}
    predicted_edge_set: set[frozenset[str]] = set()
    unmapped_edges = 0
    for first, second in predicted_edges:
        if first not in predicted_to_teacher or second not in predicted_to_teacher:
            unmapped_edges += 1
            continue
        mapped = frozenset((str(predicted_to_teacher[first]), str(predicted_to_teacher[second])))
        if len(mapped) != 2:
            continue
        predicted_edge_set.add(mapped)
    edge_true_positive = len(predicted_edge_set & teacher_edge_set)
    edge_scores = _prf(edge_true_positive, len(predicted_edge_set) + unmapped_edges, len(teacher_edge_set))
    predicted_invariants = graph_invariants(predicted_node_ids, predicted_edges)
    teacher_invariants = graph_invariants(list(teacher_nodes), [tuple(edge) for edge in teacher_edges])
    return {
        "node": node_scores,
        "edge": edge_scores,
        "unmapped_predicted_edges": unmapped_edges,
        "predicted_invariants": predicted_invariants,
        "teacher_invariants": teacher_invariants,
        "connected_component_error": predicted_invariants["connected_components"] - teacher_invariants["connected_components"],
        "cycle_rank_error": predicted_invariants["cycle_rank"] - teacher_invariants["cycle_rank"],
    }


def instance_mapped_graph_f1(
    *,
    predicted_node_ids: Sequence[Any],
    predicted_edges: Sequence[Sequence[Any]],
    predicted_to_teacher: Mapping[Any, str],
    teacher_node_ids: Sequence[str],
    teacher_edges: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Graph F1 that penalizes duplicate and unmapped predicted instances.

    ``mapped_graph_f1`` is retained for historical evidence. This stricter
    paper metric allows at most one true-positive predicted node per Teacher
    identity and one true-positive predicted edge per Teacher edge; duplicate
    nodes/edges remain in the prediction denominator.
    """

    predicted_ids = tuple(predicted_node_ids)
    if len(set(predicted_ids)) != len(predicted_ids):
        raise ValueError("predicted node IDs must be unique")
    teacher_nodes = {str(value) for value in teacher_node_ids}
    mapped_valid = {
        str(predicted_to_teacher[node])
        for node in predicted_ids
        if node in predicted_to_teacher and str(predicted_to_teacher[node]) in teacher_nodes
    }
    node_scores = _prf(len(mapped_valid), len(predicted_ids), len(teacher_nodes))
    duplicate_or_unmapped_nodes = len(predicted_ids) - len(mapped_valid)

    teacher_edge_set = {frozenset((str(edge[0]), str(edge[1]))) for edge in teacher_edges}
    mapped_edge_instances: list[frozenset[str] | None] = []
    for edge in predicted_edges:
        if len(edge) != 2 or edge[0] not in set(predicted_ids) or edge[1] not in set(predicted_ids):
            raise ValueError("predicted edge references an unknown node")
        if edge[0] not in predicted_to_teacher or edge[1] not in predicted_to_teacher:
            mapped_edge_instances.append(None)
            continue
        mapped = frozenset((str(predicted_to_teacher[edge[0]]), str(predicted_to_teacher[edge[1]])))
        mapped_edge_instances.append(mapped if len(mapped) == 2 else None)
    unique_true_edges = {
        edge for edge in mapped_edge_instances if edge is not None and edge in teacher_edge_set
    }
    edge_scores = _prf(len(unique_true_edges), len(predicted_edges), len(teacher_edge_set))
    duplicate_or_unmapped_edges = len(predicted_edges) - len(unique_true_edges)
    predicted_invariants = graph_invariants(predicted_ids, predicted_edges)
    teacher_invariants = graph_invariants(list(teacher_nodes), [tuple(edge) for edge in teacher_edges])
    return {
        "node": node_scores,
        "edge": edge_scores,
        "node_true_positive": len(mapped_valid),
        "predicted_node_count": len(predicted_ids),
        "teacher_node_count": len(teacher_nodes),
        "edge_true_positive": len(unique_true_edges),
        "predicted_edge_count": len(predicted_edges),
        "teacher_edge_count": len(teacher_edge_set),
        "duplicate_or_unmapped_predicted_nodes": duplicate_or_unmapped_nodes,
        "duplicate_or_unmapped_predicted_edges": duplicate_or_unmapped_edges,
        "predicted_invariants": predicted_invariants,
        "teacher_invariants": teacher_invariants,
        "connected_component_error": predicted_invariants["connected_components"] - teacher_invariants["connected_components"],
        "cycle_rank_error": predicted_invariants["cycle_rank"] - teacher_invariants["cycle_rank"],
    }


def relative_mae_improvement(baseline_mae: float, method_mae: float) -> float:
    if not math.isfinite(baseline_mae) or baseline_mae <= 0.0 or not math.isfinite(method_mae) or method_mae < 0.0:
        raise ValueError("MAE values must be finite and baseline positive")
    return float((baseline_mae - method_mae) / baseline_mae)


__all__ = [
    "association_metrics",
    "continuous_geometry_mae",
    "graph_invariants",
    "instance_mapped_graph_f1",
    "macro_f1",
    "mapped_graph_f1",
    "relative_mae_improvement",
]
