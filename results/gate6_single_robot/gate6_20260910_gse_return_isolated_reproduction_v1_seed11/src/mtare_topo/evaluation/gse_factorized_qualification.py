"""Frozen-threshold qualification helpers for Factorized GSE association."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


DECISION_EVENTS = frozenset(("junction", "terminal"))


def runtime_candidate_pairs(
    rows: Sequence[Mapping[str, object]],
    association_valid: np.ndarray,
    sensor_xyz_m: np.ndarray,
    world_sequence_row: np.ndarray,
    *,
    maximum_distance_m: float = 16.0,
) -> dict[str, np.ndarray]:
    """Build strictly-past candidates, keeping the nearest view per identity."""

    valid = np.asarray(association_valid, dtype=np.bool_)
    xyz = np.asarray(sensor_xyz_m, dtype=np.float64)
    world_row = np.asarray(world_sequence_row, dtype=np.int64)
    if (
        len(rows) == 0
        or valid.shape != (len(rows),)
        or xyz.shape != (len(rows), 3)
        or world_row.shape != (len(rows),)
        or not np.all(np.isfinite(xyz))
        or not np.isfinite(maximum_distance_m)
        or maximum_distance_m <= 0.0
    ):
        raise ValueError("runtime candidate population contract drift")
    parent = np.asarray([str(row["parent_id"]) for row in rows])
    identity = np.asarray([str(row["identity"]) for row in rows])
    event = np.asarray([str(row["event"]) for row in rows])
    global_index = np.asarray([int(row["global_sequence_index"]) for row in rows], dtype=np.int64)
    decision = valid & np.isin(event, tuple(DECISION_EVENTS))
    left: list[int] = []
    right: list[int] = []
    label: list[bool] = []
    distance: list[float] = []
    family: list[str] = []
    query_with_candidate = 0
    query_with_positive = 0
    for current_parent in sorted(set(parent.tolist())):
        queries = np.flatnonzero((parent == current_parent) & decision)
        order = np.lexsort((global_index[queries], world_row[queries]))
        queries = queries[order]
        if len(np.unique(world_row[queries])) != len(queries):
            raise ValueError("runtime decision queries have duplicate world sequence rows")
        past: list[int] = []
        for query in queries:
            by_identity: dict[str, tuple[float, int]] = {}
            for candidate in past:
                current_distance = float(np.linalg.norm(xyz[candidate] - xyz[query]))
                if current_distance > maximum_distance_m + 1e-12:
                    continue
                key = str(identity[candidate])
                rank = (current_distance, int(global_index[candidate]))
                if key not in by_identity or rank < (
                    by_identity[key][0], int(global_index[by_identity[key][1]])
                ):
                    by_identity[key] = (current_distance, int(candidate))
            if by_identity:
                query_with_candidate += 1
            if str(identity[query]) in by_identity:
                query_with_positive += 1
            for candidate_identity, (current_distance, candidate) in sorted(by_identity.items()):
                if world_row[candidate] >= world_row[query]:
                    raise RuntimeError("runtime association candidate is not strictly past")
                left.append(int(query))
                right.append(candidate)
                label.append(candidate_identity == str(identity[query]))
                distance.append(current_distance)
                family.append(str(current_parent).split("_", 1)[0])
            past.append(int(query))
    return {
        "left": np.asarray(left, dtype=np.int64),
        "right": np.asarray(right, dtype=np.int64),
        "label": np.asarray(label, dtype=np.uint8),
        "distance_m": np.asarray(distance, dtype=np.float32),
        "family": np.asarray(family),
        "decision_queries": np.asarray(int(np.sum(decision)), dtype=np.int64),
        "queries_with_candidate": np.asarray(query_with_candidate, dtype=np.int64),
        "queries_with_positive": np.asarray(query_with_positive, dtype=np.int64),
    }


def fixed_threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    score = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    family = np.asarray(families).astype(str)
    if (
        score.ndim != 1
        or truth.shape != score.shape
        or family.shape != score.shape
        or not np.all(np.isfinite(score))
        or not np.isfinite(threshold)
        or len(score) == 0
    ):
        raise ValueError("fixed-threshold metric inputs are invalid")
    accepted = score >= float(threshold)

    def summarize(mask: np.ndarray) -> dict[str, object]:
        positive = int(np.sum(truth & mask))
        negative = int(np.sum(~truth & mask))
        true_positive = int(np.sum(accepted & truth & mask))
        false_positive = int(np.sum(accepted & ~truth & mask))
        accepted_count = true_positive + false_positive
        return {
            "pairs": int(np.sum(mask)), "positive_support": positive,
            "negative_support": negative, "accepted": accepted_count,
            "true_positive": true_positive, "false_positive": false_positive,
            "precision_identifiable": negative > 0,
            "precision": true_positive / accepted_count if accepted_count else 0.0,
            "false_accept_rate": false_positive / accepted_count if accepted_count else 0.0,
            "recall": true_positive / positive if positive else 0.0,
            "negative_rejection_rate": 1.0 - false_positive / negative if negative else None,
        }

    result = summarize(np.ones(len(score), dtype=np.bool_))
    result["threshold"] = float(threshold)
    result["per_family"] = {
        name: summarize(family == name) for name in sorted(set(family.tolist()))
    }
    return result


def select_consensus_metric_configuration(
    balanced_acceptance: np.ndarray,
    balanced_labels: np.ndarray,
    balanced_families: np.ndarray,
    balanced_physical_mask: np.ndarray,
    runtime_acceptance: np.ndarray,
    runtime_labels: np.ndarray,
    runtime_families: np.ndarray,
    runtime_distance_m: np.ndarray,
    *,
    distance_grid_m: np.ndarray | None = None,
) -> dict[str, object]:
    """Select a robust consensus/distance contract on development data only."""

    balanced_votes = np.asarray(balanced_acceptance, dtype=np.bool_)
    runtime_votes = np.asarray(runtime_acceptance, dtype=np.bool_)
    balanced_labels = np.asarray(balanced_labels, dtype=np.uint8)
    runtime_labels = np.asarray(runtime_labels, dtype=np.uint8)
    balanced_families = np.asarray(balanced_families).astype(str)
    runtime_families = np.asarray(runtime_families).astype(str)
    physical = np.asarray(balanced_physical_mask, dtype=np.bool_)
    distance = np.asarray(runtime_distance_m, dtype=np.float64)
    grid = np.arange(.5, 16.01, .5) if distance_grid_m is None else np.asarray(distance_grid_m, dtype=np.float64)
    if (
        balanced_votes.shape != (3, len(balanced_labels))
        or runtime_votes.shape != (3, len(runtime_labels))
        or balanced_families.shape != balanced_labels.shape
        or runtime_families.shape != runtime_labels.shape
        or physical.shape != balanced_labels.shape
        or distance.shape != runtime_labels.shape
        or grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid))
        or np.any(grid <= 0.0) or np.any(grid > 16.0)
    ):
        raise ValueError("consensus metric selection input contract drift")
    candidates: list[dict[str, object]] = []
    for votes_required in (1, 2, 3):
        balanced_accept = np.sum(balanced_votes, axis=0) >= votes_required
        balanced = fixed_threshold_metrics(
            balanced_accept.astype(np.float64), balanced_labels, balanced_families, .5
        )
        physical_balanced = fixed_threshold_metrics(
            balanced_accept[physical].astype(np.float64), balanced_labels[physical],
            balanced_families[physical], .5,
        )
        balanced_margin = bool(
            balanced["accepted"] > 0 and balanced["precision"] >= .995
            and balanced["false_accept_rate"] <= .005 and balanced["recall"] >= .25
            and all(value["accepted"] > 0 and value["precision"] >= .98 and value["recall"] >= .10 for value in balanced["per_family"].values())
        )
        physical_margin = bool(
            physical_balanced["accepted"] > 0 and physical_balanced["precision"] >= .995
            and physical_balanced["false_accept_rate"] <= .005
            and physical_balanced["recall"] >= .25
            and all(value["accepted"] > 0 and value["precision"] >= .98 and value["recall"] >= .10 for value in physical_balanced["per_family"].values())
        )
        runtime_consensus = np.sum(runtime_votes, axis=0) >= votes_required
        for cap in grid:
            runtime_accept = runtime_consensus & (distance <= float(cap) + 1e-12)
            runtime = fixed_threshold_metrics(
                runtime_accept.astype(np.float64), runtime_labels, runtime_families, .5
            )
            runtime_margin = bool(
                runtime["accepted"] > 0 and runtime["precision"] >= .995
                and runtime["false_accept_rate"] <= .005 and runtime["recall"] >= .25
            )
            candidates.append({
                "votes_required": votes_required, "distance_cap_m": float(cap),
                "balanced": balanced, "balanced_physical_only": physical_balanced,
                "runtime": runtime,
                "checks": {"balanced_margin": balanced_margin, "physical_margin": physical_margin, "runtime_margin": runtime_margin},
                "qualifies": balanced_margin and physical_margin and runtime_margin,
            })
    eligible = [row for row in candidates if row["qualifies"]]
    if not eligible:
        raise RuntimeError("no consensus metric configuration satisfies the selection margin")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["runtime"]["recall"]), float(row["balanced"]["recall"]),
            -float(row["distance_cap_m"]), int(row["votes_required"]),
        ),
    )
    return {
        "selection_rule": "maximize runtime recall; tie balanced recall; tie smaller distance cap; tie larger vote count",
        "selection_margin": {"precision": .995, "false_accept_rate": .005, "recall": .25, "balanced_family_precision": .98, "balanced_family_recall": .10},
        "grid": {"votes_required": [1, 2, 3], "distance_cap_m": grid.tolist()},
        "qualifying_configurations": len(eligible), "selected": selected,
        "candidates": candidates,
    }


__all__ = [
    "DECISION_EVENTS", "fixed_threshold_metrics", "runtime_candidate_pairs",
    "select_consensus_metric_configuration",
]
