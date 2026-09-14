"""Deterministic typed 4 m metrics for variable spatial event sets."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Mapping, Sequence

import numpy as np


EVENT_TYPE_NAMES = ("terminal", "junction")


def _maximum_valid_matching(
    predicted_type: np.ndarray,
    predicted_xyz: np.ndarray,
    target_type: np.ndarray,
    target_xyz: np.ndarray,
    *,
    maximum_error_m: float,
) -> list[tuple[int, int, float]]:
    """Maximize valid match count, then minimize total error deterministically."""

    prediction_count = len(predicted_type)
    target_count = len(target_type)
    if target_count == 0 or prediction_count == 0:
        return []
    distances = np.linalg.norm(predicted_xyz[:, None] - target_xyz[None], axis=2)
    valid = (
        predicted_type[:, None] == target_type[None]
    ) & (distances <= float(maximum_error_m) + 1e-9)

    @lru_cache(maxsize=None)
    def solve(target_index: int, used_mask: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if target_index == target_count:
            return 0, 0.0, ()
        best = solve(target_index + 1, used_mask)
        for prediction_index in range(prediction_count):
            if used_mask & (1 << prediction_index) or not valid[prediction_index, target_index]:
                continue
            count, cost, pairs = solve(
                target_index + 1, used_mask | (1 << prediction_index)
            )
            candidate = (
                count + 1,
                cost + float(distances[prediction_index, target_index]),
                ((prediction_index, target_index),) + pairs,
            )
            # More matches, then lower error, then lexical pair order.
            if (
                candidate[0] > best[0]
                or (candidate[0] == best[0] and candidate[1] < best[1] - 1e-12)
                or (
                    candidate[0] == best[0]
                    and abs(candidate[1] - best[1]) <= 1e-12
                    and candidate[2] < best[2]
                )
            ):
                best = candidate
        return best

    _, _, pairs = solve(0, 0)
    return [
        (prediction, target, float(distances[prediction, target]))
        for prediction, target in pairs
    ]


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def spatial_event_set_metrics(
    *,
    confidence: np.ndarray,
    predicted_type: np.ndarray,
    predicted_xyz_m: np.ndarray,
    target_type: np.ndarray,
    target_xyz_m: np.ndarray,
    target_mask: np.ndarray,
    threshold: float,
    maximum_error_m: float = 4.0,
) -> dict[str, object]:
    """Score same-type one-to-one matches that lie within the frozen 3D cap."""

    score = np.asarray(confidence, dtype=np.float64)
    predicted_class = np.asarray(predicted_type, dtype=np.int64)
    predicted_xyz = np.asarray(predicted_xyz_m, dtype=np.float64)
    target_class = np.asarray(target_type, dtype=np.int64)
    target_xyz = np.asarray(target_xyz_m, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    rows, queries = score.shape
    if (
        rows == 0
        or predicted_class.shape != (rows, queries)
        or predicted_xyz.shape != (rows, queries, 3)
        or target_class.shape != (rows, 16)
        or target_xyz.shape != (rows, 16, 3)
        or mask.shape != (rows, 16)
        or not np.all(np.isfinite(score))
        or not np.all(np.isfinite(predicted_xyz))
        or not np.all(np.isfinite(target_xyz))
        or np.any((score < 0.0) | (score > 1.0))
        or np.any((predicted_class < 0) | (predicted_class > 1))
        or not math.isfinite(float(threshold))
        or not 0.0 <= threshold <= 1.0
        or not math.isfinite(float(maximum_error_m))
        or maximum_error_m <= 0.0
    ):
        raise ValueError("spatial event metric input contract drift")

    true_positive = 0
    predicted_count = 0
    target_count = int(mask.sum())
    type_tp = np.zeros(2, dtype=np.int64)
    type_predicted = np.zeros(2, dtype=np.int64)
    type_target = np.bincount(target_class[mask], minlength=2).astype(np.int64)
    errors: list[float] = []
    multi_tp = 0
    multi_target = 0
    exact_rows = 0
    rows_with_prediction = 0
    for row in range(rows):
        selected = np.flatnonzero(score[row] >= float(threshold))
        # Content-canonical order makes matching independent of query enumeration.
        selected = np.asarray(
            sorted(
                selected.tolist(),
                key=lambda index: (
                    int(predicted_class[row, index]),
                    *tuple(float(value) for value in predicted_xyz[row, index]),
                    -float(score[row, index]),
                ),
            ),
            dtype=np.int64,
        )
        active = np.flatnonzero(mask[row])
        predicted_count += len(selected)
        rows_with_prediction += int(len(selected) > 0)
        if len(selected):
            type_predicted += np.bincount(
                predicted_class[row, selected], minlength=2
            ).astype(np.int64)
        matches = _maximum_valid_matching(
            predicted_class[row, selected],
            predicted_xyz[row, selected],
            target_class[row, active],
            target_xyz[row, active],
            maximum_error_m=maximum_error_m,
        )
        true_positive += len(matches)
        row_types = []
        for prediction_local, target_local, error in matches:
            event_type = int(target_class[row, active[target_local]])
            type_tp[event_type] += 1
            errors.append(error)
            row_types.append(event_type)
        if len(active) >= 2:
            multi_tp += len(matches)
            multi_target += len(active)
        if len(matches) == len(active) == len(selected):
            exact_rows += 1
    false_positive = predicted_count - true_positive
    false_negative = target_count - true_positive
    precision = _safe_divide(true_positive, predicted_count)
    recall = _safe_divide(true_positive, target_count)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    error_array = np.asarray(errors, dtype=np.float64)
    per_type = {}
    for index, name in enumerate(EVENT_TYPE_NAMES):
        type_precision = _safe_divide(int(type_tp[index]), int(type_predicted[index]))
        type_recall = _safe_divide(int(type_tp[index]), int(type_target[index]))
        type_f1 = (
            2.0 * type_precision * type_recall / (type_precision + type_recall)
            if type_precision + type_recall
            else 0.0
        )
        per_type[name] = {
            "true_positive": int(type_tp[index]),
            "predicted": int(type_predicted[index]),
            "target": int(type_target[index]),
            "precision": type_precision,
            "recall": type_recall,
            "f1": type_f1,
        }
    return {
        "threshold": float(threshold),
        "maximum_error_m": float(maximum_error_m),
        "rows": rows,
        "rows_with_prediction": rows_with_prediction,
        "target_events": target_count,
        "predicted_events": predicted_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_fraction": float(exact_rows / rows),
        "multi_event_recall": _safe_divide(multi_tp, multi_target),
        "matched_localization_mae_m": float(error_array.mean()) if len(error_array) else None,
        "matched_localization_median_m": float(np.median(error_array)) if len(error_array) else None,
        "matched_localization_p90_m": float(np.quantile(error_array, 0.9)) if len(error_array) else None,
        "per_type": per_type,
    }


def select_fixed_grid_threshold(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    thresholds: Sequence[float],
    *,
    maximum_error_m: float = 4.0,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select maximum F1, then precision, recall, and highest threshold."""

    records = [
        spatial_event_set_metrics(
            confidence=predictions["confidence"],
            predicted_type=predictions["event_type"],
            predicted_xyz_m=predictions["relative_xyz_m"],
            target_type=targets["event_type_index"],
            target_xyz_m=targets["event_relative_xyz_m"],
            target_mask=targets["event_mask"],
            threshold=float(threshold),
            maximum_error_m=maximum_error_m,
        )
        for threshold in thresholds
    ]
    best = max(
        records,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
            float(row["threshold"]),
        ),
    )
    return best, records


__all__ = [
    "EVENT_TYPE_NAMES",
    "select_fixed_grid_threshold",
    "spatial_event_set_metrics",
]
