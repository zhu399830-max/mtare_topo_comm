"""Deterministic metrics for circular executable-geometry peak fields."""

from __future__ import annotations

import numpy as np


def circular_local_maxima(confidence: np.ndarray) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 180 or not np.all(np.isfinite(values)):
        raise ValueError("circular confidence must be finite [N,180]")
    # One strict side makes plateau handling deterministic and one-per-plateau.
    return (values >= np.roll(values, 1, axis=1)) & (values > np.roll(values, -1, axis=1))


def ranked_peak_metrics(confidence: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(confidence, dtype=np.float64)
    truth = np.asarray(target, dtype=np.bool_)
    if values.shape != truth.shape or values.ndim != 2 or values.shape[1] != 180:
        raise ValueError("ranked peak metric shape drift")
    candidate = circular_local_maxima(values)
    scores = values[candidate]
    labels = truth[candidate]
    positives = int(truth.sum())
    if positives <= 0 or len(scores) == 0:
        raise ValueError("ranked peak metrics require candidates and positives")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order].astype(np.int64)
    true_positive = np.cumsum(sorted_labels)
    precision = true_positive / np.arange(1, len(order) + 1)
    average_precision = float(precision[sorted_labels == 1].sum() / positives)
    return {
        "average_precision": average_precision,
        "candidate_peaks": int(candidate.sum()),
        "teacher_peaks": positives,
        "candidate_true_peaks": int(labels.sum()),
    }


def select_safe_threshold(
    confidence: np.ndarray,
    target: np.ndarray,
    *,
    precision_floor: float,
) -> dict[str, float | int] | None:
    if not 0.0 < precision_floor <= 1.0:
        raise ValueError("precision floor must be in (0,1]")
    values = np.asarray(confidence, dtype=np.float64)
    truth = np.asarray(target, dtype=np.bool_)
    candidate = circular_local_maxima(values)
    scores = values[candidate]
    labels = truth[candidate]
    positives = int(truth.sum())
    order = np.argsort(-scores, kind="stable")
    scores = scores[order]
    labels = labels[order].astype(np.int64)
    cumulative_true = np.cumsum(labels)
    group_end = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    selected = group_end + 1
    true_positive = cumulative_true[group_end]
    precision = true_positive / selected
    recall = true_positive / positives
    safe = np.flatnonzero(precision >= precision_floor)
    if not len(safe):
        return None
    best = max(safe, key=lambda index: (float(recall[index]), float(precision[index]), float(scores[group_end[index]])))
    return {
        "threshold": float(scores[group_end[best]]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "selected_peaks": int(selected[best]),
        "true_positive_peaks": int(true_positive[best]),
    }


def apply_peak_threshold(
    confidence: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    values = np.asarray(confidence, dtype=np.float64)
    truth = np.asarray(target, dtype=np.bool_)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    selected = circular_local_maxima(values) & (values >= float(threshold))
    true_positive = int((selected & truth).sum())
    false_positive = int((selected & ~truth).sum())
    false_negative = int((~selected & truth).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return selected, {
        "precision": float(precision),
        "recall": float(recall),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "selected_peaks": int(selected.sum()),
    }


def action_class_from_count(count: np.ndarray) -> np.ndarray:
    values = np.asarray(count, dtype=np.int64)
    result = np.full(values.shape, -1, dtype=np.int8)
    result[values == 1] = 0  # terminal
    result[values == 2] = 1  # corridor
    result[values >= 3] = 2  # junction
    return result


def action_macro_f1(predicted_count: np.ndarray, target_count: np.ndarray) -> dict[str, object]:
    predicted = action_class_from_count(predicted_count)
    target = action_class_from_count(target_count)
    per_class = []
    for class_index, name in enumerate(("terminal", "corridor", "junction")):
        true_positive = int(((predicted == class_index) & (target == class_index)).sum())
        false_positive = int(((predicted == class_index) & (target != class_index)).sum())
        false_negative = int(((predicted != class_index) & (target == class_index)).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        per_class.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": int((target == class_index).sum())})
    return {"macro_f1": float(np.mean([row["f1"] for row in per_class])), "per_class": per_class, "provisional_predictions": int((predicted < 0).sum())}


__all__ = [
    "action_macro_f1",
    "apply_peak_threshold",
    "circular_local_maxima",
    "ranked_peak_metrics",
    "select_safe_threshold",
]
