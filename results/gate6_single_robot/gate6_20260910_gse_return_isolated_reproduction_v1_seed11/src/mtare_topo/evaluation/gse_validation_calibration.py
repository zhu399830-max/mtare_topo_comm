"""Deterministic validation-only calibration primitives for GSE-Graph.

The functions in this module never consume test-world data and never mutate a
checkpoint.  They turn sealed validation logits/descriptors into auditable
scalar calibration parameters and complete threshold curves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    accepted: int
    true_positive: int
    false_positive: int
    precision: float
    recall: float
    false_accept_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventRejectionSelection:
    threshold: float
    maximum_uncertainty: float
    macro_f1: float
    structural_precision: float
    structural_recall: float
    accepted_structural_events: int
    rejected_structural_predictions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classification_arrays(
    scores: Sequence[float], correct: Sequence[bool], eligible_positive: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(correct, dtype=np.bool_)
    eligible = np.asarray(eligible_positive, dtype=np.bool_)
    if values.ndim != 1 or truth.shape != values.shape or eligible.shape != values.shape:
        raise ValueError("score, correctness and eligibility arrays must be aligned 1-D arrays")
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must be non-empty and finite")
    if np.any(truth & ~eligible):
        raise ValueError("a correct acceptance must be an eligible positive")
    return values, truth, eligible


def precision_constrained_threshold(
    scores: Sequence[float],
    correct: Sequence[bool],
    eligible_positive: Sequence[bool],
    *,
    minimum_precision: float = 0.98,
    maximum_false_accept_rate: float = 0.01,
) -> tuple[ThresholdSelection, list[dict[str, float | int]]]:
    """Maximize recovered true matches subject to the paper's safety gates.

    Every unique observed score is evaluated as an inclusive threshold.  The
    selected operating point is lexicographic: most correct accepted matches,
    then fewest false accepts, then highest precision, then highest threshold.
    This prevents a vacuous zero-acceptance solution and makes selection fully
    reproducible from the saved curve.
    """

    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0,1]")
    if not 0.0 <= maximum_false_accept_rate < 1.0:
        raise ValueError("maximum_false_accept_rate must be in [0,1)")
    values, truth, eligible = _classification_arrays(scores, correct, eligible_positive)
    total_positive = int(eligible.sum())
    if total_positive == 0:
        raise ValueError("calibration contains no eligible positive")

    curve: list[dict[str, float | int]] = []
    feasible: list[ThresholdSelection] = []
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    sorted_truth = truth[order]
    accepted = 0
    true_positive = 0
    for index, threshold in enumerate(sorted_values):
        accepted += 1
        true_positive += int(sorted_truth[index])
        if index + 1 < len(sorted_values) and sorted_values[index + 1] == threshold:
            continue
        false_positive = accepted - true_positive
        precision = true_positive / accepted
        recall = true_positive / total_positive
        false_accept_rate = false_positive / accepted
        point = ThresholdSelection(
            threshold=float(threshold),
            accepted=accepted,
            true_positive=true_positive,
            false_positive=false_positive,
            precision=float(precision),
            recall=float(recall),
            false_accept_rate=float(false_accept_rate),
        )
        curve.append(point.to_dict())
        if precision >= minimum_precision and false_accept_rate <= maximum_false_accept_rate:
            feasible.append(point)
    if not feasible:
        raise RuntimeError("no non-empty validation threshold satisfies the association safety gate")
    selected = max(
        feasible,
        key=lambda point: (
            point.true_positive,
            -point.false_positive,
            point.precision,
            point.threshold,
        ),
    )
    return selected, curve


def binary_f1_threshold(
    scores: Sequence[float], labels: Sequence[bool]
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Select an inclusive score threshold by maximum validation F1."""

    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.bool_)
    if values.ndim != 1 or truth.shape != values.shape or len(values) == 0:
        raise ValueError("binary calibration arrays must be aligned and non-empty")
    if not np.all(np.isfinite(values)) or not bool(truth.any()):
        raise ValueError("binary calibration requires finite scores and positive labels")
    curve: list[dict[str, float | int]] = []
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    sorted_truth = truth[order]
    total_positive = int(truth.sum())
    true_positive = 0
    accepted = 0
    for index, threshold in enumerate(sorted_values):
        accepted += 1
        true_positive += int(sorted_truth[index])
        if index + 1 < len(sorted_values) and sorted_values[index + 1] == threshold:
            continue
        false_positive = accepted - true_positive
        false_negative = total_positive - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        curve.append(
            {
                "threshold": float(threshold),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    selected = max(
        curve,
        key=lambda point: (
            float(point["f1"]),
            float(point["precision"]),
            float(point["recall"]),
            float(point["threshold"]),
        ),
    )
    return selected, curve


def causal_nearest_descriptor_records(
    descriptors: np.ndarray,
    identities: Sequence[int],
    parent_ids: Sequence[str],
    sequence_order: Sequence[int],
) -> list[dict[str, Any]]:
    """Produce past-only nearest-neighbour decisions within each parent world.

    Rows sharing a sequence order are simultaneous and cannot match each
    other. GT identity is used only to score a proposed match; it is never
    returned as model input or passed into online graph association.
    """

    vectors = np.asarray(descriptors, dtype=np.float64)
    labels = np.asarray(identities, dtype=np.int64)
    parents = np.asarray(parent_ids)
    order = np.asarray(sequence_order, dtype=np.int64)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("descriptors must be a non-empty [N,D] array")
    if labels.shape != (len(vectors),) or parents.shape != labels.shape or order.shape != labels.shape:
        raise ValueError("descriptor metadata arrays are misaligned")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("descriptors must be finite")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("descriptors must be non-zero")
    vectors = vectors / norms

    records: list[dict[str, Any]] = []
    for parent in sorted({str(value) for value in parents.tolist()}):
        indices = np.flatnonzero(parents.astype(str) == parent)
        indices = indices[np.lexsort((indices, order[indices]))]
        past: list[int] = []
        for group_order in np.unique(order[indices]):
            group = indices[order[indices] == group_order].tolist()
            valid_past = [candidate for candidate in past if int(labels[candidate]) >= 0]
            valid_queries = [query for query in group if int(labels[query]) >= 0]
            similarities = (
                vectors[valid_queries] @ vectors[valid_past].T
                if valid_queries and valid_past
                else np.empty((len(valid_queries), 0), dtype=np.float64)
            )
            for query_row, query_index in enumerate(valid_queries):
                identity = int(labels[query_index])
                if not valid_past:
                    continue
                row_similarities = similarities[query_row]
                ranked = np.argsort(-row_similarities, kind="stable")
                best_position = int(ranked[0])
                best_index = int(valid_past[best_position])
                best_score = float(row_similarities[best_position])
                second_score = float(row_similarities[int(ranked[1])]) if len(ranked) > 1 else -1.0
                records.append(
                    {
                        "parent_id": parent,
                        "query_index": int(query_index),
                        "candidate_index": best_index,
                        "sequence_order": int(order[query_index]),
                        "score": best_score,
                        "margin": best_score - second_score,
                        "correct": bool(int(labels[best_index]) == identity),
                        "eligible_positive": bool(any(int(labels[value]) == identity for value in valid_past)),
                    }
                )
            past.extend(group)
    if not records:
        raise RuntimeError("causal descriptor calibration produced no match decisions")
    return records


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def event_negative_log_likelihood(
    logits: np.ndarray, labels: Sequence[int], *, temperature: float
) -> float:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] == 0 or truth.shape != (len(values),):
        raise ValueError("event logits and labels must be aligned non-empty arrays")
    if not np.all(np.isfinite(values)) or np.any(truth < 0) or np.any(truth >= values.shape[1]):
        raise ValueError("event calibration arrays contain invalid values")
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    probabilities = _softmax(values / float(temperature))
    return float(-np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-300)).mean())


def fit_event_temperature(
    logits: np.ndarray,
    labels: Sequence[int],
    *,
    minimum_temperature: float = 0.05,
    maximum_temperature: float = 20.0,
    iterations: int = 96,
) -> dict[str, float | int]:
    """Fit one event temperature by fixed-iteration golden-section search."""

    if not 0.0 < minimum_temperature < maximum_temperature or iterations < 1:
        raise ValueError("invalid temperature search contract")
    left = math.log(minimum_temperature)
    right = math.log(maximum_temperature)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)

    def objective(log_temperature: float) -> float:
        return event_negative_log_likelihood(logits, labels, temperature=math.exp(log_temperature))

    first_value = objective(first)
    second_value = objective(second)
    for _ in range(iterations):
        if first_value <= second_value:
            right, second, second_value = second, first, first_value
            first = right - ratio * (right - left)
            first_value = objective(first)
        else:
            left, first, first_value = first, second, second_value
            second = left + ratio * (right - left)
            second_value = objective(second)
    temperature = math.exp((left + right) / 2.0)
    before = event_negative_log_likelihood(logits, labels, temperature=1.0)
    after = event_negative_log_likelihood(logits, labels, temperature=temperature)
    if after > before + 1e-10:
        temperature = 1.0
        after = before
    return {
        "temperature": float(temperature),
        "negative_log_likelihood_before": before,
        "negative_log_likelihood_after": after,
        "iterations": int(iterations),
    }


def _macro_f1_indices(truth: np.ndarray, predicted: np.ndarray, class_count: int) -> float:
    scores = []
    for index in range(class_count):
        true_positive = int(np.sum((truth == index) & (predicted == index)))
        predicted_count = int(np.sum(predicted == index))
        actual_count = int(np.sum(truth == index))
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def select_event_rejection_threshold(
    logits: np.ndarray,
    labels: Sequence[int],
    uncertainty: Sequence[float],
    *,
    temperature: float,
    corridor_index: int = 0,
) -> tuple[EventRejectionSelection, list[dict[str, float | int]]]:
    """Select one auditable threshold for confidence and uncertainty rejection.

    A predicted structural event is accepted only when both its calibrated
    probability and ``1 - uncertainty`` exceed the same threshold. Rejected
    structural predictions become ordinary corridor observations and therefore
    cannot create graph nodes. The threshold maximizes five-class macro-F1;
    ties prefer structural precision, then retention, then stronger rejection.
    """

    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    risk = np.asarray(uncertainty, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) == 0
        or truth.shape != (len(values),)
        or risk.shape != truth.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(risk))
        or np.any(risk < 0.0)
        or np.any(risk > 1.0)
        or np.any(truth < 0)
        or np.any(truth >= values.shape[1])
        or not 0 <= corridor_index < values.shape[1]
    ):
        raise ValueError("event rejection arrays violate the calibration contract")
    probabilities = _softmax(values / float(temperature))
    raw_prediction = probabilities.argmax(axis=1)
    predicted_confidence = probabilities[np.arange(len(values)), raw_prediction]
    reliability = np.minimum(predicted_confidence, 1.0 - risk)
    raw_structural = raw_prediction != corridor_index
    actual_structural = truth != corridor_index
    candidates = np.unique(np.concatenate((np.asarray((0.0, 1.0)), reliability[raw_structural])))
    curve: list[dict[str, float | int]] = []
    selections: list[EventRejectionSelection] = []
    for threshold in candidates:
        accepted = raw_structural & (reliability >= threshold)
        predicted = raw_prediction.copy()
        predicted[raw_structural & ~accepted] = corridor_index
        true_structural = int(np.sum(accepted & actual_structural))
        accepted_count = int(accepted.sum())
        actual_count = int(actual_structural.sum())
        structural_precision = true_structural / accepted_count if accepted_count else 0.0
        structural_recall = true_structural / actual_count if actual_count else 0.0
        selection = EventRejectionSelection(
            threshold=float(threshold),
            maximum_uncertainty=float(1.0 - threshold),
            macro_f1=_macro_f1_indices(truth, predicted, values.shape[1]),
            structural_precision=float(structural_precision),
            structural_recall=float(structural_recall),
            accepted_structural_events=accepted_count,
            rejected_structural_predictions=int(np.sum(raw_structural & ~accepted)),
        )
        curve.append(selection.to_dict())
        selections.append(selection)
    selected = max(
        selections,
        key=lambda item: (
            item.macro_f1,
            item.structural_precision,
            item.accepted_structural_events,
            item.threshold,
        ),
    )
    return selected, curve


__all__ = [
    "EventRejectionSelection",
    "ThresholdSelection",
    "binary_f1_threshold",
    "causal_nearest_descriptor_records",
    "event_negative_log_likelihood",
    "fit_event_temperature",
    "precision_constrained_threshold",
    "select_event_rejection_threshold",
]
