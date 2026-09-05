"""Evaluation helpers for causal exit-token transport and action-set events."""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np


DECISION_NAMES = ("corridor_or_other", "junction", "terminal")


def decision_from_visible_exit_count(count: np.ndarray) -> np.ndarray:
    """Map an executable visible-exit set to a conservative decision event.

    One exit is terminal support, three or more exits are junction support and
    two exits retain the corridor/other state.  This is an evaluation-only
    Teacher upper bound; learned models never receive the count target.
    """

    values = np.asarray(count, dtype=np.int64)
    if values.ndim != 1 or np.any((values < 1) | (values > 6)):
        raise ValueError("visible exit counts must be one-dimensional in [1,6]")
    return np.where(values >= 3, 1, np.where(values == 1, 2, 0)).astype(np.int8)


def multiclass_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    """Return deterministic three-class precision/recall/F1 and confusion."""

    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if (
        target.ndim != 1
        or prediction.shape != target.shape
        or len(target) == 0
        or np.any((target < 0) | (target > 2))
        or np.any((prediction < 0) | (prediction > 2))
    ):
        raise ValueError("decision targets and predictions must align in [0,2]")
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (target, prediction), 1)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, name in enumerate(DECISION_NAMES):
        true_positive = int(confusion[index, index])
        predicted = int(confusion[:, index].sum())
        actual = int(confusion[index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
        }
    return {
        "macro_f1": float(np.mean(f1_values)),
        "accuracy": float(np.trace(confusion) / len(target)),
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def _minimum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve a deterministic rectangular assignment with at most six tokens."""

    values = np.asarray(cost, dtype=np.float64)
    if (
        values.ndim != 2
        or min(values.shape) < 1
        or max(values.shape) > 6
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("token assignment cost must be finite and at most 6x6")
    transposed = values.shape[0] > values.shape[1]
    working = values.T if transposed else values
    rows, columns = working.shape

    @lru_cache(maxsize=None)
    def solve(row: int, used: int) -> tuple[float, tuple[int, ...]]:
        if row == rows:
            return 0.0, ()
        best: tuple[float, tuple[int, ...]] | None = None
        for column in range(columns):
            if used & (1 << column):
                continue
            tail_cost, tail = solve(row + 1, used | (1 << column))
            candidate = (float(working[row, column]) + tail_cost, (column,) + tail)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("no deterministic token assignment")
        return best

    chosen = np.asarray(solve(0, 0)[1], dtype=np.int64)
    left = np.arange(rows, dtype=np.int64)
    return (chosen, left) if transposed else (left, chosen)


def attach_teacher_exit_identities(
    predicted_tokens: np.ndarray,
    target: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Attach identities after frozen heading/width matching for evaluation.

    Teacher identity is never part of the assignment cost.  The returned
    identity is used only to score a subsequent deployment-only transport.
    """

    token = np.asarray(predicted_tokens, dtype=np.float32)
    mask = np.asarray(target["mask"], dtype=np.bool_)
    heading = np.asarray(target["heading"], dtype=np.float32)
    width = np.asarray(target["width"], dtype=np.float32)
    width_mask = np.asarray(target["width_mask"], dtype=np.bool_)
    identity = np.asarray(target["identity"], dtype=np.int64)
    count = len(token)
    if (
        token.shape != (count, 6, 40)
        or mask.shape != (count, 6)
        or heading.shape != (count, 6, 2)
        or width.shape != (count, 6)
        or width_mask.shape != (count, 6)
        or identity.shape != (count, 6)
        or np.any(width_mask & ~mask)
        or np.any(identity[~mask] != -1)
        or not np.all(np.isfinite(token))
    ):
        raise ValueError("predicted/Teacher exit-token contract drift")
    attached = np.full((count, 6), -1, dtype=np.int32)
    predicted_heading = token[:, :, 1:3]
    predicted_width = token[:, :, 3]
    for row in range(count):
        targets = np.flatnonzero(mask[row])
        cosine = np.clip(predicted_heading[row] @ heading[row, targets].T, -1.0, 1.0)
        cost = 1.0 - cosine
        cost += np.abs(
            np.log(predicted_width[row, :, None] / np.maximum(width[row, targets][None, :], 1e-6))
        ) * width_mask[row, targets][None, :]
        predictions, compact_targets = _minimum_assignment(cost)
        attached[row, predictions] = identity[row, targets[compact_targets]].astype(np.int32)
    return attached


def descriptor_transport_pair(
    previous_descriptor: np.ndarray,
    current_descriptor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate two visible token sets using descriptor cosine only."""

    previous = np.asarray(previous_descriptor, dtype=np.float64)
    current = np.asarray(current_descriptor, dtype=np.float64)
    if (
        previous.ndim != 2
        or current.ndim != 2
        or previous.shape[1:] != current.shape[1:]
        or previous.shape[1] != 32
        or not 1 <= len(previous) <= 6
        or not 1 <= len(current) <= 6
        or not np.all(np.isfinite(previous))
        or not np.all(np.isfinite(current))
    ):
        raise ValueError("descriptor transport requires two finite [1..6,32] sets")
    previous /= np.maximum(np.linalg.norm(previous, axis=1, keepdims=True), 1e-8)
    current /= np.maximum(np.linalg.norm(current, axis=1, keepdims=True), 1e-8)
    return _minimum_assignment(1.0 - np.clip(previous @ current.T, -1.0, 1.0))


__all__ = [
    "DECISION_NAMES",
    "attach_teacher_exit_identities",
    "decision_from_visible_exit_count",
    "descriptor_transport_pair",
    "multiclass_metrics",
]
