"""Read-only diagnostics for frozen exit-token validity evidence."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.optimize import linear_sum_assignment


def binary_ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    precision_floor: float = 0.995,
) -> dict[str, float | int | None]:
    """Evaluate only realizable score thresholds, including all tied scores."""

    target = np.asarray(labels, dtype=np.bool_).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if (
        target.shape != score.shape
        or len(target) == 0
        or not np.all(np.isfinite(score))
        or not 0.0 < precision_floor <= 1.0
    ):
        raise ValueError("binary ranking inputs are invalid")
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("binary ranking requires both classes")

    order = np.argsort(-score, kind="stable")
    ranked_score = score[order]
    ranked_target = target[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(ranked_score[1:] != ranked_score[:-1]), len(score) - 1]
    cumulative_true = np.cumsum(ranked_target)[ends]
    predicted = ends + 1
    precision = cumulative_true / predicted
    recall = cumulative_true / positives
    recall_increment = np.diff(np.r_[0.0, recall])
    average_precision = float(np.sum(precision * recall_increment))
    valid = np.flatnonzero(precision >= precision_floor)
    if len(valid):
        chosen = int(valid[np.argmax(recall[valid])])
        threshold = float(ranked_score[ends[chosen]])
        high_precision_recall = float(recall[chosen])
        achieved_precision = float(precision[chosen])
        selected = int(predicted[chosen])
    else:
        threshold = None
        high_precision_recall = 0.0
        achieved_precision = None
        selected = 0
    return {
        "positives": positives,
        "negatives": negatives,
        "average_precision": average_precision,
        "precision_floor": float(precision_floor),
        "recall_at_precision_floor": high_precision_recall,
        "threshold_at_precision_floor": threshold,
        "precision_at_selected_threshold": achieved_precision,
        "selected_tokens": selected,
    }


def causal_descriptor_track_mean_confidence(
    tokens: np.ndarray,
    traversal_id: np.ndarray,
    sequence_index: np.ndarray,
    *,
    history: int = 5,
) -> np.ndarray:
    """Transport six slots by past-only descriptor cosine and average confidence."""

    value = np.asarray(tokens, dtype=np.float32)
    traversal = np.asarray(traversal_id)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    if (
        value.ndim != 3
        or value.shape[1:] != (6, 40)
        or traversal.shape != (len(value),)
        or sequence.shape != (len(value),)
        or history < 1
        or not np.all(np.isfinite(value))
    ):
        raise ValueError("causal token track contract drift")
    descriptor = value[:, :, 8:40].astype(np.float64)
    descriptor /= np.maximum(np.linalg.norm(descriptor, axis=2, keepdims=True), 1e-12)
    result = np.empty((len(value), 6), dtype=np.float32)
    tracks: list[list[float]] = []
    previous_descriptor: np.ndarray | None = None
    previous_traversal: object | None = None
    previous_sequence = -2
    for row in range(len(value)):
        current_traversal = traversal[row].item() if hasattr(traversal[row], "item") else traversal[row]
        contiguous = current_traversal == previous_traversal and int(sequence[row]) == previous_sequence + 1
        if not contiguous:
            tracks = [[float(value[row, slot, 0])] for slot in range(6)]
        else:
            if previous_descriptor is None:
                raise RuntimeError("descriptor track state is missing")
            left, right = linear_sum_assignment(1.0 - np.clip(previous_descriptor @ descriptor[row].T, -1.0, 1.0))
            updated: list[list[float] | None] = [None] * 6
            for old_slot, new_slot in zip(left.tolist(), right.tolist(), strict=True):
                updated[new_slot] = (tracks[old_slot] + [float(value[row, new_slot, 0])])[-history:]
            if any(item is None for item in updated):
                raise RuntimeError("descriptor assignment did not cover six slots")
            tracks = [item for item in updated if item is not None]
        result[row] = np.asarray([np.mean(item) for item in tracks], dtype=np.float32)
        previous_descriptor = descriptor[row]
        previous_traversal = current_traversal
        previous_sequence = int(sequence[row])
    return result


_PERMUTATIONS = np.asarray(list(permutations(range(6))), dtype=np.int16)


def cross_seed_geometry_consensus_score(tokens: np.ndarray, *, chunk_size: int = 2048) -> np.ndarray:
    """Score each seed/token by frozen 20deg/1log/1m three-seed agreement.

    The integer agreement count is the primary key and the token's own
    confidence is only a deterministic tie-breaker.  No Teacher value enters.
    """

    value = np.asarray(tokens, dtype=np.float32)
    if value.ndim != 4 or value.shape[1:] != (3, 6, 40) or chunk_size < 1 or not np.all(np.isfinite(value)):
        raise ValueError("cross-seed token contract drift")
    heading = value[:, :, :, 1:3].astype(np.float64)
    heading /= np.maximum(np.linalg.norm(heading, axis=3, keepdims=True), 1e-12)
    width = value[:, :, :, 3].astype(np.float64)
    profile = value[:, :, :, 4:8].astype(np.float64)
    agreement = np.zeros(value.shape[:3], dtype=np.int8)
    for reference in range(3):
        for other in range(3):
            if other == reference:
                continue
            for start in range(0, len(value), chunk_size):
                stop = min(start + chunk_size, len(value))
                cosine = np.einsum(
                    "nid,njd->nij", heading[start:stop, reference], heading[start:stop, other]
                )
                cosine = np.clip(cosine, -1.0, 1.0)
                width_log = np.abs(
                    np.log(
                        width[start:stop, reference, :, None]
                        / np.maximum(width[start:stop, other, None, :], 1e-12)
                    )
                )
                profile_error = np.mean(
                    np.abs(
                        profile[start:stop, reference, :, None, :]
                        - profile[start:stop, other, None, :, :]
                    ),
                    axis=3,
                )
                cost = 1.0 - cosine + width_log + profile_error
                assignment_cost = np.zeros((stop - start, len(_PERMUTATIONS)), dtype=np.float64)
                for slot in range(6):
                    assignment_cost += cost[:, slot, :][:, _PERMUTATIONS[:, slot]]
                chosen = _PERMUTATIONS[np.argmin(assignment_cost, axis=1)]
                batch = np.arange(stop - start)
                for slot in range(6):
                    matched = chosen[:, slot]
                    angle = np.degrees(np.arccos(cosine[batch, slot, matched]))
                    valid = (
                        (angle <= 20.0)
                        & (width_log[batch, slot, matched] <= 1.0)
                        & (profile_error[batch, slot, matched] <= 1.0)
                    )
                    agreement[start:stop, reference, slot] += valid
    return agreement.astype(np.float32) * 2.0 + value[:, :, :, 0]


__all__ = [
    "binary_ranking_metrics",
    "causal_descriptor_track_mean_confidence",
    "cross_seed_geometry_consensus_score",
]
