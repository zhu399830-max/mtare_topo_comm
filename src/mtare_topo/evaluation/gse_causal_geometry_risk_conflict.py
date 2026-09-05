"""Deterministic primitives for causal geometry/risk-conflict audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from mtare_topo.evaluation.gse_causal_geometry_delta import (
    binary_roc_auc,
    fixed_negative_quantile_threshold,
)


CORRIDOR_EVENT = 0
CHANGE_POINT_EVENT = 4


@dataclass(frozen=True)
class FixedRiskSummary:
    lag_steps: int
    fit_pairs: int
    selection_pairs: int
    selection_corridor_frames: int
    selection_change_frames: int
    transition_vs_corridor_auc: float
    fit_corridor_threshold: float
    selection_false_positive_rate: float
    accepted_change_frames: int
    covered_change_identities: int
    total_change_identities: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def causal_lag_pairs(
    traversal_id: Iterable[str],
    sequence_index: Iterable[int],
    partition: Iterable[int],
    geometry_valid: Iterable[bool],
    *,
    lag_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact same-traversal, same-partition valid causal lag pairs."""

    traversal = np.asarray(tuple(traversal_id), dtype=str)
    sequence = np.asarray(tuple(sequence_index), dtype=np.int64)
    split = np.asarray(tuple(partition), dtype=np.int8)
    valid = np.asarray(tuple(geometry_valid), dtype=np.bool_)
    if (
        traversal.ndim != 1
        or sequence.shape != traversal.shape
        or split.shape != traversal.shape
        or valid.shape != traversal.shape
        or len(traversal) == 0
        or lag_steps <= 0
    ):
        raise ValueError("causal lag arrays must be aligned and lag_steps positive")
    if np.any(sequence < 0) or np.any(~np.isin(split, (0, 1))):
        raise ValueError("causal lag sequence/partition contract violated")
    lookup = {(str(traversal[row]), int(sequence[row])): row for row in range(len(sequence))}
    if len(lookup) != len(sequence):
        raise ValueError("duplicate traversal/sequence key")
    current: list[int] = []
    past: list[int] = []
    for row in range(len(sequence)):
        predecessor = lookup.get((str(traversal[row]), int(sequence[row]) - lag_steps))
        if predecessor is None or not valid[row] or not valid[predecessor]:
            continue
        if split[row] != split[predecessor]:
            raise ValueError("causal lag pair crosses a development partition")
        current.append(row)
        past.append(predecessor)
    return np.asarray(current, dtype=np.int64), np.asarray(past, dtype=np.int64)


def width_height_change_score(
    geometry: np.ndarray,
    current: np.ndarray,
    past: np.ndarray,
) -> np.ndarray:
    """Return the sealed observability score: max absolute width/height delta."""

    value = np.asarray(geometry, dtype=np.float64)
    current = np.asarray(current, dtype=np.int64)
    past = np.asarray(past, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 4 or current.shape != past.shape:
        raise ValueError("geometry must be [N,4] and lag indices aligned")
    if np.any(current < 0) or np.any(past < 0) or np.any(current >= len(value)) or np.any(past >= len(value)):
        raise ValueError("lag index is outside the geometry population")
    delta = value[current, :2] - value[past, :2]
    if not np.all(np.isfinite(delta)):
        raise ValueError("valid lag geometry contains nonfinite values")
    return np.max(np.abs(delta), axis=1)


def fixed_risk_summary(
    score: Iterable[float],
    current: Iterable[int],
    event: Iterable[int],
    identity: Iterable[int],
    partition: Iterable[int],
    *,
    lag_steps: int,
) -> FixedRiskSummary:
    """Evaluate a fit-frozen one-percent corridor threshold on selection."""

    value = np.asarray(tuple(score), dtype=np.float64)
    rows = np.asarray(tuple(current), dtype=np.int64)
    event_all = np.asarray(tuple(event), dtype=np.int64)
    identity_all = np.asarray(tuple(identity), dtype=np.int64)
    partition_all = np.asarray(tuple(partition), dtype=np.int8)
    if value.ndim != 1 or rows.shape != value.shape or len(value) == 0:
        raise ValueError("risk score/current must be aligned non-empty vectors")
    if event_all.shape != identity_all.shape or event_all.shape != partition_all.shape:
        raise ValueError("event/identity/partition population mismatch")
    if np.any(rows < 0) or np.any(rows >= len(event_all)) or not np.all(np.isfinite(value)):
        raise ValueError("invalid risk-score population")
    pair_event = event_all[rows]
    pair_identity = identity_all[rows]
    pair_partition = partition_all[rows]
    fit_corridor = (pair_partition == 0) & (pair_event == CORRIDOR_EVENT)
    selection_corridor = (pair_partition == 1) & (pair_event == CORRIDOR_EVENT)
    selection_change = (pair_partition == 1) & (pair_event == CHANGE_POINT_EVENT)
    if not fit_corridor.any() or not selection_corridor.any() or not selection_change.any():
        raise ValueError("fixed-risk audit requires fit corridor and selection corridor/change rows")
    threshold = fixed_negative_quantile_threshold(value[fit_corridor])
    evaluation = selection_corridor | selection_change
    accepted_change = selection_change & (value >= threshold)
    teacher_identities = set(pair_identity[selection_change].tolist())
    if any(item < 0 for item in teacher_identities):
        raise ValueError("selection change point lacks a structural identity")
    covered_identities = set(pair_identity[accepted_change].tolist())
    return FixedRiskSummary(
        lag_steps=lag_steps,
        fit_pairs=int(np.sum(pair_partition == 0)),
        selection_pairs=int(np.sum(pair_partition == 1)),
        selection_corridor_frames=int(selection_corridor.sum()),
        selection_change_frames=int(selection_change.sum()),
        transition_vs_corridor_auc=binary_roc_auc(value[evaluation], selection_change[evaluation]),
        fit_corridor_threshold=threshold,
        selection_false_positive_rate=float(np.mean(value[selection_corridor] >= threshold)),
        accepted_change_frames=int(accepted_change.sum()),
        covered_change_identities=len(covered_identities),
        total_change_identities=len(teacher_identities),
    )


def change_confidence_decomposition(
    probability: np.ndarray,
    event: Iterable[int],
) -> dict[str, float | int | list[int]]:
    """Describe binary-vs-conditional evidence on corrected change frames."""

    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(tuple(event), dtype=np.int64)
    if probability.ndim != 2 or probability.shape[1] != 5 or labels.shape != (len(probability),):
        raise ValueError("probability/event must align as [N,5]/[N]")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("event probabilities must be finite and nonnegative")
    totals = probability.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-5):
        raise ValueError("event probabilities must sum to one")
    mask = labels == CHANGE_POINT_EVENT
    if not mask.any():
        raise ValueError("confidence decomposition requires change-point rows")
    structural = 1.0 - probability[mask, CORRIDOR_EVENT]
    conditional = probability[mask, CHANGE_POINT_EVENT] / np.clip(structural, 1e-12, None)
    return {
        "change_frames": int(mask.sum()),
        "argmax_counts": np.bincount(
            np.argmax(probability[mask], axis=1), minlength=probability.shape[1]
        ).astype(int).tolist(),
        "median_structural_probability": float(np.median(structural)),
        "mean_structural_probability": float(np.mean(structural)),
        "median_conditional_change_probability": float(np.median(conditional)),
        "mean_conditional_change_probability": float(np.mean(conditional)),
        "median_final_change_probability": float(
            np.median(probability[mask, CHANGE_POINT_EVENT])
        ),
    }


def hard_negative_tail_summary(
    score: Iterable[float],
    corrected_event: Iterable[int],
    old_event: Iterable[int],
    partition: Iterable[int],
    *,
    threshold: float,
) -> dict[str, int | float]:
    """Split high-scoring selection corridor into removed-old-transition vs ordinary."""

    score = np.asarray(tuple(score), dtype=np.float64)
    corrected = np.asarray(tuple(corrected_event), dtype=np.int64)
    old = np.asarray(tuple(old_event), dtype=np.int64)
    split = np.asarray(tuple(partition), dtype=np.int8)
    if not (score.shape == corrected.shape == old.shape == split.shape) or len(score) == 0:
        raise ValueError("hard-negative arrays must align")
    if not np.isfinite(threshold) or not np.all(np.isfinite(score)):
        raise ValueError("hard-negative score/threshold must be finite")
    selection_corridor = (split == 1) & (corrected == CORRIDOR_EVENT)
    tail = selection_corridor & (score >= threshold)
    removed_transition = tail & (old == CHANGE_POINT_EVENT)
    ordinary = tail & (old != CHANGE_POINT_EVENT)
    return {
        "selection_corridor_rows": int(selection_corridor.sum()),
        "high_score_corridor_rows": int(tail.sum()),
        "removed_old_transition_rows": int(removed_transition.sum()),
        "ordinary_corridor_rows": int(ordinary.sum()),
        "removed_old_transition_fraction": (
            float(removed_transition.sum() / tail.sum()) if tail.any() else 0.0
        ),
    }


__all__ = [
    "CHANGE_POINT_EVENT",
    "CORRIDOR_EVENT",
    "FixedRiskSummary",
    "causal_lag_pairs",
    "change_confidence_decomposition",
    "fixed_risk_summary",
    "hard_negative_tail_summary",
    "width_height_change_score",
]
