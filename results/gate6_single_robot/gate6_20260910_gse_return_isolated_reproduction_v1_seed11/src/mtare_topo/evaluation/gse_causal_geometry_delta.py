"""Deterministic audits for causal geometric-change observability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BinaryScoreSummary:
    negative_count: int
    positive_count: int
    roc_auc: float
    negative_median: float
    negative_p90: float
    positive_median: float
    positive_p10: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def binary_roc_auc(score: Iterable[float], positive: Iterable[bool]) -> float:
    """Compute exact rank AUC with deterministic average ranks for ties."""

    values = np.asarray(tuple(score), dtype=np.float64)
    labels = np.asarray(tuple(positive), dtype=np.bool_)
    if values.ndim != 1 or labels.shape != values.shape or len(values) == 0:
        raise ValueError("score/positive must be aligned non-empty vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("binary score must be finite")
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("binary AUC requires both classes")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    numerator = ranks[labels].sum() - positive_count * (positive_count + 1) / 2.0
    result = float(numerator / (positive_count * negative_count))
    if not 0.0 <= result <= 1.0 or not math.isfinite(result):
        raise RuntimeError("binary AUC is invalid")
    return result


def summarize_binary_score(
    score: Iterable[float], positive: Iterable[bool]
) -> BinaryScoreSummary:
    values = np.asarray(tuple(score), dtype=np.float64)
    labels = np.asarray(tuple(positive), dtype=np.bool_)
    auc = binary_roc_auc(values, labels)
    negative = values[~labels]
    positive_values = values[labels]
    return BinaryScoreSummary(
        negative_count=len(negative),
        positive_count=len(positive_values),
        roc_auc=auc,
        negative_median=float(np.median(negative)),
        negative_p90=float(np.quantile(negative, 0.9)),
        positive_median=float(np.median(positive_values)),
        positive_p10=float(np.quantile(positive_values, 0.1)),
    )


def fixed_negative_quantile_threshold(
    fit_negative_score: Iterable[float], *, false_positive_rate: float = 0.01
) -> float:
    values = np.asarray(tuple(fit_negative_score), dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or not 0.0 < false_positive_rate < 1.0
    ):
        raise ValueError("invalid negative quantile threshold inputs")
    return float(
        np.quantile(values, 1.0 - false_positive_rate, method="higher")
    )


def signed_lag_delta(
    current: np.ndarray,
    past: np.ndarray,
) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    past = np.asarray(past, dtype=np.float64)
    if current.shape != past.shape or current.ndim != 2 or current.shape[1] != 4:
        raise ValueError("geometry arrays must be aligned [N,4]")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(past)):
        raise ValueError("geometry arrays must be finite")
    return current - past


__all__ = [
    "BinaryScoreSummary",
    "binary_roc_auc",
    "fixed_negative_quantile_threshold",
    "signed_lag_delta",
    "summarize_binary_score",
]
