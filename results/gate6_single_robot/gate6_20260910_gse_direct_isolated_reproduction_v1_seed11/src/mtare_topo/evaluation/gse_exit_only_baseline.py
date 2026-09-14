"""Fair five-event evaluation adapter for the frozen M1D exit-only baseline.

M1D predicts the historical ``interior/junction/terminal`` role vocabulary.
For the GSE paper comparison those probabilities are embedded, without
retraining or validation tuning, in the five-event vocabulary.  Consequently
the baseline can never claim a turn or geometry-transition detection.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


M1D_ROLE_NAMES = ("interior", "junction", "terminal")
M1D_TO_GSE_EVENT = {
    "interior": "corridor",
    "junction": "junction",
    "terminal": "terminal",
}


def m1d_role_probabilities_to_gse(role_probabilities: np.ndarray) -> np.ndarray:
    """Embed three frozen role probabilities into the five GSE events."""

    values = np.asarray(role_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(M1D_ROLE_NAMES):
        raise ValueError("M1D role probabilities must have shape [N,3]")
    if (
        len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("M1D role probabilities must be finite normalized distributions")
    result = np.zeros((len(values), len(EVENT_NAMES)), dtype=np.float64)
    for role_index, role_name in enumerate(M1D_ROLE_NAMES):
        result[:, EVENT_NAMES.index(M1D_TO_GSE_EVENT[role_name])] = values[:, role_index]
    return result


def m1d_role_logits_to_gse(role_logits: np.ndarray) -> np.ndarray:
    """Stable softmax followed by the fixed role-to-event embedding."""

    logits = np.asarray(role_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != len(M1D_ROLE_NAMES) or not np.all(np.isfinite(logits)):
        raise ValueError("M1D role logits must be finite with shape [N,3]")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return m1d_role_probabilities_to_gse(exponential / exponential.sum(axis=1, keepdims=True))


def five_event_scores(target_indices: Sequence[int], probabilities: np.ndarray) -> dict[str, Any]:
    """Return the same five-class metrics used for GSE training validation."""

    target = np.asarray(target_indices, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    if target.shape != (len(values),) or len(target) == 0:
        raise ValueError("event targets and probability rows must be non-empty and aligned")
    if np.any(target < 0) or np.any(target >= len(EVENT_NAMES)):
        raise ValueError("event target outside the frozen five-event vocabulary")
    if values.shape != (len(target), len(EVENT_NAMES)):
        raise ValueError("event probability shape mismatch")
    predicted = values.argmax(axis=1)
    matrix = np.zeros((len(EVENT_NAMES), len(EVENT_NAMES)), dtype=np.int64)
    np.add.at(matrix, (target, predicted), 1)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(EVENT_NAMES):
        true_positive = int(matrix[index, index])
        predicted_count = int(matrix[:, index].sum())
        support = int(matrix[index, :].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(float(f1))
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return {
        "macro_f1": float(np.mean(f1_values)),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


__all__ = [
    "M1D_ROLE_NAMES",
    "M1D_TO_GSE_EVENT",
    "five_event_scores",
    "m1d_role_logits_to_gse",
    "m1d_role_probabilities_to_gse",
]
