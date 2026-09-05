"""Identity-balanced residual decoder for rare GSE structural events."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from mtare_topo.representation.gse_open_set_association import (
    OpenSetAssociationContract,
    select_nonvacuous_threshold,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


FROZEN_FEATURE_DIM = 146
CORRECTIVE_INPUT_DIM = 2 * FROZEN_FEATURE_DIM
EVENT_COUNT = len(EVENT_NAMES)
BASELINE_EVENT_MACRO_F1 = 0.6819938211624408


class GSERareEventCorrective(nn.Module):
    """Predict a residual over the frozen three-seed mean event logits."""

    def __init__(self) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(CORRECTIVE_INPUT_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, EVENT_COUNT),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor, baseline_probability: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != CORRECTIVE_INPUT_DIM:
            raise ValueError("corrective features must be [N,292]")
        if baseline_probability.shape != (len(features), EVENT_COUNT):
            raise ValueError("baseline event probability must be [N,5]")
        if torch.any(~torch.isfinite(features)) or torch.any(~torch.isfinite(baseline_probability)):
            raise ValueError("corrective inputs must be finite")
        if torch.any(baseline_probability <= 0.0):
            raise ValueError("baseline probabilities must be strictly positive")
        return baseline_probability.log() + self.residual(features)


def identity_class_balanced_weights(event: np.ndarray, identity: np.ndarray) -> np.ndarray:
    """Give every event class and every structural identity equal expected mass."""

    event = np.asarray(event, dtype=np.int64)
    identity = np.asarray(identity, dtype=np.int64)
    if event.ndim != 1 or identity.shape != event.shape or len(event) == 0:
        raise ValueError("event/identity arrays must be aligned non-empty vectors")
    if np.any((event < 0) | (event >= EVENT_COUNT)):
        raise ValueError("event labels violate the five-class contract")
    if np.any((event == 0) != (identity < 0)):
        raise ValueError("corridor/structural identity contract drift")
    weights = np.empty(len(event), dtype=np.float64)
    class_mass = 1.0 / EVENT_COUNT
    for class_index in range(EVENT_COUNT):
        rows = np.where(event == class_index)[0]
        if len(rows) == 0:
            raise ValueError("every event class requires fit support")
        if class_index == 0:
            weights[rows] = class_mass / len(rows)
            continue
        counts = Counter(identity[rows].tolist())
        identity_mass = class_mass / len(counts)
        weights[rows] = np.asarray(
            [identity_mass / counts[int(identity[row])] for row in rows], dtype=np.float64
        )
    weights /= weights.sum()
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RuntimeError("identity-balanced weights are invalid")
    return weights


def _event_metrics(probability: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    if probability.shape != (len(truth), EVENT_COUNT):
        raise ValueError("event evaluation arrays are misaligned")
    predicted = np.argmax(probability, axis=1)
    matrix = np.zeros((EVENT_COUNT, EVENT_COUNT), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    per_class = {}
    f1_values = []
    for index, name in enumerate(EVENT_NAMES):
        tp = int(matrix[index, index])
        predicted_count = int(matrix[:, index].sum())
        actual = int(matrix[index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
        }
        f1_values.append(f1)
    return {
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion": matrix.tolist(),
        "predicted": predicted,
    }


def evaluate_rare_event_corrective(
    probability: np.ndarray,
    truth: np.ndarray,
    identity: np.ndarray,
    family: np.ndarray,
    *,
    contract: OpenSetAssociationContract | None = None,
) -> dict[str, Any]:
    """Evaluate event classes, open-set rejection and identity coverage."""

    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    identity = np.asarray(identity, dtype=np.int64)
    family = np.asarray(family).astype(str)
    if identity.shape != truth.shape or family.shape != truth.shape:
        raise ValueError("corrective labels/families are misaligned")
    event = _event_metrics(probability, truth)
    structural_score = 1.0 - probability[:, 0]
    structural_label = identity >= 0
    selection = None
    selection_error = None
    try:
        selection = select_nonvacuous_threshold(
            structural_score,
            structural_label,
            family,
            OpenSetAssociationContract() if contract is None else contract,
        )
    except RuntimeError as exc:
        selection_error = str(exc)
    coverage = {}
    if selection is not None:
        accepted = structural_score >= float(selection["threshold"])
        predicted = np.asarray(event.pop("predicted"), dtype=np.int64)
        for class_index, name in enumerate(EVENT_NAMES[1:], start=1):
            class_rows = (truth == class_index) & (identity >= 0)
            teacher = set(identity[class_rows].tolist())
            covered = set(identity[class_rows & accepted & (predicted == class_index)].tolist())
            coverage[name] = {
                "teacher_identities": len(teacher),
                "covered_identities": len(covered),
                "correct_class_identity_coverage": len(covered) / len(teacher) if teacher else 0.0,
            }
    else:
        event.pop("predicted", None)
    return {
        "event": event,
        "structural_selection": selection,
        "structural_selection_error": selection_error,
        "identity_coverage": coverage,
    }


def corrective_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    selection = metrics.get("structural_selection") or {}
    coverage = metrics.get("identity_coverage") or {}
    event_macro = float(metrics.get("event", {}).get("macro_f1", 0.0))
    requirements = {
        "event_macro_f1": event_macro >= BASELINE_EVENT_MACRO_F1 + 0.05,
        "structural_precision": float(selection.get("precision", 0.0)) >= 0.98,
        "structural_false_accept": float(selection.get("false_accept_rate", 1.0)) <= 0.01,
        "structural_recall": float(selection.get("recall", 0.0)) >= 0.40,
        "junction_identity_coverage": float(coverage.get("junction", {}).get("correct_class_identity_coverage", 0.0)) >= 0.90,
        "terminal_identity_coverage": float(coverage.get("terminal", {}).get("correct_class_identity_coverage", 0.0)) >= 0.90,
        "turn_identity_coverage": float(coverage.get("turn", {}).get("correct_class_identity_coverage", 0.0)) >= 0.40,
        "geometry_transition_identity_coverage": float(coverage.get("geometry_transition", {}).get("correct_class_identity_coverage", 0.0)) >= 0.40,
    }
    return {"passed": all(requirements.values()), "requirements": requirements}


__all__ = [
    "BASELINE_EVENT_MACRO_F1",
    "CORRECTIVE_INPUT_DIM",
    "GSERareEventCorrective",
    "corrective_gate",
    "evaluate_rare_event_corrective",
    "identity_class_balanced_weights",
]
