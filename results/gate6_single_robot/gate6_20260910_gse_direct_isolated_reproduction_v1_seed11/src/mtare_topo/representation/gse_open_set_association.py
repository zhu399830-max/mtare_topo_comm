"""Open-set association verifier for frozen GSE observations.

The verifier is deliberately downstream of the frozen perception network.  It
learns whether an observation is a structural place at all and whether two
past/current observations denote the same place.  Pair features are exactly
symmetric so replay cannot depend on query/candidate ordering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


EVENT_DIM = 5
AXIS_DIM = 3
GEOMETRY_DIM = 4
PLACE_DESCRIPTOR_DIM = 128
EXIT_SUMMARY_DIM = 5
OBSERVATION_FEATURE_DIM = (
    EVENT_DIM + AXIS_DIM + GEOMETRY_DIM + PLACE_DESCRIPTOR_DIM + 1 + EXIT_SUMMARY_DIM
)
PAIR_FEATURE_DIM = 2 * OBSERVATION_FEATURE_DIM + 1


@dataclass(frozen=True)
class OpenSetAssociationContract:
    """Frozen training and selection contract for the corrective verifier."""

    maximum_candidate_distance_m: float = 16.0
    minimum_precision: float = 0.98
    maximum_false_accept_rate: float = 0.01
    minimum_recall: float = 0.25
    minimum_family_precision: float = 0.95
    minimum_family_recall: float = 0.10

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_candidate_distance_m) or self.maximum_candidate_distance_m <= 0:
            raise ValueError("maximum_candidate_distance_m must be positive and finite")
        for name in ("minimum_precision", "maximum_false_accept_rate", "minimum_recall",
                     "minimum_family_precision", "minimum_family_recall"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _finite(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def observation_features(outputs: Mapping[str, Any]) -> np.ndarray:
    """Build deployment-available features from frozen GSE batch outputs.

    No identity, pose, parent, traversal, frame index, or teacher value is
    accepted.  Exit queries are summarized without selecting a threshold.
    """

    logits = np.asarray(outputs["event_logits"], dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != EVENT_DIM or not np.all(np.isfinite(logits)):
        raise ValueError("event_logits must be a finite [N,5] matrix")
    rows = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    axis = _finite("local_axis", outputs["local_axis"], (rows, AXIS_DIM))
    geometry = np.stack(
        tuple(np.asarray(outputs[name], dtype=np.float32) for name in
              ("width_m", "height_m", "slope_deg", "curvature_per_m")), axis=1
    )
    if geometry.shape != (rows, GEOMETRY_DIM) or not np.all(np.isfinite(geometry)):
        raise ValueError("continuous geometry outputs are invalid")
    geometry = geometry / np.asarray((30.0, 30.0, 45.0, 0.1), dtype=np.float32)
    descriptor = _finite(
        "place_descriptor", outputs["place_descriptor"], (rows, PLACE_DESCRIPTOR_DIM)
    )
    uncertainty = _finite("uncertainty", outputs["uncertainty"], (rows,)).reshape(rows, 1)
    confidence = np.asarray(outputs["exit_confidence"], dtype=np.float32)
    heading = np.asarray(outputs["exit_heading_unit"], dtype=np.float32)
    width = np.asarray(outputs["exit_opening_width_m"], dtype=np.float32)
    vertical = np.asarray(outputs["exit_vertical_profile"], dtype=np.float32)
    if (
        confidence.ndim != 2
        or heading.shape != (*confidence.shape, 2)
        or width.shape != confidence.shape
        or vertical.shape[:2] != confidence.shape
        or vertical.ndim != 3
        or not all(np.all(np.isfinite(x)) for x in (confidence, heading, width, vertical))
        or np.any((confidence < 0.0) | (confidence > 1.0))
    ):
        raise ValueError("exit-query outputs are invalid or misaligned")
    weight_sum = np.maximum(confidence.sum(axis=1), 1e-6)
    circular_resultant = np.linalg.norm(
        (heading * confidence[..., None]).sum(axis=1), axis=1
    ) / weight_sum
    exit_summary = np.stack(
        (
            confidence.mean(axis=1),
            confidence.max(axis=1),
            weight_sum / confidence.shape[1],
            circular_resultant,
            (np.abs(vertical).mean(axis=2) * confidence).sum(axis=1) / weight_sum / 10.0,
        ),
        axis=1,
    ).astype(np.float32)
    features = np.concatenate(
        (probabilities, axis, geometry, descriptor, uncertainty, exit_summary), axis=1
    ).astype(np.float32)
    if features.shape != (rows, OBSERVATION_FEATURE_DIM) or not np.all(np.isfinite(features)):
        raise RuntimeError("constructed observation features violate the frozen contract")
    return features


def symmetric_pair_features(
    left: np.ndarray,
    right: np.ndarray,
    distance_m: np.ndarray,
    *,
    maximum_candidate_distance_m: float = 16.0,
) -> np.ndarray:
    """Return order-invariant pair features: absolute delta, product, distance."""

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    distance = np.asarray(distance_m, dtype=np.float32).reshape(-1, 1)
    if (
        left.ndim != 2
        or left.shape != right.shape
        or left.shape[1] != OBSERVATION_FEATURE_DIM
        or distance.shape != (len(left), 1)
        or not all(np.all(np.isfinite(x)) for x in (left, right, distance))
        or np.any(distance < 0.0)
        or not math.isfinite(maximum_candidate_distance_m)
        or maximum_candidate_distance_m <= 0.0
    ):
        raise ValueError("pair feature inputs violate the frozen shape/value contract")
    result = np.concatenate(
        (np.abs(left - right), left * right, distance / maximum_candidate_distance_m), axis=1
    ).astype(np.float32)
    if result.shape != (len(left), PAIR_FEATURE_DIM):
        raise RuntimeError("pair feature dimension drift")
    return result


class GSEOpenSetAssociationVerifier(nn.Module):
    """Small matchability and symmetric same-place verifier."""

    def __init__(self) -> None:
        super().__init__()
        self.matchability = nn.Sequential(
            nn.Linear(OBSERVATION_FEATURE_DIM, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.pair = nn.Sequential(
            nn.Linear(PAIR_FEATURE_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            left.ndim != 2
            or left.shape != right.shape
            or left.shape[1] != OBSERVATION_FEATURE_DIM
            or pair_features.shape != (len(left), PAIR_FEATURE_DIM)
        ):
            raise ValueError("verifier tensor shapes violate the frozen contract")
        left_logit = self.matchability(left).squeeze(1)
        right_logit = self.matchability(right).squeeze(1)
        pair_logit = self.pair(pair_features).squeeze(1)
        score = torch.sigmoid(pair_logit) * torch.minimum(
            torch.sigmoid(left_logit), torch.sigmoid(right_logit)
        )
        return {
            "left_matchability_logit": left_logit,
            "right_matchability_logit": right_logit,
            "pair_logit": pair_logit,
            "association_score": score,
        }


def select_nonvacuous_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    contract: OpenSetAssociationContract,
) -> dict[str, Any]:
    """Select the lowest admissible observed score, maximizing recall.

    Every topology family must have positive support and a non-empty accepted
    set.  This makes an all-reject or almost-all-reject solution impossible.
    """

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    families = np.asarray(families).astype(str)
    if (
        scores.ndim != 1
        or labels.shape != scores.shape
        or families.shape != scores.shape
        or len(scores) == 0
        or not np.all(np.isfinite(scores))
        or np.any((scores < 0.0) | (scores > 1.0))
    ):
        raise ValueError("threshold selection arrays are invalid or misaligned")
    unique_families = tuple(sorted(set(families.tolist())))
    if any(not np.any(labels[families == family]) for family in unique_families):
        raise ValueError("every topology family must contain a positive pair")
    feasible: list[dict[str, Any]] = []
    positives = int(labels.sum())
    family_index = {family: index for index, family in enumerate(unique_families)}
    encoded_family = np.asarray([family_index[value] for value in families], dtype=np.int64)
    family_positive = np.bincount(
        encoded_family[labels], minlength=len(unique_families)
    ).astype(np.int64)
    family_true_positive = np.zeros(len(unique_families), dtype=np.int64)
    family_false_positive = np.zeros(len(unique_families), dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    true_positive = 0
    false_positive = 0
    start = 0
    while start < len(order):
        stop = start + 1
        threshold = scores[order[start]]
        while stop < len(order) and scores[order[stop]] == threshold:
            stop += 1
        group = order[start:stop]
        group_labels = labels[group]
        true_positive += int(np.sum(group_labels))
        false_positive += int(len(group) - np.sum(group_labels))
        family_true_positive += np.bincount(
            encoded_family[group[group_labels]], minlength=len(unique_families)
        )
        family_false_positive += np.bincount(
            encoded_family[group[~group_labels]], minlength=len(unique_families)
        )
        total = true_positive + false_positive
        precision = true_positive / total
        recall = true_positive / positives
        false_rate = false_positive / total
        per_family = {}
        family_ok = True
        for index, family in enumerate(unique_families):
            tp = int(family_true_positive[index])
            fp = int(family_false_positive[index])
            actual = int(family_positive[index])
            count = tp + fp
            family_precision = tp / count if count else 0.0
            family_recall = tp / actual
            per_family[family] = {
                "accepted": count,
                "precision": family_precision,
                "recall": family_recall,
            }
            family_ok &= (
                count > 0
                and family_precision >= contract.minimum_family_precision
                and family_recall >= contract.minimum_family_recall
            )
        if (
            total > 0
            and precision >= contract.minimum_precision
            and false_rate <= contract.maximum_false_accept_rate
            and recall >= contract.minimum_recall
            and family_ok
        ):
            feasible.append({
                "threshold": float(threshold),
                "accepted": total,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "precision": precision,
                "recall": recall,
                "false_accept_rate": false_rate,
                "per_family": per_family,
            })
        start = stop
    if not feasible:
        raise RuntimeError("no non-vacuous open-set association threshold satisfies the contract")
    return max(feasible, key=lambda row: (row["recall"], row["precision"], -row["threshold"]))


def select_online_candidate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    distance_m: np.ndarray,
    contract: OpenSetAssociationContract,
) -> dict[str, Any]:
    """Select a threshold only over pairs eligible for online association."""

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    families = np.asarray(families)
    distance = np.asarray(distance_m, dtype=np.float64)
    if (
        scores.ndim != 1
        or labels.shape != scores.shape
        or families.shape != scores.shape
        or distance.shape != scores.shape
        or not np.all(np.isfinite(distance))
        or np.any(distance < 0.0)
    ):
        raise ValueError("online candidate arrays are invalid or misaligned")
    eligible = distance <= contract.maximum_candidate_distance_m + 1e-12
    if not np.any(eligible):
        raise RuntimeError("online candidate domain is empty")
    selected = select_nonvacuous_threshold(
        scores[eligible], labels[eligible], families[eligible], contract
    )
    selected["candidate_domain"] = {
        "maximum_distance_m": contract.maximum_candidate_distance_m,
        "eligible_pairs": int(np.sum(eligible)),
        "excluded_pairs": int(np.sum(~eligible)),
        "eligible_positive_pairs": int(np.sum(np.asarray(labels, dtype=np.bool_)[eligible])),
        "eligible_negative_pairs": int(np.sum(~np.asarray(labels, dtype=np.bool_)[eligible])),
    }
    return selected


__all__ = [
    "GSEOpenSetAssociationVerifier",
    "OBSERVATION_FEATURE_DIM",
    "OpenSetAssociationContract",
    "PAIR_FEATURE_DIM",
    "observation_features",
    "select_online_candidate_threshold",
    "select_nonvacuous_threshold",
    "symmetric_pair_features",
]
