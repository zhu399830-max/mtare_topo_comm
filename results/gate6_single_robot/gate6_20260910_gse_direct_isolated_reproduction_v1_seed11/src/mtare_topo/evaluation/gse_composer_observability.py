"""Train-free observability primitives for the factorized GSE composers.

The functions in this module deliberately accept only deployed, interpretable
outputs.  Encoder context, event logits, descriptors, identities and pose are
not part of this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from mtare_topo.evaluation.gse_causal_geometry_delta import binary_roc_auc
from mtare_topo.evaluation.gse_token_validity_feasibility import (
    binary_ranking_metrics,
)


HISTORY_FRAMES = 5
MAX_TOKENS = 6
COUNT_CLASSES = MAX_TOKENS + 1
GEOMETRY_DIM = 4
PROFILE_DIM = 4


@dataclass(frozen=True)
class ExplicitComposerFeatures:
    """Interpretable per-observation features with no hidden-state bypass."""

    expected_count: np.ndarray
    circular_first_moment: np.ndarray
    circular_second_moment: np.ndarray
    opening_width_mean_m: np.ndarray
    vertical_profile_mean_m: np.ndarray
    geometry_uncertainty_mean: np.ndarray
    persistent_mass: np.ndarray
    reveal_mass: np.ndarray
    withdraw_mass: np.ndarray
    geometry_sequence: np.ndarray
    geometry_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        count = np.asarray(self.expected_count)
        if count.ndim != 2 or count.shape[1] != HISTORY_FRAMES:
            raise ValueError("expected_count must have shape [N,5]")
        rows = len(count)
        required = {
            "circular_first_moment": (rows, HISTORY_FRAMES, 2),
            "circular_second_moment": (rows, HISTORY_FRAMES, 2),
            "opening_width_mean_m": (rows, HISTORY_FRAMES),
            "vertical_profile_mean_m": (rows, HISTORY_FRAMES, PROFILE_DIM),
            "geometry_uncertainty_mean": (rows, HISTORY_FRAMES, 1 + PROFILE_DIM),
            "persistent_mass": (rows, HISTORY_FRAMES - 1),
            "reveal_mass": (rows, HISTORY_FRAMES - 1),
            "withdraw_mass": (rows, HISTORY_FRAMES - 1),
            "geometry_sequence": (rows, HISTORY_FRAMES, GEOMETRY_DIM),
            "geometry_valid_mask": (rows, HISTORY_FRAMES),
        }
        for name, shape in required.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if name != "geometry_valid_mask" and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class BinaryTransferMetrics:
    """A fit-only threshold and its untouched transfer-set behavior."""

    fit_roc_auc: float
    fit_average_precision: float
    fit_threshold: float | None
    fit_precision: float | None
    fit_recall: float
    transfer_roc_auc: float
    transfer_average_precision: float
    transfer_precision: float
    transfer_recall: float
    transfer_false_positive_rate: float
    transfer_identity_covered: int
    transfer_identity_total: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def binary_transfer_metrics(
    fit_label: np.ndarray,
    fit_score: np.ndarray,
    transfer_label: np.ndarray,
    transfer_score: np.ndarray,
    *,
    precision_floor: float,
    transfer_identity: Sequence[str | None] | None = None,
) -> BinaryTransferMetrics:
    """Select a realizable threshold on fit rows and apply it unchanged.

    The threshold is selected before the transfer labels are inspected.  Tied
    scores are kept as one operating point by ``binary_ranking_metrics``.
    Identities are used only after scoring to report whether at least one row
    of each positive structural event was recovered.
    """

    fit_truth = np.asarray(fit_label, dtype=np.bool_).reshape(-1)
    fit_value = np.asarray(fit_score, dtype=np.float64).reshape(-1)
    transfer_truth = np.asarray(transfer_label, dtype=np.bool_).reshape(-1)
    transfer_value = np.asarray(transfer_score, dtype=np.float64).reshape(-1)
    if (
        fit_truth.shape != fit_value.shape
        or transfer_truth.shape != transfer_value.shape
        or len(fit_truth) == 0
        or len(transfer_truth) == 0
        or not np.all(np.isfinite(fit_value))
        or not np.all(np.isfinite(transfer_value))
    ):
        raise ValueError("binary transfer rows are invalid")
    if transfer_identity is None:
        identity = np.full(len(transfer_truth), None, dtype=object)
    else:
        identity = np.asarray(tuple(transfer_identity), dtype=object)
        if identity.shape != transfer_truth.shape:
            raise ValueError("transfer identities are not row-aligned")

    fit = binary_ranking_metrics(
        fit_truth, fit_value, precision_floor=precision_floor
    )
    transfer = binary_ranking_metrics(
        transfer_truth, transfer_value, precision_floor=precision_floor
    )
    threshold = fit["threshold_at_precision_floor"]
    accepted = (
        np.zeros(len(transfer_truth), dtype=bool)
        if threshold is None
        else transfer_value >= float(threshold)
    )
    true_positive = int(np.sum(accepted & transfer_truth))
    false_positive = int(np.sum(accepted & ~transfer_truth))
    positive = int(np.sum(transfer_truth))
    negative = len(transfer_truth) - positive
    selected = true_positive + false_positive
    positive_identity = {
        str(value)
        for value in identity[transfer_truth]
        if value is not None and str(value)
    }
    covered_identity = {
        str(value)
        for value in identity[accepted & transfer_truth]
        if value is not None and str(value)
    }
    return BinaryTransferMetrics(
        fit_roc_auc=binary_roc_auc(fit_value, fit_truth),
        fit_average_precision=float(fit["average_precision"]),
        fit_threshold=None if threshold is None else float(threshold),
        fit_precision=(
            None
            if fit["precision_at_selected_threshold"] is None
            else float(fit["precision_at_selected_threshold"])
        ),
        fit_recall=float(fit["recall_at_precision_floor"]),
        transfer_roc_auc=binary_roc_auc(transfer_value, transfer_truth),
        transfer_average_precision=float(transfer["average_precision"]),
        transfer_precision=float(true_positive / selected) if selected else 0.0,
        transfer_recall=float(true_positive / positive),
        transfer_false_positive_rate=float(false_positive / negative),
        transfer_identity_covered=len(covered_identity),
        transfer_identity_total=len(positive_identity),
    )


def token_validity_weights(count_probability: np.ndarray) -> np.ndarray:
    """Return differentiable rank validity ``P(count > rank)`` for six tokens."""

    probability = np.asarray(count_probability, dtype=np.float64)
    if probability.ndim < 1 or probability.shape[-1] != COUNT_CLASSES:
        raise ValueError("count probability must end in seven classes")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("count probability must be finite and nonnegative")
    total = probability.sum(axis=-1)
    if not np.allclose(total, 1.0, rtol=0.0, atol=2e-3):
        raise ValueError("count probability rows must sum to one")
    normalized = probability / total[..., None]
    return np.stack(
        [normalized[..., rank + 1 :].sum(axis=-1) for rank in range(MAX_TOKENS)],
        axis=-1,
    )


def weighted_circular_moments(
    bearing_deg: np.ndarray, validity_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize a weighted token set without imposing token order."""

    bearing = np.asarray(bearing_deg, dtype=np.float64)
    weight = np.asarray(validity_weight, dtype=np.float64)
    if bearing.shape != weight.shape or bearing.shape[-1] != MAX_TOKENS:
        raise ValueError("bearing and token validity must be aligned [...,6]")
    if not np.all(np.isfinite(bearing)) or not np.all(np.isfinite(weight)):
        raise ValueError("circular inputs must be finite")
    if np.any(weight < 0.0) or np.any(weight > 1.0 + 1e-8):
        raise ValueError("token validity lies outside [0,1]")
    angle = np.deg2rad(np.remainder(bearing, 360.0))
    denominator = np.maximum(weight.sum(axis=-1, keepdims=True), 1e-12)
    first = np.stack(
        ((weight * np.cos(angle)).sum(axis=-1), (weight * np.sin(angle)).sum(axis=-1)),
        axis=-1,
    ) / denominator
    second = np.stack(
        (
            (weight * np.cos(2.0 * angle)).sum(axis=-1),
            (weight * np.sin(2.0 * angle)).sum(axis=-1),
        ),
        axis=-1,
    ) / denominator
    return first, second


def causal_row_history(
    traversal_id: Sequence[str],
    sequence_index: Sequence[int],
    current_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct five causal observation rows without crossing traversals."""

    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    value = np.asarray(current_value)
    if traversal.ndim != 1 or sequence.shape != traversal.shape or len(value) != len(traversal):
        raise ValueError("causal history inputs are not row-aligned")
    if value.ndim < 1 or np.any(sequence < 0):
        raise ValueError("causal history values or indices are invalid")
    lookup: dict[tuple[str, int], int] = {}
    for row, key in enumerate(zip(traversal.tolist(), sequence.tolist(), strict=True)):
        normalized = (str(key[0]), int(key[1]))
        if normalized in lookup:
            raise ValueError("duplicate traversal sequence index")
        lookup[normalized] = row
    output = np.zeros((len(value), HISTORY_FRAMES, *value.shape[1:]), dtype=value.dtype)
    valid = np.zeros((len(value), HISTORY_FRAMES), dtype=bool)
    for row, (name, index) in enumerate(zip(traversal, sequence, strict=True)):
        for history_slot, offset in enumerate(range(HISTORY_FRAMES - 1, -1, -1)):
            source = lookup.get((str(name), int(index) - offset))
            if source is not None:
                output[row, history_slot] = value[source]
                valid[row, history_slot] = True
    return output, valid


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if value.shape[:-1] != weight.shape:
        raise ValueError("weighted token value is not aligned")
    denominator = np.maximum(weight.sum(axis=-1, keepdims=True), 1e-12)
    return (value * weight[..., None]).sum(axis=-2) / denominator


def explicit_composer_features(
    prediction: Mapping[str, np.ndarray],
    *,
    geometry_sequence: np.ndarray,
    geometry_valid_mask: np.ndarray,
) -> ExplicitComposerFeatures:
    """Build the only feature bundle allowed to enter the two composers."""

    required = {
        "token_count_probability",
        "token_bearing_deg",
        "token_opening_width_m",
        "token_vertical_profile_m",
        "token_geometry_uncertainty",
        "transport_row_probability",
        "transport_reveal_probability",
    }
    missing = required - set(prediction)
    if missing:
        raise ValueError(f"missing explicit prediction arrays: {sorted(missing)}")
    count_probability = np.asarray(prediction["token_count_probability"], dtype=np.float64)
    if count_probability.ndim != 3 or count_probability.shape[1:] != (HISTORY_FRAMES, COUNT_CLASSES):
        raise ValueError("token count probability must have shape [N,5,7]")
    rows = len(count_probability)
    token_shape = (rows, HISTORY_FRAMES, MAX_TOKENS)
    bearing = np.asarray(prediction["token_bearing_deg"], dtype=np.float64)
    width = np.asarray(prediction["token_opening_width_m"], dtype=np.float64)
    profile = np.asarray(prediction["token_vertical_profile_m"], dtype=np.float64)
    uncertainty = np.asarray(prediction["token_geometry_uncertainty"], dtype=np.float64)
    row_probability = np.asarray(prediction["transport_row_probability"], dtype=np.float64)
    reveal_probability = np.asarray(prediction["transport_reveal_probability"], dtype=np.float64)
    if bearing.shape != token_shape or width.shape != token_shape:
        raise ValueError("token bearing/width shape drift")
    if profile.shape != (*token_shape, PROFILE_DIM):
        raise ValueError("token vertical profile shape drift")
    if uncertainty.shape != (*token_shape, 1 + PROFILE_DIM):
        raise ValueError("token uncertainty shape drift")
    if row_probability.shape != (rows, HISTORY_FRAMES - 1, MAX_TOKENS, MAX_TOKENS + 1):
        raise ValueError("transport row probability shape drift")
    if reveal_probability.shape != (rows, HISTORY_FRAMES - 1, MAX_TOKENS):
        raise ValueError("transport reveal probability shape drift")
    arrays = (bearing, width, profile, uncertainty, row_probability, reveal_probability)
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("explicit prediction arrays must be finite")
    if np.any(width <= 0.0) or np.any(uncertainty <= 0.0):
        raise ValueError("token width and uncertainty must be positive")
    row_total = row_probability.sum(axis=-1)
    if np.any(row_probability < 0.0) or not np.allclose(row_total, 1.0, rtol=0.0, atol=2e-3):
        raise ValueError("transport rows must be probabilities")
    if np.any(reveal_probability < 0.0) or np.any(reveal_probability > 1.0 + 1e-8):
        raise ValueError("reveal probabilities lie outside [0,1]")

    validity = token_validity_weights(count_probability)
    first, second = weighted_circular_moments(bearing, validity)
    expected_count = validity.sum(axis=-1)
    width_mean = _weighted_mean(width[..., None], validity)[..., 0]
    profile_mean = _weighted_mean(profile, validity)
    uncertainty_mean = _weighted_mean(uncertainty, validity)

    previous_validity = validity[:, :-1]
    current_validity = validity[:, 1:]
    match_weight = previous_validity[..., :, None] * current_validity[..., None, :]
    persistent = (row_probability[..., :MAX_TOKENS] * match_weight).sum(axis=(-2, -1))
    withdraw = (row_probability[..., MAX_TOKENS] * previous_validity).sum(axis=-1)
    reveal = (reveal_probability * current_validity).sum(axis=-1)

    geometry = np.asarray(geometry_sequence, dtype=np.float64)
    geometry_valid = np.asarray(geometry_valid_mask, dtype=bool)
    if geometry.shape != (rows, HISTORY_FRAMES, GEOMETRY_DIM):
        raise ValueError("geometry sequence must have shape [N,5,4]")
    if geometry_valid.shape != (rows, HISTORY_FRAMES):
        raise ValueError("geometry validity must have shape [N,5]")
    if not np.all(np.isfinite(geometry)):
        raise ValueError("geometry sequence must be finite")

    return ExplicitComposerFeatures(
        expected_count=expected_count,
        circular_first_moment=first,
        circular_second_moment=second,
        opening_width_mean_m=width_mean,
        vertical_profile_mean_m=profile_mean,
        geometry_uncertainty_mean=uncertainty_mean,
        persistent_mass=persistent,
        reveal_mass=reveal,
        withdraw_mass=withdraw,
        geometry_sequence=geometry,
        geometry_valid_mask=geometry_valid,
    )


def train_free_event_scores(features: ExplicitComposerFeatures) -> dict[str, np.ndarray]:
    """Return threshold-free diagnostic scores, not a deployable classifier."""

    count = np.asarray(features.expected_count, dtype=np.float64)
    geometry = np.asarray(features.geometry_sequence, dtype=np.float64)
    valid = np.asarray(features.geometry_valid_mask, dtype=bool)
    first = np.asarray(features.circular_first_moment, dtype=np.float64)
    current_count = count[:, -1]
    # Junctions should rank above ordinary two-exit corridors; terminals should
    # be closest to one executable exit.  These are diagnostics, not commits.
    junction = current_count
    terminal = -np.abs(current_count - 1.0)

    oldest_valid = valid[:, 0] & valid[:, -1]
    scales = np.asarray((30.0, 30.0, 45.0, 0.1), dtype=np.float64)
    delta = (geometry[:, -1] - geometry[:, 0]) / scales
    opening_delta = (
        np.asarray(features.opening_width_mean_m)[:, -1]
        - np.asarray(features.opening_width_mean_m)[:, 0]
    ) / 30.0
    profile_delta = (
        np.asarray(features.vertical_profile_mean_m)[:, -1]
        - np.asarray(features.vertical_profile_mean_m)[:, 0]
    ) / 5.0
    transition = np.sqrt(
        np.square(delta).sum(axis=-1)
        + np.square(opening_delta)
        + np.square(profile_delta).sum(axis=-1)
    )
    transition[~oldest_valid] = 0.0

    current_curvature = np.abs(geometry[:, -1, 3]) / 0.1
    dot = np.clip((first[:, 0] * first[:, -1]).sum(axis=-1), -1.0, 1.0)
    cross = first[:, 0, 0] * first[:, -1, 1] - first[:, 0, 1] * first[:, -1, 0]
    bearing_drift = np.abs(np.arctan2(cross, dot)) / np.pi
    turn = current_curvature + bearing_drift
    turn[~oldest_valid] = 0.0
    return {
        "junction": junction,
        "terminal": terminal,
        "turn": turn,
        "geometry_transition": transition,
        "history_complete": oldest_valid,
    }


def _gather_token_axis(value: np.ndarray, order: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[:3] != order.shape:
        raise ValueError("token array and canonical order are not aligned")
    index = order[(...,) + (None,) * (array.ndim - order.ndim)]
    return np.take_along_axis(array, index, axis=2)


def canonical_explicit_feature_matrix(
    prediction: Mapping[str, np.ndarray],
    *,
    geometry_sequence: np.ndarray,
    geometry_valid_mask: np.ndarray,
) -> np.ndarray:
    """Flatten the complete explicit state into a deterministic 642-D probe.

    Tokens are ordered by route-frame bearing separately in each frame.  The
    previous/current axes of every transport matrix are reordered with the
    corresponding token sets.  Extra arrays in ``prediction`` are ignored, so
    hidden context, event logits and descriptors cannot influence the result.
    """

    features = explicit_composer_features(
        prediction,
        geometry_sequence=geometry_sequence,
        geometry_valid_mask=geometry_valid_mask,
    )
    count_probability = np.asarray(prediction["token_count_probability"], dtype=np.float64)
    bearing = np.asarray(prediction["token_bearing_deg"], dtype=np.float64)
    validity = token_validity_weights(count_probability)
    order = np.argsort(np.remainder(bearing, 360.0), axis=2, kind="stable")
    canonical_bearing = _gather_token_axis(bearing, order)
    canonical_validity = _gather_token_axis(validity, order)
    canonical_width = _gather_token_axis(
        np.asarray(prediction["token_opening_width_m"], dtype=np.float64), order
    )
    canonical_profile = _gather_token_axis(
        np.asarray(prediction["token_vertical_profile_m"], dtype=np.float64), order
    )
    canonical_uncertainty = _gather_token_axis(
        np.asarray(prediction["token_geometry_uncertainty"], dtype=np.float64), order
    )
    angle = np.deg2rad(np.remainder(canonical_bearing, 360.0))
    token = np.concatenate(
        (
            canonical_validity[..., None],
            np.sin(angle)[..., None],
            np.cos(angle)[..., None],
            canonical_width[..., None] / 30.0,
            canonical_profile / 5.0,
            np.log(canonical_uncertainty),
        ),
        axis=-1,
    )

    row_probability = np.asarray(prediction["transport_row_probability"], dtype=np.float64)
    reveal_probability = np.asarray(prediction["transport_reveal_probability"], dtype=np.float64)
    previous_order = order[:, :-1]
    current_order = order[:, 1:]
    row_index = previous_order[..., None]
    row_sorted = np.take_along_axis(row_probability, row_index, axis=2)
    match_sorted = np.take_along_axis(
        row_sorted[..., :MAX_TOKENS], current_order[:, :, None, :], axis=3
    )
    transport = np.concatenate((match_sorted, row_sorted[..., MAX_TOKENS:]), axis=3)
    reveal = np.take_along_axis(reveal_probability, current_order, axis=2)

    geometry = np.asarray(features.geometry_sequence, dtype=np.float64).copy()
    geometry_valid = np.asarray(features.geometry_valid_mask, dtype=bool)
    geometry /= np.asarray((30.0, 30.0, 45.0, 0.1), dtype=np.float64)
    geometry[~geometry_valid] = 0.0
    rows = len(geometry)
    matrix = np.concatenate(
        (
            count_probability.reshape(rows, -1),
            token.reshape(rows, -1),
            transport.reshape(rows, -1),
            reveal.reshape(rows, -1),
            geometry.reshape(rows, -1),
            geometry_valid.astype(np.float64),
        ),
        axis=1,
    )
    if matrix.shape != (rows, 642) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("canonical explicit feature matrix contract drift")
    return matrix


__all__ = [
    "BinaryTransferMetrics",
    "ExplicitComposerFeatures",
    "binary_transfer_metrics",
    "canonical_explicit_feature_matrix",
    "causal_row_history",
    "explicit_composer_features",
    "token_validity_weights",
    "train_free_event_scores",
    "weighted_circular_moments",
]
