"""Shared metrics for learned and non-learning primitive-relation predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch

from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveRelationLossTargets,
    align_primitive_relation_targets,
    match_primitives,
    sample_swept_superellipse_surface,
)
from mtare_topo.representation.primitive_relation_model import (
    MAXIMUM_SLOTS,
    PrimitiveRelationPrediction,
)
from mtare_topo.semantics.primitive_relation_nonlearning import (
    NonlearningPrimitiveRelationPrediction,
)


THRESHOLD_GRID = np.linspace(0.05, 0.95, 91, dtype=np.float64)
HARD_LOGIT = 10.0


@dataclass(frozen=True)
class BinaryCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 2 * self.true_positive / denominator if denominator else 1.0

    def __add__(self, other: "BinaryCounts") -> "BinaryCounts":
        return BinaryCounts(
            self.true_positive + other.true_positive,
            self.false_positive + other.false_positive,
            self.false_negative + other.false_negative,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def align_for_evaluation(
    prediction: PrimitiveRelationPrediction,
    targets: PrimitiveRelationLossTargets,
) -> dict[str, torch.Tensor]:
    assignments = match_primitives(prediction, targets)
    return align_primitive_relation_targets(targets, assignments)


def nonlearning_batch_to_torch(
    predictions: Iterable[NonlearningPrimitiveRelationPrediction],
    *,
    device: torch.device,
) -> PrimitiveRelationPrediction:
    values = tuple(predictions)
    if not values:
        raise ValueError("non-learning prediction batch is empty")

    def stacked(name: str) -> np.ndarray:
        return np.stack([np.asarray(getattr(value, name)) for value in values])

    mask = stacked("primitive_mask").astype(bool)
    attachment = stacked("endpoint_attachment").astype(bool)
    overlap = stacked("disconnected_overlap").astype(bool)
    temporal_presence = stacked("temporal_visibility").astype(bool)
    temporal_destination = stacked("temporal_destination").astype(np.int64)
    if np.any((temporal_destination < 0) | (temporal_destination > MAXIMUM_SLOTS)):
        raise ValueError("non-learning temporal destination outside slots+dustbin")
    temporal_logits = np.full(
        (len(values), 5, MAXIMUM_SLOTS, MAXIMUM_SLOTS + 1),
        -HARD_LOGIT,
        dtype=np.float32,
    )
    np.put_along_axis(
        temporal_logits,
        temporal_destination[..., None],
        np.float32(HARD_LOGIT),
        axis=-1,
    )

    def floating(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device=device)

    return PrimitiveRelationPrediction(
        existence_logits=floating(np.where(mask, HARD_LOGIT, -HARD_LOGIT)),
        axis_control_current_sensor_m=floating(stacked("axis_control_current_sensor_m")),
        endpoint_half_axes_m=floating(stacked("endpoint_half_axes_m")),
        endpoint_shape_exponent=floating(stacked("endpoint_shape_exponent")),
        endpoint_descriptor=torch.zeros(
            len(values), MAXIMUM_SLOTS, 2, 32, device=device, dtype=torch.float32,
        ),
        geometry_uncertainty=floating(stacked("geometry_uncertainty")),
        endpoint_attachment_logits=floating(
            np.where(attachment, HARD_LOGIT, -HARD_LOGIT),
        ),
        disconnected_overlap_logits=floating(
            np.where(overlap, HARD_LOGIT, -HARD_LOGIT),
        ),
        temporal_correspondence_logits=floating(temporal_logits),
        temporal_presence_logits=floating(
            np.where(temporal_presence, HARD_LOGIT, -HARD_LOGIT),
        ),
    )


def binary_counts(predicted: torch.Tensor, target: torch.Tensor) -> BinaryCounts:
    predicted = predicted.bool().reshape(-1)
    target = target.bool().reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("binary metric shape drift")
    return BinaryCounts(
        int((predicted & target).sum()),
        int((predicted & ~target).sum()),
        int((~predicted & target).sum()),
    )


def threshold_sweep_counts(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    eligible: torch.Tensor | None = None,
    thresholds: Iterable[float] = THRESHOLD_GRID,
) -> tuple[BinaryCounts, ...]:
    probability = probability.detach().reshape(-1).cpu().numpy()
    target = target.detach().bool().reshape(-1).cpu().numpy()
    if probability.shape != target.shape:
        raise ValueError("threshold sweep shape drift")
    if eligible is not None:
        eligible = eligible.detach().bool().reshape(-1).cpu().numpy()
        if eligible.shape != probability.shape:
            raise ValueError("threshold eligibility shape drift")
    counts: list[BinaryCounts] = []
    for threshold in thresholds:
        predicted = probability >= float(threshold)
        if eligible is not None:
            predicted &= eligible
        counts.append(BinaryCounts(
            int(np.count_nonzero(predicted & target)),
            int(np.count_nonzero(predicted & ~target)),
            int(np.count_nonzero(~predicted & target)),
        ))
    return tuple(counts)


def add_sweeps(
    first: tuple[BinaryCounts, ...] | None,
    second: tuple[BinaryCounts, ...],
) -> tuple[BinaryCounts, ...]:
    if first is None:
        return second
    if len(first) != len(second):
        raise ValueError("threshold sweep length drift")
    return tuple(left + right for left, right in zip(first, second, strict=True))


def select_threshold(
    counts: tuple[BinaryCounts, ...],
    *,
    thresholds: Iterable[float] = THRESHOLD_GRID,
    minimum_precision: float | None = None,
) -> dict[str, float | int | bool]:
    thresholds = tuple(float(value) for value in thresholds)
    if len(counts) != len(thresholds):
        raise ValueError("threshold/count grid drift")
    candidates = list(range(len(counts)))
    if minimum_precision is not None:
        candidates = [index for index in candidates if counts[index].precision >= minimum_precision]
    if not candidates:
        return {
            "available": False,
            "threshold": math.nan,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
    if minimum_precision is None:
        selected = max(
            candidates,
            key=lambda index: (counts[index].f1, counts[index].precision, thresholds[index]),
        )
    else:
        selected = max(
            candidates,
            key=lambda index: (counts[index].recall, counts[index].precision, thresholds[index]),
        )
    result = counts[selected].to_dict()
    result.update({"available": True, "threshold": thresholds[selected]})
    return result


def existence_sweep(
    prediction: PrimitiveRelationPrediction,
    aligned: dict[str, torch.Tensor],
) -> tuple[BinaryCounts, ...]:
    return threshold_sweep_counts(
        torch.sigmoid(prediction.existence_logits), aligned["mask"],
    )


def _attachment_upper_mask(device: torch.device) -> torch.Tensor:
    endpoint = torch.arange(2 * MAXIMUM_SLOTS, device=device)
    primitive = torch.div(endpoint, 2, rounding_mode="floor")
    return torch.triu(
        primitive[:, None] != primitive[None, :], diagonal=1,
    )


def _attachment_observability_validity(
    aligned: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Exclude hidden matched pairs while retaining unmatched negatives."""

    matched_endpoint = aligned["mask"].repeat_interleave(2, dim=1)
    observed_endpoint = aligned["endpoint_observed"].reshape(
        len(matched_endpoint), 2 * MAXIMUM_SLOTS,
    )
    both_matched = matched_endpoint[:, :, None] & matched_endpoint[:, None, :]
    both_observed = observed_endpoint[:, :, None] & observed_endpoint[:, None, :]
    return ~both_matched | both_observed


def relation_sweeps(
    prediction: PrimitiveRelationPrediction,
    aligned: dict[str, torch.Tensor],
    *,
    existence_threshold: float,
) -> dict[str, tuple[BinaryCounts, ...]]:
    active = torch.sigmoid(prediction.existence_logits) >= float(existence_threshold)
    endpoint_active = active.repeat_interleave(2, dim=1)
    attachment_eligible = (
        endpoint_active[:, :, None]
        & endpoint_active[:, None, :]
        & _attachment_upper_mask(active.device)[None]
        & _attachment_observability_validity(aligned)
    )
    attachment_target = aligned["attachment"].reshape(
        len(active), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
    ).bool() & _attachment_upper_mask(active.device)[None] & _attachment_observability_validity(aligned)
    primitive_upper = torch.triu(
        torch.ones(MAXIMUM_SLOTS, MAXIMUM_SLOTS, dtype=torch.bool, device=active.device),
        diagonal=1,
    )
    overlap_eligible = active[:, :, None] & active[:, None, :] & primitive_upper[None]
    overlap_target = aligned["overlap"].bool() & primitive_upper[None]
    return {
        "attachment": threshold_sweep_counts(
            torch.sigmoid(prediction.endpoint_attachment_logits),
            attachment_target,
            eligible=attachment_eligible,
        ),
        "disconnected_overlap": threshold_sweep_counts(
            torch.sigmoid(prediction.disconnected_overlap_logits),
            overlap_target,
            eligible=overlap_eligible,
        ),
    }


def relation_counts(
    prediction: PrimitiveRelationPrediction,
    aligned: dict[str, torch.Tensor],
    *,
    existence_threshold: float,
    attachment_threshold: float,
    overlap_threshold: float,
) -> dict[str, BinaryCounts]:
    active = torch.sigmoid(prediction.existence_logits) >= float(existence_threshold)
    endpoint_active = active.repeat_interleave(2, dim=1)
    attachment_upper = _attachment_upper_mask(active.device)
    attachment_validity = _attachment_observability_validity(aligned)
    attachment_eligible = (
        endpoint_active[:, :, None] & endpoint_active[:, None, :]
        & attachment_validity
    )
    attachment_prediction = (
        torch.sigmoid(prediction.endpoint_attachment_logits).reshape(
            len(active), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
        ) >= float(attachment_threshold)
    ) & attachment_eligible & attachment_upper[None]
    attachment_target = aligned["attachment"].reshape(
        len(active), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
    ).bool() & attachment_upper[None] & attachment_validity
    primitive_upper = torch.triu(
        torch.ones(MAXIMUM_SLOTS, MAXIMUM_SLOTS, dtype=torch.bool, device=active.device),
        diagonal=1,
    )
    overlap_prediction = (
        torch.sigmoid(prediction.disconnected_overlap_logits) >= float(overlap_threshold)
    ) & active[:, :, None] & active[:, None, :] & primitive_upper[None]
    overlap_target = aligned["overlap"].bool() & primitive_upper[None]
    return {
        "attachment": binary_counts(attachment_prediction, attachment_target),
        "disconnected_overlap": binary_counts(overlap_prediction, overlap_target),
    }


def temporal_batch_metrics(
    prediction: PrimitiveRelationPrediction,
    aligned: dict[str, torch.Tensor],
    *,
    existence_threshold: float,
    presence_threshold: float = 0.5,
) -> dict[str, BinaryCounts | int]:
    active = torch.sigmoid(prediction.existence_logits) >= float(existence_threshold)
    predicted_presence = (
        torch.sigmoid(prediction.temporal_presence_logits) >= float(presence_threshold)
    ) & active[:, None, :]
    presence = binary_counts(predicted_presence, aligned["temporal"])
    target_destination = torch.full(
        aligned["temporal"].shape,
        MAXIMUM_SLOTS,
        dtype=torch.long,
        device=active.device,
    )
    slot = torch.arange(MAXIMUM_SLOTS, device=active.device)[None, None]
    target_destination = torch.where(aligned["temporal"].bool(), slot, target_destination)
    predicted_destination = prediction.temporal_correspondence_logits.argmax(dim=-1)
    target_pair = aligned["mask"][:, None, :].expand_as(target_destination)
    correct = (
        (predicted_destination == target_destination)
        & active[:, None, :]
        & target_pair
    )
    return {
        "presence": presence,
        "correspondence_correct": int(correct.sum()),
        "correspondence_total": int(target_pair.sum()),
    }


def _slope_and_curvature(axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    chord = axis[..., 2, :] - axis[..., 0, :]
    slope = torch.rad2deg(torch.atan2(chord[..., 2], torch.linalg.vector_norm(chord[..., :2], dim=-1).clamp_min(1e-8)))
    first = axis[..., 1, :] - axis[..., 0, :]
    second = axis[..., 2, :] - axis[..., 1, :]
    first_length = torch.linalg.vector_norm(first, dim=-1).clamp_min(1e-8)
    second_length = torch.linalg.vector_norm(second, dim=-1).clamp_min(1e-8)
    cosine = (first * second).sum(dim=-1) / (first_length * second_length)
    angle = torch.acos(cosine.clamp(-1.0, 1.0))
    curvature = angle / (0.5 * (first_length + second_length))
    return slope, curvature


def geometry_batch_totals(
    prediction: PrimitiveRelationPrediction,
    aligned: dict[str, torch.Tensor],
    *,
    existence_threshold: float,
) -> dict[str, float | int]:
    active = torch.sigmoid(prediction.existence_logits) >= float(existence_threshold)
    matched = aligned["mask"]
    qualified = active & matched
    result: dict[str, float | int] = {
        "target_primitives": int(matched.sum()),
        "predicted_primitives": int(active.sum()),
        "matched_active_primitives": int(qualified.sum()),
    }
    if not bool(qualified.any()):
        for name in (
            "axis_control_absolute_error_m", "half_axis_absolute_error_m",
            "width_absolute_error_m", "height_absolute_error_m",
            "shape_exponent_absolute_error", "slope_absolute_error_deg",
            "curvature_absolute_error_per_m", "surface_chamfer_m",
        ):
            result[name + "_sum"] = 0.0
        return result
    predicted_axis = prediction.axis_control_current_sensor_m[qualified]
    target_axis = aligned["axis"][qualified]
    predicted_axes = prediction.endpoint_half_axes_m[qualified]
    target_axes = aligned["half_axes"][qualified]
    predicted_exponent = prediction.endpoint_shape_exponent[qualified]
    target_exponent = aligned["exponent"][qualified]
    count = int(qualified.sum())
    result["axis_control_absolute_error_m_sum"] = float(
        torch.abs(predicted_axis - target_axis).mean(dim=(-1, -2)).sum()
    )
    result["half_axis_absolute_error_m_sum"] = float(
        torch.abs(predicted_axes - target_axes).mean(dim=(-1, -2)).sum()
    )
    result["width_absolute_error_m_sum"] = float(
        torch.abs(2.0 * predicted_axes[..., 0] - 2.0 * target_axes[..., 0]).mean(dim=-1).sum()
    )
    result["height_absolute_error_m_sum"] = float(
        torch.abs(2.0 * predicted_axes[..., 1] - 2.0 * target_axes[..., 1]).mean(dim=-1).sum()
    )
    result["shape_exponent_absolute_error_sum"] = float(
        torch.abs(predicted_exponent - target_exponent).mean(dim=-1).sum()
    )
    predicted_slope, predicted_curvature = _slope_and_curvature(predicted_axis)
    target_slope, target_curvature = _slope_and_curvature(target_axis)
    result["slope_absolute_error_deg_sum"] = float(torch.abs(predicted_slope - target_slope).sum())
    result["curvature_absolute_error_per_m_sum"] = float(torch.abs(predicted_curvature - target_curvature).sum())
    predicted_surface = sample_swept_superellipse_surface(
        predicted_axis, predicted_axes, predicted_exponent,
    )
    target_surface = sample_swept_superellipse_surface(
        target_axis, target_axes.clamp_min(1e-4), target_exponent.clamp_min(2.0),
    )
    distance = torch.cdist(predicted_surface, target_surface)
    chamfer = 0.5 * (distance.amin(dim=-1).mean(dim=-1) + distance.amin(dim=-2).mean(dim=-1))
    result["surface_chamfer_m_sum"] = float(chamfer.sum())
    if count != result["matched_active_primitives"]:
        raise RuntimeError("geometry metric cardinality drift")
    return result


def merge_geometry_totals(
    first: dict[str, float | int] | None,
    second: dict[str, float | int],
) -> dict[str, float | int]:
    if first is None:
        return dict(second)
    if set(first) != set(second):
        raise ValueError("geometry total inventory drift")
    return {name: first[name] + second[name] for name in first}


def finalize_geometry_totals(totals: dict[str, float | int]) -> dict[str, float | int]:
    count = int(totals["matched_active_primitives"])
    result = dict(totals)
    result["target_coverage"] = count / int(totals["target_primitives"])
    result["prediction_precision"] = count / int(totals["predicted_primitives"]) if int(totals["predicted_primitives"]) else 0.0
    for name, value in tuple(totals.items()):
        if name.endswith("_sum"):
            result[name[:-4] + "_mean"] = float(value) / count if count else math.inf
    return result


__all__ = [
    "BinaryCounts", "HARD_LOGIT", "THRESHOLD_GRID", "add_sweeps",
    "align_for_evaluation",
    "binary_counts", "existence_sweep", "finalize_geometry_totals",
    "geometry_batch_totals", "merge_geometry_totals", "relation_sweeps",
    "nonlearning_batch_to_torch", "relation_counts", "select_threshold",
    "temporal_batch_metrics",
    "threshold_sweep_counts",
]
