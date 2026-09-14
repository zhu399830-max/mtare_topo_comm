"""Validation-only calibration of one frozen GSE checkpoint output archive."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mtare_topo.evaluation.gse_exit_token_metrics import evaluate_exit_token_sets
from mtare_topo.evaluation.gse_validation_calibration import (
    causal_nearest_descriptor_records,
    fit_event_temperature,
    precision_constrained_threshold,
    select_event_rejection_threshold,
)
from mtare_topo.representation.gse_graph import (
    CURVATURE_SCALE_PER_M,
    METRIC_DISTANCE_SCALE_M,
    SLOPE_SCALE_DEG,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


REQUIRED_ARRAYS = (
    "event_logits",
    "uncertainty",
    "width_m",
    "height_m",
    "slope_deg",
    "curvature_per_m",
    "place_descriptor",
    "exit_confidence",
    "exit_heading_unit",
    "exit_opening_width_m",
    "exit_vertical_profile",
    "exit_descriptor",
    "target_event_index",
    "target_width_m",
    "target_height_m",
    "target_slope_deg",
    "target_curvature_per_m",
    "target_geometry_valid_mask",
    "target_association_identity",
    "target_association_valid_mask",
    "target_exit_mask",
    "target_exit_heading_unit",
    "target_exit_opening_width_m",
    "target_exit_width_valid_mask",
    "target_exit_vertical_profile",
    "target_exit_identity",
    "global_sequence_index",
    "parent_id",
    "observation_id",
)


def _finite_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("Pearson inputs must be aligned non-trivial vectors")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Pearson inputs must be finite")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def geometry_uncertainty_diagnostic(
    arrays: Mapping[str, np.ndarray],
    *,
    requested_bins: int = 10,
) -> dict[str, Any]:
    """Audit the learned geometry residual scale without changing calibration.

    The model optimizes ``0.5 * (residual / variance + log(variance))`` where
    variance is the squared scalar uncertainty.  This diagnostic therefore
    compares that exact predicted variance with the same masked, normalized
    per-sample Smooth-L1 residual used by training.  Event error is reported
    alongside it because deployment also uses the scalar as an observation
    rejection signal, but neither curve participates in threshold selection.
    """

    if requested_bins < 2:
        raise ValueError("requested_bins must be at least two")
    scales = np.asarray(
        (
            METRIC_DISTANCE_SCALE_M,
            METRIC_DISTANCE_SCALE_M,
            SLOPE_SCALE_DEG,
            CURVATURE_SCALE_PER_M,
        ),
        dtype=np.float64,
    )
    predicted = np.stack(
        tuple(
            np.asarray(arrays[name], dtype=np.float64)
            for name in ("width_m", "height_m", "slope_deg", "curvature_per_m")
        ),
        axis=-1,
    ) / scales
    target = np.stack(
        tuple(
            np.asarray(arrays[name], dtype=np.float64)
            for name in (
                "target_width_m",
                "target_height_m",
                "target_slope_deg",
                "target_curvature_per_m",
            )
        ),
        axis=-1,
    ) / scales
    valid = np.asarray(arrays["target_geometry_valid_mask"], dtype=np.bool_)
    uncertainty = np.asarray(arrays["uncertainty"], dtype=np.float64)
    event_logits = np.asarray(arrays["event_logits"], dtype=np.float64)
    event_target = np.asarray(arrays["target_event_index"], dtype=np.int64)
    frames = len(uncertainty)
    if (
        predicted.shape != (frames, 4)
        or target.shape != (frames, 4)
        or valid.shape != (frames, 4)
        or event_logits.shape[0] != frames
        or event_target.shape != (frames,)
        or frames < 2
        or not valid.any(axis=1).all()
        or not np.all(np.isfinite(predicted[valid]))
        or not np.all(np.isfinite(target[valid]))
        or not np.all(np.isfinite(uncertainty))
        or np.any((uncertainty < 0.0) | (uncertainty > 1.0))
    ):
        raise ValueError("geometry uncertainty diagnostic arrays are invalid or misaligned")
    absolute = np.abs(predicted - target)
    smooth_l1 = np.where(absolute < 1.0, 0.5 * np.square(absolute), absolute - 0.5)
    residual = (smooth_l1 * valid).sum(axis=1) / valid.sum(axis=1)
    predicted_variance = np.maximum(np.square(uncertainty), 1e-4)
    event_error = (np.argmax(event_logits, axis=1) != event_target).astype(np.float64)

    bin_count = min(int(requested_bins), frames)
    order = np.argsort(uncertainty, kind="stable")
    curve = []
    weighted_gap = 0.0
    for bin_index, indices in enumerate(np.array_split(order, bin_count)):
        variance_mean = float(predicted_variance[indices].mean())
        residual_mean = float(residual[indices].mean())
        gap = abs(variance_mean - residual_mean)
        weighted_gap += gap * len(indices)
        curve.append(
            {
                "bin_index": bin_index,
                "count": int(len(indices)),
                "uncertainty_min": float(uncertainty[indices].min()),
                "uncertainty_mean": float(uncertainty[indices].mean()),
                "uncertainty_max": float(uncertainty[indices].max()),
                "predicted_variance_mean": variance_mean,
                "geometry_residual_mean": residual_mean,
                "absolute_calibration_gap": gap,
                "event_error_rate": float(event_error[indices].mean()),
            }
        )
    return {
        "definition": "diagnostic_only_predicted_u_squared_vs_masked_normalized_smooth_l1_geometry_residual",
        "selection_effect": "NONE",
        "frames": frames,
        "bins": bin_count,
        "predicted_variance_mean": float(predicted_variance.mean()),
        "geometry_residual_mean": float(residual.mean()),
        "weighted_absolute_calibration_gap": float(weighted_gap / frames),
        "geometry_residual_pearson": _finite_pearson(uncertainty, residual),
        "event_error_pearson": _finite_pearson(uncertainty, event_error),
        "lowest_uncertainty_bin_event_error_rate": curve[0]["event_error_rate"],
        "highest_uncertainty_bin_event_error_rate": curve[-1]["event_error_rate"],
        "curve": curve,
    }


def selected_event_class_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    *,
    temperature: float,
    threshold: float,
    corridor_index: int = 0,
) -> dict[str, Any]:
    """Replay the frozen rejection point and retain all five event classes."""

    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    risk = np.asarray(uncertainty, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(EVENT_NAMES)
        or truth.shape != (len(values),)
        or risk.shape != truth.shape
        or len(values) == 0
        or not np.isfinite(temperature)
        or temperature <= 0.0
        or not np.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(risk))
        or np.any((risk < 0.0) | (risk > 1.0))
        or np.any((truth < 0) | (truth >= len(EVENT_NAMES)))
        or not 0 <= corridor_index < len(EVENT_NAMES)
    ):
        raise ValueError("selected event class metric arrays violate the frozen contract")
    shifted = values / float(temperature)
    shifted -= shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(values)), predicted]
    reliability = np.minimum(confidence, 1.0 - risk)
    structural = predicted != corridor_index
    predicted[structural & (reliability < float(threshold))] = corridor_index
    confusion = np.zeros((len(EVENT_NAMES), len(EVENT_NAMES)), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    per_class = {}
    f1_values = []
    for class_index, name in enumerate(EVENT_NAMES):
        true_positive = int(confusion[class_index, class_index])
        support = int(confusion[class_index].sum())
        predicted_count = int(confusion[:, class_index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "predicted": predicted_count,
        }
    return {
        "selection_effect": "NONE_REPLAY_OF_FROZEN_EVENT_POINT",
        "temperature": float(temperature),
        "threshold": float(threshold),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix_truth_rows_prediction_columns": confusion.tolist(),
    }


def per_parent_perception_diagnostic(
    arrays: Mapping[str, np.ndarray],
    *,
    temperature: float,
    threshold: float,
) -> list[dict[str, Any]]:
    """Replay one frozen event point and geometry metrics for every parent.

    This is deliberately diagnostic-only: the globally selected temperature
    and rejection threshold are replayed unchanged in every C09 parent.  The
    result cannot select a checkpoint, threshold, parent, or geometry field.
    """

    parent_ids = np.asarray(arrays["parent_id"]).astype(str)
    frames = len(parent_ids)
    predicted_geometry = np.stack(
        tuple(
            np.asarray(arrays[name], dtype=np.float64)
            for name in ("width_m", "height_m", "slope_deg", "curvature_per_m")
        ),
        axis=-1,
    )
    target_geometry = np.stack(
        tuple(
            np.asarray(arrays[name], dtype=np.float64)
            for name in (
                "target_width_m",
                "target_height_m",
                "target_slope_deg",
                "target_curvature_per_m",
            )
        ),
        axis=-1,
    )
    valid = np.asarray(arrays["target_geometry_valid_mask"], dtype=np.bool_)
    if (
        parent_ids.shape != (frames,)
        or predicted_geometry.shape != (frames, 4)
        or target_geometry.shape != (frames, 4)
        or valid.shape != (frames, 4)
        or frames == 0
        or any(not value for value in parent_ids)
        or not np.all(np.isfinite(predicted_geometry[valid]))
        or not np.all(np.isfinite(target_geometry[valid]))
    ):
        raise ValueError("per-parent perception arrays are invalid or misaligned")
    names = ("width_m", "height_m", "slope_deg", "curvature_per_m")
    records = []
    for parent_id in sorted(np.unique(parent_ids).tolist()):
        selected = parent_ids == parent_id
        parent_valid = valid[selected]
        counts = parent_valid.sum(axis=0)
        if not selected.any() or np.any(counts == 0):
            raise ValueError(f"per-parent geometry metric is empty: {parent_id}")
        absolute = np.abs(predicted_geometry[selected] - target_geometry[selected])
        event = selected_event_class_metrics(
            np.asarray(arrays["event_logits"])[selected],
            np.asarray(arrays["target_event_index"])[selected],
            np.asarray(arrays["uncertainty"])[selected],
            temperature=temperature,
            threshold=threshold,
        )
        records.append(
            {
                "parent_id": parent_id,
                "frames": int(selected.sum()),
                "event": event,
                "geometry_mae": {
                    name: float((absolute[:, index] * parent_valid[:, index]).sum() / counts[index])
                    for index, name in enumerate(names)
                },
                "geometry_valid_count": {
                    name: int(counts[index]) for index, name in enumerate(names)
                },
            }
        )
    if sum(record["frames"] for record in records) != frames:
        raise RuntimeError("per-parent diagnostics do not cover the validation population")
    return records


def calibrate_validation_outputs(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"validation output archive is missing arrays: {missing}")
    frames = len(np.asarray(arrays["event_logits"]))
    if frames == 0 or any(len(np.asarray(arrays[name])) != frames for name in REQUIRED_ARRAYS):
        raise ValueError("validation output archive arrays are empty or misaligned")
    sequence_order = np.asarray(arrays["global_sequence_index"], dtype=np.int64)
    if len(np.unique(sequence_order)) != frames:
        raise ValueError("validation sequence indices are not unique")
    parent_ids = np.asarray(arrays["parent_id"]).astype(str)
    observation_ids = np.asarray(arrays["observation_id"]).astype(str)
    if len(np.unique(observation_ids)) != frames:
        raise ValueError("validation observation IDs are not unique")

    event_logits = np.asarray(arrays["event_logits"], dtype=np.float64)
    event_labels = np.asarray(arrays["target_event_index"], dtype=np.int64)
    temperature = fit_event_temperature(event_logits, event_labels)
    event_selection, event_curve = select_event_rejection_threshold(
        event_logits,
        event_labels,
        np.asarray(arrays["uncertainty"], dtype=np.float64),
        temperature=float(temperature["temperature"]),
    )
    event_class_metrics = selected_event_class_metrics(
        event_logits,
        event_labels,
        np.asarray(arrays["uncertainty"], dtype=np.float64),
        temperature=float(temperature["temperature"]),
        threshold=float(event_selection.threshold),
    )
    if abs(event_class_metrics["macro_f1"] - event_selection.macro_f1) > 1e-12:
        raise RuntimeError("selected event class replay does not reproduce the frozen macro-F1")
    per_parent_diagnostic = per_parent_perception_diagnostic(
        arrays,
        temperature=float(temperature["temperature"]),
        threshold=float(event_selection.threshold),
    )
    uncertainty_diagnostic = geometry_uncertainty_diagnostic(arrays)

    association_identity = np.asarray(arrays["target_association_identity"], dtype=np.int64)
    association_valid = np.asarray(arrays["target_association_valid_mask"], dtype=np.bool_)
    if association_identity.shape != (frames,) or association_valid.shape != (frames,):
        raise ValueError("place-association arrays are misaligned")
    labels = np.where(association_valid, association_identity, -1)
    place_records = causal_nearest_descriptor_records(
        np.asarray(arrays["place_descriptor"], dtype=np.float64),
        labels,
        parent_ids,
        sequence_order,
    )
    place_selection, place_curve = precision_constrained_threshold(
        [float(record["score"]) for record in place_records],
        [bool(record["correct"]) for record in place_records],
        [bool(record["eligible_positive"]) for record in place_records],
    )

    exit_result = evaluate_exit_token_sets(
        confidence=np.asarray(arrays["exit_confidence"]),
        heading_unit=np.asarray(arrays["exit_heading_unit"]),
        opening_width_m=np.asarray(arrays["exit_opening_width_m"]),
        vertical_profile=np.asarray(arrays["exit_vertical_profile"]),
        descriptor=np.asarray(arrays["exit_descriptor"]),
        target_mask=np.asarray(arrays["target_exit_mask"]),
        target_heading_unit=np.asarray(arrays["target_exit_heading_unit"]),
        target_opening_width_m=np.asarray(arrays["target_exit_opening_width_m"]),
        target_width_valid_mask=np.asarray(arrays["target_exit_width_valid_mask"]),
        target_vertical_profile=np.asarray(arrays["target_exit_vertical_profile"]),
        target_identity=np.asarray(arrays["target_exit_identity"]),
        parent_ids=parent_ids,
        sequence_order=sequence_order,
    )
    exit_records = exit_result.pop("descriptor_records")
    exit_selection, exit_curve = precision_constrained_threshold(
        [float(record["score"]) for record in exit_records],
        [bool(record["correct"]) for record in exit_records],
        [bool(record["eligible_positive"]) for record in exit_records],
    )
    return {
        "schema_version": "gse_validation_calibration_seed_v1",
        "validation_frames": frames,
        "validation_parents": int(len(np.unique(parent_ids))),
        "event": {
            "temperature": temperature,
            "rejection_selection": event_selection.to_dict(),
            "rejection_curve": event_curve,
            "selected_class_metrics": event_class_metrics,
        },
        "uncertainty_diagnostic": uncertainty_diagnostic,
        "per_parent_diagnostic": {
            "selection_effect": "NONE_REPLAY_OF_GLOBAL_FROZEN_EVENT_POINT_AND_ALL_GEOMETRY_FIELDS",
            "parents": per_parent_diagnostic,
        },
        "place_association": {
            "selection": place_selection.to_dict(),
            "curve": place_curve,
            "causal_decisions": place_records,
        },
        "exit_tokens": {
            **exit_result,
            "descriptor_association": {
                "selection": exit_selection.to_dict(),
                "curve": exit_curve,
                "causal_decisions": exit_records,
            },
        },
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }


__all__ = [
    "REQUIRED_ARRAYS",
    "calibrate_validation_outputs",
    "geometry_uncertainty_diagnostic",
    "per_parent_perception_diagnostic",
    "selected_event_class_metrics",
]
