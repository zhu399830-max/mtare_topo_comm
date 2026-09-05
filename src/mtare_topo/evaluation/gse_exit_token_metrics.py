"""Permutation-invariant validation metrics for predicted GSE exit tokens."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Sequence

import numpy as np

from mtare_topo.evaluation.gse_validation_calibration import (
    binary_f1_threshold,
    causal_nearest_descriptor_records,
)
from mtare_topo.evaluation.phase3_semantic_metrics import match_headings


def _assignment(cost: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < values.shape[1] or values.shape[0] > 6:
        raise ValueError("exit assignment requires at most six predictions and predictions >= targets")

    @lru_cache(maxsize=None)
    def solve(column: int, used: int) -> tuple[float, tuple[int, ...]]:
        if column == values.shape[1]:
            return 0.0, ()
        best: tuple[float, tuple[int, ...]] | None = None
        for row in range(values.shape[0]):
            if used & (1 << row):
                continue
            tail_cost, tail = solve(column + 1, used | (1 << row))
            candidate = (float(values[row, column]) + tail_cost, (row,) + tail)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("no deterministic exit-token assignment")
        return best

    return solve(0, 0)[1]


def evaluate_exit_token_sets(
    *,
    confidence: np.ndarray,
    heading_unit: np.ndarray,
    opening_width_m: np.ndarray,
    vertical_profile: np.ndarray,
    descriptor: np.ndarray,
    target_mask: np.ndarray,
    target_heading_unit: np.ndarray,
    target_opening_width_m: np.ndarray,
    target_width_valid_mask: np.ndarray,
    target_vertical_profile: np.ndarray,
    target_identity: np.ndarray,
    parent_ids: Sequence[str],
    sequence_order: Sequence[int],
) -> dict[str, Any]:
    """Match unordered tokens and return metrics plus calibration records.

    Matching reproduces the frozen training assignment (heading plus valid
    opening-width cost). GT identities are only attached after matching for
    offline evaluation and are never an input to the matcher.
    """

    confidence = np.asarray(confidence, dtype=np.float64)
    heading_unit = np.asarray(heading_unit, dtype=np.float64)
    opening_width_m = np.asarray(opening_width_m, dtype=np.float64)
    vertical_profile = np.asarray(vertical_profile, dtype=np.float64)
    descriptor = np.asarray(descriptor, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=np.bool_)
    target_heading_unit = np.asarray(target_heading_unit, dtype=np.float64)
    target_opening_width_m = np.asarray(target_opening_width_m, dtype=np.float64)
    target_width_valid_mask = np.asarray(target_width_valid_mask, dtype=np.bool_)
    target_vertical_profile = np.asarray(target_vertical_profile, dtype=np.float64)
    target_identity = np.asarray(target_identity, dtype=np.int64)
    parent_ids = np.asarray(parent_ids)
    sequence_order = np.asarray(sequence_order, dtype=np.int64)
    if confidence.ndim != 2 or confidence.shape[0] == 0:
        raise ValueError("exit confidence must be [N,Q]")
    samples, queries = confidence.shape
    if queries > 6:
        raise ValueError("exit token contract permits at most six queries")
    expected = (samples, queries)
    shapes = {
        "heading_unit": heading_unit.shape[:2],
        "opening_width_m": opening_width_m.shape,
        "vertical_profile": vertical_profile.shape[:2],
        "descriptor": descriptor.shape[:2],
        "target_mask": target_mask.shape,
        "target_heading_unit": target_heading_unit.shape[:2],
        "target_opening_width_m": target_opening_width_m.shape,
        "target_width_valid_mask": target_width_valid_mask.shape,
        "target_vertical_profile": target_vertical_profile.shape[:2],
        "target_identity": target_identity.shape,
    }
    if any(shape != expected for shape in shapes.values()):
        raise ValueError(f"exit token arrays are misaligned: {shapes}")
    if heading_unit.shape[-1] != 2 or target_heading_unit.shape[-1] != 2:
        raise ValueError("exit heading units must have final dimension two")
    if vertical_profile.ndim != 3 or target_vertical_profile.shape != vertical_profile.shape:
        raise ValueError("predicted and target vertical profiles must align")
    if descriptor.ndim != 3 or descriptor.shape[2] == 0:
        raise ValueError("exit descriptors must be [N,Q,D]")
    if parent_ids.shape != (samples,) or sequence_order.shape != (samples,):
        raise ValueError("exit token sample metadata is misaligned")
    numeric = (
        confidence,
        heading_unit,
        opening_width_m,
        vertical_profile,
        descriptor,
        target_heading_unit,
        target_opening_width_m,
        target_vertical_profile,
    )
    if not all(np.all(np.isfinite(array)) for array in numeric):
        raise ValueError("exit token arrays must be finite")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0) or np.any(opening_width_m <= 0.0):
        raise ValueError("exit confidences or predicted widths are invalid")
    if np.any(target_width_valid_mask & ~target_mask) or np.any(target_identity[~target_mask] != -1):
        raise ValueError("exit target masks and identities disagree")
    if np.any(target_mask.sum(axis=1) > queries):
        raise ValueError("target exit count exceeds query count")

    heading_unit = heading_unit / np.maximum(np.linalg.norm(heading_unit, axis=2, keepdims=True), 1e-12)
    target_heading_unit = target_heading_unit / np.maximum(
        np.linalg.norm(target_heading_unit, axis=2, keepdims=True), 1e-12
    )
    descriptor = descriptor / np.maximum(np.linalg.norm(descriptor, axis=2, keepdims=True), 1e-12)
    matched_prediction = np.zeros(expected, dtype=np.bool_)
    matched_identity = np.full(expected, -1, dtype=np.int64)
    heading_errors: list[float] = []
    width_errors: list[float] = []
    width_log_errors: list[float] = []
    profile_errors: list[float] = []
    matched_target_count = 0
    for sample in range(samples):
        targets = np.flatnonzero(target_mask[sample])
        if len(targets) == 0:
            continue
        cosine = np.clip(heading_unit[sample] @ target_heading_unit[sample, targets].T, -1.0, 1.0)
        cost = 1.0 - cosine
        width_log = np.abs(
            np.log(
                opening_width_m[sample, :, None]
                / np.maximum(target_opening_width_m[sample, targets][None, :], 1e-12)
            )
        )
        cost = cost + width_log * target_width_valid_mask[sample, targets][None, :]
        predictions = _assignment(cost)
        for compact_target, prediction in enumerate(predictions):
            target = int(targets[compact_target])
            matched_prediction[sample, prediction] = True
            matched_identity[sample, prediction] = int(target_identity[sample, target])
            heading_errors.append(math.degrees(math.acos(float(cosine[prediction, compact_target]))))
            if target_width_valid_mask[sample, target]:
                width_errors.append(
                    abs(float(opening_width_m[sample, prediction] - target_opening_width_m[sample, target]))
                )
                width_log_errors.append(
                    abs(
                        math.log(
                            float(opening_width_m[sample, prediction])
                            / float(target_opening_width_m[sample, target])
                        )
                    )
                )
            profile_errors.append(
                float(
                    np.mean(
                        np.abs(
                            vertical_profile[sample, prediction]
                            - target_vertical_profile[sample, target]
                        )
                    )
                )
            )
            matched_target_count += 1
    if matched_target_count != int(target_mask.sum()) or not heading_errors or not profile_errors:
        raise RuntimeError("exit-token assignment did not cover every visible target")

    presence, presence_curve = binary_f1_threshold(confidence.ravel(), matched_prediction.ravel())
    predicted_mask = confidence >= float(presence["threshold"])
    predicted_count = predicted_mask.sum(axis=1)
    target_count = target_mask.sum(axis=1)
    direction_matched = 0
    direction_predicted = 0
    direction_target = 0
    direction_errors: list[float] = []
    predicted_heading_deg = np.degrees(np.arctan2(heading_unit[..., 0], heading_unit[..., 1])) % 360.0
    target_heading_deg = np.degrees(
        np.arctan2(target_heading_unit[..., 0], target_heading_unit[..., 1])
    ) % 360.0
    for sample in range(samples):
        matched, predicted, target, errors = match_headings(
            predicted_heading_deg[sample, predicted_mask[sample]].tolist(),
            target_heading_deg[sample, target_mask[sample]].tolist(),
        )
        direction_matched += int(matched)
        direction_predicted += int(predicted)
        direction_target += int(target)
        direction_errors.extend(float(error) for error in errors)
    direction_precision = direction_matched / direction_predicted if direction_predicted else 0.0
    direction_recall = direction_matched / direction_target if direction_target else 0.0
    direction_f1 = (
        2.0 * direction_precision * direction_recall / (direction_precision + direction_recall)
        if direction_precision + direction_recall
        else 0.0
    )
    valid_descriptor = matched_identity >= 0
    flat_descriptors = descriptor[valid_descriptor]
    flat_identities = matched_identity[valid_descriptor]
    flat_parents = np.repeat(parent_ids[:, None], queries, axis=1)[valid_descriptor]
    flat_order = np.repeat(sequence_order[:, None], queries, axis=1)[valid_descriptor]
    descriptor_records = causal_nearest_descriptor_records(
        flat_descriptors,
        flat_identities,
        flat_parents,
        flat_order,
    )
    return {
        "target_tokens": int(target_mask.sum()),
        "matched_tokens": matched_target_count,
        "heading_error_deg": {
            "mean": float(np.mean(heading_errors)),
            "p95": float(np.quantile(heading_errors, 0.95)),
        },
        "opening_width_error_m": {
            "mean": float(np.mean(width_errors)),
            "p95": float(np.quantile(width_errors, 0.95)),
        } if width_errors else None,
        "opening_width_log_error": {
            "mean": float(np.mean(width_log_errors)),
            "p95": float(np.quantile(width_log_errors, 0.95)),
        } if width_log_errors else None,
        "opening_width_valid_tokens": len(width_errors),
        "vertical_profile_error_m": {
            "mean": float(np.mean(profile_errors)),
            "p95": float(np.quantile(profile_errors, 0.95)),
        },
        "presence_threshold": presence,
        "presence_curve": presence_curve,
        "count_at_selected_threshold": {
            "exact_accuracy": float(np.mean(predicted_count == target_count)),
            "mean_absolute_error": float(np.mean(np.abs(predicted_count - target_count))),
        },
        "direction_at_selected_threshold": {
            "matching_tolerance_deg": 20.0,
            "matched": direction_matched,
            "predicted": direction_predicted,
            "target": direction_target,
            "precision": float(direction_precision),
            "recall": float(direction_recall),
            "f1": float(direction_f1),
            "mean_matched_angular_error_deg": (
                float(np.mean(direction_errors)) if direction_errors else None
            ),
        },
        "descriptor_records": descriptor_records,
    }


__all__ = ["evaluate_exit_token_sets"]
