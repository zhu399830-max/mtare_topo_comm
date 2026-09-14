"""Sensor-observable correction for sealed spatial structure-event sets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


VERTICAL_FOV_MIN_DEG = -15.0
VERTICAL_FOV_MAX_DEG = 15.0
MAXIMUM_EVENT_TOKENS = 16
FOV_NUMERICAL_EPSILON_DEG = 1e-5


@dataclass(frozen=True)
class ObservableEventSet:
    event_type_index: np.ndarray
    event_relative_xyz_m: np.ndarray
    event_distance_m: np.ndarray
    event_identity_index: np.ndarray
    event_mask: np.ndarray
    set_cardinality: np.ndarray
    retained_original_slot: np.ndarray
    removed_mask: np.ndarray
    elevation_deg: np.ndarray


def event_elevation_deg(relative_xyz_m: np.ndarray) -> np.ndarray:
    relative = np.asarray(relative_xyz_m, dtype=np.float64)
    if relative.ndim != 3 or relative.shape[1:] != (MAXIMUM_EVENT_TOKENS, 3):
        raise ValueError("spatial event xyz must have shape [N,16,3]")
    if not np.all(np.isfinite(relative)):
        raise ValueError("spatial event xyz must be finite")
    horizontal = np.linalg.norm(relative[..., :2], axis=-1)
    return np.degrees(np.arctan2(relative[..., 2], horizontal))


def observable_vertical_fov_mask(
    relative_xyz_m: np.ndarray,
    event_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(event_mask, dtype=bool)
    elevation = event_elevation_deg(relative_xyz_m)
    if mask.shape != elevation.shape:
        raise ValueError("spatial event mask shape drift")
    observable = mask & (
        elevation >= VERTICAL_FOV_MIN_DEG - FOV_NUMERICAL_EPSILON_DEG
    ) & (
        elevation <= VERTICAL_FOV_MAX_DEG + FOV_NUMERICAL_EPSILON_DEG
    )
    return observable, elevation


def repack_observable_event_set(arrays: Mapping[str, np.ndarray]) -> ObservableEventSet:
    required = {
        "event_type_index",
        "event_relative_xyz_m",
        "event_distance_m",
        "event_identity_index",
        "event_mask",
        "set_cardinality",
    }
    if required - set(arrays):
        raise ValueError(f"missing spatial event arrays: {sorted(required - set(arrays))}")
    event_type = np.asarray(arrays["event_type_index"], dtype=np.int8)
    relative = np.asarray(arrays["event_relative_xyz_m"], dtype=np.float32)
    distance = np.asarray(arrays["event_distance_m"], dtype=np.float32)
    identity = np.asarray(arrays["event_identity_index"], dtype=np.int32)
    mask = np.asarray(arrays["event_mask"], dtype=np.uint8)
    cardinality = np.asarray(arrays["set_cardinality"], dtype=np.uint8)
    row_count = len(cardinality)
    if (
        event_type.shape != (row_count, MAXIMUM_EVENT_TOKENS)
        or relative.shape != (row_count, MAXIMUM_EVENT_TOKENS, 3)
        or distance.shape != event_type.shape
        or identity.shape != event_type.shape
        or mask.shape != event_type.shape
        or not np.array_equal(mask.sum(axis=1), cardinality)
        or not np.all(np.isfinite(distance))
    ):
        raise ValueError("sealed spatial event array contract drift")
    active = mask.astype(bool)
    if (
        np.any(event_type[active] < 0)
        or np.any(event_type[active] > 1)
        or np.any(identity[active] < 0)
        or np.any(event_type[~active] != -1)
        or np.any(identity[~active] != -1)
        or np.any(distance[active] < 0.0)
    ):
        raise ValueError("sealed spatial event padding/value drift")

    observable, elevation = observable_vertical_fov_mask(relative, mask)
    removed = active & ~observable
    output_type = np.full_like(event_type, -1)
    output_relative = np.zeros_like(relative)
    output_distance = np.zeros_like(distance)
    output_identity = np.full_like(identity, -1)
    output_mask = np.zeros_like(mask)
    retained_original_slot = np.full_like(identity, -1)
    output_cardinality = observable.sum(axis=1).astype(np.uint8)
    for row in range(row_count):
        slots = np.flatnonzero(observable[row])
        count = len(slots)
        if count == 0:
            continue
        target = slice(0, count)
        output_type[row, target] = event_type[row, slots]
        output_relative[row, target] = relative[row, slots]
        output_distance[row, target] = distance[row, slots]
        output_identity[row, target] = identity[row, slots]
        output_mask[row, target] = 1
        retained_original_slot[row, target] = slots
    if not np.array_equal(output_mask.sum(axis=1), output_cardinality):
        raise RuntimeError("observable event repack cardinality drift")
    return ObservableEventSet(
        event_type_index=output_type,
        event_relative_xyz_m=output_relative,
        event_distance_m=output_distance,
        event_identity_index=output_identity,
        event_mask=output_mask,
        set_cardinality=output_cardinality,
        retained_original_slot=retained_original_slot,
        removed_mask=removed,
        elevation_deg=elevation.astype(np.float32),
    )


def observable_teacher_contract() -> dict[str, object]:
    return {
        "vertical_fov_deg_inclusive": [VERTICAL_FOV_MIN_DEG, VERTICAL_FOV_MAX_DEG],
        "filter_inputs": ["event_relative_xyz_m", "event_mask"],
        "forbidden_filter_inputs": [
            "model_prediction",
            "model_error",
            "checkpoint",
            "confidence",
            "C09",
            "C10",
            "M-TARE",
        ],
        "row_policy": "retain every observation; repack retained tokens in original order",
        "identity_policy": "retain original teacher-only identity indices",
    }


__all__ = [
    "MAXIMUM_EVENT_TOKENS",
    "FOV_NUMERICAL_EPSILON_DEG",
    "ObservableEventSet",
    "VERTICAL_FOV_MAX_DEG",
    "VERTICAL_FOV_MIN_DEG",
    "event_elevation_deg",
    "observable_teacher_contract",
    "observable_vertical_fov_mask",
    "repack_observable_event_set",
]
