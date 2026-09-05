"""Immutable Teacher targets for the O(E) composition-anchor head.

The student predicts one physical composition anchor for each endpoint of an
active primitive.  This module gathers the construction-program anchors in
the exact P1b primitive-slot order and expresses them in the current sensor
frame.  Construction identities never appear in the returned student target.
"""

from __future__ import annotations

import hashlib

import numpy as np

from mtare_topo.evaluation.primitive_composition_anchor_teacher import (
    MAXIMUM_PRIMITIVES,
    current_sensor_anchor_targets,
)


ENDPOINTS_PER_PRIMITIVE = 2
COORDINATES_PER_ENDPOINT = 3
TARGET_SHAPE = (
    MAXIMUM_PRIMITIVES,
    ENDPOINTS_PER_PRIMITIVE,
    COORDINATES_PER_ENDPOINT,
)


def materialize_current_sensor_anchors(
    *,
    primitive_index: np.ndarray,
    primitive_mask: np.ndarray,
    anchor_world_m: np.ndarray,
    sensor_xyz_m: np.ndarray,
    yaw_deg: np.ndarray,
) -> np.ndarray:
    """Return canonical float32 targets with exact zero inactive slots."""

    index = np.asarray(primitive_index, dtype=np.int64)
    mask = np.asarray(primitive_mask, dtype=np.uint8)
    if index.ndim != 2 or index.shape[1:] != (MAXIMUM_PRIMITIVES,):
        raise ValueError("primitive index must be [batch,32]")
    if mask.shape != index.shape:
        raise ValueError("primitive index/mask shape differs")
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("primitive mask must be binary")
    active = mask.astype(bool)
    if np.any(index[active] < 0) or np.any(index[~active] != -1):
        raise ValueError("active/inactive primitive index contract differs")

    anchors = current_sensor_anchor_targets(
        primitive_index=index,
        primitive_mask=mask,
        anchor_world_m=np.asarray(anchor_world_m, dtype=np.float64),
        sensor_xyz_m=np.asarray(sensor_xyz_m, dtype=np.float64),
        yaw_deg=np.asarray(yaw_deg, dtype=np.float64),
    )
    result = np.zeros((len(index),) + TARGET_SHAPE, dtype=np.float32)
    result[active] = anchors[active].astype(np.float32)
    validate_materialized_current_sensor_anchors(result, mask)
    return result


def validate_materialized_current_sensor_anchors(
    anchor_current_sensor_m: np.ndarray,
    primitive_mask: np.ndarray,
) -> None:
    """Fail closed on non-finite active values or nonzero inactive storage."""

    value = np.asarray(anchor_current_sensor_m)
    mask = np.asarray(primitive_mask, dtype=np.uint8)
    if value.shape != (len(mask),) + TARGET_SHAPE or mask.shape != (len(mask), 32):
        raise ValueError("composition-anchor target/mask shape differs")
    if value.dtype != np.dtype(np.float32):
        raise ValueError("composition-anchor targets must be float32")
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("primitive mask must be binary")
    active = mask.astype(bool)
    if not np.all(np.isfinite(value[active])):
        raise ValueError("active composition anchors must be finite")
    if np.any(value[~active] != 0):
        raise ValueError("inactive composition-anchor slots must be exact zero")


def canonical_anchor_digest(
    anchor_current_sensor_m: np.ndarray,
    source_global_sequence_index: np.ndarray,
) -> str:
    """Hash values and sequence identities without depending on Zarr layout."""

    anchors = np.asarray(anchor_current_sensor_m, dtype="<f4")
    sequence = np.asarray(source_global_sequence_index, dtype="<i8")
    if anchors.ndim != 4 or anchors.shape[1:] != TARGET_SHAPE:
        raise ValueError("composition-anchor digest expects [batch,32,2,3]")
    if sequence.shape != (len(anchors),):
        raise ValueError("composition-anchor sequence identity shape differs")
    digest = hashlib.sha256()
    for array in (anchors, sequence):
        shape = np.asarray(array.shape, dtype="<i8")
        digest.update(shape.tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def display_anchor_norm_histogram(
    anchor_norm_m: np.ndarray,
    edges_m: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Summarize targets for a bounded plot without imposing a data gate."""

    values = np.asarray(anchor_norm_m, dtype=np.float64)
    edges = np.asarray(edges_m, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("anchor norms must be finite nonnegative one-dimensional values")
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("histogram edges must be finite and one-dimensional")
    if edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("histogram edges must increase from zero")
    overflow = int(np.count_nonzero(values >= edges[-1]))
    display = np.minimum(values, np.nextafter(edges[-1], 0.0))
    counts = np.histogram(display, bins=edges)[0].astype(np.int64, copy=False)
    return counts, overflow, float(np.max(values, initial=0.0))


__all__ = [
    "COORDINATES_PER_ENDPOINT",
    "ENDPOINTS_PER_PRIMITIVE",
    "TARGET_SHAPE",
    "canonical_anchor_digest",
    "display_anchor_norm_histogram",
    "materialize_current_sensor_anchors",
    "validate_materialized_current_sensor_anchors",
]
