"""Observation-only diagnostics for construction-supervised endpoint links.

P1b labels an attachment whenever two primitives visible anywhere in the
five-frame window share a construction node.  This module measures the
missing condition: whether the *labelled endpoints* themselves are supported
by the cropped LiDAR-observed primitive axes.  Construction identity is used
only by this offline Teacher audit and is never a model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SUPPORT_BANDS_M = (0.10, 0.25, 0.50, 1.00, 2.00)


@dataclass(frozen=True)
class AttachmentObservationBatch:
    """One row per unique, undirected positive endpoint attachment."""

    first_gap_m: np.ndarray
    second_gap_m: np.ndarray
    observed_endpoint_separation_m: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value) for value in (
            self.first_gap_m,
            self.second_gap_m,
            self.observed_endpoint_separation_m,
        ))
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("attachment observations must be one-dimensional")
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("attachment observation lengths disagree")
        if any(not np.isfinite(value).all() or np.any(value < 0) for value in arrays):
            raise ValueError("attachment observations must be finite and nonnegative")

    @property
    def pair_count(self) -> int:
        return int(len(self.first_gap_m))

    @property
    def maximum_gap_m(self) -> np.ndarray:
        return np.maximum(self.first_gap_m, self.second_gap_m)


def world_to_sensor(
    points_world_m: np.ndarray,
    sensor_xyz_m: np.ndarray,
    yaw_deg: np.ndarray,
) -> np.ndarray:
    """Transform batched world points to the corresponding sensor frames."""

    points = np.asarray(points_world_m, dtype=np.float64)
    sensor = np.asarray(sensor_xyz_m, dtype=np.float64)
    yaw = np.asarray(yaw_deg, dtype=np.float64)
    if points.ndim < 3 or points.shape[0] != len(sensor) or sensor.shape != (len(yaw), 3):
        raise ValueError("batched points, sensor origins and yaws disagree")
    delta = points - sensor.reshape((len(sensor),) + (1,) * (points.ndim - 2) + (3,))
    angle = np.radians(yaw).reshape((len(yaw),) + (1,) * (points.ndim - 2))
    cosine, sine = np.cos(angle), np.sin(angle)
    result = delta.copy()
    result[..., 0] = cosine * delta[..., 0] + sine * delta[..., 1]
    result[..., 1] = -sine * delta[..., 0] + cosine * delta[..., 1]
    return result


def endpoint_support_gaps(
    *,
    axis_control_current_sensor_m: np.ndarray,
    primitive_index: np.ndarray,
    primitive_mask: np.ndarray,
    primitive_endpoints_world_m: np.ndarray,
    sensor_xyz_m: np.ndarray,
    yaw_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint-to-observed-crop gaps and observed endpoint positions.

    P1b axis controls are ordered ``low, middle, high`` along each realized
    primitive.  Therefore controls 0 and 2 are the nearest observed support to
    construction endpoints 0 and 1.  Their distance to the true realized
    endpoint is exactly the unobserved axial gap (up to the 0.025 m field
    sampling discretization).
    """

    axis = np.asarray(axis_control_current_sensor_m, dtype=np.float64)
    index = np.asarray(primitive_index, dtype=np.int64)
    mask = np.asarray(primitive_mask, dtype=bool)
    endpoints = np.asarray(primitive_endpoints_world_m, dtype=np.float64)
    if axis.ndim != 4 or axis.shape[2:] != (3, 3):
        raise ValueError("axis controls must have shape [batch,slots,3,3]")
    if index.shape != axis.shape[:2] or mask.shape != index.shape:
        raise ValueError("primitive index/mask shape mismatch")
    if endpoints.ndim != 3 or endpoints.shape[1:] != (2, 3):
        raise ValueError("primitive endpoints must have shape [primitives,2,3]")
    if np.any(index[mask] < 0) or np.any(index[mask] >= len(endpoints)):
        raise ValueError("active primitive index is out of range")
    if np.any(index[~mask] != -1):
        raise ValueError("inactive primitive slots must use index -1")

    safe_index = np.where(mask, index, 0)
    truth_sensor = world_to_sensor(endpoints[safe_index], sensor_xyz_m, yaw_deg)
    observed = axis[:, :, (0, 2), :]
    gaps = np.linalg.norm(observed - truth_sensor, axis=-1)
    gaps[~mask] = np.inf
    observed = observed.copy()
    observed[~mask] = np.nan
    return gaps, observed


def unique_attachment_observations(
    *,
    endpoint_neighbor: np.ndarray,
    endpoint_gap_m: np.ndarray,
    observed_endpoint_sensor_m: np.ndarray,
) -> AttachmentObservationBatch:
    """Extract unique positive pairs from compact P1b neighbor storage."""

    neighbor = np.asarray(endpoint_neighbor, dtype=np.int16)
    gaps = np.asarray(endpoint_gap_m, dtype=np.float64)
    observed = np.asarray(observed_endpoint_sensor_m, dtype=np.float64)
    if neighbor.ndim != 4 or neighbor.shape[1:] != (32, 2, 3):
        raise ValueError("endpoint neighbors must have shape [batch,32,2,3]")
    if gaps.shape != (len(neighbor), 32, 2) or observed.shape != (len(neighbor), 32, 2, 3):
        raise ValueError("endpoint evidence shape mismatch")

    pairs = unique_attachment_indices(neighbor)
    if not len(pairs):
        empty = np.empty(0, dtype=np.float64)
        return AttachmentObservationBatch(empty, empty.copy(), empty.copy())
    flattened_gap = gaps.reshape(len(gaps), 64)
    flattened_observed = observed.reshape(len(observed), 64, 3)
    row, source, destination = pairs.T
    return AttachmentObservationBatch(
        flattened_gap[row, source],
        flattened_gap[row, destination],
        np.linalg.norm(
            flattened_observed[row, source] - flattened_observed[row, destination], axis=1
        ),
    )


def unique_attachment_indices(endpoint_neighbor: np.ndarray) -> np.ndarray:
    """Return ``[row, first_flat_endpoint, second_flat_endpoint]`` once/pair."""

    neighbor = np.asarray(endpoint_neighbor, dtype=np.int16)
    if neighbor.ndim != 4 or neighbor.shape[1:] != (32, 2, 3):
        raise ValueError("endpoint neighbors must have shape [batch,32,2,3]")
    records: list[np.ndarray] = []
    for source in range(64):
        destinations = neighbor[:, source // 2, source % 2, :]
        row, column = np.nonzero(destinations > source)
        if not len(row):
            continue
        records.append(np.column_stack((
            row,
            np.full(len(row), source, dtype=np.int64),
            destinations[row, column].astype(np.int64),
        )))
    return np.concatenate(records, axis=0) if records else np.empty((0, 3), dtype=np.int64)


def summarize_attachment_observations(
    observations: AttachmentObservationBatch,
    *,
    support_bands_m: Sequence[float] = SUPPORT_BANDS_M,
) -> dict[str, object]:
    """Summarize endpoint support without selecting a favorable threshold."""

    bands = tuple(float(value) for value in support_bands_m)
    if not bands or any(value <= 0 for value in bands) or any(
        second <= first for first, second in zip(bands, bands[1:])
    ):
        raise ValueError("support bands must be positive and strictly increasing")
    maximum = observations.maximum_gap_m
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    result: dict[str, object] = {
        "unique_attachment_pairs": observations.pair_count,
        "maximum_endpoint_gap_quantiles_m": {
            f"q{int(round(value * 100)):02d}": float(np.quantile(maximum, value))
            for value in quantiles
        } if observations.pair_count else {},
        "observed_endpoint_separation_quantiles_m": {
            f"q{int(round(value * 100)):02d}": float(np.quantile(
                observations.observed_endpoint_separation_m, value
            ))
            for value in quantiles
        } if observations.pair_count else {},
        "support_bands": {},
    }
    support = result["support_bands"]
    assert isinstance(support, dict)
    for band in bands:
        # Axis controls are stored as float32; the tolerance only prevents a
        # value such as 0.10000000149 from crossing its declared 0.10 m bin.
        tolerance = max(1e-9, band * 1e-7)
        first = observations.first_gap_m <= band + tolerance
        second = observations.second_gap_m <= band + tolerance
        both = first & second
        exactly_one = first ^ second
        neither = ~(first | second)
        support[f"{band:.2f}m"] = {
            "both_endpoints": int(np.sum(both)),
            "exactly_one_endpoint": int(np.sum(exactly_one)),
            "neither_endpoint": int(np.sum(neither)),
            "both_fraction": float(np.mean(both)) if len(both) else 0.0,
            "at_least_one_fraction": float(np.mean(first | second)) if len(both) else 0.0,
        }
    return result


def merge_attachment_summaries(
    batches: Sequence[AttachmentObservationBatch],
    *,
    support_bands_m: Sequence[float] = SUPPORT_BANDS_M,
) -> Mapping[str, object]:
    """Merge task batches exactly before computing global quantiles."""

    nonempty = [value for value in batches if value.pair_count]
    if not nonempty:
        empty = np.empty(0, dtype=np.float64)
        return summarize_attachment_observations(
            AttachmentObservationBatch(empty, empty.copy(), empty.copy()),
            support_bands_m=support_bands_m,
        )
    return summarize_attachment_observations(
        AttachmentObservationBatch(
            np.concatenate([value.first_gap_m for value in nonempty]),
            np.concatenate([value.second_gap_m for value in nonempty]),
            np.concatenate([value.observed_endpoint_separation_m for value in nonempty]),
        ),
        support_bands_m=support_bands_m,
    )


__all__ = [
    "AttachmentObservationBatch",
    "SUPPORT_BANDS_M",
    "endpoint_support_gaps",
    "merge_attachment_summaries",
    "summarize_attachment_observations",
    "unique_attachment_indices",
    "unique_attachment_observations",
    "world_to_sensor",
]
