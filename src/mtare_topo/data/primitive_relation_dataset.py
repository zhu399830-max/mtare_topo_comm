"""Lossless ray provenance and visible primitive-window supervision contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from mtare_topo.teacher.primitive_provenance_field import PrimitiveRayHit
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


class PrimitiveMembershipCodebook:
    """Map exact per-ray primitive source sets to compact world-local codes.

    Code zero is the no-hit set.  Every other code retains a sorted tuple of
    primitive indices; ambiguous union surfaces are therefore never collapsed
    to an arbitrary primary primitive.
    """

    def __init__(self, primitive_ids: Sequence[str]):
        identities = tuple(str(value) for value in primitive_ids)
        if not identities or len(identities) != len(set(identities)):
            raise ValueError("primitive identities must be nonempty and unique")
        if len(identities) >= np.iinfo(np.uint16).max:
            raise ValueError("primitive count exceeds uint16 codebook contract")
        self.primitive_ids = identities
        self._primitive_to_index = {value: index for index, value in enumerate(identities)}
        self._sets: list[tuple[int, ...]] = [()]
        self._set_to_code: dict[tuple[int, ...], int] = {(): 0}

    @property
    def source_sets(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._sets)

    def encode(self, hits: Iterable[PrimitiveRayHit | None]) -> np.ndarray:
        codes = []
        for hit in hits:
            if hit is None:
                source = ()
            else:
                try:
                    source = tuple(sorted(self._primitive_to_index[value] for value in hit.source_primitive_ids))
                except KeyError as exc:
                    raise ValueError(f"hit references unknown primitive {exc.args[0]!r}") from exc
                if not source or len(source) != len(set(source)):
                    raise ValueError("hit source membership must be nonempty and unique")
            code = self._set_to_code.get(source)
            if code is None:
                if len(self._sets) >= np.iinfo(np.uint16).max:
                    raise OverflowError("world provenance codebook exceeds uint16 capacity")
                code = len(self._sets)
                self._sets.append(source)
                self._set_to_code[source] = code
            codes.append(code)
        return np.asarray(codes, dtype=np.uint16)

    def decode(self, codes: np.ndarray) -> tuple[tuple[int, ...], ...]:
        values = np.asarray(codes)
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= len(self._sets)):
            raise ValueError("provenance code is outside the codebook")
        return tuple(self._sets[int(value)] for value in values.reshape(-1))

    def cardinality(self, codes: np.ndarray) -> np.ndarray:
        sizes = np.asarray([len(value) for value in self._sets], dtype=np.uint16)
        values = np.asarray(codes)
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= len(sizes)):
            raise ValueError("provenance code is outside the codebook")
        return sizes[values]

    def as_dict(self) -> dict:
        return {
            "schema_version": "primitive_membership_codebook_v1",
            "primitive_ids": list(self.primitive_ids),
            "source_sets": [list(value) for value in self._sets],
            "zero_code_semantics": "no qualified union-surface hit",
        }


@dataclass(frozen=True)
class VisiblePrimitiveWindowTargets:
    primitive_index: np.ndarray
    mask: np.ndarray
    axis_control_current_sensor_m: np.ndarray
    endpoint_half_axes_m: np.ndarray
    endpoint_shape_exponent: np.ndarray
    support_ray_count: np.ndarray
    temporal_visibility: np.ndarray

    def __post_init__(self) -> None:
        slots = len(self.primitive_index)
        expected = {
            "mask": (slots,),
            "axis_control_current_sensor_m": (slots, 3, 3),
            "endpoint_half_axes_m": (slots, 2, 2),
            "endpoint_shape_exponent": (slots, 2),
            "support_ray_count": (slots,),
            "temporal_visibility": (5, slots),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if np.any(self.mask.astype(bool) != (self.primitive_index >= 0)):
            raise ValueError("primitive mask and padded indices disagree")
        if not np.all(np.isfinite(self.axis_control_current_sensor_m[self.mask.astype(bool)])):
            raise ValueError("active primitive axes must be finite")


def _current_sensor_transform(points: np.ndarray, origin: np.ndarray, yaw_deg: float) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    yaw = np.radians(float(yaw_deg))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    result = delta.copy()
    result[..., 0] = cosine * delta[..., 0] + sine * delta[..., 1]
    result[..., 1] = -sine * delta[..., 0] + cosine * delta[..., 1]
    return result


def visible_primitive_window_targets(
    *,
    field: SweptSuperellipseProvenanceField,
    hits_by_frame: Sequence[Sequence[PrimitiveRayHit | None]],
    current_sensor_xyz_m: np.ndarray,
    current_yaw_deg: float,
    maximum_slots: int = 8,
) -> VisiblePrimitiveWindowTargets:
    """Crop program primitives to the centreline support visible in five scans."""

    if len(hits_by_frame) != 5:
        raise ValueError("a causal primitive target requires exactly five frames")
    if maximum_slots < 1:
        raise ValueError("maximum_slots must be positive")
    identity_to_index = {value: index for index, value in enumerate(field.primitive_ids)}
    support_indices: dict[int, list[int]] = {}
    support_counts: dict[int, int] = {}
    temporal: dict[int, np.ndarray] = {}
    for frame_index, hits in enumerate(hits_by_frame):
        for hit in hits:
            if hit is None:
                continue
            point = np.asarray(hit.xyz_m, dtype=np.float64)
            for identity in hit.source_primitive_ids:
                primitive_index = identity_to_index[identity]
                operand = field.operands[primitive_index]
                _, sampled_index = operand.tree.query(point, k=1, workers=1)
                support_indices.setdefault(primitive_index, []).append(int(sampled_index))
                support_counts[primitive_index] = support_counts.get(primitive_index, 0) + 1
                temporal.setdefault(primitive_index, np.zeros(5, dtype=np.uint8))[frame_index] = 1
    active = sorted(support_indices)
    if len(active) > maximum_slots:
        raise OverflowError(f"visible primitive cardinality {len(active)} exceeds {maximum_slots} slots")
    primitive_index = np.full(maximum_slots, -1, dtype=np.int32)
    mask = np.zeros(maximum_slots, dtype=np.uint8)
    axes = np.zeros((maximum_slots, 3, 3), dtype=np.float32)
    half_axes = np.zeros((maximum_slots, 2, 2), dtype=np.float32)
    exponent = np.zeros((maximum_slots, 2), dtype=np.float32)
    counts = np.zeros(maximum_slots, dtype=np.int32)
    temporal_visibility = np.zeros((5, maximum_slots), dtype=np.uint8)
    for slot, index in enumerate(active):
        operand = field.operands[index]
        low, high = min(support_indices[index]), max(support_indices[index])
        middle_arc = .5 * (operand.arc_m[low] + operand.arc_m[high])
        middle = int(np.argmin(np.abs(operand.arc_m - middle_arc)))
        controls = operand.points[[low, middle, high]]
        fractions = operand.arc_m[[low, high]] / operand.arc_m[-1]
        parameters, shapes = operand.primitive.parameters_at_fraction(fractions)
        primitive_index[slot] = index; mask[slot] = 1
        axes[slot] = _current_sensor_transform(controls, current_sensor_xyz_m, current_yaw_deg)
        half_axes[slot] = parameters; exponent[slot] = shapes
        counts[slot] = support_counts[index]; temporal_visibility[:, slot] = temporal[index]
    return VisiblePrimitiveWindowTargets(
        primitive_index, mask, axes, half_axes, exponent, counts, temporal_visibility
    )


__all__ = [
    "PrimitiveMembershipCodebook",
    "VisiblePrimitiveWindowTargets",
    "visible_primitive_window_targets",
]
