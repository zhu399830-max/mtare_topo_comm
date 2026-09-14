"""Permutation-safe endpoint, overlap and temporal supervision for P1b."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtare_topo.teacher.primitive_construction_supervisor import PrimitiveConstructionGraph
from mtare_topo.data.primitive_frame_support import PrimitiveFrameSupport


@dataclass(frozen=True)
class PrimitiveRelationWindowTargets:
    primitive_index: np.ndarray
    primitive_mask: np.ndarray
    frame_primitive_index: np.ndarray
    frame_primitive_mask: np.ndarray
    temporal_correspondence: np.ndarray
    temporal_source_mask: np.ndarray
    endpoint_attachment: np.ndarray
    endpoint_pair_mask: np.ndarray
    disconnected_angular_overlap: np.ndarray
    primitive_pair_mask: np.ndarray

    def __post_init__(self) -> None:
        slots = len(self.primitive_index)
        expected = {
            "primitive_mask": (slots,), "frame_primitive_index": (5, slots),
            "frame_primitive_mask": (5, slots),
            "temporal_correspondence": (5, slots, slots + 1),
            "temporal_source_mask": (5, slots),
            "endpoint_attachment": (slots, 2, slots, 2),
            "endpoint_pair_mask": (slots, 2, slots, 2),
            "disconnected_angular_overlap": (slots, slots),
            "primitive_pair_mask": (slots, slots),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if np.any(self.primitive_mask.astype(bool) != (self.primitive_index >= 0)):
            raise ValueError("aggregate primitive mask/index mismatch")
        if np.any(self.frame_primitive_mask.astype(bool) != (self.frame_primitive_index >= 0)):
            raise ValueError("frame primitive mask/index mismatch")
        row_sum = np.sum(self.temporal_correspondence, axis=2)
        if np.any(row_sum[self.temporal_source_mask.astype(bool)] != 1) or np.any(row_sum[~self.temporal_source_mask.astype(bool)] != 0):
            raise ValueError("temporal correspondence must be one-hot on every active source")
        if np.any(self.endpoint_attachment & ~self.endpoint_pair_mask):
            raise ValueError("attachment target exists outside endpoint mask")
        if np.any(self.disconnected_angular_overlap & ~self.primitive_pair_mask):
            raise ValueError("overlap target exists outside pair mask")


def primitive_relation_window_targets(
    *,
    construction: PrimitiveConstructionGraph,
    primitive_ids: Sequence[str],
    source_sets_by_frame: Sequence[Sequence[tuple[int, ...]]],
    maximum_slots: int = 32,
) -> PrimitiveRelationWindowTargets:
    """Create relation targets from five exact per-ray primitive memberships.

    Temporal mappings use independently sorted frame-local slots and map each
    past primitive to the current frame's matching local slot or dustbin.
    Disconnected overlap is strictly an observation-space label: two
    non-attached primitives have LiDAR support in a common azimuth column.
    """

    if len(source_sets_by_frame) != 5 or maximum_slots < 1:
        raise ValueError("relation target requires five frames and positive capacity")
    identities = tuple(str(value) for value in primitive_ids)
    if len(identities) != len(set(identities)) or not identities:
        raise ValueError("primitive identities must be unique and nonempty")
    frame_sets: list[tuple[int, ...]] = []
    azimuth_support: list[dict[int, set[int]]] = []
    for sources in source_sets_by_frame:
        active: set[int] = set(); supports: dict[int, set[int]] = {}
        for ray_index, source in enumerate(sources):
            for primitive_index in source:
                if primitive_index < 0 or primitive_index >= len(identities):
                    raise ValueError("source set contains invalid primitive index")
                active.add(int(primitive_index)); supports.setdefault(int(primitive_index), set()).add(ray_index % 720)
        ordered = tuple(sorted(active))
        if len(ordered) > maximum_slots:
            raise OverflowError(f"frame visible primitive cardinality {len(ordered)} exceeds {maximum_slots}")
        frame_sets.append(ordered); azimuth_support.append(supports)
    aggregate = tuple(sorted(set().union(*map(set, frame_sets))))
    if len(aggregate) > maximum_slots:
        raise OverflowError(f"window visible primitive cardinality {len(aggregate)} exceeds {maximum_slots}")
    primitive_index = np.full(maximum_slots, -1, dtype=np.int32); primitive_index[:len(aggregate)] = aggregate
    primitive_mask = (primitive_index >= 0).astype(np.uint8)
    frame_index = np.full((5, maximum_slots), -1, dtype=np.int32)
    for frame, values in enumerate(frame_sets): frame_index[frame, :len(values)] = values
    frame_mask = (frame_index >= 0).astype(np.uint8)
    temporal = np.zeros((5, maximum_slots, maximum_slots + 1), dtype=np.uint8)
    current_lookup = {value: slot for slot, value in enumerate(frame_sets[-1])}
    for frame, values in enumerate(frame_sets):
        for source_slot, value in enumerate(values):
            temporal[frame, source_slot, current_lookup.get(value, maximum_slots)] = 1

    identity_to_index = {value: index for index, value in enumerate(identities)}
    aggregate_slot = {value: slot for slot, value in enumerate(aggregate)}
    attachment = np.zeros((maximum_slots, 2, maximum_slots, 2), dtype=np.uint8)
    attached_primitive_pairs: set[tuple[int, int]] = set()
    for composition in construction.compositions:
        members = []
        for endpoint in composition.member_endpoints:
            if endpoint.primitive_id not in identity_to_index:
                raise ValueError("construction endpoint is absent from primitive identities")
            primitive = identity_to_index[endpoint.primitive_id]
            if primitive in aggregate_slot:
                members.append((aggregate_slot[primitive], int(endpoint.endpoint_index)))
        for first_slot, first_endpoint in members:
            for second_slot, second_endpoint in members:
                if first_slot == second_slot and first_endpoint == second_endpoint:
                    continue
                attachment[first_slot, first_endpoint, second_slot, second_endpoint] = 1
                attached_primitive_pairs.add(tuple(sorted((aggregate[first_slot], aggregate[second_slot]))))
    endpoint_mask = np.zeros_like(attachment)
    endpoint_mask[:len(aggregate), :, :len(aggregate), :] = 1
    for slot in range(len(aggregate)): endpoint_mask[slot, :, slot, :] = 0
    pair_mask = np.zeros((maximum_slots, maximum_slots), dtype=np.uint8)
    pair_mask[:len(aggregate), :len(aggregate)] = 1; np.fill_diagonal(pair_mask, 0)
    overlap = np.zeros((maximum_slots, maximum_slots), dtype=np.uint8)
    for first_slot, first in enumerate(aggregate):
        for second_slot in range(first_slot + 1, len(aggregate)):
            second = aggregate[second_slot]
            if tuple(sorted((first, second))) in attached_primitive_pairs:
                continue
            shares_azimuth = any(
                bool(supports.get(first, set()) & supports.get(second, set()))
                for supports in azimuth_support
            )
            if shares_azimuth:
                overlap[first_slot, second_slot] = overlap[second_slot, first_slot] = 1
    return PrimitiveRelationWindowTargets(
        primitive_index=primitive_index, primitive_mask=primitive_mask,
        frame_primitive_index=frame_index, frame_primitive_mask=frame_mask,
        temporal_correspondence=temporal, temporal_source_mask=frame_mask.copy(),
        endpoint_attachment=attachment, endpoint_pair_mask=endpoint_mask,
        disconnected_angular_overlap=overlap, primitive_pair_mask=pair_mask,
    )


def primitive_relation_targets_from_frame_support(
    *,
    construction: PrimitiveConstructionGraph,
    primitive_ids: Sequence[str],
    frame_supports: Sequence[PrimitiveFrameSupport],
    maximum_slots: int = 32,
) -> PrimitiveRelationWindowTargets:
    """Build the same relation target from cached lossless frame summaries."""

    if len(frame_supports) != 5:
        raise ValueError("relation target requires five frame supports")
    primitive_count = len(primitive_ids)
    if any(len(value.support_ray_count) != primitive_count for value in frame_supports):
        raise ValueError("frame support primitive population drift")
    identities = tuple(str(value) for value in primitive_ids)
    if len(identities) != len(set(identities)) or not identities:
        raise ValueError("primitive identities must be unique and nonempty")
    frame_sets = [tuple(int(x) for x in np.flatnonzero(value.support_ray_count > 0)) for value in frame_supports]
    if any(len(value) > maximum_slots for value in frame_sets):
        raise OverflowError("frame visible primitive cardinality exceeds slot capacity")
    aggregate = tuple(sorted(set().union(*map(set, frame_sets))))
    if len(aggregate) > maximum_slots:
        raise OverflowError(f"window visible primitive cardinality {len(aggregate)} exceeds {maximum_slots}")

    primitive_index = np.full(maximum_slots, -1, dtype=np.int32); primitive_index[:len(aggregate)] = aggregate
    primitive_mask = (primitive_index >= 0).astype(np.uint8)
    frame_index = np.full((5, maximum_slots), -1, dtype=np.int32)
    for frame, values in enumerate(frame_sets): frame_index[frame, :len(values)] = values
    frame_mask = (frame_index >= 0).astype(np.uint8)
    temporal = np.zeros((5, maximum_slots, maximum_slots + 1), dtype=np.uint8)
    current_lookup = {value: slot for slot, value in enumerate(frame_sets[-1])}
    for frame, values in enumerate(frame_sets):
        for source_slot, value in enumerate(values):
            temporal[frame, source_slot, current_lookup.get(value, maximum_slots)] = 1

    identity_to_index = {value: index for index, value in enumerate(identities)}
    aggregate_slot = {value: slot for slot, value in enumerate(aggregate)}
    attachment = np.zeros((maximum_slots, 2, maximum_slots, 2), dtype=np.uint8)
    attached_pairs: set[tuple[int, int]] = set()
    for composition in construction.compositions:
        members = []
        for endpoint in composition.member_endpoints:
            if endpoint.primitive_id not in identity_to_index:
                raise ValueError("construction endpoint is absent from primitive identities")
            primitive = identity_to_index[endpoint.primitive_id]
            if primitive in aggregate_slot:
                members.append((aggregate_slot[primitive], int(endpoint.endpoint_index)))
        for first_slot, first_endpoint in members:
            for second_slot, second_endpoint in members:
                if first_slot == second_slot and first_endpoint == second_endpoint:
                    continue
                attachment[first_slot, first_endpoint, second_slot, second_endpoint] = 1
                attached_pairs.add(tuple(sorted((aggregate[first_slot], aggregate[second_slot]))))
    endpoint_mask = np.zeros_like(attachment); endpoint_mask[:len(aggregate), :, :len(aggregate), :] = 1
    for slot in range(len(aggregate)): endpoint_mask[slot, :, slot, :] = 0
    pair_mask = np.zeros((maximum_slots, maximum_slots), dtype=np.uint8); pair_mask[:len(aggregate), :len(aggregate)] = 1
    np.fill_diagonal(pair_mask, 0)
    overlap = np.zeros((maximum_slots, maximum_slots), dtype=np.uint8)
    for first_slot, first in enumerate(aggregate):
        for second_slot in range(first_slot + 1, len(aggregate)):
            second = aggregate[second_slot]
            if tuple(sorted((first, second))) in attached_pairs:
                continue
            if any(np.any(frame.azimuth_support_packed[first] & frame.azimuth_support_packed[second]) for frame in frame_supports):
                overlap[first_slot, second_slot] = overlap[second_slot, first_slot] = 1
    return PrimitiveRelationWindowTargets(
        primitive_index=primitive_index, primitive_mask=primitive_mask,
        frame_primitive_index=frame_index, frame_primitive_mask=frame_mask,
        temporal_correspondence=temporal, temporal_source_mask=frame_mask.copy(),
        endpoint_attachment=attachment, endpoint_pair_mask=endpoint_mask,
        disconnected_angular_overlap=overlap, primitive_pair_mask=pair_mask,
    )


__all__ = [
    "PrimitiveRelationWindowTargets", "primitive_relation_window_targets",
    "primitive_relation_targets_from_frame_support",
]
