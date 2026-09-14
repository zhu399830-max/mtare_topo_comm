"""Lossless compact storage for dense in-memory primitive relation targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mtare_topo.data.primitive_relation_targets import PrimitiveRelationWindowTargets


@dataclass(frozen=True)
class PackedPrimitiveRelationTargets:
    temporal_destination: np.ndarray
    endpoint_neighbor: np.ndarray
    disconnected_overlap_packed: np.ndarray

    def __post_init__(self) -> None:
        if self.temporal_destination.shape != (5, 32):
            raise ValueError("temporal destinations must be 5x32")
        if self.endpoint_neighbor.shape != (32, 2, 3):
            raise ValueError("endpoint neighbors must be 32x2x3")
        if self.disconnected_overlap_packed.shape != (32, 4):
            raise ValueError("packed overlap must be 32x4 bytes")
        if self.temporal_destination.dtype != np.int8 or self.endpoint_neighbor.dtype != np.int8 or self.disconnected_overlap_packed.dtype != np.uint8:
            raise ValueError("packed relation dtypes drift")


def pack_primitive_relation_targets(target: PrimitiveRelationWindowTargets) -> PackedPrimitiveRelationTargets:
    if len(target.primitive_index) != 32:
        raise ValueError("formal P1b storage requires exactly 32 slots")
    temporal = np.full((5, 32), -1, dtype=np.int8)
    for frame, slot in np.argwhere(target.temporal_source_mask > 0):
        destinations = np.flatnonzero(target.temporal_correspondence[frame, slot])
        if len(destinations) != 1:
            raise RuntimeError("active temporal source is not one-hot")
        temporal[frame, slot] = np.int8(destinations[0])
    neighbors = np.full((32, 2, 3), -1, dtype=np.int8)
    for slot in range(32):
        for endpoint in range(2):
            destinations = np.argwhere(target.endpoint_attachment[slot, endpoint] > 0)
            if len(destinations) > 3:
                raise OverflowError("endpoint incidence exceeds frozen degree-4 construction contract")
            flattened = [int(other_slot) * 2 + int(other_endpoint) for other_slot, other_endpoint in destinations]
            neighbors[slot, endpoint, :len(flattened)] = np.asarray(flattened, dtype=np.int8)
    overlap = np.packbits(target.disconnected_angular_overlap.astype(np.uint8), axis=1, bitorder="little")
    return PackedPrimitiveRelationTargets(temporal, neighbors, overlap)


def unpack_temporal_correspondence(packed: PackedPrimitiveRelationTargets) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((5, 32, 33), dtype=np.uint8); mask = (packed.temporal_destination >= 0).astype(np.uint8)
    for frame, slot in np.argwhere(mask > 0): values[frame, slot, int(packed.temporal_destination[frame, slot])] = 1
    return values, mask


def unpack_endpoint_attachment(packed: PackedPrimitiveRelationTargets) -> np.ndarray:
    result = np.zeros((32, 2, 32, 2), dtype=np.uint8)
    for slot in range(32):
        for endpoint in range(2):
            for destination in packed.endpoint_neighbor[slot, endpoint]:
                if destination >= 0: result[slot, endpoint, int(destination) // 2, int(destination) % 2] = 1
    return result


def unpack_disconnected_overlap(packed: PackedPrimitiveRelationTargets) -> np.ndarray:
    return np.unpackbits(packed.disconnected_overlap_packed, axis=1, count=32, bitorder="little").astype(np.uint8)


__all__ = [
    "PackedPrimitiveRelationTargets", "pack_primitive_relation_targets",
    "unpack_temporal_correspondence", "unpack_endpoint_attachment",
    "unpack_disconnected_overlap",
]
