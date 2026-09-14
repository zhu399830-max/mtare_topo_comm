"""Deterministic Teacher rasterization and descriptor batches for GSE-Graph."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np


BEARING_BINS = 180
RELATION_STEPS = 4
RELATION_NAMES = ("persistent", "reveal", "withdraw")


def nearest_bearing_bin(heading_unit: np.ndarray) -> tuple[int, float]:
    heading = np.asarray(heading_unit, dtype=np.float64)
    if heading.shape != (2,) or not np.all(np.isfinite(heading)):
        raise ValueError("heading_unit must be finite [sin,cos]")
    norm = float(np.linalg.norm(heading))
    if not 0.998 <= norm <= 1.002:
        raise ValueError("heading_unit must have unit norm")
    bearing = float(np.degrees(np.arctan2(heading[0], heading[1])) % 360.0)
    index = int(np.floor((bearing + 1.0) / 2.0)) % BEARING_BINS
    residual = (bearing - 2.0 * index + 180.0) % 360.0 - 180.0
    if not -1.00001 <= residual <= 1.00001:
        raise RuntimeError("nearest bearing residual outside one degree")
    return index, residual


@dataclass(frozen=True)
class WorldRelationTeacher:
    relation_index: np.ndarray
    branch_presence_mask: np.ndarray
    branch_heading_residual_deg: np.ndarray
    branch_opening_width_m: np.ndarray
    branch_width_valid_mask: np.ndarray
    branch_vertical_profile_m: np.ndarray
    branch_identity: np.ndarray
    relation_valid_pair_mask: np.ndarray

    def population(self) -> dict[str, int]:
        valid = self.relation_index >= 0
        counts = {
            name: int(((self.relation_index[..., index] == 1) & valid[..., index]).sum())
            for index, name in enumerate(RELATION_NAMES)
        }
        negatives = {
            f"{name}_negative": int(((self.relation_index[..., index] == 0) & valid[..., index]).sum())
            for index, name in enumerate(RELATION_NAMES)
        }
        return {
            "valid_pair_positions": int(self.relation_valid_pair_mask.sum()),
            "complete_relation_rows": int(self.relation_valid_pair_mask.all(axis=1).sum()),
            "simultaneous_reveal_withdraw_bins": int(
                ((self.relation_index[..., 1] == 1) & (self.relation_index[..., 2] == 1)).sum()
            ),
            **counts, **negatives,
            "current_branch_tokens": int(self.branch_presence_mask.sum()),
        }


def materialize_world_relation_teacher(
    *,
    local_frame_references: np.ndarray,
    traversal_id: Sequence[str],
    exit_mask: np.ndarray,
    exit_identity: np.ndarray,
    exit_heading_unit: np.ndarray,
    exit_opening_width_m: np.ndarray,
    exit_width_valid_mask: np.ndarray,
    exit_vertical_profile_m: np.ndarray,
) -> WorldRelationTeacher:
    """Rasterize only objective same-traversal observation-level relations."""
    references = np.asarray(local_frame_references, dtype=np.int64)
    traversal = np.asarray(traversal_id, dtype=str)
    mask = np.asarray(exit_mask, dtype=bool)
    identity = np.asarray(exit_identity, dtype=np.int64)
    heading = np.asarray(exit_heading_unit, dtype=np.float32)
    width = np.asarray(exit_opening_width_m, dtype=np.float32)
    width_valid = np.asarray(exit_width_valid_mask, dtype=bool)
    profile = np.asarray(exit_vertical_profile_m, dtype=np.float32)
    rows = len(references)
    if (
        references.shape != (rows, 5) or traversal.shape != (rows,)
        or mask.shape != (rows, 6) or identity.shape != (rows, 6)
        or heading.shape != (rows, 6, 2) or width.shape != (rows, 6)
        or width_valid.shape != (rows, 6) or profile.shape != (rows, 6, 4)
        or len(np.unique(references[:, -1])) != rows
    ):
        raise ValueError("world relation Teacher input shape/identity drift")
    if np.any(width_valid & ~mask):
        raise ValueError("exit width-valid mask must be a subset of exit mask")
    current_lookup = {int(value): row for row, value in enumerate(references[:, -1])}
    relation = np.full((rows, RELATION_STEPS, BEARING_BINS, 3), -1, dtype=np.int8)
    relation_valid = np.zeros((rows, RELATION_STEPS), dtype=bool)
    presence = np.zeros((rows, BEARING_BINS), dtype=bool)
    residual = np.zeros((rows, BEARING_BINS), dtype=np.float32)
    raster_width = np.zeros((rows, BEARING_BINS), dtype=np.float32)
    raster_width_valid = np.zeros((rows, BEARING_BINS), dtype=bool)
    raster_profile = np.zeros((rows, BEARING_BINS, 4), dtype=np.float32)
    raster_identity = np.full((rows, BEARING_BINS), -1, dtype=np.int64)

    def exit_map(row: int) -> dict[int, tuple[np.ndarray, int]]:
        return {
            int(identity[row, slot]): (heading[row, slot], int(slot))
            for slot in np.flatnonzero(mask[row])
        }

    for row in range(rows):
        for slot in np.flatnonzero(mask[row]):
            bearing_index, bearing_residual = nearest_bearing_bin(heading[row, slot])
            if presence[row, bearing_index]:
                raise RuntimeError(f"current exit bearing-bin collision at row {row}, bin {bearing_index}")
            presence[row, bearing_index] = True
            residual[row, bearing_index] = bearing_residual
            raster_width[row, bearing_index] = width[row, slot]
            raster_width_valid[row, bearing_index] = width_valid[row, slot]
            raster_profile[row, bearing_index] = profile[row, slot]
            raster_identity[row, bearing_index] = identity[row, slot]
        for step, (previous_frame, current_frame) in enumerate(zip(references[row, :-1], references[row, 1:])):
            previous_row = current_lookup.get(int(previous_frame))
            current_row = current_lookup.get(int(current_frame))
            if (
                previous_row is None or current_row is None
                or traversal[previous_row] != traversal[row]
                or traversal[current_row] != traversal[row]
            ):
                continue
            relation[row, step] = 0
            relation_valid[row, step] = True
            previous = exit_map(previous_row); current = exit_map(current_row)
            for state, keys, source in (
                (0, set(previous) & set(current), current),
                (1, set(current) - set(previous), current),
                (2, set(previous) - set(current), previous),
            ):
                for key in sorted(keys):
                    bearing_index, _ = nearest_bearing_bin(source[key][0])
                    if relation[row, step, bearing_index, state] == 1:
                        raise RuntimeError(
                            f"same-channel relation bearing-bin collision at row {row}, step {step}, "
                            f"bin {bearing_index}, channel {state}"
                        )
                    relation[row, step, bearing_index, state] = 1
    return WorldRelationTeacher(
        relation_index=relation,
        branch_presence_mask=presence,
        branch_heading_residual_deg=residual,
        branch_opening_width_m=raster_width,
        branch_width_valid_mask=raster_width_valid,
        branch_vertical_profile_m=raster_profile,
        branch_identity=raster_identity,
        relation_valid_pair_mask=relation_valid,
    )


def inverse_sqrt_class_weights(counts: Sequence[int]) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("class counts must be positive finite one-dimensional values")
    weight = 1.0 / np.sqrt(values)
    weight /= weight.mean()
    return weight.astype(np.float32)


@dataclass(frozen=True, order=True)
class DescriptorRow:
    association_identity: int
    parent_id: str
    row: int
    branch_identity: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.association_identity < 0 or self.row < 0 or not self.parent_id or not self.branch_identity:
            raise ValueError("descriptor row requires structural identity, source and branches")


def _stable_rotation(identity: int, seed: int, epoch: int, length: int) -> int:
    payload = f"gse-axis-descriptor:{identity}:{seed}:{epoch}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % length


def descriptor_identity_batches(
    records: Iterable[DescriptorRow],
    *,
    seed: int,
    epoch: int,
    paired_identities_per_batch: int = 48,
    maximum_batch_rows: int = 128,
) -> list[tuple[DescriptorRow, ...]]:
    """Sample two full-star views per pairable identity and singleton negatives."""
    if seed not in (0, 1, 2) or epoch < 0 or paired_identities_per_batch < 2:
        raise ValueError("descriptor batch schedule parameter drift")
    grouped: dict[int, list[DescriptorRow]] = defaultdict(list)
    for record in records:
        grouped[record.association_identity].append(record)
    if len(grouped) < 2:
        raise ValueError("descriptor schedule requires at least two identities")
    paired = []
    singleton = []
    for identity in sorted(grouped):
        rows = sorted(grouped[identity])
        maximum_star = max(len(row.branch_identity) for row in rows)
        full = [row for row in rows if len(row.branch_identity) == maximum_star]
        if len(full) < 2:
            singleton.append(full[0])
            continue
        start = _stable_rotation(identity, seed, epoch, len(full))
        first = full[start]
        second = full[(start + 1) % len(full)]
        if first.branch_identity != second.branch_identity:
            raise RuntimeError(f"association identity {identity} lacks an exact canonical full branch star")
        paired.append((identity, first, second))
    if len(paired) < 2:
        raise ValueError("descriptor schedule requires at least two pairable identities")
    generator = np.random.default_rng(np.random.SeedSequence((seed, epoch, 20260829)))
    paired = [paired[index] for index in generator.permutation(len(paired))]
    singleton = [singleton[index] for index in generator.permutation(len(singleton))] if singleton else []
    chunks = [paired[start : start + paired_identities_per_batch] for start in range(0, len(paired), paired_identities_per_batch)]
    if len(chunks[-1]) == 1:
        chunks[-1].insert(0, chunks[-2].pop())
    batches = [[row for _, first, second in chunk for row in (first, second)] for chunk in chunks]
    for index, record in enumerate(singleton):
        target = index % len(batches)
        if len(batches[target]) >= maximum_batch_rows:
            choices = [i for i, batch in enumerate(batches) if len(batch) < maximum_batch_rows]
            if not choices:
                raise RuntimeError("descriptor singleton capacity exhausted")
            target = choices[0]
        batches[target].append(record)
    result = []
    for batch in batches:
        identities = Counter(row.association_identity for row in batch)
        if len(batch) > maximum_batch_rows or sum(value >= 2 for value in identities.values()) < 2:
            raise RuntimeError("descriptor batch lacks two positive identities or exceeds capacity")
        branch_counts = Counter(branch for row in batch for branch in row.branch_identity)
        if sum(value >= 2 for value in branch_counts.values()) < 2:
            raise RuntimeError("descriptor batch lacks repeated positive branch identities")
        result.append(tuple(batch))
    return result


__all__ = [
    "DescriptorRow", "WorldRelationTeacher", "descriptor_identity_batches",
    "inverse_sqrt_class_weights", "materialize_world_relation_teacher",
    "nearest_bearing_bin",
]
