from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mtare_topo.data.gse_axis_anchored_event_relation_training import (
    DescriptorRow,
    descriptor_identity_batches,
    inverse_sqrt_class_weights,
    materialize_world_relation_teacher,
    nearest_bearing_bin,
)


def _heading(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    return np.asarray([np.sin(radians), np.cos(radians)], dtype=np.float32)


def _world() -> dict:
    rows = 6
    references = np.asarray([np.arange(index, index + 5) for index in range(rows)], dtype=np.int64)
    mask = np.zeros((rows, 6), dtype=np.uint8)
    identity = np.full((rows, 6), -1, dtype=np.int64)
    heading = np.zeros((rows, 6, 2), dtype=np.float32)
    exits = [
        ((10, 0.0), (20, 90.0)),
        ((10, 0.0), (20, 90.0)),
        ((10, 0.0), (30, -90.0)),
        ((10, 2.0), (30, -88.0)),
        ((10, 2.0), (30, -88.0)),
        ((10, 2.0), (30, -88.0)),
    ]
    for row, values in enumerate(exits):
        for slot, (key, degrees) in enumerate(values):
            mask[row, slot] = 1; identity[row, slot] = key; heading[row, slot] = _heading(degrees)
    width = np.full((rows, 6), 4.0, dtype=np.float32)
    width_valid = mask.copy()
    profile = np.zeros((rows, 6, 4), dtype=np.float32)
    return {
        "local_frame_references": references,
        "traversal_id": ["t0"] * rows,
        "exit_mask": mask,
        "exit_identity": identity,
        "exit_heading_unit": heading,
        "exit_opening_width_m": width,
        "exit_width_valid_mask": width_valid,
        "exit_vertical_profile_m": profile,
    }


def test_nearest_bearing_bin_wrap_and_residual() -> None:
    assert nearest_bearing_bin(_heading(359.5)) == (0, pytest.approx(-0.5))
    assert nearest_bearing_bin(_heading(91.1))[0] == 46
    with pytest.raises(ValueError, match="unit norm"):
        nearest_bearing_bin(np.asarray([1.0, 1.0]))


def test_relation_teacher_masks_missing_early_observations_without_dropping_rows() -> None:
    teacher = materialize_world_relation_teacher(**_world())
    assert teacher.relation_index.shape == (6, 4, 180, 3)
    assert teacher.branch_presence_mask.shape == (6, 180)
    assert not teacher.relation_valid_pair_mask[0].any()
    assert teacher.relation_valid_pair_mask[4].all()
    assert np.all(teacher.relation_index[0] == -1)
    assert teacher.branch_presence_mask[0].sum() == 2
    population = teacher.population()
    assert population["current_branch_tokens"] == 12
    assert population["reveal"] >= 1
    assert population["withdraw"] >= 1
    assert population["complete_relation_rows"] == 2


def test_relation_teacher_is_deterministic_and_traversal_isolated() -> None:
    values = _world()
    first = materialize_world_relation_teacher(**values)
    second = materialize_world_relation_teacher(**values)
    assert np.array_equal(first.relation_index, second.relation_index)
    changed = dict(values); changed["traversal_id"] = ["a", "a", "a", "b", "b", "b"]
    isolated = materialize_world_relation_teacher(**changed)
    assert isolated.relation_valid_pair_mask.sum() < first.relation_valid_pair_mask.sum()


def test_relation_teacher_allows_reveal_and_withdraw_in_same_bin() -> None:
    values = _world()
    values["exit_heading_unit"][1, 1] = _heading(-90.0)
    teacher = materialize_world_relation_teacher(**values)
    overlap = (teacher.relation_index[..., 1] == 1) & (teacher.relation_index[..., 2] == 1)
    assert overlap.any()


def test_inverse_sqrt_weights_are_positive_normalized_and_fixed() -> None:
    weights = inverse_sqrt_class_weights([100, 25, 4, 1])
    np.testing.assert_allclose(weights.mean(), 1.0)
    assert np.all(weights > 0)
    assert np.all(np.diff(weights) > 0)
    with pytest.raises(ValueError, match="positive"):
        inverse_sqrt_class_weights([10, 0])


def _records() -> list[DescriptorRow]:
    result = []
    for identity in range(1, 101):
        star = (1000 + 2 * identity, 1001 + 2 * identity)
        result.extend((
            DescriptorRow(identity, f"p{identity % 3}", 3 * identity, star),
            DescriptorRow(identity, f"p{identity % 3}", 3 * identity + 1, star),
            DescriptorRow(identity, f"p{identity % 3}", 3 * identity + 2, star[:1]),
        ))
    for identity in range(101, 111):
        result.append(DescriptorRow(identity, "singleton", identity, (2000 + identity,)))
    return result


def test_descriptor_batches_are_identity_balanced_complete_and_deterministic() -> None:
    records = _records()
    first = descriptor_identity_batches(records, seed=0, epoch=0)
    second = descriptor_identity_batches(records, seed=0, epoch=0)
    assert first == second
    assert first != descriptor_identity_batches(records, seed=0, epoch=1)
    selected = [row for batch in first for row in batch]
    association = Counter(row.association_identity for row in selected)
    assert all(association[key] == 2 for key in range(1, 101))
    assert all(association[key] == 1 for key in range(101, 111))
    for batch in first:
        identities = Counter(row.association_identity for row in batch)
        branches = Counter(branch for row in batch for branch in row.branch_identity)
        assert sum(value >= 2 for value in identities.values()) >= 2
        assert sum(value >= 2 for value in branches.values()) >= 2
        assert len(batch) <= 128


def test_descriptor_batches_reject_noncanonical_full_stars() -> None:
    rows = [
        DescriptorRow(1, "p", 0, (1, 2)), DescriptorRow(1, "p", 1, (1, 3)),
        DescriptorRow(2, "p", 2, (4, 5)), DescriptorRow(2, "p", 3, (4, 5)),
    ]
    with pytest.raises(RuntimeError, match="canonical"):
        descriptor_identity_batches(rows, seed=0, epoch=0)
