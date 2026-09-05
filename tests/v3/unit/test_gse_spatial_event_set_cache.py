from __future__ import annotations

import numpy as np

from mtare_topo.data.gse_spatial_event_set_cache import (
    SpatialEventTeacherArrays,
    fit_only_loss_weights,
)


def _synthetic_teacher() -> SpatialEventTeacherArrays:
    # Bypass formal population constants only to exercise the immutable
    # dataclass helpers would require 188,126 rows; formal join is covered by
    # the readiness executor.  Construct without __init__ for local slicing.
    teacher = object.__new__(SpatialEventTeacherArrays)
    object.__setattr__(teacher, "global_sequence_index", np.arange(4, dtype=np.int64))
    object.__setattr__(teacher, "partition_code", np.asarray((0, 0, 1, 1), dtype=np.uint8))
    event_type = np.full((4, 16), -1, dtype=np.int8)
    identity = np.full((4, 16), -1, dtype=np.int32)
    mask = np.zeros((4, 16), dtype=np.uint8)
    relative = np.zeros((4, 16, 3), dtype=np.float32)
    event_type[0, :2] = (0, 1); identity[0, :2] = (3, 4); mask[0, :2] = 1
    object.__setattr__(teacher, "event_type_index", event_type)
    object.__setattr__(teacher, "event_relative_xyz_m", relative)
    object.__setattr__(teacher, "event_identity_index", identity)
    object.__setattr__(teacher, "event_mask", mask)
    object.__setattr__(teacher, "set_cardinality", mask.sum(axis=1).astype(np.uint8))
    object.__setattr__(teacher, "parent_id", np.asarray(("a", "a", "b", "b")))
    return teacher


def test_teacher_target_slicing_preserves_fixed_slots():
    teacher = _synthetic_teacher()
    targets = teacher.targets(np.asarray((0, 2), dtype=np.int64))
    assert targets["event_type_index"].shape == (2, 16)
    assert targets["event_relative_xyz_m"].shape == (2, 16, 3)
    assert targets["event_mask"][0].sum() == 2
    assert targets["event_mask"][1].sum() == 0

