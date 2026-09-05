from __future__ import annotations

import numpy as np
import pytest

from evaluate_gse_structured_polar_multidepth_capacity_v1 import (
    _second_depth_recall,
    _second_depth_target_mask,
)


def _targets(radii: tuple[float, ...]) -> dict[str, np.ndarray]:
    mask = np.zeros((1, 16), dtype=bool)
    event_type = np.full((1, 16), -1, dtype=np.int64)
    identity = np.full((1, 16), -1, dtype=np.int64)
    xyz = np.zeros((1, 16, 3), dtype=np.float32)
    for index, radius in enumerate(radii):
        angle = np.deg2rad(10.2)
        mask[0, index] = True
        event_type[0, index] = index % 2
        identity[0, index] = 100 + index
        xyz[0, index] = (radius * np.cos(angle), radius * np.sin(angle), 0.0)
    return {
        "event_mask": mask,
        "event_type_index": event_type,
        "event_identity_index": identity,
        "event_relative_xyz_m": xyz,
    }


def _predictions(targets: dict[str, np.ndarray], source_index: int) -> dict[str, np.ndarray]:
    confidence = np.zeros((1, 16), dtype=np.float32)
    event_type = np.zeros((1, 16), dtype=np.int8)
    xyz = np.zeros((1, 16, 3), dtype=np.float32)
    confidence[0, 0] = 0.9
    event_type[0, 0] = targets["event_type_index"][0, source_index]
    xyz[0, 0] = targets["event_relative_xyz_m"][0, source_index]
    return {"confidence": confidence, "event_type": event_type, "relative_xyz_m": xyz}


def test_second_depth_mask_marks_only_farther_same_bin_target() -> None:
    targets = _targets((4.0, 12.0))
    observed = _second_depth_target_mask(targets)
    assert observed.tolist() == [[False, True] + [False] * 14]


def test_second_depth_recall_uses_common_typed_four_metre_matching() -> None:
    targets = _targets((4.0, 12.0))
    far = _second_depth_recall(_predictions(targets, 1), targets, threshold=0.5)
    near = _second_depth_recall(_predictions(targets, 0), targets, threshold=0.5)
    assert far == {"true_positive": 1, "target": 1, "recall": 1.0}
    assert near == {"true_positive": 0, "target": 1, "recall": 0.0}


def test_second_depth_mask_rejects_three_same_bin_targets() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        _second_depth_target_mask(_targets((4.0, 8.0, 12.0)))
