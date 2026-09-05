from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.evaluation.gse_spatial_event_set_metrics import (
    select_fixed_grid_threshold,
    spatial_event_set_metrics,
)


def _case():
    confidence = np.asarray(((0.9, 0.8, 0.1), (0.7, 0.2, 0.1)), dtype=np.float32)
    predicted_type = np.asarray(((0, 1, 0), (1, 0, 1)), dtype=np.int64)
    predicted_xyz = np.asarray(
        (((2.0, 0.0, 0.0), (4.0, 0.0, 0.0), (20.0, 0.0, 0.0)),
         ((1.0, 0.0, 0.0), (8.0, 0.0, 0.0), (9.0, 0.0, 0.0))),
        dtype=np.float32,
    )
    target_type = np.full((2, 16), -1, dtype=np.int64)
    target_xyz = np.zeros((2, 16, 3), dtype=np.float32)
    target_mask = np.zeros((2, 16), dtype=np.uint8)
    target_type[0, :2] = (0, 1); target_xyz[0, 0, 0] = 2.5; target_xyz[0, 1, 0] = 4.5
    target_type[1, 0] = 1; target_xyz[1, 0, 0] = 1.5
    target_mask[0, :2] = 1; target_mask[1, 0] = 1
    return confidence, predicted_type, predicted_xyz, target_type, target_xyz, target_mask


def test_typed_four_metre_matching_and_multi_event_recall():
    values = _case()
    result = spatial_event_set_metrics(
        confidence=values[0], predicted_type=values[1], predicted_xyz_m=values[2],
        target_type=values[3], target_xyz_m=values[4], target_mask=values[5],
        threshold=0.5,
    )
    assert result["true_positive"] == 3
    assert result["false_positive"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["multi_event_recall"] == 1.0
    assert result["matched_localization_mae_m"] == pytest.approx(0.5)


def test_wrong_type_and_more_than_four_metres_are_not_matches():
    values = list(_case())
    values[1] = values[1].copy(); values[1][0, 0] = 1
    values[2] = values[2].copy(); values[2][0, 0, 0] = -10.0; values[2][0, 1, 0] = 20.0
    result = spatial_event_set_metrics(
        confidence=values[0], predicted_type=values[1], predicted_xyz_m=values[2],
        target_type=values[3], target_xyz_m=values[4], target_mask=values[5],
        threshold=0.5,
    )
    assert result["true_positive"] == 1
    assert result["false_positive"] == 2
    assert result["false_negative"] == 2


def test_threshold_grid_has_deterministic_tie_break():
    values = _case()
    predictions = {"confidence": values[0], "event_type": values[1], "relative_xyz_m": values[2]}
    targets = {"event_type_index": values[3], "event_relative_xyz_m": values[4], "event_mask": values[5]}
    best, records = select_fixed_grid_threshold(predictions, targets, (0.5, 0.6))
    assert len(records) == 2
    assert best["threshold"] == 0.6
