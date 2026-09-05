from __future__ import annotations

import numpy as np

from execute_gse_structured_polar_objectness_failure_attribution_v1r import (
    _seed_record,
    _target_bins,
    _target_to_prediction,
)


def _case():
    mask = np.zeros((1, 16), dtype=bool); mask[0, :2] = True
    target_type = np.full((1, 16), -1, dtype=np.int64); target_type[0, :2] = (0, 1)
    target_xyz = np.zeros((1, 16, 3), dtype=np.float32); target_xyz[0, 0] = (4, 0, 0); target_xyz[0, 1] = (12, 0, 0)
    targets = {"event_mask": mask, "event_type_index": target_type, "event_relative_xyz_m": target_xyz}
    confidence = np.zeros((1, 16), dtype=np.float32); confidence[0, :2] = (0.99, 0.98)
    predicted_type = np.zeros((1, 16), dtype=np.int8); predicted_type[0, 1] = 1
    predicted_xyz = np.zeros((1, 16, 3), dtype=np.float32); predicted_xyz[0, :2] = target_xyz[0, :2]
    bins = np.arange(16, dtype=np.int16)[None]; bins[0, 1] = bins[0, 0]
    slots = np.zeros((1, 16), dtype=np.int8); slots[0, 1] = 1
    predictions = {"confidence": confidence, "event_type": predicted_type, "relative_xyz_m": predicted_xyz, "azimuth_bin": bins, "depth_slot": slots}
    second = np.zeros((1, 16), dtype=bool); second[0, 1] = True
    return predictions, targets, second


def test_target_to_prediction_preserves_depth_slot_identity() -> None:
    predictions, targets, _ = _case()
    mapping = _target_to_prediction(predictions, targets)
    assert mapping[0, :2].tolist() == [0, 1]


def test_seed_record_recognizes_distinct_slot_pair() -> None:
    predictions, targets, second = _case()
    record = _seed_record(predictions, targets, second, _target_bins(targets))
    assert record["matched_second_depth_by_prediction_slot"] == {"0": 0, "1": 1}
    assert record["same_bin_pairs_matched_with_distinct_prediction_slots"] == 1
    assert record["slot1_second_depth_utilized"] is True
