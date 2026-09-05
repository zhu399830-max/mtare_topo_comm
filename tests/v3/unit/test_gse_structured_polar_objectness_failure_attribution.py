from __future__ import annotations

import numpy as np

from execute_gse_structured_polar_objectness_failure_attribution_v1 import (
    _matching_labels,
    _one_per_bin,
    _ranking_diagnostic,
)


def _targets() -> dict[str, np.ndarray]:
    mask = np.zeros((1, 16), dtype=bool)
    mask[0, :2] = True
    event_type = np.full((1, 16), -1, dtype=np.int64)
    event_type[0, :2] = (0, 1)
    xyz = np.zeros((1, 16, 3), dtype=np.float32)
    xyz[0, 0] = (4.0, 0.0, 0.0)
    xyz[0, 1] = (12.0, 0.0, 0.0)
    return {"event_mask": mask, "event_type_index": event_type, "event_relative_xyz_m": xyz}


def _predictions() -> dict[str, np.ndarray]:
    confidence = np.zeros((1, 16), dtype=np.float32)
    confidence[0, :3] = (0.9, 0.8, 0.7)
    event_type = np.zeros((1, 16), dtype=np.int8)
    event_type[0, 1] = 1
    xyz = np.zeros((1, 16, 3), dtype=np.float32)
    xyz[0, 0] = (4.0, 0.0, 0.0)
    xyz[0, 1] = (12.0, 0.0, 0.0)
    xyz[0, 2] = (30.0, 0.0, 0.0)
    azimuth = np.arange(16, dtype=np.int16)[None]
    azimuth[0, 1] = azimuth[0, 0]
    slot = np.zeros((1, 16), dtype=np.int8)
    slot[0, 1] = 1
    return {"confidence": confidence, "event_type": event_type, "relative_xyz_m": xyz, "azimuth_bin": azimuth, "depth_slot": slot}


def test_matching_labels_marks_typed_candidates_and_targets() -> None:
    predictions = _predictions()
    selected = np.zeros((1, 16), dtype=bool)
    selected[0, :3] = True
    prediction_match, target_match, errors = _matching_labels(predictions, _targets(), selected)
    assert prediction_match[0, :3].tolist() == [True, True, False]
    assert target_match[0, :2].tolist() == [True, True]
    assert errors.tolist() == [0.0, 0.0]


def test_one_per_bin_keeps_higher_confidence_candidate() -> None:
    corrected, suppressed = _one_per_bin(_predictions())
    assert suppressed == 1
    assert corrected["confidence"][0, 0] == np.float32(0.9)
    assert corrected["confidence"][0, 1] == np.float32(0.0)


def test_ranking_diagnostic_reports_perfect_order() -> None:
    confidence = np.asarray([[0.9, 0.8, 0.2, 0.1]], dtype=np.float32)
    labels = np.asarray([[True, True, False, False]])
    result = _ranking_diagnostic(confidence, labels)
    assert result["average_precision"] == 1.0
    assert result["precision_at_candidate_recall_at_least_0p25"] == 1.0
