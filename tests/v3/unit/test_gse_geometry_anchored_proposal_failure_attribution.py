from __future__ import annotations

import numpy as np

from execute_gse_geometry_anchored_proposal_failure_attribution_v1 import (
    _nms,
    _population_summary,
    _proposal_support,
    _threshold_diagnostics,
)


def _fixture():
    confidence = np.zeros((2, 16), dtype=np.float32)
    event_type = np.ones((2, 16), dtype=np.int8)
    xyz = np.zeros((2, 16, 3), dtype=np.float32)
    confidence[0, 0] = 0.99
    event_type[0, 0] = 0
    xyz[0, 0] = (3.0, 0.0, 0.0)
    confidence[1, :2] = (0.99, 0.98)
    event_type[1, :2] = 0
    xyz[1, 0] = (5.0, 0.0, 0.0)
    xyz[1, 1] = (5.2, 0.0, 0.0)
    target_type = np.full((2, 16), -1, dtype=np.int64)
    target_xyz = np.zeros((2, 16, 3), dtype=np.float32)
    target_mask = np.zeros((2, 16), dtype=np.uint8)
    target_type[1, 0] = 0
    target_xyz[1, 0] = (5.0, 0.0, 0.0)
    target_mask[1, 0] = 1
    predictions = {"confidence": confidence, "event_type": event_type, "relative_xyz_m": xyz}
    targets = {
        "event_type_index": target_type,
        "event_relative_xyz_m": target_xyz,
        "event_mask": target_mask,
    }
    return predictions, targets


def test_empty_row_false_positive_and_cardinality_are_explicit():
    predictions, targets = _fixture()
    result = _threshold_diagnostics(predictions, targets, 0.95)
    assert result["empty_target_rows"] == 1
    assert result["empty_row_false_positive_events"] == 1
    assert result["predicted_cardinality_histogram"] == {"1": 1, "2": 1}
    assert result["target_cardinality_histogram"] == {"0": 1, "1": 1}


def test_nms_and_all_query_support_are_deterministic():
    predictions, targets = _fixture()
    filtered, counts = _nms(predictions, 0.95)
    assert counts == {
        "selected_before": 3,
        "suppressed": 1,
        "suppressed_fraction": 1 / 3,
        "selected_after": 2,
    }
    assert filtered["confidence"][1, 0] == np.float32(0.99)
    assert filtered["confidence"][1, 1] == np.float32(0.0)
    support = _proposal_support(predictions, targets)
    assert support["target_count"] == 1
    assert support["typed_independent_recall"] == 1.0
    assert support["typed_one_to_one_oracle"]["recall"] == 1.0
    assert support["assignment_conflict_targets"] == 0


def test_uint_mask_population_uses_explicit_integer_cardinality():
    _, targets = _fixture()
    targets["event_identity_index"] = np.full((2, 16), -1, dtype=np.int32)
    targets["event_identity_index"][1, 0] = 7
    result = _population_summary(targets)
    assert result == {
        "worlds": 20,
        "observations": 2,
        "target_tokens": 1,
        "target_types": [1, 0],
        "cardinality": [1, 1, 0, 0, 0, 0],
        "event_identities": 1,
    }
