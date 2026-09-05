from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.gse_validation_evaluator import calibrate_validation_outputs


def _archive() -> dict[str, np.ndarray]:
    frames, queries = 10, 2
    headings = np.tile(np.asarray((((0.0, 1.0), (1.0, 0.0)),)), (frames, 1, 1))
    event_labels = np.tile(np.arange(5, dtype=np.int64), 2)
    event_logits = np.full((frames, 5), -5.0)
    event_logits[np.arange(frames), event_labels] = 5.0
    descriptors = np.eye(5, dtype=np.float64)[event_labels]
    exit_descriptors = np.tile(np.asarray((((1.0, 0.0), (0.0, 1.0)),)), (frames, 1, 1))
    exit_identity = np.tile(np.asarray(((100, 200),)), (frames, 1))
    return {
        "event_logits": event_logits,
        "uncertainty": np.full(frames, 0.05),
        "width_m": np.linspace(4.0, 4.5, frames),
        "height_m": np.linspace(3.0, 3.5, frames),
        "slope_deg": np.linspace(0.0, 2.5, frames),
        "curvature_per_m": np.linspace(0.0, 0.01, frames),
        "place_descriptor": descriptors,
        "exit_confidence": np.full((frames, queries), 0.9),
        "exit_heading_unit": headings,
        "exit_opening_width_m": np.full((frames, queries), 4.0),
        "exit_vertical_profile": np.zeros((frames, queries, 4)),
        "exit_descriptor": exit_descriptors,
        "target_event_index": event_labels,
        "target_width_m": np.linspace(4.1, 4.6, frames),
        "target_height_m": np.linspace(3.1, 3.6, frames),
        "target_slope_deg": np.linspace(0.1, 2.6, frames),
        "target_curvature_per_m": np.linspace(0.001, 0.011, frames),
        "target_geometry_valid_mask": np.ones((frames, 4), dtype=np.uint8),
        "target_association_identity": event_labels + 10,
        "target_association_valid_mask": np.ones(frames, dtype=np.uint8),
        "target_exit_mask": np.ones((frames, queries), dtype=np.uint8),
        "target_exit_heading_unit": headings,
        "target_exit_opening_width_m": np.full((frames, queries), 4.0),
        "target_exit_width_valid_mask": np.ones((frames, queries), dtype=np.uint8),
        "target_exit_vertical_profile": np.zeros((frames, queries, 4)),
        "target_exit_identity": exit_identity,
        "global_sequence_index": np.arange(frames),
        "parent_id": np.asarray(("a",) * frames),
        "observation_id": np.asarray(tuple(f"o{index}" for index in range(frames))),
    }


def test_validation_evaluator_calibrates_all_online_interfaces() -> None:
    result = calibrate_validation_outputs(_archive())
    assert result["validation_frames"] == 10
    assert result["event"]["rejection_selection"]["macro_f1"] == 1.0
    class_metrics = result["event"]["selected_class_metrics"]
    assert class_metrics["selection_effect"] == "NONE_REPLAY_OF_FROZEN_EVENT_POINT"
    assert class_metrics["macro_f1"] == result["event"]["rejection_selection"]["macro_f1"]
    assert class_metrics["per_class"]["corridor"]["f1"] == 1.0
    assert class_metrics["per_class"]["junction"]["f1"] == 1.0
    assert sum(sum(row) for row in class_metrics["confusion_matrix_truth_rows_prediction_columns"]) == 10
    diagnostic = result["uncertainty_diagnostic"]
    assert diagnostic["selection_effect"] == "NONE"
    assert diagnostic["frames"] == 10
    assert diagnostic["bins"] == 10
    assert sum(row["count"] for row in diagnostic["curve"]) == 10
    assert diagnostic["weighted_absolute_calibration_gap"] >= 0.0
    per_parent = result["per_parent_diagnostic"]
    assert per_parent["selection_effect"].startswith("NONE_REPLAY")
    assert len(per_parent["parents"]) == 1
    assert per_parent["parents"][0]["parent_id"] == "a"
    assert per_parent["parents"][0]["frames"] == 10
    assert per_parent["parents"][0]["event"]["macro_f1"] == 1.0
    assert set(per_parent["parents"][0]["geometry_mae"]) == {
        "width_m",
        "height_m",
        "slope_deg",
        "curvature_per_m",
    }
    assert result["place_association"]["selection"]["precision"] == 1.0
    assert result["exit_tokens"]["presence_threshold"]["f1"] == 1.0
    assert result["exit_tokens"]["descriptor_association"]["selection"]["precision"] == 1.0
    assert result["strict_test_worlds_read"] == 0


def test_validation_evaluator_rejects_missing_archive_field() -> None:
    arrays = _archive()
    del arrays["event_logits"]
    try:
        calibrate_validation_outputs(arrays)
    except ValueError as error:
        assert "missing arrays" in str(error)
    else:
        raise AssertionError("missing validation output was accepted")


def test_uncertainty_diagnostic_rejects_empty_geometry_mask() -> None:
    arrays = _archive()
    arrays["target_geometry_valid_mask"][2] = 0
    try:
        calibrate_validation_outputs(arrays)
    except ValueError as error:
        assert "geometry uncertainty diagnostic" in str(error)
    else:
        raise AssertionError("sample without a geometry target was accepted")


def test_selected_event_class_metrics_rejects_threshold_drift() -> None:
    arrays = _archive()
    try:
        from mtare_topo.evaluation.gse_validation_evaluator import selected_event_class_metrics

        selected_event_class_metrics(
            arrays["event_logits"],
            arrays["target_event_index"],
            arrays["uncertainty"],
            temperature=1.0,
            threshold=1.1,
        )
    except ValueError as error:
        assert "frozen contract" in str(error)
    else:
        raise AssertionError("out-of-range event threshold was accepted")
