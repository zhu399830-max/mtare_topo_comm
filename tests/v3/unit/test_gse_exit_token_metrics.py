from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.evaluation.gse_exit_token_metrics import evaluate_exit_token_sets


def test_exit_token_metrics_match_sets_and_exclude_same_frame_descriptor_candidates() -> None:
    confidence = np.asarray(((0.9, 0.8, 0.1), (0.95, 0.85, 0.1)), dtype=np.float64)
    heading = np.asarray(
        (
            ((0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)),
            ((0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)),
        )
    )
    width = np.asarray(((4.0, 3.0, 1.0), (4.0, 3.0, 1.0)))
    profile = np.zeros((2, 3, 4), dtype=np.float64)
    descriptor = np.asarray(
        (
            ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
            ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
        )
    )
    target_mask = np.asarray(((1, 1, 0), (1, 1, 0)), dtype=np.uint8)
    target_identity = np.asarray(((10, 20, -1), (10, 20, -1)), dtype=np.int64)
    result = evaluate_exit_token_sets(
        confidence=confidence,
        heading_unit=heading,
        opening_width_m=width,
        vertical_profile=profile,
        descriptor=descriptor,
        target_mask=target_mask,
        target_heading_unit=heading,
        target_opening_width_m=width,
        target_width_valid_mask=target_mask,
        target_vertical_profile=profile,
        target_identity=target_identity,
        parent_ids=("world", "world"),
        sequence_order=(0, 1),
    )
    assert result["matched_tokens"] == 4
    assert result["heading_error_deg"]["mean"] == pytest.approx(0.0)
    assert result["opening_width_error_m"]["mean"] == pytest.approx(0.0)
    assert result["opening_width_log_error"]["p95"] == pytest.approx(0.0)
    assert result["vertical_profile_error_m"]["p95"] == pytest.approx(0.0)
    assert result["presence_threshold"]["f1"] == pytest.approx(1.0)
    assert result["count_at_selected_threshold"]["exact_accuracy"] == pytest.approx(1.0)
    assert result["direction_at_selected_threshold"]["f1"] == pytest.approx(1.0)
    assert result["direction_at_selected_threshold"]["mean_matched_angular_error_deg"] == pytest.approx(0.0)
    assert len(result["descriptor_records"]) == 2
    assert all(record["correct"] for record in result["descriptor_records"])


def test_exit_token_metrics_reject_mask_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="masks and identities"):
        evaluate_exit_token_sets(
            confidence=np.asarray(((0.9,),)),
            heading_unit=np.asarray((((0.0, 1.0),),)),
            opening_width_m=np.asarray(((4.0,),)),
            vertical_profile=np.zeros((1, 1, 4)),
            descriptor=np.asarray((((1.0, 0.0),),)),
            target_mask=np.asarray(((0,),)),
            target_heading_unit=np.asarray((((0.0, 1.0),),)),
            target_opening_width_m=np.asarray(((0.0,),)),
            target_width_valid_mask=np.asarray(((0,),)),
            target_vertical_profile=np.zeros((1, 1, 4)),
            target_identity=np.asarray(((3,),)),
            parent_ids=("world",),
            sequence_order=(0,),
        )
