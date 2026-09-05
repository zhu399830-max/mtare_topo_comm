import numpy as np
import pytest
import torch

from mtare_topo.evaluation.primitive_composition_anchor_failure_attribution import (
    binned_error_summary,
    categorical_error_summary,
    diagnose_attribution,
    symmetric_negative_pair_distance,
)


def test_symmetric_negative_pair_distance_is_exact_and_nearest_first():
    points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    score = symmetric_negative_pair_distance(points)
    assert torch.equal(score, score.transpose(1, 2))
    assert score[0, 0, 1] > score[0, 0, 2]
    with pytest.raises(ValueError):
        symmetric_negative_pair_distance(torch.zeros(2, 3))


def test_binned_and_categorical_error_summaries_preserve_counts_and_improvement():
    raw = np.asarray([2.0, 4.0, 10.0])
    anchor = np.asarray([1.0, 3.0, 8.0])
    bins = binned_error_summary(raw, anchor, np.asarray([1.0, 6.0, 12.0]), np.asarray([0, 5, 10, 20]))
    assert [row["count"] for row in bins] == [1, 1, 1]
    assert bins[0]["relative_improvement"] == 0.5
    categories = categorical_error_summary(raw, anchor, np.asarray([0, 1, 1]), {0: "a", 1: "b"})
    assert [row["count"] for row in categories] == [1, 2]


def _seed(teacher=True, predicted=False, gaussian=True, evidence=True):
    def condition(value):
        return {"safe": {"true_positive": 1 if value else 0}}
    return {"conditions": {
        "proposal_oracle_teacher_anchor_distance": condition(teacher),
        "proposal_oracle_teacher_anchor_gaussian": condition(gaussian),
        "proposal_oracle_teacher_anchor_safe": condition(evidence),
        "proposal_oracle_predicted_anchor_distance": condition(predicted),
        "proposal_oracle_model_compatibility": condition(predicted),
        "deployed_model_safe": condition(False),
    }}


def test_diagnosis_selects_coordinate_bottleneck_before_calibration():
    result = diagnose_attribution([_seed(), _seed(), _seed(teacher=False)])
    assert result["diagnosis"] == "PREDICTED_COMPOSITION_ANCHOR_COORDINATES_ARE_PRIMARY_BOTTLENECK"
    assert result["safe_passing_seeds"]["teacher_anchor_distance"] == 2


def test_diagnosis_stops_invalid_teacher_before_model_changes():
    result = diagnose_attribution([_seed(teacher=False) for _ in range(3)])
    assert result["decision"] == "STOP_COMPOSITION_ANCHOR_TARGET_AND_REASSESS_TEACHER"
