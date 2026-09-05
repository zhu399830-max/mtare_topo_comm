from types import SimpleNamespace

import numpy as np
import torch

from mtare_topo.evaluation.primitive_predicted_geometry_association import (
    predicted_geometry_association_slice,
)


def _fixture():
    axis = torch.zeros(1, 32, 3, 3)
    # endpoint 1 of primitive 0 and endpoint 0 of primitive 1 coincide.
    axis[0, 0, 2] = torch.tensor((1.0, 0.0, 0.0))
    axis[0, 1, 0] = torch.tensor((1.0, 0.0, 0.0))
    axis[0, 2] = 10.0
    attachment = torch.zeros(1, 32, 2, 32, 2, dtype=torch.bool)
    attachment[0, 0, 1, 1, 0] = True
    observed = torch.zeros(1, 32, 2, dtype=torch.bool)
    observed[0, 0, 1] = True; observed[0, 1, 0] = True
    mask = torch.zeros(1, 32, dtype=torch.bool); mask[0, :2] = True
    overlap = torch.zeros(1, 32, 32, dtype=torch.bool)
    overlap[0, 0, 2] = overlap[0, 2, 0] = True
    logits = torch.full((1, 32, 2, 32, 2), -4.0)
    logits[0, 0, 1, 1, 0] = 4.0
    prediction = SimpleNamespace(
        axis_control_current_sensor_m=axis,
        endpoint_attachment_logits=logits,
    )
    aligned = {
        "attachment": attachment, "endpoint_observed": observed,
        "mask": mask, "overlap": overlap,
    }
    return prediction, aligned, mask


def test_geometry_score_recovers_coincident_observable_endpoints():
    prediction, aligned, mask = _fixture()
    result = predicted_geometry_association_slice(prediction, aligned, mask)
    assert result.all_observable_target_pairs == 1
    assert result.eligible_pairs == 1
    assert result.target.tolist() == [True]
    assert result.geometry_score.tolist() == [0.0]
    assert result.learned_pair_score[0] > .98


def test_redundant_proposal_is_retained_as_negative():
    prediction, aligned, mask = _fixture()
    oracle = mask.clone()
    deployed = mask.clone(); deployed[0, 2] = True
    first = predicted_geometry_association_slice(prediction, aligned, oracle)
    second = predicted_geometry_association_slice(prediction, aligned, deployed)
    assert first.eligible_pairs == 1
    assert second.eligible_pairs > first.eligible_pairs
    assert second.all_observable_target_pairs == 1
    assert np.any(second.overlap_hard_negative)


def test_hidden_matched_pair_is_unknown_not_negative():
    prediction, aligned, mask = _fixture()
    aligned["endpoint_observed"][0, 1, 0] = False
    result = predicted_geometry_association_slice(prediction, aligned, mask)
    assert result.all_observable_target_pairs == 0
    assert result.eligible_pairs == 0
