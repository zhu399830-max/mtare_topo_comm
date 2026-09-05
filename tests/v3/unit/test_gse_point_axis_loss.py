"""Synthetic geometry assignment: no data, labels, relation or checkpoint I/O."""
import inspect

import pytest
import torch

from mtare_topo.representation.gse_point_axis_loss import axis_set_loss, axis_set_metrics


def _case():
    target = torch.tensor([[[[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]],
                             [[0., 4., 0.], [1., 4., 0.], [3., 4., 0.]],
                             [[float("nan")] * 3] * 3]], dtype=torch.float64)
    pred = torch.full((1, 4, 3, 3), 50., dtype=torch.float64)
    pred[0, 0] = target[0, 1]
    pred[0, 2] = target[0, 0]
    mask = torch.tensor([[True, True, False]])
    return pred, target, mask


def test_exact_match_reports_unmatched_queries_without_detector_claims():
    pred, target, mask = _case()
    assert axis_set_loss(pred, target, mask) == 0
    result = axis_set_metrics(pred, target, mask)
    assert len(result) == 1
    row = result[0]
    assert row["n_targets"] == 2 and row["n_predictions"] == 4 and row["unmatched_predictions"] == 2
    assert row["coordinate_mae_m"] == row["point_mean_euclidean_m"] == 0
    assert [(m["prediction_index"], m["target_index"]) for m in row["matching"]] == [(0, 1), (2, 0)]
    assert set(row) == {"n_targets", "n_predictions", "unmatched_predictions", "coordinate_mae_m",
                        "point_mean_euclidean_m", "matching"}


def test_endpoint_reversal_matches_geometry_without_identity():
    pred, target, mask = _case()
    pred = pred.flip(2)
    result = axis_set_metrics(pred, target, mask)
    assert axis_set_loss(pred, target, mask) == 0
    assert all(m["reversed"] for m in result[0]["matching"])


def test_predictions_and_target_permutations_keep_fitting_scores():
    pred, target, mask = _case()
    pred[..., 2] += .6
    expected = axis_set_loss(pred, target, mask)
    altered = axis_set_loss(pred[:, [3, 2, 0, 1]], target[:, [2, 1, 0]], mask[:, [2, 1, 0]])
    torch.testing.assert_close(altered, expected)
    original = axis_set_metrics(pred, target, mask)[0]
    changed = axis_set_metrics(pred[:, [3, 2, 0, 1]], target[:, [2, 1, 0]], mask[:, [2, 1, 0]])[0]
    assert changed["coordinate_mae_m"] == pytest.approx(original["coordinate_mae_m"])
    assert changed["point_mean_euclidean_m"] == pytest.approx(original["point_mean_euclidean_m"])


def test_exact_cost_ties_do_not_add_perturbation_or_change_minimum_loss():
    pred = torch.zeros(1, 3, 3, 3, dtype=torch.float64)
    target = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
    target[0, 0, :, 0] = -1
    target[0, 1, :, 0] = 1
    mask = torch.ones(1, 2, dtype=torch.bool)
    expected = 1 / 150
    assert axis_set_loss(pred, target, mask) == pytest.approx(expected)
    assert axis_set_loss(pred[:, [2, 1, 0]], target.flip(1), mask) == pytest.approx(expected)
    assert axis_set_metrics(pred, target, mask) == axis_set_metrics(pred, target, mask)


def test_loss_averages_observations_not_total_target_population():
    pred = torch.zeros(2, 2, 3, 3, dtype=torch.float64)
    target = torch.zeros_like(pred)
    pred[0] = 1.
    pred[1] = 3.
    mask = torch.tensor([[True, False], [True, True]])
    # Observation errors 1 and 3 -> mean 2, not (1 + 3 + 3) / 3.
    assert axis_set_loss(pred, target, mask) == pytest.approx(2 / 50)


def test_matched_prediction_gradients_are_finite_unmatched_gradients_zero():
    pred, target, mask = _case()
    pred[..., 2] += .5
    pred.requires_grad_()
    target.requires_grad_()
    loss = axis_set_loss(pred, target, mask)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and torch.isfinite(target.grad).all()
    assert pred.grad[0, [0, 2]].abs().sum() > 0
    assert not pred.grad[0, [1, 3]].any()
    assert not target.grad[~mask].any()


def test_metrics_use_coordinate_and_point_distance_not_interchangeably():
    pred, target, mask = _case()
    pred[..., 2] += 3.
    result = axis_set_metrics(pred, target, mask)[0]
    assert result["coordinate_mae_m"] == 1
    assert result["point_mean_euclidean_m"] == 3
    assert axis_set_loss(pred, target, mask) == pytest.approx(1 / 50)


def test_interface_has_no_identity_relation_shape_or_threshold_arguments():
    for function in (axis_set_loss, axis_set_metrics):
        assert list(inspect.signature(function).parameters) == ["pred", "target", "mask"]
        with pytest.raises(TypeError):
            function(*_case(), teacher_identity=[123])


@pytest.mark.parametrize("kind", ["prediction_nan", "target_nan", "prediction_inf", "target_inf", "unmatched_nan"])
def test_any_prediction_or_active_target_nonfinite_fails(kind):
    pred, target, mask = _case()
    value = float("inf") if kind.endswith("inf") else float("nan")
    if kind.startswith("target"):
        target[0, 0, 0, 0] = value
    else:
        pred[0, 1 if kind == "unmatched_nan" else 0, 0, 0] = value
    for function in (axis_set_loss, axis_set_metrics):
        with pytest.raises(ValueError):
            function(pred, target, mask)


@pytest.mark.parametrize("kind", ["uint_mask", "shape_mask", "empty_row", "too_many", "pred_shape", "target_shape", "dtype"])
def test_invalid_shapes_mask_and_population_fail(kind):
    pred, target, mask = _case()
    if kind == "uint_mask":
        mask = mask.to(torch.uint8)
    elif kind == "shape_mask":
        mask = mask[:, :1]
    elif kind == "empty_row":
        mask[:] = False
    elif kind == "too_many":
        pred = pred[:, :1]
    elif kind == "pred_shape":
        pred = pred[..., :2]
    elif kind == "target_shape":
        target = target[..., :2]
    else:
        target = target.float()
    for function in (axis_set_loss, axis_set_metrics):
        with pytest.raises(ValueError):
            function(pred, target, mask)


def test_float32_supported_and_masked_nan_target_does_not_enter_cost():
    pred, target, mask = _case()
    loss = axis_set_loss(pred.float(), target.float(), mask)
    assert loss.dtype == torch.float32 and loss == 0
