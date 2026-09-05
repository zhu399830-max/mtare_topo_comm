"""Oracle-weighted operator contracts, not learned segmentation success."""
import math

import pytest
import torch

from mtare_topo.representation.gse_point_axis_readout import point_axis_vote


def _inputs(batch=1, slots=2, count=4):
    points = torch.zeros(batch, count, 3, dtype=torch.float64)
    points[..., 0] = torch.arange(count, dtype=torch.float64)
    offsets = torch.zeros_like(points)
    controls = torch.zeros(batch, slots, 3, count, dtype=torch.float64)
    membership = torch.zeros(batch, slots + 1, count, dtype=torch.float64)
    valid = torch.ones(batch, count, dtype=torch.bool)
    return points, offsets, controls, membership, valid


def test_oracle_point_membership_separates_layers_inside_same_azimuth_pool():
    points, offsets, controls, membership, valid = _inputs()
    points[0] = torch.tensor([[-1., 0., -2.], [1., 0., -2.],
                              [-1., 0., 2.], [1., 0., 2.]], dtype=torch.float64)
    membership.fill_(-80.)
    membership[0, 0, :2] = 80.
    membership[0, 1, 2:] = 80.
    result = point_axis_vote(points, offsets, controls, membership, valid)
    torch.testing.assert_close(result.axis_control_m[0, 0, :, 2], torch.full((3,), -2., dtype=torch.float64))
    torch.testing.assert_close(result.axis_control_m[0, 1, :, 2], torch.full((3,), 2., dtype=torch.float64))
    assert result.control_weights.shape == (1, 2, 3, 4)
    assert result.membership_probability.shape == (1, 3, 4)
    torch.testing.assert_close(result.membership_probability.sum(dim=1), torch.ones(1, 4, dtype=torch.float64))


def test_point_to_axis_offsets_escape_one_sided_surface_convex_hull():
    points, offsets, controls, membership, valid = _inputs(slots=1, count=2)
    points[0] = torch.tensor([[-1., 2., 0.], [1., 2., 0.]], dtype=torch.float64)
    raw_result = point_axis_vote(points, offsets, controls, membership, valid)
    offsets[..., 1] = -2.
    shifted = point_axis_vote(points, offsets, controls, membership, valid)
    torch.testing.assert_close(raw_result.axis_control_m[..., 1], torch.full((1, 1, 3), 2., dtype=torch.float64))
    torch.testing.assert_close(shifted.axis_control_m, torch.zeros(1, 1, 3, 3, dtype=torch.float64))


def test_axis_loss_has_nonzero_finite_offset_gradient():
    values = list(_inputs(slots=1))
    values[1].requires_grad_()
    result = point_axis_vote(*values)
    loss = (result.axis_control_m - 2.).square().mean()
    loss.backward()
    gradient = values[1].grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def _nonuniform():
    generator = torch.Generator().manual_seed(831)
    points, offsets, controls, membership, valid = _inputs(batch=2, slots=2, count=7)
    for value in (points, offsets, controls, membership):
        value.copy_(torch.randn(value.shape, generator=generator, dtype=value.dtype))
    valid[1, 6] = False
    return points, offsets, controls, membership, valid


def test_point_permutation_preserves_geometry_and_permutates_point_fields():
    points, offsets, controls, membership, valid = _nonuniform()
    order = torch.tensor([6, 2, 5, 0, 3, 1, 4])
    result = point_axis_vote(points, offsets, controls, membership, valid)
    changed = point_axis_vote(points[:, order], offsets[:, order], controls[..., order], membership[..., order], valid[:, order])
    for key in ("axis_control_m", "vote_variance_m2", "effective_points"):
        torch.testing.assert_close(getattr(changed, key), getattr(result, key))
    torch.testing.assert_close(changed.control_weights, result.control_weights[..., order])
    torch.testing.assert_close(changed.membership_probability, result.membership_probability[..., order])
    assert torch.equal(changed.direction_defined, result.direction_defined)


def test_slot_permutation_moves_foreground_but_keeps_background_last():
    points, offsets, controls, membership, valid = _nonuniform()
    result = point_axis_vote(points, offsets, controls, membership, valid)
    changed = point_axis_vote(points, offsets, controls[:, [1, 0]], membership[:, [1, 0, 2]], valid)
    for key in ("axis_control_m", "vote_variance_m2", "effective_points", "control_weights"):
        torch.testing.assert_close(getattr(changed, key), getattr(result, key)[:, [1, 0]])
    torch.testing.assert_close(changed.membership_probability, result.membership_probability[:, [1, 0, 2]])
    assert torch.equal(changed.direction_defined, result.direction_defined[:, [1, 0]])


def test_operator_rigid_transform_rotates_offsets_without_translating_them():
    points, offsets, controls, membership, valid = _nonuniform()
    cosine, sine = math.cos(.6), math.sin(.6)
    rotation = torch.tensor([[cosine, 0., sine], [0., 1., 0.], [-sine, 0., cosine]], dtype=torch.float64)
    translation = torch.tensor([13., -4., 2.], dtype=torch.float64)
    result = point_axis_vote(points, offsets, controls, membership, valid)
    changed = point_axis_vote(points @ rotation.T + translation, offsets @ rotation.T, controls, membership, valid)
    torch.testing.assert_close(changed.axis_control_m, result.axis_control_m @ rotation.T + translation)
    for key in ("control_weights", "membership_probability", "vote_variance_m2", "effective_points"):
        torch.testing.assert_close(getattr(changed, key), getattr(result, key))
    assert torch.equal(changed.direction_defined, result.direction_defined)


def test_background_reduces_relative_weight_of_its_point_without_a_hard_gate():
    points, offsets, controls, membership, valid = _inputs(slots=1, count=2)
    original = point_axis_vote(points, offsets, controls, membership, valid)
    membership[0, -1, 0] = 80.
    changed = point_axis_vote(points, offsets, controls, membership, valid)
    assert original.control_weights[0, 0, 0, 0] == .5
    assert changed.control_weights[0, 0, 0, 0] < 1e-30
    assert changed.control_weights[0, 0, 0, 1] > .999999
    assert changed.membership_probability[0, -1, 0] > .999999


def test_uniform_background_dominance_is_not_misreported_as_an_existence_decision():
    points, offsets, controls, membership, valid = _inputs(slots=1, count=2)
    original = point_axis_vote(points, offsets, controls, membership, valid)
    membership[:, -1] = 80.
    changed = point_axis_vote(points, offsets, controls, membership, valid)
    torch.testing.assert_close(changed.control_weights, original.control_weights)
    assert changed.membership_probability[:, 0].max() < 1e-30
    # Conditional weights still sum to 1. A later graph interface must retain low membership confidence.
    torch.testing.assert_close(changed.control_weights.sum(dim=-1), torch.ones(1, 1, 3, dtype=torch.float64))


def test_invalid_nan_points_offsets_and_logits_never_pollute_outputs_or_gradients():
    values = list(_inputs(slots=1, count=3))
    values[-1][0, -1] = False
    for index in (0, 1):
        values[index][0, -1] = float("nan")
    for index in (2, 3):
        values[index][..., -1] = float("nan")
    for value in values[:-1]:
        value.requires_grad_()
    result = point_axis_vote(*values)
    for key in ("axis_control_m", "control_weights", "membership_probability", "vote_variance_m2", "effective_points"):
        assert torch.isfinite(getattr(result, key)).all(), key
    assert not result.control_weights[..., -1].any()
    loss = result.axis_control_m.square().sum() + result.vote_variance_m2.sum() + result.membership_probability.square().sum()
    loss.backward()
    for index, value in enumerate(values[:-1]):
        assert value.grad is not None and torch.isfinite(value.grad).all(), index
        invalid_gradient = value.grad[0, -1] if index in (0, 1) else value.grad[..., -1]
        assert not invalid_gradient.any(), index


@pytest.mark.parametrize("field", [0, 1, 2, 3])
def test_nonfinite_fields_on_valid_points_are_rejected(field):
    values = list(_inputs())
    values[field].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        point_axis_vote(*values)


def test_batch_with_an_empty_observation_is_rejected():
    values = list(_inputs(batch=2))
    values[-1][1] = False
    with pytest.raises(ValueError):
        point_axis_vote(*values)


def test_degenerate_controls_are_retained_and_direction_is_marked_undefined():
    values = _inputs(slots=1, count=2)
    result = point_axis_vote(*values)
    assert result.axis_control_m.shape == (1, 1, 3, 3)
    assert result.direction_defined.shape == (1, 1)
    assert result.direction_defined.dtype == torch.bool
    assert not result.direction_defined.any()


def test_control_variance_and_effective_sample_size_have_analytic_values():
    points, offsets, controls, membership, valid = _inputs(slots=1, count=2)
    points[0, :, 0] = torch.tensor([0., 2.], dtype=torch.float64)
    controls[..., 1] = math.log(3.)
    result = point_axis_vote(points, offsets, controls, membership, valid)
    torch.testing.assert_close(result.axis_control_m[..., 0], torch.full((1, 1, 3), 1.5, dtype=torch.float64))
    torch.testing.assert_close(result.control_weights, torch.tensor([[[[.25, .75]] * 3]], dtype=torch.float64))
    torch.testing.assert_close(result.vote_variance_m2, torch.full((1, 1, 3), .75, dtype=torch.float64))
    torch.testing.assert_close(result.effective_points, torch.full((1, 1, 3), 1.6, dtype=torch.float64))


def test_one_observed_point_can_vote_for_two_candidate_axes_without_hard_identity():
    points, _, controls, membership, valid = _inputs(slots=2, count=1)
    points[0, 0] = torch.tensor((0., 2., 0.), dtype=torch.float64)
    offsets = torch.zeros(1, 2, 1, 3, dtype=torch.float64)
    offsets[0, 0, 0, 1] = -2.
    offsets[0, 1, 0, 2] = 3.
    result = point_axis_vote(points, offsets, controls, membership, valid)
    torch.testing.assert_close(result.axis_control_m[0, 0], torch.zeros(3, 3, dtype=torch.float64))
    torch.testing.assert_close(result.axis_control_m[0, 1],
                               torch.tensor([[0., 2., 3.]] * 3, dtype=torch.float64))
    assert not result.direction_defined.any()


def test_broadcast_shared_offsets_preserve_existing_operator_results():
    points, offsets, controls, membership, valid = _nonuniform()
    shared = point_axis_vote(points, offsets, controls, membership, valid)
    per_slot = point_axis_vote(points, offsets[:, None].expand(-1, controls.shape[1], -1, -1),
                               controls, membership, valid)
    for key in ("axis_control_m", "control_weights", "membership_probability",
                "vote_variance_m2", "effective_points", "membership_support"):
        torch.testing.assert_close(getattr(per_slot, key), getattr(shared, key))
    assert torch.equal(per_slot.direction_defined, shared.direction_defined)


def test_slot_permutation_must_also_permute_candidate_specific_offsets():
    points, shared, controls, membership, valid = _nonuniform()
    offsets = torch.stack((shared, shared + torch.tensor([0., 3., 1.], dtype=torch.float64)), dim=1)
    result = point_axis_vote(points, offsets, controls, membership, valid)
    changed = point_axis_vote(points, offsets[:, [1, 0]], controls[:, [1, 0]],
                              membership[:, [1, 0, 2]], valid)
    for key in ("axis_control_m", "control_weights", "vote_variance_m2", "effective_points", "membership_support"):
        torch.testing.assert_close(getattr(changed, key), getattr(result, key)[:, [1, 0]])
    torch.testing.assert_close(changed.membership_probability, result.membership_probability[:, [1, 0, 2]])


def test_rigid_transform_equivariance_holds_for_per_slot_vector_offsets():
    points, shared, controls, membership, valid = _nonuniform()
    offsets = torch.stack((shared, -2. * shared), dim=1)
    cosine, sine = math.cos(.4), math.sin(.4)
    rotation = torch.tensor([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]], dtype=torch.float64)
    translation = torch.tensor([2., -7., 4.], dtype=torch.float64)
    result = point_axis_vote(points, offsets, controls, membership, valid)
    changed = point_axis_vote(points @ rotation.T + translation, offsets @ rotation.T,
                              controls, membership, valid)
    torch.testing.assert_close(changed.axis_control_m, result.axis_control_m @ rotation.T + translation)
    for key in ("control_weights", "membership_probability", "vote_variance_m2", "effective_points"):
        torch.testing.assert_close(getattr(changed, key), getattr(result, key))


def test_invalid_nan_candidate_specific_offsets_never_pollute_output_or_gradient():
    points, shared, controls, membership, valid = _inputs(slots=2, count=3)
    offsets = shared[:, None].expand(-1, 2, -1, -1).clone()
    valid[0, -1] = False
    offsets[0, :, -1] = float("nan")
    offsets.requires_grad_()
    result = point_axis_vote(points, offsets, controls, membership, valid)
    assert torch.isfinite(result.axis_control_m).all()
    assert torch.isfinite(result.vote_variance_m2).all()
    result.axis_control_m.square().sum().backward()
    assert torch.isfinite(offsets.grad).all()
    assert not offsets.grad[0, :, -1].any()
    assert offsets.grad[0, :, :-1].abs().sum() > 0


@pytest.mark.parametrize("shape", [(1, 2, 4, 2), (1, 3, 4, 3), (1, 1, 4, 3), (1, 2, 3, 4)])
def test_wrong_candidate_offset_shape_is_rejected(shape):
    points, _, controls, membership, valid = _inputs(slots=2, count=4)
    with pytest.raises(ValueError):
        point_axis_vote(points, torch.zeros(shape, dtype=torch.float64), controls, membership, valid)


def test_nonfinite_candidate_offset_on_valid_point_is_rejected():
    points, _, controls, membership, valid = _inputs(slots=2, count=4)
    offsets = torch.zeros(1, 2, 4, 3, dtype=torch.float64)
    offsets[0, 1, 1, 2] = float("nan")
    with pytest.raises(ValueError):
        point_axis_vote(points, offsets, controls, membership, valid)
