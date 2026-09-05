from dataclasses import replace

import pytest
import torch

from mtare_topo.representation.gse_composition import (
    CompositionInput, GeometricCompositionHead, masked_structure_losses, shared_frame_features,
    paired_structure_consistency,
)


def fixture():
    p = torch.tensor([[[2., 0., 0.], [-2., 0., 0.], [0., 2., 0.]]])
    return CompositionInput(p, p / 2, torch.ones(1, 3, 2), torch.ones(1, 3),
                            torch.ones(1, 3), torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool))


def test_common_yaw_preserves_features_and_structure():
    value = fixture()
    angle = torch.tensor(.73)
    c, s = angle.cos(), angle.sin()
    rotation = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    rotated = replace(value, positions_m=value.positions_m @ rotation.T, tangents=value.tangents @ rotation.T)
    first, second = shared_frame_features(value), shared_frame_features(rotated)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    torch.manual_seed(0)
    model = GeometricCompositionHead().eval()
    torch.testing.assert_close(model(value).event_logits, model(rotated).event_logits)


def test_permutation_preserves_events_and_permutes_ports():
    value, order = fixture(), torch.tensor([2, 0, 1])
    shuffled = CompositionInput(**{k: v[:, order] for k, v in value.__dict__.items()})
    model = GeometricCompositionHead().eval()
    a, b = model(value), model(shuffled)
    torch.testing.assert_close(a.event_logits, b.event_logits)
    torch.testing.assert_close(a.port_membership_logits[:, order], b.port_membership_logits)


def test_branch_angle_survives_individual_radial_frames():
    # Every individual token has identical range, height and radial tangent,
    # but the 60-degree and 90-degree branch layouts MUST be distinguishable.
    original = fixture()
    p = original.positions_m.clone()
    p[0, 2] = torch.tensor([1., 3.**.5, 0.])
    changed = replace(original, positions_m=p, tangents=p / 2)
    a, b = shared_frame_features(original), shared_frame_features(changed)
    torch.testing.assert_close(a[0], b[0])
    assert not torch.allclose(a[1], b[1])


def test_stacked_nonincident_geometry_is_distinct():
    original = fixture()
    p = original.positions_m.clone()
    p[0, 2, 2] = 4
    a, b = shared_frame_features(original), shared_frame_features(replace(original, positions_m=p))
    assert b[1][0, 0, 2, 1] == pytest.approx(4 / 50)
    assert not torch.allclose(a[1], b[1])


def test_unknown_and_padding_do_not_create_structure():
    value = fixture()
    model = GeometricCompositionHead().eval()
    empty = replace(value, valid=torch.zeros_like(value.valid))
    result = model(empty)
    assert not result.supported.any()
    assert torch.equal(result.uncertainty, torch.ones(1))
    assert torch.equal(result.event_logits, torch.zeros(1, 3))
    assert torch.isfinite(result.port_membership_logits).all()


def test_masked_padding_geometry_cannot_change_prediction():
    value = replace(fixture(), valid=torch.tensor([[True, True, False]]))
    p = value.positions_m.clone()
    p[0, 2] = torch.tensor([40., -50., 100.])
    model = GeometricCompositionHead().eval()
    torch.testing.assert_close(model(value).event_logits, model(replace(value, positions_m=p)).event_logits)


def test_repeat_and_exact_symmetry_are_not_identity_signals():
    value = fixture()
    model = GeometricCompositionHead().eval()
    a, b = model(value), model(value)
    assert torch.equal(a.event_logits, b.event_logits)
    assert torch.equal(a.port_membership_logits, b.port_membership_logits)
    # No identity, world ID, or teacher grouping exists in the forward schema.
    assert set(value.__dict__) == {"positions_m", "tangents", "half_axes_m", "shape_exponent",
                                   "confidence", "uncertainty", "valid"}


def test_unknown_targets_are_not_negatives():
    model = GeometricCompositionHead()
    prediction = model(fixture())
    losses = masked_structure_losses(prediction, torch.tensor([-999]), torch.tensor([False]),
                                    torch.full((1, 3), float("nan")), torch.zeros(1, 3, dtype=torch.bool))
    assert losses["total"].item() == 0
    losses["total"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_event_and_port_gradients_reach_relational_encoder():
    model = GeometricCompositionHead()
    losses = masked_structure_losses(model(fixture()), torch.tensor([1]), torch.tensor([True]),
                                    torch.tensor([[1., 1., 0.]]), torch.ones(1, 3, dtype=torch.bool))
    losses["total"].backward()
    for group in (model.pair_encoder, model.endpoint_encoder, model.event_head, model.port_head, model.uncertainty_head):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters())


@pytest.mark.parametrize("field,value", [("confidence", -1.), ("uncertainty", 2.), ("half_axes_m", 0.)])
def test_invalid_input_refused(field, value):
    original = fixture()
    with pytest.raises(ValueError):
        shared_frame_features(replace(original, **{field: torch.full_like(getattr(original, field), value)}))


def test_realization_consistency_uses_only_common_visible_structure():
    model = GeometricCompositionHead()
    value = fixture()
    first = model(value)
    second = model(replace(value, half_axes_m=value.half_axes_m * 2))
    pairs = torch.tensor([[[0, 0], [-1, -1], [2, 2]]])
    # Changed dimensions do not themselves incur a size-matching loss.
    losses = paired_structure_consistency(first, second, pairs,
                                          torch.tensor([[True, False, True]]), torch.tensor([True]))
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_realization_unknown_targets_and_masked_ports_not_penalized():
    prediction = GeometricCompositionHead()(fixture())
    losses = paired_structure_consistency(prediction, prediction,
                                          torch.full((1, 3, 2), -1, dtype=torch.long),
                                          torch.zeros(1, 3, dtype=torch.bool), torch.tensor([False]))
    assert losses["total"].item() == 0


def test_invalid_known_realization_pair_rejected():
    prediction = GeometricCompositionHead()(fixture())
    with pytest.raises(ValueError, match="outside"):
        paired_structure_consistency(prediction, prediction, torch.tensor([[[-1, 0]]]),
                                     torch.tensor([[True]]), torch.tensor([True]))


def test_invalid_known_targets_rejected():
    prediction = GeometricCompositionHead()(fixture())
    with pytest.raises(ValueError, match="known event"):
        masked_structure_losses(prediction, torch.tensor([3]), torch.tensor([True]),
                                torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool))


def test_teacher_port_mask_cannot_resurrect_unsupported_predictions():
    value = replace(fixture(), valid=torch.tensor([[True, True, False]]))
    prediction = GeometricCompositionHead()(value)
    a = masked_structure_losses(prediction, torch.tensor([1]), torch.tensor([True]),
                                torch.tensor([[1., 1., float("nan")]]), torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(a["total"])
