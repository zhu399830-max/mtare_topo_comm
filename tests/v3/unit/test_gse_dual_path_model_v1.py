"""Synthetic information-flow tests, not training or detection qualification."""
from dataclasses import fields, replace
import inspect

import pytest
import torch

from mtare_topo.representation.gse_dual_path_model_v1 import (
    DIM, EVENT_ORDER, DualPathModelV1, DualPathPrediction, PrimitiveGeometry,
)


def model(path="C"):
    torch.manual_seed(19)
    return DualPathModelV1(path).double().eval()


def observation(batch=2, count=11):
    generator = torch.Generator().manual_seed(77)
    return (torch.randn(batch, count, 3, generator=generator, dtype=torch.float64) * 4.,
            torch.randn(batch, count, DIM, generator=generator, dtype=torch.float64),
            torch.ones(batch, count, dtype=torch.bool))


def tensors(prediction):
    for field in fields(prediction):
        value = getattr(prediction, field.name)
        if isinstance(value, PrimitiveGeometry):
            yield from tensors(value)
        else:
            yield value


def assert_close_predictions(a, b):
    for x, y in zip(tensors(a), tensors(b)):
        torch.testing.assert_close(x, y, atol=2e-9, rtol=2e-9)


def structure_loss(prediction):
    return (prediction.structure_position_m.square().mean() / 2500. +
            prediction.structure_event_logits.square().mean() +
            prediction.portal_position_m.square().mean() / 2500. +
            prediction.portal_membership_probability[..., 0].mean())


def adapter_gradient(m):
    gradients = [p.grad for p in m.shared_point_adapter.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    return sum(float(g.abs().sum()) for g in gradients)


@pytest.mark.parametrize("path", ["A", "B", "C"])
def test_shapes_all_candidates_positive_dimensions_and_unknown_channel(path):
    m = model(path)
    p = m(*observation())
    assert p.primitives.axis_control_m.shape == (2, 32, 3, 3)
    assert p.primitives.half_axes_m.shape == (2, 32, 2, 2)
    assert p.primitives.exponent.shape == (2, 32, 2)
    assert p.structure_position_m.shape == (2, 32, 3)
    assert p.structure_event_logits.shape == (2, 32, 4)
    assert p.portal_position_m.shape == (2, 64, 3)
    assert p.portal_dimensions_m.shape == (2, 64, 2)
    assert p.portal_dimension_evidence_probability.shape == (2, 64, 2)
    assert p.portal_membership_probability.shape == (2, 64, 33)
    assert EVENT_ORDER[-1] == "no_object"
    assert p.observation_supported.all()
    assert (p.portal_dimensions_m > 0).all()
    assert (p.primitives.half_axes_m > 0).all()
    assert ((p.primitives.exponent >= 2) & (p.primitives.exponent <= 10)).all()
    torch.testing.assert_close(p.structure_presence_probability, 1-p.structure_event_logits.softmax(-1)[..., 3])
    torch.testing.assert_close(p.portal_membership_probability.sum(-1), torch.ones(2, 64).double())
    for value in tensors(p):
        assert torch.isfinite(value).all()
    for probability in (p.primitives.confidence_probability, p.structure_presence_probability,
                        p.portal_presence_probability, p.portal_dimension_evidence_probability):
        assert ((probability >= 0) & (probability <= 1)).all()
    assert len(m.primitive_decoder.layers) == len(m.structure_decoder.layers) == 2
    assert m.structure_decoder.layers[0].multihead_attn.num_heads == 4
    assert m.structure_queries.shape == (32, 128)
    assert m.portal_queries.shape == (64, 128)


def test_ab_same_initialization_layout_and_full_forward():
    a, b, c = model("A"), model("B"), model("C")
    assert a.state_dict().keys() == b.state_dict().keys() == c.state_dict().keys()
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
        assert torch.equal(value, c.state_dict()[key])
    inputs = observation()
    assert_close_predictions(a(*inputs), b(*inputs))


@pytest.mark.parametrize("path", ["A", "B"])
def test_ab_structure_never_reads_geometry_prediction(path):
    m = model(path)
    points, feature, valid, support, centroid = m.encode_observation(*observation())
    geometry = m.predict_geometry(points, feature, valid, support)
    baseline = m.decode_structure(feature, valid, support, centroid, geometry)
    # Deliberately invalid geometry fields: A/B must not inspect these at all.
    changed = replace(geometry, axis_control_m=torch.full_like(geometry.axis_control_m, float("nan")))
    p = m.decode_structure(feature, valid, support, centroid, changed)
    assert torch.equal(baseline.structure_event_logits, p.structure_event_logits)
    assert torch.equal(baseline.portal_membership_probability, p.portal_membership_probability)
    assert torch.equal(baseline.structure_position_m, p.structure_position_m)


@pytest.mark.parametrize("path", ["A", "B", "C"])
def test_structure_gradient_reaches_shared_adapter_not_frozen_context(path):
    m = model(path)
    xyz, context, valid = observation()
    context.requires_grad_(True)
    structure_loss(m(xyz, context, valid)).backward()
    assert adapter_gradient(m) > 0
    assert context.grad is None


def test_b_auxiliary_axis_loss_reaches_same_shared_adapter():
    m = model("B")
    p = m(*observation())
    (p.primitives.axis_control_m.square().mean() / 2500. +
        p.primitives.half_axes_m.square().mean()).backward()
    assert adapter_gradient(m) > 0
    assert m.axis_readout.offset.weight.grad.abs().sum() > 0
    assert m.structure_event.weight.grad is None


@pytest.mark.parametrize("path", ["A", "B", "C"])
def test_default_float32_forward_backward_is_finite(path):
    m = model(path).float()
    xyz, context, valid = observation()
    p = m(xyz.float(), context.float(), valid)
    (structure_loss(p) + p.primitives.axis_control_m.square().mean() / 2500.).backward()
    assert adapter_gradient(m) > 0
    assert all(torch.isfinite(v).all() for v in tensors(p))


@pytest.mark.parametrize("path", ["A", "B", "C"])
def test_point_set_permutation_invariance(path):
    m = model(path)
    xyz, context, valid = observation()
    permutation = torch.tensor([5, 1, 10, 8, 2, 0, 3, 9, 4, 7, 6])
    assert_close_predictions(m(xyz, context, valid),
        m(xyz[:, permutation], context[:, permutation], valid[:, permutation]))


def test_c_predicted_primitive_set_permutation_invariance():
    m = model()
    points, feature, valid, support, centroid = m.encode_observation(*observation())
    geometry = m.predict_geometry(points, feature, valid, support)
    permutation = torch.randperm(32)
    shuffled = PrimitiveGeometry(**{field.name: getattr(geometry, field.name)[:, permutation]
                                  for field in fields(geometry)})
    base = m.decode_structure(feature, valid, support, centroid, geometry)
    other = m.decode_structure(feature, valid, support, centroid, shuffled)
    for field in fields(base):
        if field.name != "primitives":
            torch.testing.assert_close(getattr(base, field.name), getattr(other, field.name), atol=2e-9, rtol=2e-9)


def test_permuting_primitive_queries_permutes_geometry_not_structure():
    m = model()
    inputs = observation()
    base = m(*inputs)
    permutation = torch.randperm(32)
    with torch.no_grad():
        m.primitive_queries.copy_(m.primitive_queries[permutation].clone())
    other = m(*inputs)
    for field in fields(base.primitives):
        torch.testing.assert_close(getattr(base.primitives, field.name)[:, permutation],
            getattr(other.primitives, field.name), atol=2e-9, rtol=2e-9)
    for field in fields(base):
        if field.name != "primitives":
            torch.testing.assert_close(getattr(base, field.name), getattr(other, field.name), atol=2e-9, rtol=2e-9)


def test_c_geometry_intervention_changes_prediction_and_raw_bypass_survives():
    m = model()
    points, feature, valid, support, centroid = m.encode_observation(*observation())
    geometry = m.predict_geometry(points, feature, valid, support)
    base = m.decode_structure(feature, valid, support, centroid, geometry)
    moved = replace(geometry, axis_control_m=geometry.axis_control_m + 20.)
    assert not torch.allclose(base.structure_event_logits,
        m.decode_structure(feature, valid, support, centroid, moved).structure_event_logits)
    low_confidence = replace(geometry, confidence_probability=torch.zeros_like(geometry.confidence_probability))
    raw1 = m.decode_structure(feature, valid, support, centroid, low_confidence)
    raw2 = m.decode_structure(feature * 1.1, valid, support, centroid, low_confidence)
    assert not torch.allclose(raw1.structure_event_logits, raw2.structure_event_logits)


def test_raw_xyz_is_not_lost_when_frozen_context_is_constant():
    m = model("A")
    xyz, context, valid = observation()
    context.zero_()
    p = m(xyz, context, valid)
    # Preserve centroid and context, alter relative layout only.
    changed = xyz.clone()
    changed[:, 0, 0] += 4.
    changed[:, 1, 0] -= 4.
    q = m(changed, context, valid)
    assert not torch.allclose(p.structure_event_logits, q.structure_event_logits)
    assert not torch.allclose(p.primitives.axis_control_m, q.primitives.axis_control_m)


def test_portal_queries_are_independent_not_primitive_crop_ends():
    m = model("A")
    inputs = observation()
    p = m(*inputs)
    with torch.no_grad():
        m.portal_queries[0].add_(torch.linspace(-1., 1., DIM))
    q = m(*inputs)
    assert torch.equal(p.primitives.axis_control_m, q.primitives.axis_control_m)
    assert not torch.allclose(p.portal_position_m, q.portal_position_m)


@pytest.mark.parametrize("path", ["A", "B", "C"])
def test_empty_rows_unknown_not_background_and_invalid_nan_safe(path):
    m = model(path)
    xyz, context, valid = observation()
    valid[0] = False
    valid[1, -3:] = False
    clean = m(xyz, context, valid)
    xyz[~valid] = float("nan")
    context[~valid] = float("nan")
    xyz.requires_grad_(True)
    p = m(xyz, context, valid)
    assert_close_predictions(clean, p)
    assert p.observation_supported.tolist() == [False, True]
    assert not p.primitives.observation_supported[0].any()
    assert not p.portal_direction_defined[0].any()
    assert p.portal_membership_probability[0, :, -1].eq(1).all()
    assert p.portal_membership_probability[0, :, :-1].eq(0).all()
    assert p.structure_event_logits[0].eq(0).all()  # no artificial no-object target
    assert p.structure_presence_probability[0].eq(0).all()
    assert p.portal_dimension_evidence_probability[0].eq(0).all()
    structure_loss(p).backward()
    assert torch.isfinite(xyz.grad).all()
    assert xyz.grad[~valid].eq(0).all()


def test_entire_batch_empty_finite_and_unknown():
    m = model()
    xyz, context, valid = observation()
    valid.zero_()
    xyz.fill_(float("nan"))
    context.fill_(float("nan"))
    p = m(xyz, context, valid)
    assert not p.observation_supported.any()
    assert all(torch.isfinite(v).all() for v in tensors(p))


@pytest.mark.parametrize("field", ["xyz", "context"])
def test_nonfinite_valid_input_rejected(field):
    xyz, context, valid = observation()
    (xyz if field == "xyz" else context)[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite valid"):
        model()(xyz, context, valid)


@pytest.mark.parametrize("error", ["context_shape", "mask_dtype", "context_dtype", "empty_n"])
def test_input_contract_rejected(error):
    xyz, context, valid = observation()
    if error == "context_shape":
        context = context[..., :-1]
    elif error == "mask_dtype":
        valid = valid.long()
    elif error == "context_dtype":
        context = context.float()
    else:
        xyz, context, valid = xyz[:, :0], context[:, :0], valid[:, :0]
    with pytest.raises(ValueError):
        model()(xyz, context, valid)


def test_no_teacher_or_identity_forward_and_fixed_branch_choices():
    assert list(inspect.signature(DualPathModelV1.forward).parameters) == [
        "self", "points_xyz_m", "frozen_point_context", "valid"]
    with pytest.raises(ValueError, match="path"):
        DualPathModelV1("oracle")


def test_generic_network_does_not_claim_strict_yaw_equivariance():
    # A representation can rotate while an unconstrained MLP is NOT equivariant.
    m = model("A")
    xyz, context, valid = observation()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=xyz.dtype)
    original = m(xyz, context, valid)
    rotated = m(xyz @ rotation.T, context, valid)
    assert not torch.allclose(rotated.portal_position_m, original.portal_position_m @ rotation.T)
