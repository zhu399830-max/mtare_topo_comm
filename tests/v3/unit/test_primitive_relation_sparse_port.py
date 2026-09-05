from __future__ import annotations

import math

import torch

from mtare_topo.representation.primitive_relation_losses import PrimitiveRelationLossTargets
from mtare_topo.representation.primitive_relation_model import MAXIMUM_SLOTS
from mtare_topo.representation.primitive_relation_sparse_port_losses import (
    sparse_cardinality_objective,
    sparse_port_relation_losses,
)
from mtare_topo.representation.primitive_relation_sparse_port_model import (
    SparsePortRelationNet,
    invariant_endpoint_pair_geometry,
    outward_endpoint_tangents,
)


def student(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(410)
    ranges = 0.1 + 0.8 * torch.rand(batch, 5, 16, 720, generator=generator)
    valid = torch.ones_like(ranges)
    range_valid = torch.stack((ranges, valid), dim=2)
    translation = torch.zeros(batch, 5, 3)
    translation[:, 0] = torch.tensor((-1.0, 0.2, 0.1))
    translation[:, 1] = torch.tensor((-0.75, 0.1, 0.05))
    translation[:, 2] = torch.tensor((-0.5, 0.05, 0.0))
    translation[:, 3] = torch.tensor((-0.25, 0.0, 0.0))
    yaw = torch.zeros(batch, 5)
    yaw[:, :4] = torch.tensor((-4.0, -3.0, -2.0, -1.0))
    return range_valid, translation, yaw


def targets(batch: int = 1) -> PrimitiveRelationLossTargets:
    mask = torch.zeros(batch, 32); mask[:, :3] = 1
    axis = torch.zeros(batch, 32, 3, 3)
    axis[:, 0] = torch.tensor(((-3.0, 0.0, 0.0), (-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    axis[:, 1] = torch.tensor(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    axis[:, 2] = torch.tensor(((-3.0, 0.0, 2.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 2.0)))
    half_axes = torch.zeros(batch, 32, 2, 2)
    half_axes[:, :3] = torch.tensor(((1.2, 0.9), (1.0, 0.8)))
    exponent = torch.zeros(batch, 32, 2)
    exponent[:, 0] = 2.0; exponent[:, 1] = 6.0; exponent[:, 2] = torch.tensor((2.0, 8.0))
    temporal = torch.zeros(batch, 5, 32); temporal[:, :, :3] = 1; temporal[:, 0, 2] = 0
    attachment = torch.zeros(batch, 32, 2, 32, 2)
    attachment[:, 0, 1, 1, 0] = 1; attachment[:, 1, 0, 0, 1] = 1
    overlap = torch.zeros(batch, 32, 32); overlap[:, 0, 2] = 1; overlap[:, 2, 0] = 1
    return PrimitiveRelationLossTargets(
        mask, axis, half_axes, exponent, temporal, attachment, overlap,
    )


def permute_targets(
    value: PrimitiveRelationLossTargets,
    permutation: torch.Tensor,
) -> PrimitiveRelationLossTargets:
    return PrimitiveRelationLossTargets(
        value.primitive_mask[:, permutation],
        value.axis_control_current_sensor_m[:, permutation],
        value.endpoint_half_axes_m[:, permutation],
        value.endpoint_shape_exponent[:, permutation],
        value.temporal_visibility[:, :, permutation],
        value.endpoint_attachment[:, permutation][:, :, :, permutation],
        value.disconnected_overlap[:, permutation][:, :, permutation],
    )


def reverse_target(
    value: PrimitiveRelationLossTargets,
    slot: int,
) -> PrimitiveRelationLossTargets:
    axis = value.axis_control_current_sensor_m.clone()
    half_axes = value.endpoint_half_axes_m.clone()
    exponent = value.endpoint_shape_exponent.clone()
    attachment = value.endpoint_attachment.clone()
    axis[:, slot] = axis[:, slot].flip(1)
    half_axes[:, slot] = half_axes[:, slot].flip(1)
    exponent[:, slot] = exponent[:, slot].flip(1)
    attachment[:, slot] = attachment[:, slot].flip(1)
    attachment[:, :, :, slot] = attachment[:, :, :, slot].flip(3)
    return PrimitiveRelationLossTargets(
        value.primitive_mask,
        axis,
        half_axes,
        exponent,
        value.temporal_visibility,
        attachment,
        value.disconnected_overlap,
    )


def test_outward_tangent_reversal_is_endpoint_permutation() -> None:
    axis = targets().axis_control_current_sensor_m[:, :3]
    direct = outward_endpoint_tangents(axis)
    reversed_value = outward_endpoint_tangents(axis.flip(-2))
    torch.testing.assert_close(reversed_value, direct.flip(-2), atol=0.0, rtol=0.0)


def test_pair_geometry_is_symmetric_and_rotation_invariant() -> None:
    position = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 1.0, 0.5], [-1.0, 3.0, 2.0]]])
    tangent = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.2, 0.0], [0.0, -1.0, 0.2]]]), dim=-1)
    half_axes = torch.tensor([[[1.0, 0.8], [1.2, 0.7], [0.9, 1.1]]])
    exponent = torch.tensor([[2.0, 4.0, 8.0]])
    descriptor = torch.randn(1, 3, 32, generator=torch.Generator().manual_seed(2))
    reference = invariant_endpoint_pair_geometry(position, tangent, half_axes, exponent, descriptor)
    torch.testing.assert_close(reference, reference.transpose(1, 2), atol=1e-7, rtol=1e-7)
    angle = math.radians(53.0)
    rotation = torch.tensor(((math.cos(angle), -math.sin(angle), 0.0), (math.sin(angle), math.cos(angle), 0.0), (0.0, 0.0, 1.0)))
    changed = invariant_endpoint_pair_geometry(position @ rotation.T, tangent @ rotation.T, half_axes, exponent, descriptor)
    torch.testing.assert_close(changed, reference, atol=2e-7, rtol=2e-7)


def test_pair_geometry_exposes_facing_not_only_distance() -> None:
    position = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    axes = torch.ones(1, 2, 2)
    exponent = torch.full((1, 2), 2.0)
    descriptor = torch.ones(1, 2, 32)
    facing = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    parallel = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    a = invariant_endpoint_pair_geometry(position, facing, axes, exponent, descriptor)
    b = invariant_endpoint_pair_geometry(position, parallel, axes, exponent, descriptor)
    assert float(a[0, 0, 1, 2]) > float(b[0, 0, 1, 2])
    assert float(a[0, 0, 1, 0]) == float(b[0, 0, 1, 0])


def test_sparse_cardinality_gradients_suppress_redundant_slots() -> None:
    logits = torch.zeros(1, 32, requires_grad=True)
    matched = torch.zeros(1, 32, dtype=torch.bool); matched[:, :3] = True
    terms = sparse_cardinality_objective(logits, matched)
    assert set(terms) == {"binary", "cardinality", "redundant_description", "total"}
    terms["total"].backward()
    assert bool((logits.grad[:, :3] < 0.0).all())
    assert bool((logits.grad[:, 3:] > 0.0).all())
    assert float(terms["redundant_description"].detach()) > 0.0


def test_model_has_no_v1_dense_relation_heads() -> None:
    model = SparsePortRelationNet()
    assert not hasattr(model, "attachment_head")
    assert not hasattr(model, "overlap_head")
    names = {name for name, _ in model.named_parameters()}
    assert any(name.startswith("endpoint_relation_transformer") for name in names)
    assert any(name.startswith("attachment_uncertainty_head") for name in names)


def test_output_schema_finiteness_symmetry_and_uncertainty() -> None:
    torch.manual_seed(3); model = SparsePortRelationNet().eval()
    with torch.no_grad(): prediction = model(*student())
    expected = {
        "existence_logits": (1, 32),
        "axis_control_current_sensor_m": (1, 32, 3, 3),
        "endpoint_half_axes_m": (1, 32, 2, 2),
        "endpoint_shape_exponent": (1, 32, 2),
        "endpoint_descriptor": (1, 32, 2, 32),
        "geometry_uncertainty": (1, 32),
        "endpoint_attachment_logits": (1, 32, 2, 32, 2),
        "endpoint_attachment_uncertainty": (1, 32, 2, 32, 2),
        "disconnected_overlap_logits": (1, 32, 32),
        "disconnected_overlap_uncertainty": (1, 32, 32),
        "temporal_correspondence_logits": (1, 5, 32, 33),
        "temporal_presence_logits": (1, 5, 32),
    }
    for name, shape in expected.items():
        value = getattr(prediction, name)
        assert tuple(value.shape) == shape
        assert bool(torch.isfinite(value).all())
    torch.testing.assert_close(
        prediction.endpoint_attachment_logits,
        prediction.endpoint_attachment_logits.permute(0, 3, 4, 1, 2),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        prediction.disconnected_overlap_logits,
        prediction.disconnected_overlap_logits.transpose(1, 2),
        atol=0.0,
        rtol=0.0,
    )
    for value in (
        prediction.endpoint_attachment_uncertainty,
        prediction.disconnected_overlap_uncertainty,
    ):
        assert bool(((value >= 0.0) & (value <= 1.0)).all())
    parameters = sum(value.numel() for value in model.parameters())
    assert 2_000_000 < parameters < 6_000_000


def test_query_permutation_equivariance_includes_uncertainty() -> None:
    torch.manual_seed(5); model = SparsePortRelationNet().eval(); inputs = student()
    permutation = torch.randperm(MAXIMUM_SLOTS, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        reference = model(*inputs)
        changed = model(*inputs, query_permutation=permutation)
    for name in (
        "existence_logits", "axis_control_current_sensor_m", "endpoint_half_axes_m",
        "endpoint_shape_exponent", "endpoint_descriptor", "geometry_uncertainty",
    ):
        torch.testing.assert_close(
            getattr(changed, name), getattr(reference, name)[:, permutation],
            atol=3e-5, rtol=3e-5,
        )
    for name in ("endpoint_attachment_logits", "endpoint_attachment_uncertainty"):
        expected = getattr(reference, name)[:, permutation][:, :, :, permutation]
        torch.testing.assert_close(getattr(changed, name), expected, atol=3e-5, rtol=3e-5)
    for name in ("disconnected_overlap_logits", "disconnected_overlap_uncertainty"):
        expected = getattr(reference, name)[:, permutation][:, :, permutation]
        torch.testing.assert_close(getattr(changed, name), expected, atol=3e-5, rtol=3e-5)


def test_six_families_all_parameters_receive_finite_gradient() -> None:
    torch.manual_seed(13); model = SparsePortRelationNet(); inputs = student()
    losses = sparse_port_relation_losses(model(*inputs), targets(), inputs[0])
    assert set(losses) == {
        "primitive_set_parameters", "surface_reconstruction", "ray_free_space",
        "port_relations", "temporal_equivariance", "uncertainty_calibration", "total",
    }
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    losses["total"].backward()
    gradients = [value.grad for value in model.parameters()]
    assert all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
    assert all(float(value.abs().max()) > 0.0 for value in gradients if value is not None)


def test_loss_target_permutation_and_endpoint_reversal_invariant() -> None:
    torch.manual_seed(17); model = SparsePortRelationNet().eval(); inputs = student()
    prediction = model(*inputs); target = targets()
    permutation = torch.cat((torch.tensor((2, 0, 1)), torch.arange(3, 32)))
    reference = sparse_port_relation_losses(prediction, target, inputs[0])
    permuted = sparse_port_relation_losses(
        prediction, permute_targets(target, permutation), inputs[0],
    )
    reversed_value = sparse_port_relation_losses(
        prediction, reverse_target(target, 0), inputs[0],
    )
    for name in reference:
        torch.testing.assert_close(reference[name], permuted[name], atol=1e-6, rtol=0.0)
        torch.testing.assert_close(reference[name], reversed_value[name], atol=1e-6, rtol=0.0)


def test_forbidden_student_shape_fails_closed() -> None:
    model = SparsePortRelationNet(); scans, translation, yaw = student()
    try:
        model(scans[..., :-1], translation, yaw)
    except ValueError as error:
        assert "range_valid" in str(error)
    else:
        raise AssertionError("invalid student scan shape did not fail closed")
