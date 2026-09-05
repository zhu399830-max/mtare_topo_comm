from dataclasses import replace
import inspect

import torch
from torch.nn import functional as F

from mtare_topo.representation.primitive_composition_anchor_model import (
    COMPOSITION_ANCHOR_OUTPUTS_PER_ENDPOINT,
    CompositionAnchorPrediction,
    CompositionAnchorResidualHead,
    FrozenObservableCompositionAnchorNet,
    CompositionAnchorModelPrediction,
    composition_anchor_safe_score,
    composition_anchor_losses,
    endpoint_local_frame,
    gaussian_anchor_compatibility_logits,
)
from mtare_topo.representation.primitive_composition_anchor_training import (
    align_composition_anchor_targets,
    composition_anchor_training_losses,
)
from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveAssignment,
)
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationPrediction,
)


def _prediction(batch: int = 1) -> ObservableSparsePortRelationPrediction:
    generator = torch.Generator().manual_seed(17)
    axis = torch.zeros(batch, 32, 3, 3)
    slot = torch.arange(32, dtype=torch.float32)[None, :, None]
    axis[:, :, 0] = torch.cat((slot, torch.zeros_like(slot), torch.zeros_like(slot)), dim=-1)
    axis[:, :, 1] = axis[:, :, 0] + torch.tensor((0.5, 0.0, 0.05))
    axis[:, :, 2] = axis[:, :, 0] + torch.tensor((1.0, 0.0, 0.10))
    descriptor = F.normalize(torch.randn(batch, 32, 2, 32, generator=generator), dim=-1)
    attachment = torch.zeros(batch, 32, 2, 32, 2)
    overlap = torch.zeros(batch, 32, 32)
    return ObservableSparsePortRelationPrediction(
        existence_logits=torch.zeros(batch, 32),
        axis_control_current_sensor_m=axis,
        endpoint_half_axes_m=torch.ones(batch, 32, 2, 2),
        endpoint_shape_exponent=torch.full((batch, 32, 2), 2.5),
        endpoint_descriptor=descriptor,
        geometry_uncertainty=torch.full((batch, 32), 0.1),
        endpoint_evidence_logits=torch.zeros(batch, 32, 2),
        endpoint_attachment_logits=attachment,
        endpoint_attachment_uncertainty=attachment.clone(),
        disconnected_overlap_logits=overlap,
        disconnected_overlap_uncertainty=overlap.clone(),
        temporal_correspondence_logits=torch.zeros(batch, 5, 32, 33),
        temporal_presence_logits=torch.zeros(batch, 5, 32),
    )


def _rotate_yaw(value: torch.Tensor, degrees: float) -> torch.Tensor:
    angle = torch.deg2rad(torch.tensor(degrees, dtype=value.dtype, device=value.device))
    rotation = torch.stack((
        torch.stack((torch.cos(angle), -torch.sin(angle), torch.zeros_like(angle))),
        torch.stack((torch.sin(angle), torch.cos(angle), torch.zeros_like(angle))),
        torch.tensor((0.0, 0.0, 1.0), dtype=value.dtype, device=value.device),
    ))
    return torch.einsum("...j,kj->...k", value, rotation)


def test_anchor_output_is_linear_in_endpoint_count_and_finite() -> None:
    torch.manual_seed(19)
    output = CompositionAnchorResidualHead()(_prediction())
    output.validate()
    assert output.anchor_current_sensor_m.shape == (1, 32, 2, 3)
    assert output.scale_m.shape == (1, 32, 2)
    assert output.residual_local_m.numel() + output.scale_m.numel() == (
        64 * COMPOSITION_ANCHOR_OUTPUTS_PER_ENDPOINT
    )
    assert output.compatibility_logits.shape == (1, 64, 64)


def test_safe_score_is_symmetric_and_uses_endpoint_evidence() -> None:
    torch.manual_seed(20)
    primitive = replace(
        _prediction(),
        endpoint_evidence_logits=torch.linspace(-5.0, 5.0, 64).reshape(1, 32, 2),
    )
    composition = CompositionAnchorResidualHead()(primitive)
    first = composition_anchor_safe_score(
        CompositionAnchorModelPrediction(primitive=primitive, composition=composition),
    )
    changed = replace(primitive, endpoint_evidence_logits=primitive.endpoint_evidence_logits.clone())
    changed.endpoint_evidence_logits[:, 0, 0] = -20.0
    second = composition_anchor_safe_score(
        CompositionAnchorModelPrediction(primitive=changed, composition=composition),
    )
    assert torch.equal(first, first.transpose(1, 2))
    evidence = torch.sigmoid(primitive.endpoint_evidence_logits).reshape(1, 64)
    expected = (
        torch.sigmoid(composition.compatibility_logits)
        * (evidence[:, :, None] * evidence[:, None, :])
    )
    assert torch.equal(first, expected)
    assert bool(torch.isfinite(first).all())
    assert torch.all(second[:, 0] < first[:, 0])


def test_anchor_head_is_yaw_equivariant() -> None:
    torch.manual_seed(23)
    model = CompositionAnchorResidualHead().eval()
    source = _prediction()
    rotated = replace(
        source,
        axis_control_current_sensor_m=_rotate_yaw(
            source.axis_control_current_sensor_m, 73.0,
        ),
    )
    with torch.no_grad():
        first = model(source)
        second = model(rotated)
    torch.testing.assert_close(
        second.anchor_current_sensor_m,
        _rotate_yaw(first.anchor_current_sensor_m, 73.0),
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(second.residual_local_m, first.residual_local_m, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(second.scale_m, first.scale_m, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(second.compatibility_logits, first.compatibility_logits, atol=2e-5, rtol=2e-5)


def test_anchor_head_is_slot_permutation_and_endpoint_reversal_equivariant() -> None:
    torch.manual_seed(29)
    model = CompositionAnchorResidualHead().eval()
    source = _prediction()
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(3))
    permuted = replace(
        source,
        existence_logits=source.existence_logits[:, permutation],
        axis_control_current_sensor_m=source.axis_control_current_sensor_m[:, permutation],
        endpoint_half_axes_m=source.endpoint_half_axes_m[:, permutation],
        endpoint_shape_exponent=source.endpoint_shape_exponent[:, permutation],
        endpoint_descriptor=source.endpoint_descriptor[:, permutation],
        geometry_uncertainty=source.geometry_uncertainty[:, permutation],
        endpoint_evidence_logits=source.endpoint_evidence_logits[:, permutation],
    )
    reversed_value = replace(
        source,
        axis_control_current_sensor_m=source.axis_control_current_sensor_m.flip(2),
        endpoint_half_axes_m=source.endpoint_half_axes_m.flip(2),
        endpoint_shape_exponent=source.endpoint_shape_exponent.flip(2),
        endpoint_descriptor=source.endpoint_descriptor.flip(2),
        endpoint_evidence_logits=source.endpoint_evidence_logits.flip(2),
    )
    with torch.no_grad():
        reference = model(source)
        changed = model(permuted)
        reversed_output = model(reversed_value)
    torch.testing.assert_close(
        changed.anchor_current_sensor_m,
        reference.anchor_current_sensor_m[:, permutation],
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        reversed_output.anchor_current_sensor_m,
        reference.anchor_current_sensor_m.flip(2),
        atol=2e-6,
        rtol=2e-6,
    )


def _loss_targets(output: CompositionAnchorPrediction):
    primitive = torch.zeros(1, 32, dtype=torch.bool)
    primitive[:, :3] = True
    observed = torch.zeros(1, 32, 2, dtype=torch.bool)
    observed[:, :3] = True
    anchors = output.anchor_current_sensor_m.detach().clone()
    shared = torch.tensor((1.5, 0.2, 0.0))
    anchors[0, 0, 1] = shared
    anchors[0, 1, 0] = shared
    attachment = torch.zeros(1, 32, 2, 32, 2)
    attachment[0, 0, 1, 1, 0] = 1.0
    attachment[0, 1, 0, 0, 1] = 1.0
    overlap = torch.zeros(1, 32, 32)
    overlap[0, 0, 2] = overlap[0, 2, 0] = 1.0
    return anchors, primitive, observed, attachment, overlap


def test_anchor_loss_has_finite_nonzero_gradients_for_every_parameter() -> None:
    torch.manual_seed(31)
    model = CompositionAnchorResidualHead()
    output = model(_prediction())
    anchors, primitive, observed, attachment, overlap = _loss_targets(output)
    losses = composition_anchor_losses(
        output,
        anchor_target_current_sensor_m=anchors,
        primitive_mask=primitive,
        endpoint_observed=observed,
        attachment_target=attachment,
        disconnected_overlap=overlap,
    )
    assert set(losses) == {
        "anchor_nll", "uncertainty_calibration", "compatibility",
        "overlap_hard_negative", "relation", "total",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert all(bool((value != 0).any()) for value in gradients)


def test_negative_compatibility_gradient_separates_anchors() -> None:
    anchors = torch.zeros(1, 32, 2, 3, requires_grad=True)
    anchors.data[0, 1, 0, 0] = 0.1
    scale = torch.full((1, 32, 2), 0.2)
    temperature = torch.zeros((), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    logits = gaussian_anchor_compatibility_logits(
        anchors, scale, log_temperature=temperature, bias=bias,
    )
    loss = F.softplus(logits[0, 1, 2])
    loss.backward()
    before = torch.linalg.vector_norm(anchors.detach()[0, 0, 1] - anchors.detach()[0, 1, 0])
    after_value = anchors.detach() - 0.01 * anchors.grad
    after = torch.linalg.vector_norm(after_value[0, 0, 1] - after_value[0, 1, 0])
    assert after > before


def test_student_head_interface_contains_no_teacher_identity() -> None:
    parameters = set(inspect.signature(CompositionAnchorResidualHead.forward).parameters)
    assert parameters == {"self", "prediction"}
    forbidden = {"world", "node_id", "primitive_id", "construction", "absolute_pose"}
    assert not parameters & forbidden


def test_near_vertical_endpoint_frame_has_finite_yaw_equivariant_fallback() -> None:
    axis = torch.zeros(1, 32, 3, 3)
    axis[:, :, 1, 2] = 0.5
    axis[:, :, 2, 2] = 1.0
    first = endpoint_local_frame(axis)
    second = endpoint_local_frame(_rotate_yaw(axis, 37.0))
    assert torch.isfinite(first).all()
    torch.testing.assert_close(second, _rotate_yaw(first, 37.0), atol=1e-7, rtol=0.0)


def test_collapsed_segment_uses_sensor_polar_equivariant_frame() -> None:
    axis = torch.zeros(1, 32, 3, 3)
    axis[..., 0] = 2.0
    first = endpoint_local_frame(axis)
    rotated_axis = _rotate_yaw(axis, 51.0)
    second = endpoint_local_frame(rotated_axis)
    assert torch.isfinite(first).all()
    assert torch.linalg.vector_norm(first[..., 0, :], dim=-1).min() > 0.99
    assert torch.equal(first[..., 2, 2], torch.ones_like(first[..., 2, 2]))
    torch.testing.assert_close(second, _rotate_yaw(first, 51.0), atol=2e-6, rtol=2e-6)


def test_teacher_anchor_alignment_respects_query_mapping_and_reversal() -> None:
    from tests.v3.unit.test_primitive_relation_model import _targets

    targets = _targets()
    targets = replace(targets, endpoint_observed=targets.primitive_mask[:, :, None].expand(-1, -1, 2))
    anchors = targets.axis_control_current_sensor_m[:, :, (0, 2)].clone()
    mapping = torch.full((32,), -1, dtype=torch.long)
    mapping[:3] = torch.tensor((2, 0, 1))
    reversed_slots = torch.zeros(32, dtype=torch.bool)
    reversed_slots[1] = True
    aligned = align_composition_anchor_targets(
        anchors, targets, (PrimitiveAssignment(mapping, reversed_slots),),
    )
    torch.testing.assert_close(aligned.anchor_current_sensor_m[0, 0], anchors[0, 2])
    torch.testing.assert_close(aligned.anchor_current_sensor_m[0, 1], anchors[0, 0].flip(0))
    torch.testing.assert_close(aligned.anchor_current_sensor_m[0, 2], anchors[0, 1])
    assert aligned.primitive_mask[0, :3].all()
    assert aligned.endpoint_observed[0, :3].all()


def test_frozen_wrapper_updates_only_linear_anchor_head() -> None:
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, *args, query_permutation=None):
            value = _prediction()
            if query_permutation is None:
                return value
            permutation = query_permutation
            return replace(
                value,
                existence_logits=value.existence_logits[:, permutation],
                axis_control_current_sensor_m=value.axis_control_current_sensor_m[:, permutation],
                endpoint_half_axes_m=value.endpoint_half_axes_m[:, permutation],
                endpoint_shape_exponent=value.endpoint_shape_exponent[:, permutation],
                endpoint_descriptor=value.endpoint_descriptor[:, permutation],
                geometry_uncertainty=value.geometry_uncertainty[:, permutation],
                endpoint_evidence_logits=value.endpoint_evidence_logits[:, permutation],
            )

    torch.manual_seed(37)
    model = FrozenObservableCompositionAnchorNet(Backbone()).train()
    output = model(None, None, None).composition
    anchors, primitive, observed, attachment, overlap = _loss_targets(output)
    losses = composition_anchor_training_losses(
        output,
        type("Aligned", (), {
            "anchor_current_sensor_m": anchors,
            "primitive_mask": primitive,
            "endpoint_observed": observed,
            "attachment": attachment,
            "disconnected_overlap": overlap,
        })(),
    )
    losses["total"].backward()
    assert model.backbone.scale.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.anchor_head.parameters()
    )
