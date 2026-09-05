from dataclasses import replace
import inspect

import torch
from torch.nn import functional as F

from mtare_topo.representation.primitive_local_composition_slot_model import (
    ASSIGNMENT_CLASS_COUNT,
    COMPOSITION_SLOT_COUNT,
    FrozenObservableLocalCompositionSlotNet,
    LocalCompositionSlotHead,
    LocalCompositionSlotModelPrediction,
    LocalCompositionSlotPrediction,
    composition_slot_relation_probability,
    decode_local_composition_slots,
    local_composition_slot_safe_score,
)
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationPrediction,
)


def _primitive(batch: int = 1) -> ObservableSparsePortRelationPrediction:
    generator = torch.Generator().manual_seed(17)
    axis = torch.zeros(batch, 32, 3, 3)
    x = torch.arange(32, dtype=torch.float32)[None, :, None].expand(batch, -1, -1)
    axis[:, :, 0] = torch.cat((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
    axis[:, :, 1] = axis[:, :, 0] + torch.tensor((0.4, 0.2, 0.05))
    axis[:, :, 2] = axis[:, :, 0] + torch.tensor((0.8, 0.4, 0.10))
    descriptor = F.normalize(
        torch.randn(batch, 32, 2, 32, generator=generator), dim=-1,
    )
    attachment = torch.zeros(batch, 32, 2, 32, 2)
    overlap = torch.zeros(batch, 32, 32)
    return ObservableSparsePortRelationPrediction(
        existence_logits=torch.linspace(2.0, -2.0, 32)[None].expand(batch, -1),
        axis_control_current_sensor_m=axis,
        endpoint_half_axes_m=torch.ones(batch, 32, 2, 2),
        endpoint_shape_exponent=torch.full((batch, 32, 2), 2.5),
        endpoint_descriptor=descriptor,
        geometry_uncertainty=torch.full((batch, 32), 0.1),
        endpoint_evidence_logits=torch.ones(batch, 32, 2),
        endpoint_attachment_logits=attachment,
        endpoint_attachment_uncertainty=attachment.clone(),
        disconnected_overlap_logits=overlap,
        disconnected_overlap_uncertainty=overlap.clone(),
        temporal_correspondence_logits=torch.zeros(batch, 5, 32, 33),
        temporal_presence_logits=torch.zeros(batch, 5, 32),
    )


def _permute_primitive(
    value: ObservableSparsePortRelationPrediction,
    permutation: torch.Tensor,
) -> ObservableSparsePortRelationPrediction:
    endpoint_permutation = (
        permutation[:, None] * 2 + torch.arange(2)[None]
    ).reshape(-1)
    attachment = value.endpoint_attachment_logits.reshape(1, 64, 64)
    attachment_uncertainty = value.endpoint_attachment_uncertainty.reshape(1, 64, 64)
    return replace(
        value,
        existence_logits=value.existence_logits[:, permutation],
        axis_control_current_sensor_m=value.axis_control_current_sensor_m[:, permutation],
        endpoint_half_axes_m=value.endpoint_half_axes_m[:, permutation],
        endpoint_shape_exponent=value.endpoint_shape_exponent[:, permutation],
        endpoint_descriptor=value.endpoint_descriptor[:, permutation],
        geometry_uncertainty=value.geometry_uncertainty[:, permutation],
        endpoint_evidence_logits=value.endpoint_evidence_logits[:, permutation],
        endpoint_attachment_logits=attachment[:, endpoint_permutation][
            :, :, endpoint_permutation
        ].reshape(1, 32, 2, 32, 2),
        endpoint_attachment_uncertainty=attachment_uncertainty[:, endpoint_permutation][
            :, :, endpoint_permutation
        ].reshape(1, 32, 2, 32, 2),
        disconnected_overlap_logits=value.disconnected_overlap_logits[:, permutation][
            :, :, permutation
        ],
        disconnected_overlap_uncertainty=value.disconnected_overlap_uncertainty[:, permutation][
            :, :, permutation
        ],
        temporal_correspondence_logits=value.temporal_correspondence_logits[:, :, permutation],
        temporal_presence_logits=value.temporal_presence_logits[:, :, permutation],
    )


def test_local_composition_slot_shapes_are_linear_and_finite() -> None:
    torch.manual_seed(19)
    prediction = LocalCompositionSlotHead()(_primitive())
    prediction.validate()
    assert prediction.assignment_logits.shape == (1, 64, 33)
    assert prediction.assignment_logits.numel() == 64 * ASSIGNMENT_CLASS_COUNT
    assert prediction.slot_presence_logits.shape == (1, COMPOSITION_SLOT_COUNT)


def test_primitive_query_permutation_only_permutes_endpoint_axis() -> None:
    torch.manual_seed(23)
    model = LocalCompositionSlotHead().eval()
    source = _primitive()
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(3))
    endpoint_permutation = (
        permutation[:, None] * 2 + torch.arange(2)[None]
    ).reshape(-1)
    with torch.no_grad():
        reference = model(source)
        changed = model(_permute_primitive(source, permutation))
    torch.testing.assert_close(
        changed.assignment_logits,
        reference.assignment_logits[:, endpoint_permutation],
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        changed.slot_presence_logits,
        reference.slot_presence_logits,
        atol=3e-6,
        rtol=3e-6,
    )


def test_composition_slot_permutation_only_permutes_slot_outputs() -> None:
    torch.manual_seed(29)
    model = LocalCompositionSlotHead().eval()
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        reference = model(_primitive())
        changed = model(_primitive(), composition_slot_permutation=permutation)
    torch.testing.assert_close(
        changed.assignment_logits[..., :32],
        reference.assignment_logits[..., permutation],
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        changed.assignment_logits[..., 32],
        reference.assignment_logits[..., 32],
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        changed.slot_presence_logits,
        reference.slot_presence_logits[:, permutation],
        atol=3e-6,
        rtol=3e-6,
    )


def test_relation_probability_is_symmetric_and_slot_permutation_invariant() -> None:
    torch.manual_seed(31)
    model = LocalCompositionSlotHead().eval()
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        first = composition_slot_relation_probability(model(_primitive()))
        second = composition_slot_relation_probability(
            model(_primitive(), composition_slot_permutation=permutation),
        )
    assert torch.equal(first, first.transpose(1, 2))
    assert torch.equal(torch.diagonal(first, dim1=1, dim2=2), torch.zeros(1, 64))
    torch.testing.assert_close(second, first, atol=3e-7, rtol=3e-6)


def test_safe_score_uses_evidence_and_uncertainty() -> None:
    torch.manual_seed(37)
    head = LocalCompositionSlotHead().eval()
    primitive = _primitive()
    with torch.no_grad():
        composition = head(primitive)
    first = local_composition_slot_safe_score(
        LocalCompositionSlotModelPrediction(primitive, composition),
    )
    changed_primitive = replace(
        primitive,
        endpoint_evidence_logits=primitive.endpoint_evidence_logits.clone(),
    )
    changed_primitive.endpoint_evidence_logits[:, 0, 0] = -20.0
    second = local_composition_slot_safe_score(
        LocalCompositionSlotModelPrediction(changed_primitive, composition),
    )
    assert torch.equal(first, first.transpose(1, 2))
    assert torch.all(second[:, 0, 1:] < first[:, 0, 1:])
    assert first[0, 0, 0] == second[0, 0, 0] == 0.0


def test_decode_rejects_dustbin_low_margin_and_low_evidence() -> None:
    logits = torch.full((1, 64, 33), -8.0)
    logits[:, :, 32] = 8.0
    logits[0, 0, 2] = 12.0
    logits[0, 1, 2] = 12.0
    logits[0, 2, 3] = logits[0, 2, 4] = 9.0
    logits[0, 3, 5] = 12.0
    primitive = replace(_primitive(), endpoint_evidence_logits=torch.ones(1, 32, 2) * 8.0)
    primitive.endpoint_evidence_logits.reshape(1, 64)[0, 3] = -8.0
    probability = torch.softmax(logits, dim=-1)
    entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum(-1) / torch.log(
        torch.tensor(33.0)
    )
    composition = LocalCompositionSlotPrediction(
        assignment_logits=logits,
        slot_presence_logits=torch.ones(1, 32) * 8.0,
        endpoint_uncertainty=entropy,
    )
    labels, attachment = decode_local_composition_slots(
        LocalCompositionSlotModelPrediction(primitive, composition),
        assignment_probability_threshold=0.8,
        assignment_margin_threshold=0.2,
        endpoint_evidence_threshold=0.8,
    )
    assert labels[0, 0] == labels[0, 1] == 2
    assert labels[0, 2] == labels[0, 3] == -1
    assert attachment[0, 0, 1] and attachment[0, 1, 0]
    assert int(attachment.sum()) == 2


def test_student_forward_interfaces_contain_no_teacher_identity() -> None:
    head_parameters = set(inspect.signature(LocalCompositionSlotHead.forward).parameters)
    model_parameters = set(inspect.signature(FrozenObservableLocalCompositionSlotNet.forward).parameters)
    forbidden = {
        "world", "node_id", "primitive_id", "construction", "absolute_pose",
        "attachment_target", "endpoint_observed", "teacher_slot",
    }
    assert not head_parameters & forbidden
    assert not model_parameters & forbidden


def test_invalid_composition_slot_permutation_is_rejected() -> None:
    permutation = torch.zeros(32, dtype=torch.long)
    try:
        LocalCompositionSlotHead()(
            _primitive(), composition_slot_permutation=permutation,
        )
    except ValueError as exc:
        assert "bijection" in str(exc)
    else:
        raise AssertionError("non-bijective composition-slot permutation was accepted")
