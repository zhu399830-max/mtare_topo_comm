import torch

from mtare_topo.representation.primitive_local_composition_slot_model import (
    LocalCompositionSlotHead,
    LocalCompositionSlotPrediction,
)
from mtare_topo.representation.primitive_local_composition_slot_training import (
    UNKNOWN_LABEL,
    align_local_composition_slot_targets,
    local_composition_slot_losses,
)
from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveAssignment,
    PrimitiveRelationLossTargets,
)
from tests.v3.unit.test_primitive_local_composition_slot_model import _primitive


def _targets(*, nonclique: bool = False, overlap_violation: bool = False):
    primitive = torch.zeros(1, 32)
    primitive[:, :4] = 1
    axis = torch.zeros(1, 32, 3, 3)
    half_axes = torch.ones(1, 32, 2, 2)
    exponent = torch.ones(1, 32, 2) * 2.5
    temporal = torch.zeros(1, 5, 32)
    attachment = torch.zeros(1, 32, 2, 32, 2)

    def connect(first: int, second: int) -> None:
        a, ae = divmod(first, 2)
        b, be = divmod(second, 2)
        attachment[0, a, ae, b, be] = 1
        attachment[0, b, be, a, ae] = 1

    connect(0, 3)
    connect(4, 7)
    if nonclique:
        connect(7, 1)
    overlap = torch.zeros(1, 32, 32)
    overlap[0, 0, 2] = overlap[0, 2, 0] = 1
    if overlap_violation:
        overlap[0, 0, 1] = overlap[0, 1, 0] = 1
    observed = torch.zeros(1, 32, 2)
    observed[:, :4] = 1
    value = PrimitiveRelationLossTargets(
        primitive_mask=primitive,
        axis_control_current_sensor_m=axis,
        endpoint_half_axes_m=half_axes,
        endpoint_shape_exponent=exponent,
        temporal_visibility=temporal,
        endpoint_attachment=attachment,
        disconnected_overlap=overlap,
        endpoint_observed=observed,
    )
    mapping = torch.full((32,), -1, dtype=torch.long)
    mapping[:4] = torch.arange(4)
    assignment = PrimitiveAssignment(mapping, torch.zeros(32, dtype=torch.bool))
    return value, (assignment,)


def test_aligned_slot_targets_are_lossless_with_observed_dustbin() -> None:
    targets, assignments = _targets()
    aligned = align_local_composition_slot_targets(targets, assignments)
    aligned.validate()
    assert aligned.cluster_count.tolist() == [2]
    assert aligned.labels[0, 0] == aligned.labels[0, 3] >= 0
    assert aligned.labels[0, 4] == aligned.labels[0, 7] >= 0
    assert aligned.labels[0, 1] == -1
    assert torch.all(aligned.labels[0, 8:] == UNKNOWN_LABEL)


def test_nonclique_teacher_relation_is_rejected() -> None:
    targets, assignments = _targets(nonclique=True)
    try:
        align_local_composition_slot_targets(targets, assignments)
    except ValueError as exc:
        assert "not a clique" in str(exc)
    else:
        raise AssertionError("nonclique Teacher relation was accepted")


def test_disconnected_overlap_same_cluster_is_rejected() -> None:
    targets, assignments = _targets(overlap_violation=True)
    try:
        align_local_composition_slot_targets(targets, assignments)
    except ValueError as exc:
        assert "disconnected overlap" in str(exc)
    else:
        raise AssertionError("disconnected overlap was merged")


def test_loss_is_invariant_to_predicted_slot_permutation() -> None:
    torch.manual_seed(41)
    targets, assignments = _targets()
    aligned = align_local_composition_slot_targets(targets, assignments)
    head = LocalCompositionSlotHead().eval()
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        first = head(_primitive())
        second = head(_primitive(), composition_slot_permutation=permutation)
    first_losses = local_composition_slot_losses(first, aligned)
    second_losses = local_composition_slot_losses(second, aligned)
    for name in first_losses:
        torch.testing.assert_close(first_losses[name], second_losses[name], atol=2e-6, rtol=2e-6)


def test_every_head_parameter_gets_finite_nonzero_gradient() -> None:
    torch.manual_seed(43)
    targets, assignments = _targets()
    aligned = align_local_composition_slot_targets(targets, assignments)
    head = LocalCompositionSlotHead()
    prediction = head(_primitive())
    losses = local_composition_slot_losses(prediction, aligned)
    assert set(losses) == {
        "cluster_assignment", "dustbin_assignment", "slot_presence",
        "overlap_rejection", "total",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    gradients = [parameter.grad for parameter in head.parameters()]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert all(bool((value != 0).any()) for value in gradients)


def test_loss_rejects_prediction_target_batch_mismatch() -> None:
    targets, assignments = _targets()
    aligned = align_local_composition_slot_targets(targets, assignments)
    prediction = LocalCompositionSlotPrediction(
        assignment_logits=torch.zeros(2, 64, 33),
        slot_presence_logits=torch.zeros(2, 32),
        endpoint_uncertainty=torch.ones(2, 64),
    )
    try:
        local_composition_slot_losses(prediction, aligned)
    except ValueError as exc:
        assert "batch mismatch" in str(exc)
    else:
        raise AssertionError("prediction/target batch mismatch was accepted")
