from dataclasses import replace

import torch

from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
    safe_attachment_score,
)
from mtare_topo.representation.primitive_relation_observable_training import observable_primitive_relation_losses
from mtare_topo.representation.primitive_relation_sparse_port_losses import (
    sparse_port_relation_losses,
)
from tests.v3.unit.test_primitive_relation_model import _student, _targets


def _observable_targets():
    targets = _targets()
    observed = torch.zeros(1, 32, 2)
    observed[:, :3] = torch.tensor(((1.0, 1.0), (1.0, 0.0), (0.0, 1.0)))
    return replace(targets, endpoint_observed=observed)


def test_observable_model_schema_equivariance_and_safe_score() -> None:
    torch.manual_seed(101); model = ObservableSparsePortRelationNet().eval(); inputs = _student()
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        reference = model(*inputs)
        changed = model(*inputs, query_permutation=permutation)
    assert reference.endpoint_evidence_logits.shape == (1, 32, 2)
    assert torch.isfinite(reference.endpoint_evidence_logits).all()
    assert torch.allclose(changed.endpoint_evidence_logits, reference.endpoint_evidence_logits[:, permutation], atol=2e-5, rtol=2e-5)
    score = safe_attachment_score(reference)
    assert score.shape == (1, 32, 2, 32, 2)
    assert bool(((score >= 0.0) & (score <= 1.0)).all())


def test_hidden_attachment_target_does_not_become_negative() -> None:
    torch.manual_seed(103); model = ObservableSparsePortRelationNet().eval(); inputs = _student(); prediction = model(*inputs)
    first = _observable_targets()
    attachment = first.endpoint_attachment.clone()
    # Endpoint (primitive 1, endpoint 1) is unobserved, so this arbitrary
    # hidden label must have no effect on the masked attachment loss.
    attachment[:, 0, 0, 1, 1] = 1; attachment[:, 1, 1, 0, 0] = 1
    second = replace(first, endpoint_attachment=attachment)
    loss_first = observable_primitive_relation_losses(prediction, first, inputs[0])
    loss_second = observable_primitive_relation_losses(prediction, second, inputs[0])
    assert torch.allclose(loss_first["port_relations"], loss_second["port_relations"], atol=1e-7, rtol=0.0)


def test_observable_loss_preserves_sparse_cardinality_objective() -> None:
    torch.manual_seed(105); model = ObservableSparsePortRelationNet().eval(); inputs = _student()
    prediction = model(*inputs); targets = _observable_targets()
    observable = observable_primitive_relation_losses(prediction, targets, inputs[0])
    sparse = sparse_port_relation_losses(prediction, targets, inputs[0])
    assert torch.equal(
        observable["primitive_set_parameters"],
        sparse["primitive_set_parameters"],
    )


def test_observable_loss_has_finite_nonzero_evidence_head_gradients() -> None:
    torch.manual_seed(107); model = ObservableSparsePortRelationNet(); inputs = _student(); prediction = model(*inputs)
    losses = observable_primitive_relation_losses(prediction, _observable_targets(), inputs[0])
    assert set(losses) == {
        "primitive_set_parameters", "surface_reconstruction", "ray_free_space",
        "port_relations", "temporal_equivariance", "uncertainty_calibration", "total",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    gradients = [parameter.grad for parameter in model.endpoint_evidence_head.parameters()]
    assert all(value is not None and torch.isfinite(value).all() and bool((value != 0).any()) for value in gradients)


def test_endpoint_observability_reverses_with_teacher_geometry() -> None:
    torch.manual_seed(109); model = ObservableSparsePortRelationNet().eval(); inputs = _student(); prediction = model(*inputs)
    target = _observable_targets()
    axis = target.axis_control_current_sensor_m.clone(); axes = target.endpoint_half_axes_m.clone()
    exponent = target.endpoint_shape_exponent.clone(); attachment = target.endpoint_attachment.clone()
    observed = target.endpoint_observed.clone()
    axis[:, 0] = axis[:, 0].flip(1); axes[:, 0] = axes[:, 0].flip(1); exponent[:, 0] = exponent[:, 0].flip(1)
    attachment[:, 0] = attachment[:, 0].flip(1); attachment[:, :, :, 0] = attachment[:, :, :, 0].flip(3)
    observed[:, 0] = observed[:, 0].flip(1)
    reversed_target = replace(target, axis_control_current_sensor_m=axis, endpoint_half_axes_m=axes, endpoint_shape_exponent=exponent, endpoint_attachment=attachment, endpoint_observed=observed)
    first = observable_primitive_relation_losses(prediction, target, inputs[0])
    second = observable_primitive_relation_losses(prediction, reversed_target, inputs[0])
    for name in first:
        assert torch.allclose(first[name], second[name], atol=1e-6, rtol=0.0), name
