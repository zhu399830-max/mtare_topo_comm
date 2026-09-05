from dataclasses import replace
import inspect

import torch

from mtare_topo.representation.primitive_endpoint_relation_metric_decoding import decode_complete_link_clusters
from mtare_topo.representation.primitive_endpoint_relation_metric_model import (
    EndpointRelationMetricHead,
    endpoint_relation_pair_score,
)
from mtare_topo.representation.primitive_endpoint_relation_metric_training import endpoint_relation_metric_losses
from mtare_topo.representation.primitive_local_composition_slot_training import LocalCompositionSlotTargets
from tests.v3.unit.test_primitive_local_composition_slot_model import _permute_primitive, _primitive


def _targets() -> LocalCompositionSlotTargets:
    labels = torch.full((1, 64), -2, dtype=torch.long)
    labels[0, :2] = 0; labels[0, 2:4] = 1; labels[0, 4] = -1
    observed = labels != -2
    overlap = torch.zeros(1, 64, 64, dtype=torch.bool)
    overlap[0, 0, 2] = overlap[0, 2, 0] = True
    return LocalCompositionSlotTargets(labels, observed, overlap, torch.tensor([2]))


def test_metric_shapes_are_symmetric_and_linear_output() -> None:
    torch.manual_seed(101); prediction = EndpointRelationMetricHead()(_primitive())
    prediction.validate(); score = endpoint_relation_pair_score(prediction)
    assert prediction.embedding.shape == (1, 64, 64)
    assert torch.equal(score, score.transpose(1, 2))
    assert torch.equal(torch.diagonal(score, dim1=1, dim2=2), torch.zeros(1, 64))


def test_primitive_permutation_only_permutes_endpoint_relation_axes() -> None:
    torch.manual_seed(103); head = EndpointRelationMetricHead().eval(); source = _primitive()
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(5))
    endpoint_permutation = (permutation[:, None] * 2 + torch.arange(2)[None]).reshape(-1)
    with torch.no_grad():
        first = head(source); second = head(_permute_primitive(source, permutation))
    torch.testing.assert_close(second.embedding, first.embedding[:, endpoint_permutation], atol=3e-6, rtol=3e-6)
    expected = first.pair_logits[:, endpoint_permutation][:, :, endpoint_permutation]
    torch.testing.assert_close(second.pair_logits, expected, atol=3e-6, rtol=3e-6)


def test_yaw_rotation_leaves_metric_unchanged() -> None:
    torch.manual_seed(107); head = EndpointRelationMetricHead().eval(); source = _primitive()
    angle = torch.tensor(0.73); cosine, sine = torch.cos(angle), torch.sin(angle)
    axis = source.axis_control_current_sensor_m.clone(); x, y = axis[..., 0].clone(), axis[..., 1].clone()
    axis[..., 0] = cosine * x - sine * y; axis[..., 1] = sine * x + cosine * y
    with torch.no_grad(): first = head(source); second = head(replace(source, axis_control_current_sensor_m=axis))
    torch.testing.assert_close(second.pair_logits, first.pair_logits, atol=3e-6, rtol=3e-6)


def test_real_loss_components_have_finite_nonzero_gradients() -> None:
    torch.manual_seed(109); head = EndpointRelationMetricHead(); prediction = head(_primitive())
    losses = endpoint_relation_metric_losses(prediction, _targets()); losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    gradients = [parameter.grad for parameter in head.parameters()]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert sum(float(value.abs().sum()) for value in gradients) > 0


def test_complete_link_refuses_chain_merge_and_is_deterministic() -> None:
    score = torch.zeros(1, 5, 5)
    score[0, 0, 1] = score[0, 1, 0] = .9
    score[0, 1, 2] = score[0, 2, 1] = .9
    score[0, 0, 2] = score[0, 2, 0] = .2
    score[0, 3, 4] = score[0, 4, 3] = .8
    active = torch.ones(1, 5, dtype=torch.bool)
    first = decode_complete_link_clusters(score, active, confidence_threshold=.7)
    second = decode_complete_link_clusters(score, active, confidence_threshold=.7)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert first[1][0, 0, 1] and not first[1][0, 0, 2] and first[1][0, 3, 4]


def test_forward_interface_has_no_teacher_or_identity() -> None:
    parameters = set(inspect.signature(EndpointRelationMetricHead.forward).parameters)
    assert not parameters & {"teacher", "world", "node_id", "primitive_id", "attachment", "absolute_pose"}
