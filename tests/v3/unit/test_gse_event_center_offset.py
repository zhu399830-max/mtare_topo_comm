import pytest
import torch

from mtare_topo.representation.gse_event_center_offset import (
    EventCenterOffsetHead,
    EventCenterVectorHead,
    event_center_offset_loss,
    event_center_paired_loss,
)


def test_offset_head_is_bounded_and_differentiable():
    model = EventCenterOffsetHead()
    context = torch.randn(8, 128, requires_grad=True)
    predicted = model(context)
    assert tuple(predicted.shape) == (8,)
    assert bool((predicted.abs() <= 12.0).all())
    predicted.sum().backward()
    assert context.grad is not None


def test_vector_head_inherits_scalar_longitudinal_output_exactly():
    scalar = EventCenterOffsetHead()
    vector = EventCenterVectorHead()
    vector.initialize_from_scalar_state(scalar.state_dict())
    context = torch.randn(7, 128)
    scalar_output = scalar(context)
    vector_output = vector(context)
    assert torch.allclose(vector_output[:, 0], scalar_output)
    assert torch.equal(vector_output[:, 1:], torch.zeros(7, 2))


def test_offset_loss_requires_both_decision_classes():
    predicted = torch.zeros(4)
    target = torch.tensor([-2.0, 2.0, -3.0, 3.0])
    result = event_center_offset_loss(predicted, target, torch.tensor([1, 1, 2, 2]))
    assert result["total"] > 0
    with pytest.raises(ValueError):
        event_center_offset_loss(predicted, target, torch.ones(4, dtype=torch.long))


def test_paired_loss_penalizes_cross_view_center_disagreement():
    target = torch.tensor([2.0, -2.0, 2.0, -2.0])
    event = torch.tensor([1, 1, 2, 2])
    sensor = torch.tensor([
        [0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [4.0, 1.0, 0.0],
    ])
    tangent = torch.tensor([[1.0, 0.0, 0.0]] * 4)
    pair = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    good = event_center_paired_loss(target, target, event, sensor, tangent, pair)
    bad = event_center_paired_loss(
        torch.tensor([1.0, -1.0, 1.0, -1.0]), target, event, sensor, tangent, pair
    )
    assert good["relative_center"].item() == 0.0
    assert bad["relative_center"].item() > 0.0
    assert bad["total"].item() > good["total"].item()
