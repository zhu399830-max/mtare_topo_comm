import pytest
import numpy as np
import torch

from mtare_topo.representation.gse_directional_structural_event import (
    DirectionalStructuralEventHead,
    directional_identity_balanced_weights,
    directional_structural_event_loss,
)
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet


def test_zero_initialized_directional_head_reproduces_frozen_event_probability():
    torch.manual_seed(4)
    head = DirectionalStructuralEventHead()
    azimuth = torch.randn(3, 5, 128, 18)
    context = torch.randn(3, 128)
    baseline = torch.randn(3, 5)
    output = head(azimuth, context, baseline)
    assert output["event_probability"].shape == (3, 5)
    assert torch.allclose(output["event_probability"].sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.allclose(output["event_probability"], torch.softmax(baseline, dim=1), atol=1e-6)


def test_directional_head_losses_are_separate_and_differentiable():
    torch.manual_seed(7)
    head = DirectionalStructuralEventHead()
    output = head(torch.randn(5, 5, 128, 20), torch.randn(5, 128), torch.randn(5, 5))
    losses = directional_structural_event_loss(output, torch.tensor([0, 1, 2, 3, 4]))
    assert set(losses) == {"structural", "conditional_event", "total"}
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert head.structural_residual[-1].weight.grad is not None
    assert head.class_residual[-1].weight.grad is not None


def test_directional_head_rejects_invalid_or_nonfinite_inputs():
    head = DirectionalStructuralEventHead()
    with pytest.raises(ValueError, match=r"\[B,5,128,A\]"):
        head(torch.zeros(2, 4, 128, 20), torch.zeros(2, 128), torch.zeros(2, 5))
    bad = torch.zeros(2, 5, 128, 20)
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        head(bad, torch.zeros(2, 128), torch.zeros(2, 5))


def test_frozen_encoder_exposes_directional_features_without_changing_forward_contract():
    torch.manual_seed(1)
    model = GeometrySemanticEventNet().eval()
    scans = torch.rand(1, 5, 2, 16, 720)
    scans[:, :, 1] = (scans[:, :, 1] > 0.5).float()
    with torch.inference_mode():
        features = model.encode_causal_features(scans)
        output = model(scans)
    assert features["context"].shape == (1, 128)
    assert features["azimuth_sequence"].shape == (1, 5, 128, 180)
    assert features["directional"].shape == (1, 128, 180)
    assert output["event_logits"].shape == (1, 5)


def test_directional_sampling_balances_binary_classes_and_structural_identities():
    event = np.asarray([0, 0, 0, 1, 1, 1, 2, 3, 4])
    identity = np.asarray([-1, -1, -1, 10, 10, 11, 20, 30, 40])
    hardness = np.asarray([0.1, 0.2, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    weight = directional_identity_balanced_weights(event, identity, hardness)
    assert weight.sum() == pytest.approx(1.0)
    assert weight[event == 0].sum() == pytest.approx(0.5)
    for class_index in range(1, 5):
        assert weight[event == class_index].sum() == pytest.approx(0.125)
    assert weight[(event == 1) & (identity == 10)].sum() == pytest.approx(0.0625)
    assert weight[(event == 1) & (identity == 11)].sum() == pytest.approx(0.0625)
    assert weight[2] > weight[0]
