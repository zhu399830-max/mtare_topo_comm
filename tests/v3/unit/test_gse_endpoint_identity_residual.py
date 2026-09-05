from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from mtare_topo.representation.gse_endpoint_identity_residual import (
    EndpointIdentityStructuralResidual,
    corrected_decision_distribution,
    identity_balanced_endpoint_mil_loss,
)


def test_zero_residual_preserves_base_decision_distribution() -> None:
    torch.manual_seed(4)
    context = torch.randn(7, 128)
    structural = torch.randn(7)
    conditional = torch.randn(7, 2)
    base_probability = torch.cat((
        (1 - torch.sigmoid(structural)).unsqueeze(1),
        torch.sigmoid(structural).unsqueeze(1) * torch.softmax(conditional, dim=1),
    ), dim=1)
    residual = EndpointIdentityStructuralResidual()
    output = corrected_decision_distribution({
        "structural_logit": structural,
        "conditional_decision_logits": conditional,
        "causal_context": context,
    }, residual)
    torch.testing.assert_close(output["structural_logit"], structural, rtol=0, atol=0)
    torch.testing.assert_close(output["conditional_decision_logits"], conditional, rtol=0, atol=0)
    torch.testing.assert_close(output["causal_context"], context, rtol=0, atol=0)
    torch.testing.assert_close(output["decision_probability"], base_probability, rtol=1e-7, atol=1e-7)
    assert sum(parameter.numel() for parameter in residual.parameters()) == 129


def test_identity_mean_does_not_overweight_identity_with_more_episodes() -> None:
    # Three positive episodes: identity0 owns episodes 0/1, identity1 owns 2.
    structural = torch.tensor([4.0, -1.0, 3.0, -2.0, 2.0, -4.0], requires_grad=True)
    conditional = torch.tensor([[4.0, 0.0]] * 5 + [[0.0, 0.0]], requires_grad=True)
    target = torch.tensor([1, 1, 1, 1, 1, 0])
    episode = torch.tensor([0, 0, 1, 1, 2, -1])
    loss = identity_balanced_endpoint_mil_loss(
        structural_logit=structural,
        conditional_decision_logits=conditional,
        decision_target=target,
        episode_id=episode,
        endpoint_identity_by_episode=torch.tensor([0, 0, 1]),
    )
    episode0 = -(torch.nn.functional.logsigmoid(structural[:2]) + torch.nn.functional.log_softmax(conditional[:2], dim=1)[:, 0]).amax()
    episode1 = -(torch.nn.functional.logsigmoid(structural[2:4]) + torch.nn.functional.log_softmax(conditional[2:4], dim=1)[:, 0]).amax()
    episode2 = -(torch.nn.functional.logsigmoid(structural[4:5]) + torch.nn.functional.log_softmax(conditional[4:5], dim=1)[:, 0]).amax()
    expected = ((episode0 + episode1) / 2 + episode2) / 2
    torch.testing.assert_close(loss["endpoint_identity_mean"], expected)
    loss["total"].backward()
    assert structural.grad is not None


def test_identity_mil_rejects_noncompact_endpoint_codes() -> None:
    try:
        identity_balanced_endpoint_mil_loss(
            structural_logit=torch.tensor([1.0, -1.0]),
            conditional_decision_logits=torch.zeros(2, 2),
            decision_target=torch.tensor([1, 0]), episode_id=torch.tensor([0, -1]),
            endpoint_identity_by_episode=torch.tensor([2]),
        )
    except ValueError as exc:
        assert "compact" in str(exc)
    else:
        raise AssertionError("noncompact identity codes were accepted")


def test_action_probability_embeds_in_canonical_five_events() -> None:
    source = Path(__file__).resolve().parents[3] / "tools/v3/train_gse_endpoint_identity_residual_v1.py"
    spec = importlib.util.spec_from_file_location("endpoint_residual_trainer", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    probability, uncertainty = module._probability(
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    assert probability.shape == (2, 5)
    assert np.all(probability[:, 3:] == 0.0)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.all(np.isfinite(uncertainty))
