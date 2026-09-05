from __future__ import annotations

import numpy as np
import torch

from mtare_topo.representation.gse_relational_exit_transport_event import (
    RelationalExitTokenTransportEventModel,
    relational_event_episode_loss,
)


def _inputs(batch: int = 4):
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randn(batch, 5, 3, 6, 40, generator=generator)
    tokens[..., 0] = torch.sigmoid(tokens[..., 0])
    heading = torch.nn.functional.normalize(tokens[..., 1:3], dim=-1)
    tokens[..., 1:3] = heading
    tokens[..., 3] = torch.exp(tokens[..., 3].clamp(-2, 2))
    geometry = torch.randn(batch, 5, 3, 8, generator=generator)
    mask = torch.ones(batch, 5, dtype=torch.bool)
    return tokens, geometry, mask


def test_token_and_seed_permutation_invariance() -> None:
    torch.manual_seed(5)
    model = RelationalExitTokenTransportEventModel().eval()
    tokens, geometry, mask = _inputs(2)
    with torch.no_grad():
        reference = model(tokens, geometry, mask)["event_probability"]
        permuted = tokens.clone()
        for time in range(5):
            for seed in range(3):
                order = torch.randperm(6)
                permuted[:, time, seed] = permuted[:, time, seed, order]
        seed_order = torch.tensor([2, 0, 1])
        result = model(permuted[:, :, seed_order], geometry[:, :, seed_order], mask)["event_probability"]
    assert torch.allclose(reference, result, atol=2e-6, rtol=0.0)


def test_masked_past_does_not_leak() -> None:
    torch.manual_seed(7)
    model = RelationalExitTokenTransportEventModel().eval()
    tokens, geometry, mask = _inputs(2)
    mask[:, :2] = False
    with torch.no_grad():
        reference = model(tokens, geometry, mask)["event_probability"]
        tokens[:, :2] = _inputs(2)[0][:, :2] * 2.0
        tokens[:, :2, :, :, 0] = torch.sigmoid(tokens[:, :2, :, :, 0])
        tokens[:, :2, :, :, 1:3] = torch.nn.functional.normalize(tokens[:, :2, :, :, 1:3], dim=-1)
        tokens[:, :2, :, :, 3] = torch.exp(tokens[:, :2, :, :, 3].clamp(-2, 2))
        geometry[:, :2] = geometry[:, :2] * -9.0
        result = model(tokens, geometry, mask)["event_probability"]
    assert torch.equal(reference, result)


def test_output_transport_and_finite_episode_backward() -> None:
    torch.manual_seed(11)
    model = RelationalExitTokenTransportEventModel()
    tokens, geometry, mask = _inputs(5)
    outputs = model(tokens, geometry, mask)
    assert outputs["transport"].shape == (5, 4, 3, 6, 6)
    assert torch.allclose(outputs["transport"].sum(dim=-2), torch.ones(5, 4, 3, 6), atol=1e-6)
    target = torch.tensor([0, 0, 1, 1, 2])
    episode = torch.tensor([-1, -1, 0, 0, 1])
    loss = relational_event_episode_loss(outputs, target, episode)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
