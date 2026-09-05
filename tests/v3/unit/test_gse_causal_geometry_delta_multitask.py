from __future__ import annotations

import torch

from mtare_topo.representation.gse_causal_geometry_delta import (
    CausalGeometryDeltaEventHead,
    causal_geometry_delta_event_loss,
    geometry_delta_weighted_mae,
)


def _inputs(batch: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(batch, 5, 128, 24, generator=generator),
        torch.randn(batch, 128, generator=generator),
        torch.randn(batch, 5, generator=generator),
    )


def test_zero_initialized_multitask_head_preserves_baseline_event_distribution() -> None:
    head = CausalGeometryDeltaEventHead()
    azimuth, context, logits = _inputs()
    output = head(azimuth, context, logits)
    assert torch.allclose(output["event_probability"], torch.softmax(logits, dim=1), atol=1e-6)
    assert torch.equal(output["geometry_delta"], torch.zeros(3, 4))


def test_multitask_loss_uses_only_valid_geometry_rows_and_backpropagates() -> None:
    head = CausalGeometryDeltaEventHead()
    azimuth, context, logits = _inputs()
    output = head(azimuth, context, logits)
    target = torch.tensor(
        [[1.0, 0.2, -0.1, 0.001], [float("nan")] * 4, [-2.0, 0.4, 0.3, -0.002]]
    )
    loss = causal_geometry_delta_event_loss(
        output,
        torch.tensor([0, 1, 4]),
        target,
        torch.tensor([True, False, True]),
    )
    assert set(loss) == {"structural", "conditional_event", "geometry_delta", "total"}
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert head.geometry_delta_head[-1].weight.grad is not None
    assert head.event_head.change_encoder[0].weight.grad is not None


def test_weighted_mae_is_zero_only_for_exact_delta() -> None:
    target = torch.tensor([[1.0, 0.2, -0.1, 0.001]])
    assert geometry_delta_weighted_mae(target, target).item() == 0.0
    assert geometry_delta_weighted_mae(torch.zeros_like(target), target).item() > 0.0
