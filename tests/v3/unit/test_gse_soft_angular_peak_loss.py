from __future__ import annotations

import torch

from mtare_topo.representation.gse_soft_angular_peak_loss import (
    circular_soft_peak_target,
    soft_angular_hard_negative_presence_loss,
)


def _presence() -> torch.Tensor:
    value = torch.zeros(2, 180, dtype=torch.uint8)
    value[0, [0, 40]] = 1
    value[1, [10, 14, 100]] = 1
    return value


def test_soft_target_preserves_centers_close_peaks_wrap_and_rotation():
    presence = _presence()
    target = circular_soft_peak_target(presence)
    assert torch.equal(target[presence.bool()], torch.ones(int(presence.sum())))
    assert target[0, 179] > 0 and target[0, 5] == 0
    assert target[1, 10] == 1 and target[1, 14] == 1
    shifted = circular_soft_peak_target(torch.roll(presence, 17, dims=1))
    assert torch.equal(shifted, torch.roll(target, 17, dims=1))


def test_correct_centers_rank_better_than_shifted_far_and_ghost_logits():
    presence = _presence()[:1]
    correct = torch.full((1, 180), -5.0)
    correct[presence.bool()] = 5.0
    shifted = torch.roll(correct, 1, dims=1)
    far = torch.roll(correct, 12, dims=1)
    ghost = correct.clone(); ghost[0, 100] = 6.0
    values = [soft_angular_hard_negative_presence_loss(item, presence)["total"] for item in (correct, shifted, far, ghost)]
    assert values[0] < values[1] < values[2]
    assert values[0] < values[3]


def test_high_confidence_hard_negative_receives_nonzero_suppressing_gradient():
    presence = _presence()[:1]
    logits = torch.full((1, 180), -3.0, requires_grad=True)
    with torch.no_grad():
        logits[presence.bool()] = 3.0
        logits[0, 100] = 4.0
    loss = soft_angular_hard_negative_presence_loss(logits, presence)
    loss["total"].backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 100] > 0
    assert logits.grad[0, 0] < 0


def test_loss_is_finite_for_realistic_random_batch():
    torch.manual_seed(91)
    logits = torch.randn(8, 180, requires_grad=True)
    presence = torch.zeros(8, 180, dtype=torch.uint8)
    for row in range(8):
        presence[row, [row * 7 % 180, (row * 7 + 70) % 180]] = 1
    losses = soft_angular_hard_negative_presence_loss(logits, presence)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert torch.isfinite(logits.grad).all()


def test_vectorized_loss_matches_explicit_rows_and_gradients():
    torch.manual_seed(17)
    logits = torch.randn(4, 180, dtype=torch.float64, requires_grad=True)
    presence = torch.zeros(4, 180, dtype=torch.uint8)
    presence[0, 3] = 1
    presence[1, [0, 179]] = 1
    presence[2, [20, 80, 140]] = 1
    presence[3, [10, 50, 90, 130]] = 1

    vectorized = soft_angular_hard_negative_presence_loss(logits, presence)["total"]
    vectorized_gradient = torch.autograd.grad(vectorized, logits, retain_graph=True)[0]
    explicit = torch.stack(
        [
            soft_angular_hard_negative_presence_loss(
                logits[row : row + 1], presence[row : row + 1]
            )["total"]
            for row in range(len(logits))
        ]
    ).mean()
    explicit_gradient = torch.autograd.grad(explicit, logits)[0]

    assert torch.allclose(vectorized, explicit, atol=1e-12, rtol=1e-12)
    assert torch.allclose(vectorized_gradient, explicit_gradient, atol=1e-12, rtol=1e-12)
