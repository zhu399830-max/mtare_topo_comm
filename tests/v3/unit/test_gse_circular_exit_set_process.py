from __future__ import annotations

import torch

from mtare_topo.representation.gse_circular_exit_set_process import (
    CausalCircularExitSetProcessNet,
    circular_exit_set_process_loss,
    decode_circular_exit_set,
)


def _decode_outputs(bins: list[list[int]]) -> dict[str, torch.Tensor]:
    batch = len(bins)
    logits = torch.full((batch, 180), -8.0)
    count_logits = torch.full((batch, 4), -8.0)
    for row, values in enumerate(bins):
        count_logits[row, len(values) - 1] = 8.0
        for rank, value in enumerate(values):
            logits[row, value] = 8.0 - rank
    return {
        "exit_mass": torch.softmax(logits, dim=-1),
        "exit_count_probability": torch.softmax(count_logits, dim=-1),
        "peak_heading_residual_deg": torch.zeros(batch, 180),
        "peak_opening_width_m": torch.ones(batch, 180),
        "peak_vertical_profile_m": torch.zeros(batch, 180, 4),
        "peak_descriptor": torch.zeros(batch, 180, 32),
        "peak_geometry_uncertainty": torch.ones(batch, 180, 6),
    }


def test_decode_preserves_wrap_nearby_exits_and_exact_cardinality() -> None:
    decoded = decode_circular_exit_set(_decode_outputs([[0], [179, 1], [10, 50, 90], [0, 2, 80, 120]]))
    assert torch.equal(decoded.count, torch.tensor([1, 2, 3, 4]))
    assert torch.equal(decoded.valid_mask.sum(dim=1), decoded.count)
    assert set(decoded.bin_index[1, :2].tolist()) == {179, 1}
    assert set(decoded.bin_index[3].tolist()) == {0, 2, 80, 120}


def test_set_mass_budget_penalizes_shift_and_ghost_without_slot_order() -> None:
    target = torch.zeros(1, 180, dtype=torch.uint8)
    target[0, [0, 40]] = 1
    correct = torch.full((1, 180), -8.0); correct[0, [0, 40]] = 8.0
    shifted = torch.roll(correct, 1, dims=1)
    ghost = correct.clone(); ghost[0, 100] = 10.0
    def nll(logits: torch.Tensor) -> torch.Tensor:
        return -(torch.log_softmax(logits, -1) * target).sum() / target.sum()
    assert nll(correct) < nll(shifted)
    assert nll(correct) < nll(ghost)
    assert torch.equal(target, target.flip(0))


def test_model_rotation_count_and_finite_backward_on_synthetic_targets() -> None:
    torch.manual_seed(3)
    model = CausalCircularExitSetProcessNet()
    scans = torch.zeros(4, 5, 2, 16, 720)
    scans[:, :, 1] = 1.0
    targets = {
        "presence": torch.zeros(4, 180, dtype=torch.uint8),
        "heading_residual_deg": torch.zeros(4, 180),
        "opening_width_m": torch.ones(4, 180),
        "width_valid_mask": torch.zeros(4, 180, dtype=torch.uint8),
        "vertical_profile_m": torch.zeros(4, 180, 4),
        "local_axis": torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1),
        "geometry": torch.ones(4, 4),
        "geometry_valid_mask": torch.ones(4, 4, dtype=torch.uint8),
    }
    for row in range(4):
        indices = torch.arange(row + 1) * 30
        targets["presence"][row, indices] = 1
        targets["width_valid_mask"][row, indices] = 1
    outputs = model(scans)
    loss = circular_exit_set_process_loss(outputs, targets)
    loss["total"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        base = model(scans)
        rotated = model(torch.roll(scans, 40, dims=-1))
    assert torch.allclose(rotated["exit_mass"], torch.roll(base["exit_mass"], 10, dims=1), atol=2e-6)
    assert torch.allclose(rotated["exit_count_probability"], base["exit_count_probability"], atol=2e-6)
