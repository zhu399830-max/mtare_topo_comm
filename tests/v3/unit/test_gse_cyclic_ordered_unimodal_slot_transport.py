from __future__ import annotations

import torch

from mtare_topo.representation.gse_cyclic_ordered_unimodal_slot_transport import (
    CONTRACT,
    CyclicOrderedUnimodalSlotTransportNet,
    cyclic_ordered_unimodal_slot_loss,
    cyclic_orders,
    discrete_von_mises,
)
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import BRANCH_SLICE


def _targets() -> dict[str, torch.Tensor]:
    presence = torch.zeros(1, 180, dtype=torch.uint8)
    presence[0, [10, 50, 100]] = 1
    return {
        "presence": presence,
        "heading_residual_deg": torch.zeros(1, 180),
        "opening_width_m": torch.zeros(1, 180),
        "width_valid_mask": torch.zeros(1, 180, dtype=torch.uint8),
        "vertical_profile_m": torch.zeros(1, 180, 4),
        "local_axis": torch.tensor([[1.0, 0.0, 0.0]]),
        "geometry": torch.zeros(1, 4),
        "geometry_valid_mask": torch.ones(1, 4, dtype=torch.uint8),
    }


def _outputs(slot_bins: tuple[int, int, int], *, require_grad: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    logits = torch.full((1, 10, 180), -8.0)
    for slot, bearing_bin in zip(range(BRANCH_SLICE[3].start, BRANCH_SLICE[3].stop), slot_bins, strict=True):
        logits[0, slot, bearing_bin] = 8.0
    logits.requires_grad_(require_grad)
    mass = torch.softmax(logits, dim=-1)
    azimuth = torch.arange(180) * 2 * torch.pi / 180
    cosine = (mass * torch.cos(azimuth)).sum(-1)
    sine = (mass * torch.sin(azimuth)).sum(-1)
    outputs = {
        "slot_log_mass": torch.log_softmax(logits, dim=-1),
        "slot_bearing_deg": torch.remainder(torch.rad2deg(torch.atan2(sine, cosine)), 360.0),
        "slot_opening_width_m": torch.zeros(1, 10),
        "slot_vertical_profile_m": torch.zeros(1, 10, 4),
        "exit_count_logits": torch.tensor([[-8.0, -8.0, 8.0, -8.0]]),
        "local_axis": torch.tensor([[1.0, 0.0, 0.0]]),
        "width_m": torch.zeros(1), "height_m": torch.zeros(1),
        "slope_deg": torch.zeros(1), "curvature_per_m": torch.zeros(1),
    }
    return outputs, logits


def test_contract_has_no_free_query_and_cyclic_group_only() -> None:
    assert CONTRACT.free_query_existence is False
    assert cyclic_orders(3) == ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    assert (0, 2, 1) not in cyclic_orders(3)


def test_discrete_von_mises_is_normalized_unimodal_and_stable() -> None:
    azimuth = torch.arange(180, dtype=torch.float64) * 2 * torch.pi / 180
    mean = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    kappa = torch.tensor([[0.0, 1000.0]], dtype=torch.float64)
    log_mass, mass, bearing, concentration = discrete_von_mises(mean, kappa, azimuth)
    assert bool(torch.isfinite(log_mass).all())
    assert torch.allclose(mass.sum(-1), torch.ones(1, 2, dtype=torch.float64), atol=1e-12)
    assert concentration[0, 0] < 1e-12
    assert concentration[0, 1] > 0.999
    assert abs(float(bearing[0, 1]) - 90.0) < 1e-10
    assert int(torch.argmax(mass[0, 1])) == 45


def test_cyclic_assignment_penalizes_reversal() -> None:
    correct, _ = _outputs((10, 50, 100))
    reversed_output, _ = _outputs((10, 100, 50))
    correct_loss = cyclic_ordered_unimodal_slot_loss(correct, _targets())
    reversed_loss = cyclic_ordered_unimodal_slot_loss(reversed_output, _targets())
    assert correct_loss["assignment"] < reversed_loss["assignment"]
    assert correct_loss["total"] < reversed_loss["total"]


def test_duplicate_slots_receive_missing_mode_gradient() -> None:
    duplicate, logits = _outputs((10, 10, 50), require_grad=True)
    loss = cyclic_ordered_unimodal_slot_loss(duplicate, _targets())["total"]
    loss.backward()
    branch = BRANCH_SLICE[3]
    missing_gradient = min(float(logits.grad[0, slot, 100]) for slot in range(branch.start, branch.stop))
    assert missing_gradient < 0.0


def test_forward_shapes_and_parameter_contract() -> None:
    torch.manual_seed(7)
    model = CyclicOrderedUnimodalSlotTransportNet().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 788618
    assert not any("query" in name or "objectness" in name for name, _ in model.named_parameters())
    scans = torch.rand(2, 5, 2, 16, 720)
    scans[:, :, 1] = (scans[:, :, 1] > 0.5).float()
    with torch.no_grad():
        output = model(scans)
    assert output["slot_mass"].shape == (2, 10, 180)
    assert torch.allclose(output["slot_mass"].sum(-1), torch.ones(2, 10), atol=1e-6)
    assert bool(torch.isfinite(output["slot_kappa"]).all())
    assert torch.equal(output["decoded_valid_mask"].sum(-1), output["decoded_count"])
