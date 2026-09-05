from __future__ import annotations

import inspect
import math

import pytest
import torch

from mtare_topo.representation.gse_circular_peak_geometry_model import (
    CircularPeakGeometryConfig,
    CircularPeakGeometrySemanticNet,
    balanced_peak_presence_loss,
    circular_peak_geometry_input_contract,
    circular_peak_geometry_loss,
)


def _scans(batch: int = 3) -> torch.Tensor:
    torch.manual_seed(31)
    ranges = 0.006 + 0.994 * torch.rand(batch, 5, 1, 16, 720)
    valid = torch.ones_like(ranges)
    return torch.cat((ranges, valid), dim=2)


def _targets(batch: int = 3) -> dict[str, torch.Tensor]:
    presence = torch.zeros(batch, 180, dtype=torch.uint8)
    presence[0, [0, 70]] = 1
    presence[1, [4, 90, 179]] = 1
    presence[2, [23]] = 1
    width_valid = presence.clone()
    width_valid[1, 90] = 0
    residual = torch.zeros(batch, 180)
    residual[presence.bool()] = torch.linspace(-0.8, 0.8, int(presence.sum()))
    width = torch.zeros(batch, 180)
    width[presence.bool()] = torch.linspace(2.0, 30.0, int(presence.sum()))
    profile = torch.zeros(batch, 180, 4)
    profile[presence.bool()] = torch.linspace(-4.0, 4.0, int(presence.sum()) * 4).reshape(-1, 4)
    return {
        "presence": presence,
        "heading_residual_deg": residual,
        "opening_width_m": width,
        "width_valid_mask": width_valid,
        "vertical_profile_m": profile,
        "local_axis": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.1], [-1.0, 0.0, -0.1]]),
        "geometry": torch.tensor([[5.0, 4.0, 0.0, 0.01], [6.0, 5.0, 4.0, 0.02], [4.0, 3.0, -3.0, 0.03]]),
        "geometry_valid_mask": torch.ones(batch, 4, dtype=torch.uint8),
    }


def test_typed_dense_interface_has_no_query_or_identity_input():
    model = CircularPeakGeometrySemanticNet()
    outputs = model(_scans(batch=2))
    assert tuple(inspect.signature(model.forward).parameters) == ("scans",)
    assert outputs["peak_presence_logits"].shape == (2, 180)
    assert outputs["peak_heading_residual_deg"].shape == (2, 180)
    assert outputs["peak_vertical_profile_m"].shape == (2, 180, 4)
    assert outputs["peak_descriptor"].shape == (2, 180, 32)
    assert outputs["peak_geometry_uncertainty"].shape == (2, 180, 6)
    assert outputs["local_axis"].shape == (2, 3)
    assert not any("query" in name for name, _ in model.named_parameters())
    contract = circular_peak_geometry_input_contract()
    assert contract["peak_layout"] == (180,)
    assert "exit_identity" in contract["forbidden_inputs"]


def test_aligned_rotation_is_circularly_equivariant():
    torch.manual_seed(37)
    model = CircularPeakGeometrySemanticNet().eval()
    scans = _scans(batch=1)
    shift_columns = 40
    shift_bins = 10
    angle = 2.0 * math.pi * shift_bins / 180
    with torch.no_grad():
        base = model(scans)
        rotated = model(torch.roll(scans, shift_columns, dims=-1))
    for name in (
        "peak_presence_logits", "peak_heading_residual_deg", "peak_opening_width_m",
        "peak_vertical_profile_m", "peak_descriptor", "peak_geometry_uncertainty",
    ):
        assert torch.max(torch.abs(rotated[name] - torch.roll(base[name], shift_bins, dims=1))) < 3e-5
    axis = base["local_axis"]
    expected_xy = torch.stack(
        (math.cos(angle) * axis[:, 0] - math.sin(angle) * axis[:, 1],
         math.sin(angle) * axis[:, 0] + math.cos(angle) * axis[:, 1]), dim=-1
    )
    assert torch.max(torch.abs(rotated["local_axis"][:, :2] - expected_xy)) < 3e-5
    assert torch.max(torch.abs(rotated["local_axis"][:, 2] - axis[:, 2])) < 3e-5
    for name in ("width_m", "height_m", "slope_deg", "curvature_per_m", "place_descriptor"):
        assert torch.max(torch.abs(rotated[name] - base[name])) < 3e-5


def test_masked_multitask_loss_is_finite_and_ignores_invalid_targets():
    torch.manual_seed(41)
    model = CircularPeakGeometrySemanticNet()
    scans = _scans()
    targets = _targets()
    outputs = model(scans)
    first = circular_peak_geometry_loss(outputs, targets)
    changed = {name: value.clone() for name, value in targets.items()}
    changed["opening_width_m"][1, 90] = 1e6
    changed["vertical_profile_m"][~changed["presence"].bool()] = 1e6
    second = circular_peak_geometry_loss(outputs, changed)
    assert torch.equal(first["total"], second["total"])
    first["total"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert all(torch.isfinite(value) for value in first.values())


def test_balanced_presence_requires_both_classes():
    logits = torch.zeros(2, 180)
    with pytest.raises(ValueError, match="positive and negative"):
        balanced_peak_presence_loss(logits, torch.zeros_like(logits))


def test_invalid_scan_and_config_fail_closed():
    scans = _scans(batch=1)
    scans[0, 0, 1, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        CircularPeakGeometrySemanticNet()(scans)
    with pytest.raises(ValueError, match="config is frozen"):
        CircularPeakGeometryConfig(bearing_bins=90)
