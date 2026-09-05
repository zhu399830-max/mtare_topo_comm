from __future__ import annotations

import math

import pytest
import torch

from mtare_topo.representation.gse_geometry_anchored_event import (
    GeometryAnchoredSpatialEventEncoder,
    geometry_anchored_input_contract,
)
from mtare_topo.representation.gse_spatial_event_set import spatial_event_set_loss


def _scans(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260828)
    ranges = 0.3 + 49.7 * torch.rand(batch, 5, 16, 720)
    valid = (torch.rand(batch, 5, 16, 720) > 0.2).to(torch.uint8)
    ranges = torch.where(valid.bool(), ranges, torch.full_like(ranges, 50.0))
    return ranges, valid


def _targets(cardinalities: tuple[int, ...]) -> dict[str, torch.Tensor]:
    batch = len(cardinalities)
    event_type = torch.full((batch, 16), -1, dtype=torch.int64)
    relative = torch.zeros(batch, 16, 3)
    identity = torch.full((batch, 16), -1, dtype=torch.int64)
    mask = torch.zeros(batch, 16, dtype=torch.uint8)
    for row, count in enumerate(cardinalities):
        for index in range(count):
            event_type[row, index] = index % 2
            relative[row, index] = torch.tensor((2.0 + index, 0.5 * index, 0.1 * index))
            identity[row, index] = row * 100 + index
            mask[row, index] = 1
    return {
        "event_type_index": event_type,
        "event_relative_xyz_m": relative,
        "event_identity_index": identity,
        "event_mask": mask,
    }


def test_pose_free_typed_interface_and_physical_range_bound() -> None:
    model = GeometryAnchoredSpatialEventEncoder().eval()
    ranges, valid = _scans()
    with torch.no_grad():
        output = model(ranges, valid)
    assert output["event_presence_logits"].shape == (2, 16)
    assert output["event_type_logits"].shape == (2, 16, 2)
    assert output["event_relative_xyz_m"].shape == (2, 16, 3)
    assert output["event_descriptor"].shape == (2, 16, 64)
    assert output["event_attention"].shape == (2, 16, 180)
    assert torch.all(output["event_radial_distance_m"] <= output["event_free_range_anchor_m"])
    assert torch.all(output["event_free_range_anchor_m"] <= 50.0001)
    contract = geometry_anchored_input_contract()
    assert contract["student_inputs"] == ("five_frame_range_m", "five_frame_valid_mask")
    assert "world_pose" in contract["forbidden_inputs"]
    with pytest.raises(TypeError):
        model(ranges, valid, torch.zeros(2, 3))


def test_circular_scan_roll_rotates_xy_and_preserves_semantics() -> None:
    model = GeometryAnchoredSpatialEventEncoder().eval()
    ranges, valid = _scans(batch=1)
    column_roll = 36
    with torch.no_grad():
        original = model(ranges, valid)
        rolled = model(
            torch.roll(ranges, shifts=column_roll, dims=-1),
            torch.roll(valid, shifts=column_roll, dims=-1),
        )
    angle = 2.0 * math.pi * column_roll / 720
    xyz = original["event_relative_xyz_m"]
    expected = torch.stack(
        (
            math.cos(angle) * xyz[..., 0] - math.sin(angle) * xyz[..., 1],
            math.sin(angle) * xyz[..., 0] + math.cos(angle) * xyz[..., 1],
            xyz[..., 2],
        ),
        dim=-1,
    )
    assert rolled["event_relative_xyz_m"] == pytest.approx(expected, abs=6e-4)
    assert rolled["event_free_range_anchor_m"] == pytest.approx(
        original["event_free_range_anchor_m"], abs=2e-5
    )
    for name in (
        "event_presence_logits",
        "event_type_logits",
        "event_descriptor",
        "event_uncertainty_m",
        "event_radial_fraction",
    ):
        assert rolled[name] == pytest.approx(original[name], abs=2e-5)


def test_empty_through_five_event_loss_and_backward_are_finite() -> None:
    model = GeometryAnchoredSpatialEventEncoder()
    ranges, valid = _scans(batch=6)
    output = model(ranges, valid)
    losses = spatial_event_set_loss(output, _targets((0, 1, 2, 3, 4, 5)))
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_invalid_scan_contract_fails_closed() -> None:
    model = GeometryAnchoredSpatialEventEncoder()
    ranges, valid = _scans(batch=1)
    with pytest.raises(ValueError, match="input contract"):
        model(ranges[..., :-1], valid[..., :-1])
    bad = ranges.clone()
    bad[..., 0] = 51.0
    with pytest.raises(ValueError, match="input contract"):
        model(bad, valid)
    bad_valid = valid.float()
    bad_valid[..., 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        model(ranges, bad_valid)


def test_free_range_anchor_uses_causal_and_one_bin_local_envelope() -> None:
    model = GeometryAnchoredSpatialEventEncoder().eval()
    ranges = torch.ones(1, 5, 16, 720)
    valid = torch.ones_like(ranges, dtype=torch.uint8)
    ranges[:, 0, :, 40:44] = 20.0
    with torch.no_grad():
        _, profile = model.polar_inputs(ranges, valid)
    assert profile[0, 9:12] == pytest.approx(torch.full((3,), 20.25), abs=1e-6)
    assert profile[0, 8] == pytest.approx(1.25, abs=1e-6)
    assert profile[0, 12] == pytest.approx(1.25, abs=1e-6)
