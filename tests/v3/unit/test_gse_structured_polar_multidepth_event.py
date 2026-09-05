from __future__ import annotations

import inspect
import math

import pytest
import torch

from mtare_topo.representation.gse_structured_polar_multidepth_event import (
    StructuredPolarMultiDepthEventEncoder,
    rasterize_structured_polar_targets,
    structured_polar_multidepth_input_contract,
    structured_polar_multidepth_loss,
)


def _scan(batch: int = 2):
    torch.manual_seed(11)
    ranges = 0.3 + 49.7 * torch.rand(batch, 5, 16, 720)
    valid = torch.ones_like(ranges, dtype=torch.uint8)
    return ranges, valid


def _targets(batch: int = 2, third_same_bin: bool = False):
    event_type = torch.full((batch, 16), -1, dtype=torch.long)
    xyz = torch.zeros(batch, 16, 3)
    identity = torch.full((batch, 16), -1, dtype=torch.long)
    mask = torch.zeros(batch, 16, dtype=torch.uint8)
    angle = math.radians(0.75)
    distances = [4.0, 42.0] + ([46.0] if third_same_bin else [])
    for index, distance in enumerate(distances):
        event_type[0, index] = index % 2
        xyz[0, index] = torch.tensor((distance * math.cos(angle), distance * math.sin(angle), 0.0))
        identity[0, index] = 100 + index
        mask[0, index] = 1
    return {
        "event_type_index": event_type,
        "event_relative_xyz_m": xyz,
        "event_identity_index": identity,
        "event_mask": mask,
    }


def test_dense_and_topk_interface_is_range_bounded_and_pose_free():
    model = StructuredPolarMultiDepthEventEncoder()
    assert sum(parameter.numel() for parameter in model.parameters()) == 172430
    assert tuple(inspect.signature(model.forward).parameters) == ("range_m", "valid_mask")
    ranges, valid = _scan()
    outputs = model(ranges, valid)
    assert outputs["dense_event_presence_logits"].shape == (2, 180, 2)
    assert outputs["dense_event_relative_xyz_m"].shape == (2, 180, 2, 3)
    assert outputs["event_relative_xyz_m"].shape == (2, 16, 3)
    assert outputs["event_azimuth_bin_index"].shape == (2, 16)
    assert torch.max(outputs["dense_event_radial_distance_m"] - outputs["free_range_profile_m"][:, :, None]) <= 0
    contract = structured_polar_multidepth_input_contract()
    assert contract["dense_layout"] == (180, 2)
    assert "TNG_identity" in contract["forbidden_inputs"]


def test_same_bearing_targets_are_sorted_into_two_radial_slots():
    targets = _targets()
    profile = torch.full((2, 180), 50.0)
    dense = rasterize_structured_polar_targets(targets, profile)
    assert dense["event_mask"][0, 0].tolist() == [True, True]
    assert dense["event_identity_index"][0, 0].tolist() == [100, 101]
    assert torch.allclose(dense["event_radial_fraction"][0, 0], torch.tensor([0.08, 0.84]))
    assert int(dense["event_mask"].sum()) == 2
    with pytest.raises(ValueError, match="exceeds two depth slots"):
        rasterize_structured_polar_targets(_targets(third_same_bin=True), profile)


def test_dense_rotation_and_topk_coordinates_are_equivariant():
    model = StructuredPolarMultiDepthEventEncoder().eval()
    ranges, valid = _scan(batch=1)
    shift_columns = 40
    shift_bins = 10
    angle = 2.0 * math.pi * shift_columns / 720
    with torch.no_grad():
        base = model(ranges, valid)
        rotated = model(torch.roll(ranges, shift_columns, -1), torch.roll(valid, shift_columns, -1))
    assert torch.max(torch.abs(rotated["dense_event_presence_logits"] - torch.roll(base["dense_event_presence_logits"], shift_bins, 1))) < 2e-5
    xyz = base["dense_event_relative_xyz_m"]
    expected_xyz = torch.stack((math.cos(angle) * xyz[..., 0] - math.sin(angle) * xyz[..., 1], math.sin(angle) * xyz[..., 0] + math.cos(angle) * xyz[..., 1], xyz[..., 2]), dim=-1)
    assert torch.max(torch.abs(rotated["dense_event_relative_xyz_m"] - torch.roll(expected_xyz, shift_bins, 1))) < 2e-4
    expected_top = torch.stack((math.cos(angle) * base["event_relative_xyz_m"][..., 0] - math.sin(angle) * base["event_relative_xyz_m"][..., 1], math.sin(angle) * base["event_relative_xyz_m"][..., 0] + math.cos(angle) * base["event_relative_xyz_m"][..., 1], base["event_relative_xyz_m"][..., 2]), dim=-1)
    assert torch.max(torch.abs(rotated["event_relative_xyz_m"] - expected_top)) < 2e-4
    assert torch.equal(rotated["event_azimuth_bin_index"], (base["event_azimuth_bin_index"] + shift_bins) % 180)


def test_empty_and_multidepth_dense_loss_has_finite_backward_and_replay():
    torch.manual_seed(29)
    model = StructuredPolarMultiDepthEventEncoder()
    ranges, valid = _scan()
    outputs = model(ranges, valid)
    targets = rasterize_structured_polar_targets(_targets(), outputs["free_range_profile_m"].detach())
    losses = structured_polar_multidepth_loss(outputs, targets, presence_positive_weight=10.0)
    losses["total"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert all(torch.isfinite(value) for value in losses.values())
    model.eval()
    with torch.no_grad():
        first = model(ranges, valid)
        second = model(ranges, valid)
    for name in ("dense_event_presence_logits", "dense_event_relative_xyz_m", "event_confidence", "event_relative_xyz_m", "event_azimuth_bin_index", "event_depth_slot_index"):
        assert torch.equal(first[name], second[name])


def test_invalid_lidar_contract_fails_closed():
    model = StructuredPolarMultiDepthEventEncoder()
    ranges, valid = _scan(batch=1)
    ranges[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="input contract"):
        model(ranges, valid)
