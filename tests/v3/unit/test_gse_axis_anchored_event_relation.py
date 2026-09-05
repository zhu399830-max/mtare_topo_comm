from __future__ import annotations

import inspect
import math

import pytest
import torch

from mtare_topo.representation.gse_axis_anchored_event_relation import (
    AxisAnchoredEventRelationNet,
    axis_anchored_descriptor_loss,
    axis_anchored_event_relation_core_loss,
    axis_anchored_event_relation_input_contract,
    axis_anchored_event_relation_loss,
    causal_branch_union_probability,
    reverse_axis_field,
)


def _scans(batch: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(913)
    ranges = 0.01 + 0.99 * torch.rand(batch, 5, 1, 16, 720, generator=generator)
    return torch.cat((ranges, torch.ones_like(ranges)), dim=2)


def _targets(batch: int = 4) -> dict[str, torch.Tensor]:
    relation = torch.zeros(batch, 4, 180, 3, dtype=torch.long)
    relation[:, :, 10, 0] = 1
    relation[0:2, 1:, 70, 1] = 1
    relation[2:, 0, 120, 1] = 1
    relation[2:, 1:, 120, 2] = 1
    relation[:, 0, 150, 2] = 1
    relation[0, 2, 150, 1] = 1
    present = torch.zeros(batch, 180, dtype=torch.bool)
    present[:, 10] = True
    present[0:2, 70] = True
    width_valid = present.clone()
    width_valid[0, 70] = False
    residual = torch.zeros(batch, 180)
    residual[present] = torch.linspace(-0.8, 0.8, int(present.sum()))
    width = torch.zeros(batch, 180)
    width[present] = torch.linspace(2.0, 8.0, int(present.sum()))
    profile = torch.zeros(batch, 180, 4)
    profile[present] = torch.linspace(-2.0, 2.0, int(present.sum()) * 4).reshape(-1, 4)
    branch_identity = torch.full((batch, 180), -1, dtype=torch.long)
    branch_identity[:, 10] = 100
    branch_identity[0:2, 70] = 200
    return {
        "event_index": torch.tensor([0, 1, 2, 4]),
        "relation_index": relation,
        "branch_heading_residual_deg": residual,
        "branch_opening_width_m": width,
        "branch_width_valid_mask": width_valid,
        "branch_vertical_profile_m": profile,
        "branch_identity": branch_identity,
        "local_axis": torch.tensor([
            [1.0, 0.0, 0.0], [0.99, 0.0, 0.1], [1.0, 0.0, -0.1], [1.0, 0.0, 0.0]
        ]),
        "geometry": torch.tensor([
            [5.0, 4.0, 0.0, 0.01], [6.0, 5.0, 4.0, 0.02],
            [4.0, 3.0, -3.0, 0.03], [7.0, 6.0, 2.0, 0.04],
        ]),
        "geometry_valid_mask": torch.ones(batch, 4, dtype=torch.bool),
        "association_identity": torch.tensor([10, 10, 20, 20]),
    }


def test_forward_is_typed_and_identity_free() -> None:
    model = AxisAnchoredEventRelationNet()
    output = model(_scans(2))
    assert tuple(inspect.signature(model.forward).parameters) == ("scans",)
    assert output["event_probability"].shape == (2, 5)
    assert output["relation_probability_sequence"].shape == (2, 4, 180, 3)
    assert output["branch_union_probability"].shape == (2, 180)
    assert output["branch_descriptor"].shape == (2, 180, 32)
    assert output["place_descriptor"].shape == (2, 128)
    contract = axis_anchored_event_relation_input_contract()
    assert contract["forward_parameters"] == ("scans",)
    assert "association_identity" in contract["forbidden_forward_inputs"]
    assert contract["edge_rule"] == "physical traversal only"


def test_aligned_rotation_is_equivariant_and_event_invariant() -> None:
    torch.manual_seed(29)
    model = AxisAnchoredEventRelationNet().eval()
    scans = _scans(1)
    shift_columns = 40
    shift_bins = 10
    angle = 2.0 * math.pi * shift_bins / 180
    with torch.no_grad():
        base = model(scans)
        rotated = model(torch.roll(scans, shift_columns, dims=-1))
    for name in ("relation_logits_sequence", "relation_probability_sequence"):
        assert torch.max(torch.abs(rotated[name] - torch.roll(base[name], shift_bins, dims=2))) < 3e-5
    for name in (
        "branch_relation_logits", "branch_union_probability", "branch_heading_residual_deg",
        "branch_opening_width_m", "branch_vertical_profile_m", "branch_descriptor",
        "branch_geometry_uncertainty",
    ):
        assert torch.max(torch.abs(rotated[name] - torch.roll(base[name], shift_bins, dims=1))) < 3e-5
    for name in ("event_probability", "width_m", "height_m", "slope_deg", "curvature_per_m", "place_descriptor"):
        assert torch.max(torch.abs(rotated[name] - base[name])) < 3e-5
    axis = base["local_axis"]
    expected = torch.stack((
        math.cos(angle) * axis[:, 0] - math.sin(angle) * axis[:, 1],
        math.sin(angle) * axis[:, 0] + math.cos(angle) * axis[:, 1],
    ), dim=-1)
    assert torch.max(torch.abs(rotated["local_axis"][:, :2] - expected)) < 3e-5


def test_batch_permutation_is_numerically_recoverable() -> None:
    torch.manual_seed(31)
    model = AxisAnchoredEventRelationNet().eval()
    scans = _scans(4)
    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(order)
    with torch.no_grad():
        reference = model(scans)
        permuted = model(scans[order])
    for name in ("event_probability", "branch_relation_probability", "branch_union_probability", "place_descriptor"):
        assert torch.max(torch.abs(reference[name] - permuted[name][inverse])) < 3e-6


def test_reverse_axis_transform_is_an_involution() -> None:
    values = torch.arange(2 * 180 * 4).reshape(2, 180, 4)
    reversed_once = reverse_axis_field(values, bearing_dimension=1)
    assert torch.equal(reverse_axis_field(reversed_once, bearing_dimension=1), values)
    with pytest.raises(ValueError, match="180"):
        reverse_axis_field(torch.zeros(2, 90))


def test_past_only_branch_union_uses_reveal_persist_withdraw() -> None:
    relation = torch.zeros(1, 4, 180, 3)
    relation[0, 0, 5] = torch.tensor([0.0, 1.0, 0.0])
    relation[0, 1, 5] = torch.tensor([1.0, 0.0, 0.0])
    relation[0, 2, 5] = torch.tensor([1.0, 0.0, 0.0])
    relation[0, 3, 5] = torch.tensor([0.0, 0.0, 1.0])
    relation[0, 3, 9] = torch.tensor([0.0, 1.0, 0.0])
    relation[0, 3, 5, 1] = 1.0
    occupancy = causal_branch_union_probability(relation)
    assert occupancy[0, 5] == 1.0
    assert occupancy[0, 9] == 1.0


def test_multitask_loss_and_all_gradients_are_finite() -> None:
    torch.manual_seed(37)
    model = AxisAnchoredEventRelationNet()
    output = model(_scans())
    losses = axis_anchored_event_relation_loss(output, _targets())
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_invalid_scan_and_relation_population_fail_closed() -> None:
    scans = _scans(1)
    scans[0, 0, 1, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        AxisAnchoredEventRelationNet()(scans)
    model = AxisAnchoredEventRelationNet()
    targets = _targets()
    targets["relation_index"].zero_()
    with pytest.raises(ValueError, match="persistent/reveal/withdraw"):
        axis_anchored_event_relation_loss(model(_scans()), targets)


def test_training_core_masks_unlabeled_early_relations_and_splits_identity_loss() -> None:
    torch.manual_seed(41)
    model = AxisAnchoredEventRelationNet()
    outputs = model(_scans())
    targets = _targets()
    targets["relation_index"][:, :2] = -1
    targets["branch_presence_mask"] = targets["branch_identity"] >= 0
    targets["event_class_weight"] = torch.tensor([0.2, 0.8, 1.0, 2.0, 0.5])
    targets["relation_positive_rate"] = torch.tensor([0.01, 0.001, 0.001])
    core = axis_anchored_event_relation_core_loss(outputs, targets)
    descriptor = axis_anchored_descriptor_loss(outputs, targets)
    (core["total"] + descriptor["total"]).backward()
    assert all(torch.isfinite(value) for value in core.values())
    assert all(torch.isfinite(value) for value in descriptor.values())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
