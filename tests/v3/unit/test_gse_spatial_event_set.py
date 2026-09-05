from __future__ import annotations

import math

import pytest
import torch

from mtare_topo.representation.gse_spatial_event_set import (
    SpatialEventSetConfig,
    SpatialEventSetDecoder,
    spatial_event_set_assignments,
    spatial_event_set_loss,
    validate_spatial_event_targets,
)


def _features(batch=2, azimuth=24):
    torch.manual_seed(7)
    return {
        "context": torch.randn(batch, 128),
        "directional": torch.randn(batch, 128, azimuth),
    }


def _targets(cardinalities=(0, 3)):
    batch = len(cardinalities)
    event_type = torch.full((batch, 16), -1, dtype=torch.int8)
    relative = torch.zeros(batch, 16, 3)
    identity = torch.full((batch, 16), -1, dtype=torch.int32)
    mask = torch.zeros(batch, 16, dtype=torch.uint8)
    for row, count in enumerate(cardinalities):
        for index in range(count):
            event_type[row, index] = index % 2
            relative[row, index] = torch.tensor((3.0 + index, -1.0 + index, 0.2 * index))
            identity[row, index] = 100 * row + index
            mask[row, index] = 1
    return {
        "event_type_index": event_type,
        "event_relative_xyz_m": relative,
        "event_identity_index": identity,
        "event_mask": mask,
    }


def test_frozen_config_and_output_interface():
    decoder = SpatialEventSetDecoder()
    output = decoder(_features())
    assert output["event_presence_logits"].shape == (2, 16)
    assert output["event_type_logits"].shape == (2, 16, 2)
    assert output["event_relative_xyz_m"].shape == (2, 16, 3)
    assert output["event_descriptor"].shape == (2, 16, 64)
    assert output["event_uncertainty_m"].shape == (2, 16)
    assert torch.all(torch.linalg.vector_norm(output["event_relative_xyz_m"], dim=-1) <= 50.0001)
    with pytest.raises(ValueError):
        SpatialEventSetConfig(query_count=5)


def test_empty_and_one_to_five_event_sets_match_and_backpropagate():
    decoder = SpatialEventSetDecoder()
    for cardinality in range(6):
        output = decoder(_features(batch=1))
        targets = _targets((cardinality,))
        assignments = spatial_event_set_assignments(output, targets)
        assert len(assignments[0]) == cardinality
        losses = spatial_event_set_loss(output, targets)
        assert all(torch.isfinite(value) for value in losses.values())
        decoder.zero_grad(set_to_none=True)
        losses["total"].backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in decoder.parameters()
        )


def test_matching_and_loss_are_query_permutation_invariant():
    decoder = SpatialEventSetDecoder()
    outputs = decoder(_features())
    targets = _targets((2, 4))
    original = spatial_event_set_loss(outputs, targets)
    permutation = torch.tensor([7, 3, 15, 0, 9, 4, 2, 13, 5, 10, 1, 14, 6, 8, 12, 11])
    permuted = {
        name: value[:, permutation] if value.ndim >= 2 and value.shape[1] == 16 else value
        for name, value in outputs.items()
    }
    changed = spatial_event_set_loss(permuted, targets)
    for name in ("total", "presence", "event_type", "position", "descriptor", "uncertainty"):
        assert float(changed[name].detach()) == pytest.approx(float(original[name].detach()), abs=1e-6)


def test_circular_feature_roll_rotates_xy_and_preserves_other_predictions():
    decoder = SpatialEventSetDecoder().eval()
    features = _features(batch=1, azimuth=24)
    with torch.no_grad():
        original = decoder(features)
        rolled = decoder(
            {
                "context": features["context"],
                "directional": torch.roll(features["directional"], shifts=4, dims=-1),
            }
        )
    angle = 2.0 * math.pi * 4 / 24
    expected_forward = math.cos(angle) * original["event_relative_xyz_m"][..., 0] - math.sin(angle) * original["event_relative_xyz_m"][..., 1]
    expected_left = math.sin(angle) * original["event_relative_xyz_m"][..., 0] + math.cos(angle) * original["event_relative_xyz_m"][..., 1]
    assert rolled["event_relative_xyz_m"][..., 0] == pytest.approx(expected_forward, abs=2e-4)
    assert rolled["event_relative_xyz_m"][..., 1] == pytest.approx(expected_left, abs=2e-4)
    assert rolled["event_relative_xyz_m"][..., 2] == pytest.approx(original["event_relative_xyz_m"][..., 2], abs=2e-4)
    for name in ("event_presence_logits", "event_type_logits", "event_descriptor", "event_uncertainty_m"):
        assert rolled[name] == pytest.approx(original[name], abs=2e-5)


def test_padding_and_duplicate_identity_contract_fails_closed():
    targets = _targets((2,))
    validate_spatial_event_targets(targets)
    targets["event_type_index"][0, 5] = 0
    with pytest.raises(ValueError, match="padding sentinel"):
        validate_spatial_event_targets(targets)
    targets = _targets((2,))
    targets["event_identity_index"][0, 1] = targets["event_identity_index"][0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_spatial_event_targets(targets)


def test_descriptor_loss_uses_repeated_teacher_identity_without_identity_input():
    decoder = SpatialEventSetDecoder()
    outputs = decoder(_features(batch=3))
    targets = _targets((1, 1, 1))
    targets["event_identity_index"][1, 0] = targets["event_identity_index"][0, 0]
    losses = spatial_event_set_loss(outputs, targets)
    assert losses["descriptor"] > 0.0
    assert torch.isfinite(losses["total"])


def test_fit_only_presence_and_type_weights_are_supported_and_validated():
    decoder = SpatialEventSetDecoder()
    outputs = decoder(_features(batch=2))
    targets = _targets((1, 2))
    losses = spatial_event_set_loss(
        outputs,
        targets,
        presence_positive_weight=21.86559723394846,
        event_type_class_weights=torch.tensor((2.1448713, 0.6519876)),
    )
    assert torch.isfinite(losses["total"])
    with pytest.raises(ValueError, match="presence positive"):
        spatial_event_set_loss(outputs, targets, presence_positive_weight=0.0)
    with pytest.raises(ValueError, match="class weights"):
        spatial_event_set_loss(outputs, targets, event_type_class_weights=torch.ones(3))
