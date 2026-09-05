from __future__ import annotations

import pytest
import torch

from mtare_topo.representation.gse_spatial_event_center import (
    SpatialEventCenterDecoder,
    SpatialLongitudinalCorrector,
    circular_coordinate_summary,
)


def _inputs(batch: int = 3):
    return (
        torch.randn(batch, 5, 128, 36),
        torch.randn(batch, 5, 128, 2),
        torch.randn(batch, 5, 128),
        torch.linspace(-2.0, 2.0, batch),
    )


def test_spatial_decoder_preserves_frozen_longitudinal_and_starts_at_zero_transverse():
    model = SpatialEventCenterDecoder()
    inputs = _inputs()
    output = model(*inputs)
    assert output.shape == (3, 3)
    torch.testing.assert_close(output[:, 0], inputs[-1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[:, 1:], torch.zeros(3, 2), rtol=0.0, atol=0.0)
    output[:, 1:].sum().backward()
    assert model.transverse_head[-1].weight.grad is not None


def test_circular_coordinate_summary_retains_left_right_sign():
    left = torch.zeros(1, 1, 36)
    right = torch.zeros(1, 1, 36)
    left[0, 0, 9] = 1.0
    right[0, 0, 27] = 1.0
    left_summary = circular_coordinate_summary(left)
    right_summary = circular_coordinate_summary(right)
    torch.testing.assert_close(left_summary[:, :3], right_summary[:, :3], atol=1e-6, rtol=0.0)
    assert left_summary[0, 3] > 0.0
    assert right_summary[0, 3] < 0.0


def test_spatial_decoder_rejects_contract_drift():
    model = SpatialEventCenterDecoder()
    azimuth, elevation, pooled, longitudinal = _inputs(2)
    with pytest.raises(ValueError, match="azimuth_sequence"):
        model(azimuth[..., :-1], elevation, pooled, longitudinal)
    with pytest.raises(ValueError, match="support"):
        model(azimuth, elevation, pooled, torch.tensor([13.0, 0.0]))


def test_longitudinal_corrector_starts_exactly_at_spatial_decoder_and_freezes_it():
    base = SpatialEventCenterDecoder()
    corrector = SpatialLongitudinalCorrector(base)
    inputs = _inputs(4)
    expected = base(*inputs)
    actual = corrector(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    corrector.freeze_spatial_decoder()
    assert not any(parameter.requires_grad for parameter in base.parameters())
    actual[:, 0].sum().backward()
    assert corrector.longitudinal_residual[-1].weight.grad is not None


def test_longitudinal_corrector_remains_in_frozen_support():
    model = SpatialLongitudinalCorrector()
    with torch.no_grad():
        model.longitudinal_residual[-1].bias.fill_(10.0)
    azimuth, elevation, pooled, _ = _inputs(2)
    output = model(azimuth, elevation, pooled, torch.tensor([11.5, -11.5]))
    assert bool((output[:, 0].abs() <= 12.0).all())
