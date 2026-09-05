from __future__ import annotations

import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_slope_corrective import (
    PhysicsGuidedSlopeResidualNet,
    SlopeCorrectiveConfig,
    slope_corrective_loss,
    slope_sequence_features,
)
from mtare_topo.data.gse_slope_corrective_dataset import (
    corrective_partition,
    fit_normalization,
    normalize_features,
    validate_causal_references,
)


class _Predictor:
    def __init__(self) -> None:
        self.index = 0

    def predict(self, range_m, valid_mask):
        del range_m, valid_mask
        index = self.index
        self.index += 1
        return {
            "slope_deg": float(index - 2),
            "width_m": 4.0 + index,
            "height_m": 3.0 + index,
            "curvature_per_m": 0.01 * index,
            "local_return_count": 100 + index,
            "populated_sections": 5,
        }


def test_sequence_features_are_five_frame_causal_and_prior_is_mean():
    ranges = np.ones((5, 16, 720), dtype=np.float32)
    valid = np.ones_like(ranges, dtype=np.uint8)
    features, prior = slope_sequence_features(ranges, valid, predictor=_Predictor())
    assert features.shape == (5, 6)
    assert features[:, 0].tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert prior == pytest.approx(0.0)


def test_sequence_features_reject_wrong_history_shape():
    with pytest.raises(ValueError, match="five"):
        slope_sequence_features(
            np.ones((4, 16, 720), dtype=np.float32),
            np.ones((4, 16, 720), dtype=np.uint8),
            predictor=_Predictor(),
        )


def test_zero_initialized_residual_preserves_physics_prior():
    model = PhysicsGuidedSlopeResidualNet()
    features = torch.zeros((3, 5, 6))
    prior = torch.tensor([-4.0, 0.0, 7.0])
    output = model(features, prior)
    assert torch.equal(output["slope_deg"], prior)
    assert torch.equal(output["correction_deg"], torch.zeros_like(prior))
    assert bool(torch.all(output["error_scale_deg"] > 0.0))


def test_residual_is_bounded_and_loss_has_finite_gradients():
    torch.manual_seed(7)
    model = PhysicsGuidedSlopeResidualNet()
    features = torch.randn((8, 5, 6))
    prior = torch.linspace(-8.0, 8.0, 8)
    target = prior + torch.linspace(-2.0, 2.0, 8)
    output = model(features, prior)
    assert bool(torch.all(torch.abs(output["correction_deg"]) <= 10.0))
    losses = slope_corrective_loss(output, target)
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())


def test_model_rejects_pose_or_arbitrary_feature_shapes():
    model = PhysicsGuidedSlopeResidualNet()
    with pytest.raises(ValueError, match=r"\[B,5,6\]"):
        model(torch.zeros((2, 5, 7)), torch.zeros(2))


def test_corrective_partition_is_world_disjoint_and_rejects_c09():
    assert corrective_partition("S01_flat_tree_small_C01") == "fit"
    assert corrective_partition("S10_3d_complex_C06") == "fit"
    assert corrective_partition("S01_flat_tree_small_C07") == "selection"
    assert corrective_partition("S10_3d_complex_C08") == "selection"
    with pytest.raises(ValueError, match="approved"):
        corrective_partition("S01_flat_tree_small_C09")
    with pytest.raises(ValueError, match="approved"):
        corrective_partition("forged_C01")


def test_fit_only_normalization_round_trip():
    first = np.arange(60, dtype=np.float32).reshape(2, 5, 6)
    second = np.arange(60, 120, dtype=np.float32).reshape(2, 5, 6)
    mean, scale = fit_normalization((first, second))
    normalized = normalize_features(np.concatenate((first, second)), mean, scale)
    assert normalized.shape == (4, 5, 6)
    assert np.mean(normalized, axis=(0, 1)) == pytest.approx(np.zeros(6), abs=1e-6)
    assert np.std(normalized, axis=(0, 1)) == pytest.approx(np.ones(6), abs=1e-6)


def test_masks_and_numeric_inputs_fail_closed():
    ranges = np.ones((5, 16, 720), dtype=np.float32)
    with pytest.raises(ValueError, match="bool or uint8"):
        slope_sequence_features(ranges, np.ones_like(ranges), predictor=_Predictor())
    invalid = np.ones_like(ranges, dtype=np.uint8)
    invalid[0, 0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        slope_sequence_features(ranges, invalid, predictor=_Predictor())
    model = PhysicsGuidedSlopeResidualNet()
    features = torch.zeros((2, 5, 6))
    features[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(features, torch.zeros(2))


def test_config_and_loss_use_the_explicit_residual_limit():
    with pytest.raises(ValueError, match="positive"):
        SlopeCorrectiveConfig(maximum_residual_deg=float("nan"))
    model = PhysicsGuidedSlopeResidualNet(SlopeCorrectiveConfig(maximum_residual_deg=2.5))
    output = model(torch.zeros((2, 5, 6)), torch.zeros(2))
    target = torch.ones(2)
    loss = slope_corrective_loss(output, target, maximum_residual_deg=2.5)
    assert bool(torch.isfinite(loss["total"]))
    with pytest.raises(ValueError, match="positive"):
        slope_corrective_loss(output, target, maximum_residual_deg=float("nan"))


def test_normalization_rejects_nonfinite_features():
    block = np.ones((2, 5, 6), dtype=np.float32)
    block[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_normalization((block,))


def test_causal_reference_contract_allows_local_index_reset_between_traversals():
    references = np.asarray([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int64)
    local_frame_index = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64)
    global_frame_index = np.arange(100, 110, dtype=np.int64)
    validate_causal_references(
        references,
        global_frame_index[references],
        global_frame_index,
        local_frame_index,
        frame_count=10,
    )


def test_causal_reference_contract_rejects_sequence_crossing_a_local_reset():
    references = np.asarray([[2, 3, 4, 5, 6]], dtype=np.int64)
    local_frame_index = np.asarray([0, 1, 2, 3, 4, 0, 1], dtype=np.int64)
    global_frame_index = np.arange(7, dtype=np.int64)
    with pytest.raises(RuntimeError, match="causal"):
        validate_causal_references(
            references,
            global_frame_index[references],
            global_frame_index,
            local_frame_index,
            frame_count=7,
        )
