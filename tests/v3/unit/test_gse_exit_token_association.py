import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_exit_token_association import (
    EXIT_TOKEN_PAIR_FEATURE_DIM,
    GSEExitTokenAssociationVerifier,
    TOKEN_AWARE_PAIR_FEATURE_DIM,
    exit_token_pair_features,
    hard_negative_tail_loss,
)
from mtare_topo.representation.gse_open_set_association import OBSERVATION_FEATURE_DIM


def _tokens(rows=3, seed=0):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, size=(rows, 6))
    heading = np.stack((np.sin(angle), np.cos(angle)), axis=2).astype(np.float32)
    descriptor = rng.normal(size=(rows, 6, 32)).astype(np.float32)
    descriptor /= np.linalg.norm(descriptor, axis=2, keepdims=True)
    return {
        "exit_confidence": rng.uniform(0.05, 0.95, size=(rows, 6)).astype(np.float32),
        "exit_heading_unit": heading,
        "exit_opening_width_m": rng.uniform(1.0, 8.0, size=(rows, 6)).astype(np.float32),
        "exit_vertical_profile": rng.normal(size=(rows, 6, 4)).astype(np.float32),
        "exit_descriptor": descriptor,
    }


def _permute(tokens, permutation):
    return {name: value[:, permutation] for name, value in tokens.items()}


def _rotate(tokens, radians):
    result = {name: value.copy() for name, value in tokens.items()}
    sine, cosine = np.sin(radians), np.cos(radians)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float32)
    result["exit_heading_unit"] = result["exit_heading_unit"] @ rotation.T
    return result


def test_exit_token_features_are_symmetric_and_have_frozen_shape():
    left = _tokens(seed=1)
    right = _tokens(seed=2)
    forward = exit_token_pair_features(left, right)
    reverse = exit_token_pair_features(right, left)
    assert forward.shape == (3, EXIT_TOKEN_PAIR_FEATURE_DIM)
    np.testing.assert_allclose(forward, reverse, rtol=0.0, atol=2e-7)


def test_exit_token_features_are_independently_permutation_invariant():
    left = _tokens(seed=3)
    right = _tokens(seed=4)
    expected = exit_token_pair_features(left, right)
    observed = exit_token_pair_features(
        _permute(left, [3, 0, 5, 1, 4, 2]),
        _permute(right, [1, 5, 2, 4, 0, 3]),
    )
    np.testing.assert_allclose(expected, observed, rtol=0.0, atol=2e-7)


def test_exit_token_features_ignore_independent_global_yaw():
    left = _tokens(seed=5)
    right = _tokens(seed=6)
    expected = exit_token_pair_features(left, right)
    observed = exit_token_pair_features(_rotate(left, 0.71), _rotate(right, -1.19))
    np.testing.assert_allclose(expected, observed, rtol=0.0, atol=3e-7)


def test_exit_token_features_reject_nonunit_descriptors():
    left = _tokens(seed=7)
    right = _tokens(seed=8)
    left["exit_descriptor"][0, 0] *= 2.0
    with pytest.raises(ValueError, match="descriptors"):
        exit_token_pair_features(left, right)


def test_token_aware_verifier_is_lightweight_and_shape_checked():
    model = GSEExitTokenAssociationVerifier()
    assert model.pair[0].in_features == TOKEN_AWARE_PAIR_FEATURE_DIM
    assert sum(parameter.numel() for parameter in model.parameters()) == 112898
    left = torch.zeros(2, OBSERVATION_FEATURE_DIM)
    pair = torch.zeros(2, TOKEN_AWARE_PAIR_FEATURE_DIM)
    output = model(left, left, pair)
    assert output["association_score"].shape == (2,)
    with pytest.raises(ValueError, match="contract"):
        model(left, left, pair[:, :-1])


def test_tail_risk_loss_targets_low_positives_and_high_negatives():
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    safe = hard_negative_tail_loss(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels)
    risky = hard_negative_tail_loss(torch.tensor([0.9, 0.4, 0.8, 0.1]), labels)
    assert torch.isfinite(safe)
    assert risky > safe
    with pytest.raises(RuntimeError, match="both classes"):
        hard_negative_tail_loss(torch.tensor([0.5, 0.6]), torch.ones(2))
