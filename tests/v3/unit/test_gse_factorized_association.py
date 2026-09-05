import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_factorized_association import (
    FACTORIZED_PAIR_FEATURE_DIM,
    NO_ROUTE_PAIR_FEATURE_DIM,
    FactorizedAssociationVerifier,
    factorized_pair_features,
    learned_geometry_profiles,
    parameter_count,
)


def _inputs(rows=4):
    rng = np.random.default_rng(4)
    observation = rng.normal(size=(rows, 146)).astype(np.float32)
    descriptor = observation[:, 12:140]
    observation[:, 12:140] = descriptor / np.linalg.norm(descriptor, axis=1, keepdims=True)
    observation[:, 140] = np.linspace(.1, .4, rows)
    confidence = rng.uniform(.1, .9, size=(rows, 6)).astype(np.float32)
    heading_angle = rng.uniform(-np.pi, np.pi, size=(rows, 6))
    heading = np.stack((np.sin(heading_angle), np.cos(heading_angle)), axis=2).astype(np.float32)
    exit_descriptor = rng.normal(size=(rows, 6, 32)).astype(np.float32)
    exit_descriptor /= np.linalg.norm(exit_descriptor, axis=2, keepdims=True)
    token = {
        "exit_confidence": confidence,
        "exit_heading_unit": heading,
        "exit_opening_width_m": rng.uniform(2, 12, size=(rows, 6)).astype(np.float32),
        "exit_vertical_profile": rng.normal(size=(rows, 6, 4)).astype(np.float32),
        "exit_descriptor": exit_descriptor,
    }
    references = np.asarray([
        [-1, -1, -1, -1, 0], [-1, -1, -1, 0, 1],
        [-1, -1, 0, 1, 2], [-1, 0, 1, 2, 3],
    ])
    mask = references >= 0
    return observation, token, references, mask


def test_learned_profiles_use_only_referenced_causal_rows():
    observation, _, references, mask = _inputs()
    profile = learned_geometry_profiles(observation, references, mask)
    np.testing.assert_allclose(profile[1, :4], observation[:2, 8:12].mean(0))
    assert profile[0, 8] == pytest.approx(.2)
    assert profile[3, 8] == pytest.approx(.8)


def test_factorized_features_are_pair_symmetric():
    observation, token, references, mask = _inputs()
    profile = learned_geometry_profiles(observation, references, mask)
    left = np.asarray([0, 1]); right = np.asarray([2, 3])
    forward = factorized_pair_features(observation, token, profile, left, right)
    reverse = factorized_pair_features(observation, token, profile, right, left)
    np.testing.assert_allclose(forward, reverse, atol=1e-6)
    assert forward.shape == (2, FACTORIZED_PAIR_FEATURE_DIM)


def test_factorized_features_are_token_permutation_invariant():
    observation, token, references, mask = _inputs()
    profile = learned_geometry_profiles(observation, references, mask)
    expected = factorized_pair_features(observation, token, profile, np.asarray([0]), np.asarray([1]))
    adjusted = {name: value.copy() for name, value in token.items()}
    permutation = np.asarray([5, 2, 0, 4, 1, 3])
    for name in adjusted:
        adjusted[name][1] = adjusted[name][1, permutation]
    observed = factorized_pair_features(observation, adjusted, profile, np.asarray([0]), np.asarray([1]))
    np.testing.assert_allclose(expected, observed, atol=1e-6)


def test_no_route_ablation_has_frozen_dimension_and_model_shape():
    observation, token, references, mask = _inputs()
    profile = learned_geometry_profiles(observation, references, mask)
    features = factorized_pair_features(
        observation, token, profile, np.asarray([0, 1]), np.asarray([2, 3]),
        include_route_geometry=False,
    )
    assert features.shape == (2, NO_ROUTE_PAIR_FEATURE_DIM)
    model = FactorizedAssociationVerifier(include_route_geometry=False)
    assert model(torch.from_numpy(features)).shape == (2,)
    assert parameter_count(False) < parameter_count(True) < 10_000


def test_factorized_builder_rejects_out_of_range_rows():
    observation, token, references, mask = _inputs()
    profile = learned_geometry_profiles(observation, references, mask)
    with pytest.raises(ValueError, match="indices"):
        factorized_pair_features(observation, token, profile, np.asarray([0]), np.asarray([9]))
