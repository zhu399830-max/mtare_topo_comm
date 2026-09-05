import numpy as np
import pytest
import torch

from mtare_topo.representation.gse_open_set_association import (
    GSEOpenSetAssociationVerifier,
    OBSERVATION_FEATURE_DIM,
    OpenSetAssociationContract,
    PAIR_FEATURE_DIM,
    observation_features,
    select_nonvacuous_threshold,
    select_online_candidate_threshold,
    symmetric_pair_features,
)


def _outputs(rows: int = 3):
    rng = np.random.default_rng(7)
    heading = rng.normal(size=(rows, 6, 2)).astype(np.float32)
    heading /= np.linalg.norm(heading, axis=2, keepdims=True)
    return {
        "event_logits": rng.normal(size=(rows, 5)),
        "local_axis": np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (rows, 1)),
        "width_m": np.full(rows, 4.0),
        "height_m": np.full(rows, 3.0),
        "slope_deg": np.zeros(rows),
        "curvature_per_m": np.full(rows, 0.01),
        "place_descriptor": rng.normal(size=(rows, 128)),
        "uncertainty": np.full(rows, 0.2),
        "exit_confidence": rng.uniform(size=(rows, 6)),
        "exit_heading_unit": heading,
        "exit_opening_width_m": rng.uniform(1.0, 4.0, size=(rows, 6)),
        "exit_vertical_profile": rng.normal(size=(rows, 6, 4)),
    }


def test_observation_features_have_frozen_dimension_and_no_metadata():
    features = observation_features(_outputs())
    assert features.shape == (3, OBSERVATION_FEATURE_DIM)
    assert np.isfinite(features).all()


def test_symmetric_pair_features_are_exactly_order_invariant():
    features = observation_features(_outputs())
    forward = symmetric_pair_features(features[:2], features[1::-1], np.asarray([2.0, 3.0]))
    reverse = symmetric_pair_features(features[1::-1], features[:2], np.asarray([2.0, 3.0]))
    np.testing.assert_array_equal(forward, reverse)
    assert forward.shape == (2, PAIR_FEATURE_DIM)


def test_verifier_score_is_order_invariant():
    torch.manual_seed(4)
    model = GSEOpenSetAssociationVerifier().eval()
    features = observation_features(_outputs(rows=2))
    pairs = symmetric_pair_features(features[:1], features[1:], np.asarray([4.0]))
    left = torch.from_numpy(features[:1])
    right = torch.from_numpy(features[1:])
    with torch.no_grad():
        a = model(left, right, torch.from_numpy(pairs))["association_score"]
        b = model(right, left, torch.from_numpy(pairs))["association_score"]
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_nonvacuous_selector_requires_recall_and_every_family():
    scores = np.asarray([0.99, 0.98, 0.97, 0.96, 0.20, 0.10])
    labels = np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)
    families = np.asarray(["S01", "S01", "S02", "S02", "S01", "S02"])
    selected = select_nonvacuous_threshold(scores, labels, families, OpenSetAssociationContract())
    assert selected["precision"] == 1.0
    assert selected["recall"] == 1.0
    assert selected["accepted"] == 4


def test_nonvacuous_selector_rejects_all_reject_solution():
    with pytest.raises(RuntimeError, match="non-vacuous"):
        select_nonvacuous_threshold(
            np.asarray([0.9, 0.6, 0.8, 0.7]),
            np.asarray([1, 1, 0, 0], dtype=bool),
            np.asarray(["S01", "S02", "S01", "S02"]),
            OpenSetAssociationContract(minimum_precision=1.0, minimum_recall=1.0),
        )


def test_contract_rejects_invalid_probability():
    with pytest.raises(ValueError):
        OpenSetAssociationContract(minimum_recall=1.1)


def test_online_selector_excludes_pairs_outside_candidate_radius():
    scores = np.asarray([0.99, 0.98, 0.97, 0.96, 1.0])
    labels = np.asarray([1, 1, 1, 1, 0], dtype=bool)
    families = np.asarray(["S01", "S01", "S02", "S02", "S01"])
    distance = np.asarray([1.0, 2.0, 3.0, 4.0, 20.0])
    selected = select_online_candidate_threshold(
        scores, labels, families, distance, OpenSetAssociationContract()
    )
    assert selected["precision"] == 1.0
    assert selected["recall"] == 1.0
    assert selected["candidate_domain"] == {
        "maximum_distance_m": 16.0,
        "eligible_pairs": 4,
        "excluded_pairs": 1,
        "eligible_positive_pairs": 4,
        "eligible_negative_pairs": 0,
    }


def test_online_selector_fails_when_candidate_domain_is_empty():
    with pytest.raises(RuntimeError, match="empty"):
        select_online_candidate_threshold(
            np.asarray([0.9]),
            np.asarray([1], dtype=bool),
            np.asarray(["S01"]),
            np.asarray([20.0]),
            OpenSetAssociationContract(),
        )
