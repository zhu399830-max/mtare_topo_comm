import numpy as np
import pytest

from mtare_topo.evaluation.gse_composer_observability import (
    binary_transfer_metrics,
    canonical_explicit_feature_matrix,
    causal_row_history,
    explicit_composer_features,
    token_validity_weights,
    train_free_event_scores,
    weighted_circular_moments,
)


def _prediction(rows: int = 2) -> dict[str, np.ndarray]:
    count = np.zeros((rows, 5, 7), dtype=np.float64)
    count[..., 2] = 1.0
    row = np.zeros((rows, 4, 6, 7), dtype=np.float64)
    row[..., 0] = 1.0
    return {
        "token_count_probability": count,
        "token_bearing_deg": np.broadcast_to(
            np.asarray((0, 60, 120, 180, 240, 300), dtype=np.float64), (rows, 5, 6)
        ).copy(),
        "token_opening_width_m": np.full((rows, 5, 6), 4.0),
        "token_vertical_profile_m": np.zeros((rows, 5, 6, 4)),
        "token_geometry_uncertainty": np.ones((rows, 5, 6, 5)),
        "transport_row_probability": row,
        "transport_reveal_probability": np.zeros((rows, 4, 6)),
    }


def test_token_validity_is_probability_count_exceeds_rank() -> None:
    probability = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]])
    weight = token_validity_weights(probability)
    assert np.allclose(weight, [[0.9, 0.7, 0.4, 0.0, 0.0, 0.0]])
    assert np.isclose(weight.sum(), np.arange(7) @ probability[0])


def test_circular_moments_are_permutation_and_rotation_equivariant() -> None:
    bearing = np.asarray([[[10.0, 80.0, 190.0, 0.0, 0.0, 0.0]]])
    weight = np.asarray([[[1.0, 0.7, 0.2, 0.0, 0.0, 0.0]]])
    first, second = weighted_circular_moments(bearing, weight)
    order = np.asarray((2, 0, 1, 5, 3, 4))
    permuted = weighted_circular_moments(bearing[..., order], weight[..., order])
    assert np.allclose(first, permuted[0])
    assert np.allclose(second, permuted[1])
    rotated = weighted_circular_moments(bearing + 40.0, weight)[0]
    angle = np.deg2rad(40.0)
    matrix = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    assert np.allclose(rotated, first @ matrix.T)


def test_causal_history_never_crosses_traversal() -> None:
    traversal = ["a", "a", "a", "b", "b"]
    sequence = [0, 1, 2, 0, 1]
    value = np.arange(5)[:, None]
    history, valid = causal_row_history(traversal, sequence, value)
    assert history[2, -3:, 0].tolist() == [0, 1, 2]
    assert valid[2].tolist() == [False, False, True, True, True]
    assert history[4, -2:, 0].tolist() == [3, 4]
    assert valid[4].tolist() == [False, False, False, True, True]


def test_explicit_features_reject_hidden_or_missing_data() -> None:
    prediction = _prediction()
    geometry = np.zeros((2, 5, 4))
    valid = np.ones((2, 5), dtype=bool)
    features = explicit_composer_features(
        prediction, geometry_sequence=geometry, geometry_valid_mask=valid
    )
    assert np.allclose(features.expected_count, 2.0)
    assert features.vertical_profile_mean_m.shape == (2, 5, 4)
    invalid = dict(prediction)
    invalid.pop("token_bearing_deg")
    with pytest.raises(ValueError, match="missing explicit"):
        explicit_composer_features(
            invalid, geometry_sequence=geometry, geometry_valid_mask=valid
        )


def test_train_free_scores_respond_to_the_intended_explicit_factor() -> None:
    prediction = _prediction()
    geometry = np.zeros((2, 5, 4))
    geometry[1, -1] = np.asarray((6.0, 3.0, 0.0, 0.05))
    valid = np.ones((2, 5), dtype=bool)
    features = explicit_composer_features(
        prediction, geometry_sequence=geometry, geometry_valid_mask=valid
    )
    score = train_free_event_scores(features)
    assert score["geometry_transition"][1] > score["geometry_transition"][0]
    assert score["turn"][1] > score["turn"][0]
    assert np.all(score["history_complete"])


def test_transition_score_uses_token_metric_geometry() -> None:
    prediction = _prediction()
    prediction["token_opening_width_m"][1, -1] = 14.0
    prediction["token_vertical_profile_m"][1, -1, :, 2] = 3.0
    geometry = np.zeros((2, 5, 4))
    valid = np.ones((2, 5), dtype=bool)
    features = explicit_composer_features(
        prediction, geometry_sequence=geometry, geometry_valid_mask=valid
    )
    score = train_free_event_scores(features)
    assert score["geometry_transition"][1] > score["geometry_transition"][0]


def test_canonical_full_matrix_is_642d_and_ignores_forbidden_arrays() -> None:
    prediction = _prediction()
    geometry = np.zeros((2, 5, 4))
    valid = np.ones((2, 5), dtype=bool)
    reference = canonical_explicit_feature_matrix(
        prediction, geometry_sequence=geometry, geometry_valid_mask=valid
    )
    assert reference.shape == (2, 642)
    with_forbidden = dict(prediction)
    with_forbidden["event_logits"] = np.random.default_rng(0).normal(size=(2, 5))
    with_forbidden["token_descriptor"] = np.random.default_rng(1).normal(
        size=(2, 5, 6, 32)
    )
    candidate = canonical_explicit_feature_matrix(
        with_forbidden, geometry_sequence=geometry, geometry_valid_mask=valid
    )
    assert np.array_equal(reference, candidate)


def test_canonical_transport_tracks_bearing_order() -> None:
    prediction = _prediction(rows=1)
    prediction["token_bearing_deg"][0, 0] = [300, 200, 100, 0, 50, 150]
    prediction["token_bearing_deg"][0, 1] = [150, 50, 0, 100, 200, 300]
    row = np.zeros((6, 7), dtype=np.float64)
    row[np.arange(6), np.arange(6)] = 1.0
    prediction["transport_row_probability"][0, 0] = row
    geometry = np.zeros((1, 5, 4))
    valid = np.ones((1, 5), dtype=bool)
    matrix = canonical_explicit_feature_matrix(
        prediction, geometry_sequence=geometry, geometry_valid_mask=valid
    )
    # 35 count + 390 token values; the first canonical 6x7 transport follows.
    first_transport = matrix[0, 425 : 425 + 42].reshape(6, 7)
    assert np.allclose(first_transport.sum(axis=1), 1.0)
    expected_columns = [2, 4, 0, 5, 1, 3]
    assert first_transport.argmax(axis=1).tolist() == expected_columns


def test_binary_transfer_threshold_is_fit_only_and_counts_identities() -> None:
    fit_label = np.asarray([1, 1, 0, 0, 0], dtype=bool)
    fit_score = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1])
    transfer_label = np.asarray([1, 1, 1, 0, 0], dtype=bool)
    transfer_score = np.asarray([0.85, 0.80, 0.1, 0.82, 0.05])
    metrics = binary_transfer_metrics(
        fit_label,
        fit_score,
        transfer_label,
        transfer_score,
        precision_floor=1.0,
        transfer_identity=["a", "b", "b", None, None],
    )
    assert metrics.fit_threshold == pytest.approx(0.8)
    assert metrics.fit_precision == pytest.approx(1.0)
    assert metrics.transfer_precision == pytest.approx(2.0 / 3.0)
    assert metrics.transfer_recall == pytest.approx(2.0 / 3.0)
    assert metrics.transfer_false_positive_rate == pytest.approx(0.5)
    assert metrics.transfer_identity_covered == 2
    assert metrics.transfer_identity_total == 2


def test_binary_transfer_keeps_tied_scores_at_one_operating_point() -> None:
    metrics = binary_transfer_metrics(
        np.asarray([1, 0, 1, 0], dtype=bool),
        np.asarray([0.9, 0.9, 0.5, 0.1]),
        np.asarray([1, 0], dtype=bool),
        np.asarray([0.9, 0.2]),
        precision_floor=0.75,
    )
    # The tied 0.9 pair has precision 0.5 and cannot be split after seeing
    # labels, so no fit operating point reaches the requested precision.
    assert metrics.fit_threshold is None
    assert metrics.fit_recall == 0.0
    assert metrics.transfer_precision == 0.0
    assert metrics.transfer_recall == 0.0
