from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.representation.gse_geometry_conditioned_risk import (
    GeometryConditionedRiskReadout,
    causal_geometry_risk_features,
    fit_geometry_conditioned_risk_readout,
    fit_weighted_standardizer,
    identity_balanced_event_weights,
)


def _probability(rows: int) -> np.ndarray:
    value = np.tile(np.asarray([0.5, 0.2, 0.1, 0.1, 0.1]), (rows, 1))
    return value


def test_causal_features_use_same_traversal_past_only_and_mark_missing() -> None:
    geometry = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [10.0, 10.0, 10.0, 10.0],
            [4.0, 8.0, 12.0, 16.0],
        ]
    )
    result = causal_geometry_risk_features(
        geometry,
        (0.1, 0.2, 0.3, 0.4),
        _probability(4),
        ("a", "a", "b", "a"),
        (0, 2, 0, 4),
        (0, 0, 0, 0),
        (True, True, True, True),
        lags=(2, 4),
    )
    assert result.history_available.tolist() == [[0, 0], [1, 0], [0, 0], [1, 1]]
    assert np.allclose(result.values[1, :4], [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(result.values[3, :4], [2.0, 4.0, 6.0, 8.0])
    assert np.allclose(result.values[3, 9:13], [3.0, 6.0, 9.0, 12.0])


def test_causal_features_refuse_partition_crossing() -> None:
    with pytest.raises(ValueError, match="partition"):
        causal_geometry_risk_features(
            np.zeros((2, 4)),
            (0.0, 0.0),
            _probability(2),
            ("a", "a"),
            (0, 2),
            (0, 1),
            (True, True),
            lags=(2,),
        )


def test_causal_features_decompose_structural_and_conditional_probability() -> None:
    probability = np.asarray([[0.8, 0.1, 0.04, 0.04, 0.02]])
    result = causal_geometry_risk_features(
        np.zeros((1, 4)),
        (0.25,),
        probability,
        ("a",),
        (0,),
        (0,),
        (True,),
        lags=(2,),
    )
    assert np.isclose(result.values[0, -6], 0.2)
    assert np.allclose(result.values[0, -5:-1], [0.5, 0.2, 0.2, 0.1])
    assert np.isclose(result.values[0, -1], 0.25)


def test_identity_weights_equalize_classes_and_structural_identities() -> None:
    # Every class is present. Junction identity j0 has two frames while j1 has
    # one; both identities must receive equal total mass.
    event = (0, 0, 1, 1, 1, 2, 3, 4, 4, 0)
    identity = (None, None, "j0", "j0", "j1", "t0", "u0", "g0", "g0", None)
    parent = ("w0", "w1", "w0", "w0", "w1", "w0", "w0", "w0", "w0", "sel")
    fit = (True, True, True, True, True, True, True, True, True, False)
    weights = identity_balanced_event_weights(event, identity, parent, fit)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[-1] == 0.0
    for event_index in range(5):
        mask = np.asarray(event) == event_index
        assert np.isclose(weights[mask].sum(), 0.2)
    assert np.isclose(weights[2] + weights[3], weights[4])
    assert np.isclose(weights[0], weights[1])


def test_identity_weights_reject_missing_structural_identity() -> None:
    with pytest.raises(ValueError, match="identity"):
        identity_balanced_event_weights(
            (0, 1, 2, 3, 4),
            (None, None, "t", "u", "g"),
            ("w", "w", "w", "w", "w"),
            (True, True, True, True, True),
        )


def test_weighted_standardizer_ignores_zero_weight_selection_rows() -> None:
    standardizer = fit_weighted_standardizer(
        np.asarray([[1.0, 3.0], [3.0, 7.0], [1000.0, 1000.0]]),
        (0.5, 0.5, 0.0),
    )
    assert np.allclose(standardizer.mean, [2.0, 5.0])
    assert np.allclose(standardizer.scale, [1.0, 2.0])
    assert np.allclose(standardizer.transform(np.asarray([[2.0, 5.0]])), [[0.0, 0.0]])


def test_frozen_risk_readout_produces_five_class_probabilities() -> None:
    standardizer = fit_weighted_standardizer(
        np.asarray([[0.0, 0.0], [2.0, 2.0]]),
        (0.5, 0.5),
    )
    readout = GeometryConditionedRiskReadout(
        standardizer=standardizer,
        coefficient=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        ),
        intercept=np.zeros(5),
        feature_names=("a", "b"),
    )
    probability = readout.predict_probability(np.asarray([[1.0, 1.0], [2.0, 0.0]]))
    assert probability.shape == (2, 5)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.argmax(probability[1]) == 1


def test_low_capacity_risk_fit_is_deterministic_and_excludes_zero_weight_rows() -> None:
    # Five compact clusters, repeated so every class is represented.  The final
    # extreme row has zero weight and must not influence normalization or fit.
    centre = np.asarray([[-4.0, 0.0], [-2.0, 2.0], [0.0, -2.0], [2.0, 2.0], [4.0, 0.0]])
    values = np.concatenate((np.repeat(centre, 3, axis=0), [[1000.0, 1000.0]]), axis=0)
    labels = np.concatenate((np.repeat(np.arange(5), 3), [0]))
    weights = np.concatenate((np.full(15, 1.0 / 15.0), [0.0]))
    from mtare_topo.representation.gse_geometry_conditioned_risk import CausalRiskFeatures

    features = CausalRiskFeatures(
        values=values,
        history_available=np.zeros((len(values), 1), dtype=np.uint8),
        names=("x", "y"),
    )
    first, summary = fit_geometry_conditioned_risk_readout(
        features, labels, weights, l2_strength=1e-3
    )
    second, _ = fit_geometry_conditioned_risk_readout(
        features, labels, weights, l2_strength=1e-3
    )
    assert summary.converged
    assert summary.observations == 15
    assert np.allclose(first.coefficient, second.coefficient, atol=0.0, rtol=0.0)
    assert np.array_equal(
        np.argmax(first.predict_probability(centre), axis=1), np.arange(5)
    )
    assert np.all(first.standardizer.mean < 10.0)
