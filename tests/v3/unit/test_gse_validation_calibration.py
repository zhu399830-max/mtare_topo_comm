from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.evaluation.gse_validation_calibration import (
    binary_f1_threshold,
    causal_nearest_descriptor_records,
    event_negative_log_likelihood,
    fit_event_temperature,
    precision_constrained_threshold,
    select_event_rejection_threshold,
)


def test_temperature_fit_is_deterministic_and_does_not_increase_nll() -> None:
    logits = np.asarray(((8.0, 0.0), (0.0, 8.0), (8.0, 0.0), (0.0, 8.0)))
    labels = np.asarray((0, 1, 1, 0))
    first = fit_event_temperature(logits, labels)
    second = fit_event_temperature(logits, labels)
    assert first == second
    assert first["negative_log_likelihood_after"] <= first["negative_log_likelihood_before"]
    assert event_negative_log_likelihood(logits, labels, temperature=float(first["temperature"])) == pytest.approx(
        first["negative_log_likelihood_after"]
    )


def test_precision_threshold_maximizes_safe_true_matches() -> None:
    selected, curve = precision_constrained_threshold(
        (0.99, 0.98, 0.97, 0.96, 0.20),
        (True, True, True, False, False),
        (True, True, True, False, False),
        minimum_precision=0.98,
        maximum_false_accept_rate=0.01,
    )
    assert selected.threshold == pytest.approx(0.97)
    assert selected.true_positive == 3
    assert selected.false_positive == 0
    assert len(curve) == 5


def test_precision_threshold_refuses_vacuous_or_unsafe_result() -> None:
    with pytest.raises(RuntimeError, match="no non-empty"):
        precision_constrained_threshold(
            (0.9, 0.8),
            (False, False),
            (True, True),
        )


def test_causal_descriptor_records_use_only_past_same_parent() -> None:
    descriptors = np.asarray(
        (
            (1.0, 0.0),
            (1.0, 0.01),
            (0.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
        ),
        dtype=np.float64,
    )
    records = causal_nearest_descriptor_records(
        descriptors,
        identities=(10, 10, 20, 30, 30),
        parent_ids=("a", "a", "a", "b", "b"),
        sequence_order=(0, 1, 2, 3, 4),
    )
    assert [(row["query_index"], row["candidate_index"]) for row in records] == [(1, 0), (2, 1), (4, 3)]
    assert records[0]["correct"] is True
    assert records[1]["correct"] is False
    assert records[1]["eligible_positive"] is False
    assert records[2]["correct"] is True


def test_causal_descriptor_records_never_match_simultaneous_tokens() -> None:
    records = causal_nearest_descriptor_records(
        np.asarray(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0))),
        identities=(1, 1, 1),
        parent_ids=("a", "a", "a"),
        sequence_order=(0, 0, 1),
    )
    assert len(records) == 1
    assert records[0]["query_index"] == 2
    assert records[0]["candidate_index"] in (0, 1)


def test_event_rejection_prefers_removing_confidently_wrong_high_risk_event() -> None:
    logits = np.asarray(
        (
            (5.0, 0.0),
            (0.0, 5.0),
            (0.0, 5.0),
            (0.0, 5.0),
        )
    )
    labels = np.asarray((0, 1, 1, 0))
    selected, curve = select_event_rejection_threshold(
        logits,
        labels,
        uncertainty=(0.05, 0.05, 0.05, 0.90),
        temperature=1.0,
        corridor_index=0,
    )
    assert selected.accepted_structural_events == 2
    assert selected.rejected_structural_predictions == 1
    assert selected.structural_precision == pytest.approx(1.0)
    assert selected.macro_f1 == pytest.approx(1.0)
    assert curve


def test_binary_f1_threshold_is_inclusive_and_deterministic() -> None:
    selected, curve = binary_f1_threshold(
        scores=(0.95, 0.80, 0.70, 0.20),
        labels=(True, True, False, False),
    )
    assert selected["threshold"] == pytest.approx(0.80)
    assert selected["f1"] == pytest.approx(1.0)
    assert len(curve) == 4
