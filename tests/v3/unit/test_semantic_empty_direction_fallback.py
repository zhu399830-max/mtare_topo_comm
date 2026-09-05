from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components
from mtare_topo.integration.online_topology_runtime import SemanticPrediction
from mtare_topo.integration.semantic_fallback import apply_composite_v9_semantics, apply_empty_direction_fallback


def _prediction(logits: np.ndarray) -> SemanticPrediction:
    return SemanticPrediction(
        direction_logits=logits,
        count_probabilities=np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float64),
        role_probabilities=np.asarray([0, 0, 1], dtype=np.float64),
        z_role=np.arange(128, dtype=np.float64),
    )


class FakeBaseline:
    def __init__(self, headings):
        self.headings = headings

    def predict(self, range_m, valid_mask, elevation_deg):
        assert range_m.shape == valid_mask.shape == (16, 720)
        assert elevation_deg.shape == (16,)
        return {"headings_robot_deg": self.headings, "branch_count": len(self.headings)}


def test_nonempty_learned_prediction_never_calls_fallback() -> None:
    logits = np.full(720, -20.0)
    logits[20] = 20.0
    class Forbidden:
        def predict(self, *args):
            raise AssertionError("B0 must not run for nonempty learned semantics")
    prediction = _prediction(logits)
    corrected, audit = apply_empty_direction_fallback(prediction, np.ones((16, 720)), np.ones((16, 720), dtype=bool), baseline=Forbidden())
    assert corrected is prediction
    assert audit.learned_empty is False and audit.fallback_used is False


def test_empty_learned_prediction_uses_explicit_b0_semantics_and_retains_embedding() -> None:
    prediction = _prediction(np.full(720, -20.0))
    corrected, audit = apply_empty_direction_fallback(
        prediction,
        np.ones((16, 720)),
        np.ones((16, 720), dtype=bool),
        baseline=FakeBaseline([45.0, 180.0, 359.0]),
    )
    assert audit.learned_empty and audit.fallback_used
    assert decode_direction_components(np.asarray(corrected.direction_logits), 0.5) == [45.0, 180.0, 359.0]
    assert np.argmax(corrected.count_probabilities) == 2
    assert np.argmax(corrected.role_probabilities) == 1
    assert corrected.branch_count_override is None
    assert np.array_equal(corrected.z_role, prediction.z_role)


def test_empty_b0_does_not_invent_an_exit() -> None:
    prediction = _prediction(np.full(720, -20.0))
    corrected, audit = apply_empty_direction_fallback(
        prediction,
        np.ones((16, 720)),
        np.ones((16, 720), dtype=bool),
        baseline=FakeBaseline([]),
    )
    assert corrected is prediction
    assert audit.learned_empty and not audit.fallback_used


def test_v9_nonempty_direction_still_uses_b0_count_and_fixed_role() -> None:
    logits = np.full(720, -20.0); logits[20] = 20.0
    prediction = _prediction(logits)
    corrected, fallback, audit = apply_composite_v9_semantics(
        prediction, np.ones((16, 720)), np.ones((16, 720), dtype=bool),
        baseline=FakeBaseline([0.0, 120.0, 240.0]),
    )
    assert np.array_equal(corrected.direction_logits, logits)
    assert np.argmax(corrected.count_probabilities) + 1 == 3
    assert np.argmax(corrected.role_probabilities) == 1
    assert corrected.branch_count_override == 3
    assert not fallback.fallback_used
    assert audit.b0_branch_count == 3 and audit.neural_count_role_ignored


def test_v9_empty_direction_uses_b0_direction_and_preserves_raw_zero_count() -> None:
    prediction = _prediction(np.full(720, -20.0))
    corrected, fallback, _ = apply_composite_v9_semantics(
        prediction, np.ones((16, 720)), np.ones((16, 720), dtype=bool),
        baseline=FakeBaseline([45.0, 180.0]),
    )
    assert fallback.fallback_used
    assert decode_direction_components(np.asarray(corrected.direction_logits), 0.5) == [45.0, 180.0]
    zero, zero_fallback, audit = apply_composite_v9_semantics(
        prediction, np.ones((16, 720)), np.ones((16, 720), dtype=bool),
        baseline=FakeBaseline([]),
    )
    assert zero.branch_count_override == 0
    assert np.argmax(zero.role_probabilities) == 2
    assert zero_fallback.learned_empty and not zero_fallback.fallback_used
    assert audit.b0_branch_count == 0


def test_v9_preserves_b0_count_above_neural_six_class_limit() -> None:
    logits = np.full(720, -20.0); logits[0] = 20.0
    headings = [float(index * 45) for index in range(8)]
    corrected, _, audit = apply_composite_v9_semantics(
        _prediction(logits), np.ones((16, 720)), np.ones((16, 720), dtype=bool),
        baseline=FakeBaseline(headings),
    )
    assert corrected.branch_count_override == 8
    assert audit.b0_branch_count == 8
