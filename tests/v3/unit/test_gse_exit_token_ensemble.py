from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.representation.gse_exit_token_ensemble import FrozenWorldAssociationScores


def test_world_score_table_is_symmetric_and_applies_frozen_threshold() -> None:
    code = (np.asarray([4], dtype=np.uint64) << np.uint64(32)) | np.asarray([9], dtype=np.uint64)
    table = FrozenWorldAssociationScores(
        code,
        np.asarray([3.5], dtype=np.float32),
        np.asarray([[0.94], [0.95], [0.96]], dtype=np.float32),
        threshold=0.9431912302970886,
        maximum_candidate_distance_m=16.0,
        node_keys=np.asarray([4, 9]),
        node_seed_scores=np.asarray([[0.999, 0.4], [0.995, 0.5], [0.990, 0.6]], dtype=np.float32),
    )
    forward = table.evaluate_pair(4, 9, 3.5)
    reverse = table.evaluate_pair(9, 4, 3.5)
    assert forward == reverse
    assert forward["accepted"] is True
    assert forward["ensemble_score"] == pytest.approx(0.95)
    assert table.evaluate_node(4)["accepted"] is True
    assert table.evaluate_node(9)["accepted"] is False


def test_world_score_table_rejects_outside_domain_and_distance_drift() -> None:
    code = (np.asarray([1], dtype=np.uint64) << np.uint64(32)) | np.asarray([2], dtype=np.uint64)
    table = FrozenWorldAssociationScores(
        code,
        np.asarray([2.0], dtype=np.float32),
        np.asarray([[0.99], [0.99], [0.99]], dtype=np.float32),
        threshold=0.9431912302970886,
        maximum_candidate_distance_m=16.0,
    )
    assert table.evaluate_pair(1, 2, 17.0)["rejection_reason"] == "outside_16m_candidate_domain"
    with pytest.raises(RuntimeError, match="distance drifted"):
        table.evaluate_pair(1, 2, 3.0)


def test_event_node_gate_uses_frozen_product_score_and_mean_event_probability() -> None:
    code = (np.asarray([4], dtype=np.uint64) << np.uint64(32)) | np.asarray([9], dtype=np.uint64)
    event = np.asarray(
        [
            [[0.01, 0.80, 0.10, 0.05, 0.04], [0.70, 0.10, 0.10, 0.05, 0.05]],
            [[0.02, 0.78, 0.10, 0.05, 0.05], [0.75, 0.08, 0.08, 0.05, 0.04]],
            [[0.00, 0.82, 0.08, 0.05, 0.05], [0.80, 0.05, 0.05, 0.05, 0.05]],
        ],
        dtype=np.float32,
    )
    table = FrozenWorldAssociationScores(
        code,
        np.asarray([3.5], dtype=np.float32),
        np.asarray([[0.99], [0.99], [0.99]], dtype=np.float32),
        threshold=0.9431912302970886,
        maximum_candidate_distance_m=16.0,
        node_keys=np.asarray([4, 9]),
        node_seed_scores=np.asarray([[0.99, 0.99], [0.99, 0.99], [0.99, 0.99]], dtype=np.float32),
        node_event_probabilities=event,
    )
    structural = table.evaluate_node(4)
    corridor = table.evaluate_node(9)
    assert structural["accepted"] is True
    assert structural["combined_score"] == pytest.approx(0.99 * 0.99)
    assert np.argmax(structural["mean_event_probability"]) == 1
    assert corridor["accepted"] is False
    assert corridor["rejection_reason"] == "below_frozen_event_node_threshold"
