from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.gse_factorized_qualification import (
    fixed_threshold_metrics,
    runtime_candidate_pairs,
    select_consensus_metric_configuration,
)


def _row(index: int, identity: str, *, event: str = "junction") -> dict[str, object]:
    return {
        "parent_id": "S01_demo_C09", "identity": identity, "event": event,
        "global_sequence_index": index,
    }


def test_runtime_candidates_are_past_nearest_per_identity() -> None:
    rows = [_row(10, "a"), _row(11, "a"), _row(12, "b"), _row(13, "a")]
    result = runtime_candidate_pairs(
        rows, np.ones(4, dtype=np.uint8),
        np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=float),
        np.asarray([0, 1, 2, 3]), maximum_distance_m=1.5,
    )
    pairs = list(zip(result["left"].tolist(), result["right"].tolist(), result["label"].tolist()))
    assert pairs == [(1, 0, 1), (2, 1, 0), (3, 1, 1), (3, 2, 0)]
    assert int(result["queries_with_candidate"]) == 3
    assert int(result["queries_with_positive"]) == 2


def test_fixed_metrics_marks_zero_negative_family_unidentifiable() -> None:
    result = fixed_threshold_metrics(
        np.asarray([.9, .8, .1]), np.asarray([1, 1, 0]),
        np.asarray(["S02", "S02", "S01"]), .5,
    )
    assert result["precision"] == 1.0
    assert result["per_family"]["S02"]["precision_identifiable"] is False
    assert result["per_family"]["S02"]["negative_rejection_rate"] is None


def test_consensus_selector_uses_margin_and_deterministic_tie_break() -> None:
    # Two votes accept all positives and reject all negatives.  A third vote
    # loses recall, so the selected contract must be 2-of-3.  Runtime recall is
    # identical at 1.0 and 1.5 m, hence the smaller cap wins.
    labels = np.asarray([1] * 20 + [0] * 20, dtype=np.uint8)
    families = np.asarray(["S01"] * 40)
    votes = np.stack((labels == 1, labels == 1, np.zeros(40, dtype=bool)))
    runtime_labels = labels.copy()
    distance = np.asarray([.8] * 20 + [.9] * 20)
    runtime_votes = votes.copy()
    selected = select_consensus_metric_configuration(
        votes, labels, families, np.ones(40, dtype=bool),
        runtime_votes, runtime_labels, families, distance,
        distance_grid_m=np.asarray([1.0, 1.5]),
    )["selected"]
    assert selected["votes_required"] == 2
    assert selected["distance_cap_m"] == 1.0
