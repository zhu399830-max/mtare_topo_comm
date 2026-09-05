from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.gse_relational_commit_policy import (
    EventCommitPolicy,
    evaluate_event_commits,
    extract_event_commits,
    merge_event_commits,
)


def _probability(junction: list[float], terminal: list[float] | None = None) -> np.ndarray:
    junction_array = np.asarray(junction, dtype=np.float64)
    terminal_array = np.zeros_like(junction_array) if terminal is None else np.asarray(terminal, dtype=np.float64)
    one = np.stack((1.0 - junction_array - terminal_array, junction_array, terminal_array), axis=1)
    return np.stack((one, one, one))


def test_stability_and_release_are_past_only() -> None:
    probability = _probability([0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9])
    policy = EventCommitPolicy(1, 0.8, 2, 2, 2)
    commits = extract_event_commits(
        probability,
        np.asarray(["a"] * 7),
        np.arange(7),
        np.ones(7, dtype=np.bool_),
        policy,
    )
    assert [item.row for item in commits] == [2, 6]


def test_short_gap_does_not_duplicate_commit() -> None:
    probability = _probability([0.9, 0.9, 0.1, 0.9, 0.9])
    policy = EventCommitPolicy(1, 0.8, 2, 2, 2)
    commits = extract_event_commits(
        probability, np.asarray(["a"] * 5), np.arange(5),
        np.ones(5, dtype=np.bool_), policy,
    )
    assert len(commits) == 1


def test_evaluation_counts_duplicate_corridor_and_type_errors() -> None:
    junction = extract_event_commits(
        _probability([0.9, 0.1, 0.9]), np.asarray(["a", "b", "c"]),
        np.zeros(3, dtype=np.int64), np.ones(3, dtype=np.bool_),
        EventCommitPolicy(1, 0.8, 2, 1, 2),
    )
    terminal = extract_event_commits(
        _probability([0.0, 0.0, 0.0], [0.0, 0.9, 0.0]),
        np.asarray(["a", "b", "c"]), np.zeros(3, dtype=np.int64),
        np.ones(3, dtype=np.bool_), EventCommitPolicy(2, 0.8, 2, 1, 2),
    )
    metrics = evaluate_event_commits(
        merge_event_commits(junction, terminal),
        np.asarray([1, 0, 2]), np.asarray([0, -1, 1]),
        np.ones(3, dtype=np.bool_),
    )
    assert metrics["correct_unique_episodes"] == 1
    assert metrics["failure_decomposition"] == {
        "duplicate_same_episode": 0,
        "corridor_false": 1,
        "wrong_event": 1,
    }
