import numpy as np

from mtare_topo.evaluation.gse_trace_commit_failure_funnel import (
    classify_identity_failure,
    classify_relation_failure,
    cross_trace_pair_state,
    true_relations,
)


def test_cross_trace_state_and_identity_classification() -> None:
    traversal = np.asarray(["a", "b", "c"])
    xyz = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    state = cross_trace_pair_state(
        [0, 1, 2], traversal, xyz, np.ones(3, dtype=bool), {(0, 1): True},
    )
    assert state["distinct_traversals"] == 3
    assert state["has_within_radius_pair"]
    assert state["has_accepted_pair"]
    assert classify_identity_failure([0, 1], state, []) == "ambiguity_or_commit_logic"
    assert classify_identity_failure([0, 1], state, [4]) == "committed_unique"
    assert classify_identity_failure([0, 1], state, [4, 7]) == "committed_duplicate"


def test_true_relation_and_failure_stages() -> None:
    traversal = ["t", "t", "t", "u"]
    sequence = [0, 1, 2, 0]
    identity = ["A", None, "B", "A"]
    event = ["junction", "interior", "terminal", "junction"]
    relations = true_relations(range(4), traversal, sequence, identity, event)
    assert relations == {("A", "B"): {"t"}}
    assert classify_relation_failure(
        ("A", "B"), {"t"}, {"A": [0], "B": [2]},
        {"A": [1], "B": [2]}, {("A", "t"): [0], ("B", "t"): [2]},
        {("A", "B")},
    ) == "recovered"
    assert classify_relation_failure(
        ("A", "B"), {"t"}, {"A": [0], "B": [2]},
        {"A": [1], "B": [2]}, {("A", "t"): [0]}, set(),
    ) == "no_common_committed_trace"
