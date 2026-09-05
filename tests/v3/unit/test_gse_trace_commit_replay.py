from mtare_topo.topology.gse_trace_commit_replay import (
    ProposalTrigger,
    replay_trace_commits,
    score_trace_replay,
)


def _trigger(row, traversal, x, identity="n0", event="junction"):
    return ProposalTrigger(
        row=row, world="W", order=row, traversal_id=traversal,
        sequence_index=row, event=event, confidence=.9, uncertainty=.1,
        xyz_m=(x, 0.0, 0.0), teacher_identity=identity,
    )


def test_two_independent_traces_are_required_to_commit():
    triggers = [_trigger(0, "a", 0.0), _trigger(1, "a", .1), _trigger(2, "b", .2)]
    accepted = {(0, 1): True, (0, 2): True, (1, 2): True}
    replay = replay_trace_commits(
        triggers, accepted, association_valid_rows={0, 1, 2}
    )
    assert len(replay["hypotheses"]) == 1
    assert replay["hypotheses"][0]["committed"] is True
    assert replay["decision_trace"][1]["independent_traces"] == 1
    assert replay["decision_trace"][2]["independent_traces"] == 2


def test_ambiguous_association_fails_closed():
    triggers = [_trigger(0, "a", 0.0), _trigger(1, "b", 1.0, "n1"), _trigger(2, "c", .5)]
    replay = replay_trace_commits(
        triggers, {(0, 2): True, (1, 2): True},
        association_valid_rows={0, 1, 2}, commit_immediately=True,
    )
    assert len(replay["hypotheses"]) == 3
    assert replay["hypotheses"][2]["ambiguous_on_creation"] is True


def test_edges_require_one_completed_trace_with_two_committed_endpoints():
    triggers = [
        _trigger(0, "trace", 0.0, "n0"),
        _trigger(1, "trace", 3.0, "n1", event="terminal"),
    ]
    replay = replay_trace_commits(
        triggers, {}, association_valid_rows={0, 1}, commit_immediately=True
    )
    assert len(replay["edges"]) == 1
    metrics = score_trace_replay(
        replay, true_nodes={("W", "n0"), ("W", "n1")},
        true_trace_relations={(('W', 'n0'), ('W', 'n1'))},
    )
    assert metrics["node_f1"] == 1.0
    assert metrics["edge_f1"] == 1.0


def test_completed_trace_commits_only_first_to_last_endpoint():
    triggers = [
        _trigger(0, "trace", 0.0, "n0"),
        _trigger(1, "trace", 2.0, "n1", event="terminal"),
        _trigger(2, "trace", 4.0, "n2"),
    ]
    replay = replay_trace_commits(
        triggers, {}, association_valid_rows={0, 1, 2}, commit_immediately=True
    )
    assert len(replay["edges"]) == 1
    assert replay["edges"][0]["from_hypothesis"] == 0
    assert replay["edges"][0]["to_hypothesis"] == 2
    assert replay["edges"][0]["commit_contract"] == "completed_traversal_first_last_endpoints_v1"


def test_deferred_ambiguity_requires_nonoverlapping_position_intervals():
    triggers = [
        _trigger(0, "a", 0.0, "n0"),
        _trigger(1, "b", 2.0, "n1"),
        _trigger(2, "c", .4, "n0"),
        _trigger(3, "d", .5, "n0"),
    ]
    replay = replay_trace_commits(
        triggers, {(0, 2): True, (1, 2): True, (2, 3): True},
        association_valid_rows={0, 1, 2, 3}, commit_immediately=True,
    )
    assert replay["hypotheses"][2]["merged_into"] == 0
    assert replay["row_to_hypothesis"][2] == 0
    assert replay["row_to_hypothesis"][3] == 0
    assert any(value["action"] == "deferred_associate" for value in replay["decision_trace"])


def test_deferred_ambiguity_stays_split_when_uncertainty_intervals_overlap():
    triggers = [
        _trigger(0, "a", 0.0, "n0"),
        _trigger(1, "b", 2.0, "n1"),
        ProposalTrigger(
            row=2, world="W", order=2, traversal_id="c", sequence_index=2,
            event="junction", confidence=.9, uncertainty=.1, xyz_m=(.4, 0.0, 0.0),
            teacher_identity="n0", position_uncertainty_m=1.0,
        ),
        _trigger(3, "d", .5, "n0"),
    ]
    replay = replay_trace_commits(
        triggers, {(0, 2): True, (1, 2): True, (2, 3): True},
        association_valid_rows={0, 1, 2, 3}, commit_immediately=True,
    )
    assert replay["hypotheses"][2]["merged_into"] is None
    assert not any(value["action"] == "deferred_associate" for value in replay["decision_trace"])
