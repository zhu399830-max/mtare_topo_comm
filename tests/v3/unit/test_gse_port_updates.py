from dataclasses import fields, replace

import pytest

from mtare_topo.topology.gse_port_updates import (
    GraphPort, ObservedPort, PortGeometry, PortMatchEvidence, PortTable, update_ports,
)


GEOMETRY = PortGeometry((1., 0., 0.), 3., 4., .9)


def observed(local=99, width=5.):
    return ObservedPort(local, replace(GEOMETRY, width_m=width))


def match(local=99, candidates=(3,), verified=True, node=7, t=2.):
    return PortMatchEvidence(node, t, local, candidates, verified)


def table(state="observed"):
    return PortTable(7, 1., (GraphPort(3, GEOMETRY, state, 1.),))


@pytest.mark.parametrize("state", ["observed", "attempted", "traversed", "temporarily_failed", "failed"])
def test_unique_match_updates_geometry_and_preserves_execution(state):
    original = table(state)
    result = update_ports(original, timestamp_s=2., observations=(observed(),), evidence=(match(),))
    assert result.table.ports == (GraphPort(3, observed().geometry, state, 2.),)
    assert result.decisions[0].action == "updated_verified_unique"
    assert original.ports[0].geometry == GEOMETRY
    assert original.timestamp_s == 1.


def test_ids_are_not_identity_and_unmatched_adds_deterministically():
    # Equal local IDs across observations are not correspondence evidence.
    result = update_ports(table("traversed"), timestamp_s=2., observations=(observed(3), observed(90)),
                          evidence=(match(90, (), False), match(3, (), False)))
    assert [p.graph_local_id for p in result.table.ports] == [3, 4, 5]
    assert [(d.observation_local_id, d.graph_local_id) for d in result.decisions] == [(3, 4), (90, 5)]
    assert result.table.ports[0].execution_state == "traversed"


def test_multiple_candidates_keep_pending_without_duplicate_port():
    original = PortTable(7, 1., (GraphPort(3, GEOMETRY, "traversed", 1.),
                               GraphPort(4, GEOMETRY, "failed", 1.)))
    result = update_ports(original, timestamp_s=2., observations=(observed(),),
                          evidence=(match(candidates=(4, 3)),))
    assert result.table.ports == original.ports
    assert result.pending_observations == (observed(),)
    assert result.decisions[0].candidate_graph_ids == (3, 4)
    assert result.decisions[0].action == "pending_ambiguous"


@pytest.mark.parametrize("other_verified", [True, False])
def test_two_observations_cannot_overwrite_same_port(other_verified):
    result = update_ports(table(), timestamp_s=2., observations=(observed(1), observed(2)),
                          evidence=(match(1), match(2, verified=other_verified)))
    assert result.table.ports == table().ports
    assert all(d.action == "pending_ambiguous" for d in result.decisions)
    assert len(result.pending_observations) == 2


def test_unverified_unique_is_pending():
    result = update_ports(table(), timestamp_s=2., observations=(observed(),),
                          evidence=(match(verified=False),))
    assert result.table.ports == table().ports
    assert result.decisions[0].action == "pending_unverified"


def test_order_invariance_and_causal_prefix():
    observations = (observed(1), observed(2))
    evidence = (match(1), match(2, (), False))
    a = update_ports(table(), timestamp_s=2., observations=observations, evidence=evidence)
    b = update_ports(table(), timestamp_s=2., observations=observations[::-1], evidence=evidence[::-1])
    assert a == b
    prefix = a
    end = update_ports(a.table, timestamp_s=3., observations=(observed(9),),
                       evidence=(match(9, (4,), t=3.),))
    assert a == prefix and end.table.timestamp_s == 3.
    assert a.table.ports[1].last_observed_s == 2.


def test_empty_observation_preserves_ports():
    result = update_ports(table(), timestamp_s=2., observations=(), evidence=())
    assert result.table.ports == table().ports and result.decisions == ()


@pytest.mark.parametrize("evidence", [(), (match(), match()), (match(0),),
                                        (match(node=8),), (match(t=3.),), (match(candidates=(999,)),)])
def test_bad_evidence_fails_without_mutation(evidence):
    original = table()
    with pytest.raises(ValueError):
        update_ports(original, timestamp_s=2., observations=(observed(),), evidence=evidence)
    assert original == table()


@pytest.mark.parametrize("timestamp", [0., 1., float("nan"), float("inf")])
def test_noncausal_or_nonfinite_timestamp_rejected(timestamp):
    with pytest.raises(ValueError):
        update_ports(table(), timestamp_s=timestamp, observations=(), evidence=())


@pytest.mark.parametrize("kwargs", [dict(geometry_verified="False"), dict(candidate_graph_ids=(3, 3)),
                                    dict(candidate_graph_ids=(True,)), dict(node_id=True)])
def test_evidence_rejects_truthy_or_nonunique_identifiers(kwargs):
    with pytest.raises(ValueError):
        replace(match(), **kwargs)


@pytest.mark.parametrize("kwargs", [dict(direction_node=(0., 0., 0.)), dict(width_m=0.),
                                    dict(height_m=float("nan")), dict(confidence=2.)])
def test_bad_geometry_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(GEOMETRY, **kwargs)


def test_schema_does_not_contain_teacher_identity_or_edge_claim():
    names = {f.name for cls in (ObservedPort, GraphPort, PortMatchEvidence, PortTable) for f in fields(cls)}
    assert not names & {"world", "teacher_id", "traversal_id", "true_node_id", "edge_id"}


def test_conflicting_single_and_multicandidate_both_pending():
    original = PortTable(7, 1., (GraphPort(3, GEOMETRY, "traversed", 1.),
                               GraphPort(4, GEOMETRY, "failed", 1.)))
    result = update_ports(original, timestamp_s=2., observations=(observed(1), observed(2)),
                          evidence=(match(1), match(2, (3, 4))))
    assert result.table.ports == original.ports
    assert all(d.action == "pending_ambiguous" for d in result.decisions)
