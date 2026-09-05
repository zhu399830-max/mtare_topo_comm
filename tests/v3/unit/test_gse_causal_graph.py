from dataclasses import fields

import pytest

from mtare_topo.topology.gse_causal_graph import (
    AssociationEvidence, CausalStructuralGraph, PoseEstimate, StructuralObservation, StructuralPort,
)


PORT = StructuralPort(0, (1., 0., 0.), 3., 3., 1.)


def observe(graph, t, x, event="junction", candidates=(), **kwargs):
    return graph.update(StructuralObservation(t, event, 1., 0., (PORT,)),
                        PoseEstimate(t, (x, 0., 0.), (.01, .01, .01)), candidates, **kwargs)


def node(graph, t, x, candidates=()):
    observe(graph, t, x)
    return observe(graph, t + 1, x, candidates=candidates)


def route():
    graph = CausalStructuralGraph(stable_frames=2)
    node(graph, 0, 0)
    graph.depart(0)
    observe(graph, 2, 2, "corridor")
    node(graph, 3, 4)
    graph.depart(0)
    observe(graph, 5, 6, "corridor")
    node(graph, 6, 8)
    return graph


def test_first_visit_confirms_without_revisit():
    graph = CausalStructuralGraph(stable_frames=2)
    assert observe(graph, 0, 0).node_id is None
    assert observe(graph, 1, 0).action == "local_confirmed"
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0


def test_consecutive_nodes_not_only_first_and_last():
    graph = route()
    assert len(graph.nodes) == 3
    assert [(e.source_node, e.target_node) for e in graph.edges] == [(0, 1), (1, 2)]
    assert [e.length_m for e in graph.edges] == [4., 4.]
    assert [e.trace_id for e in graph.edges] == [0, 1]
    assert graph.nodes[0]["port_states"][0] == "traversed"
    assert graph.nodes[2]["port_states"][0] == "observed"


def test_prefix_replay_and_snapshot_immutability():
    graph = CausalStructuralGraph(stable_frames=2)
    node(graph, 0, 0)
    graph.depart(0)
    observe(graph, 2, 2, "corridor")
    node(graph, 3, 4)
    snapshot = graph.snapshot()
    graph.depart(0)
    observe(graph, 5, 6, "corridor")
    node(graph, 6, 8)
    prefix = CausalStructuralGraph(stable_frames=2)
    node(prefix, 0, 0)
    prefix.depart(0)
    observe(prefix, 2, 2, "corridor")
    node(prefix, 3, 4)
    assert prefix.snapshot() == snapshot
    assert graph.snapshot() == route().snapshot()


def test_no_departure_cannot_create_edge():
    graph = CausalStructuralGraph(stable_frames=2)
    node(graph, 0, 0)
    observe(graph, 2, 1, "corridor")
    node(graph, 3, 4)
    assert len(graph.nodes) == 2 and not graph.edges


def test_similarity_without_registration_never_merges():
    graph = CausalStructuralGraph(stable_frames=2)
    node(graph, 0, 0)
    graph.depart(0)
    observe(graph, 2, 1, "corridor")
    candidate = AssociationEvidence(0, 4, False, True, True, True)
    update = node(graph, 3, 4, (candidate,))
    assert update.action == "local_confirmed"
    assert len(graph.nodes) == 2 and len(graph.edges) == 1


def test_ambiguous_merge_keeps_new_node_and_traversal():
    graph = route()
    graph.depart(0)
    observe(graph, 8, 10, "corridor")
    candidates = tuple(AssociationEvidence(i, 10, True, True, True, True) for i in (0, 1))
    update = node(graph, 9, 12, candidates)
    assert update.action == "local_confirmed_merge_ambiguous"
    assert len(graph.nodes) == 4 and len(graph.edges) == 3
    assert graph.nodes[3]["alias_candidates"] == [0, 1]


def test_verified_revisit_closes_cycle_without_fake_reverse_edge():
    graph = route()
    graph.depart(0)
    observe(graph, 8, 4, "corridor")
    candidate = AssociationEvidence(0, 10, True, True, True, True)
    update = node(graph, 9, 0, (candidate,))
    assert update.action == "verified_revisit"
    assert len(graph.nodes) == 3 and len(graph.edges) == 3
    assert (graph.edges[-1].source_node, graph.edges[-1].target_node) == (2, 0)


def test_discontinuous_trace_rejected_but_node_kept():
    graph = CausalStructuralGraph(stable_frames=2)
    node(graph, 0, 0)
    graph.depart(0)
    observe(graph, 2, 2, "corridor", continuous=False)
    result = node(graph, 3, 4)
    assert result.action.endswith("trace_rejected")
    assert len(graph.nodes) == 2 and not graph.edges
    assert graph.nodes[0]["port_states"][0] == "temporarily_failed"


def test_event_dwell_does_not_duplicate_or_make_zero_length_edge():
    graph = CausalStructuralGraph(stable_frames=2)
    node(graph, 0, 0)
    graph.depart(0)
    for t in range(2, 10):
        observe(graph, t, 0)
    assert len(graph.nodes) == 1 and not graph.edges


def test_future_evidence_and_reordered_input_fail_without_mutation():
    graph = CausalStructuralGraph(stable_frames=2)
    node(graph, 0, 0)
    before = graph.snapshot()
    with pytest.raises(ValueError, match="strictly causal"):
        observe(graph, 1, 0)
    with pytest.raises(ValueError, match="stale/future"):
        observe(graph, 2, 0, candidates=(AssociationEvidence(0, 99, True, True, True, True),))
    assert graph.snapshot() == before


def test_teacher_and_traversal_identity_not_in_runtime_schema():
    names = {f.name for t in (StructuralObservation, PoseEstimate, AssociationEvidence) for f in fields(t)}
    assert not names & {"world", "teacher_identity", "traversal_id", "node_identity", "future_pose"}


def test_unobserved_departure_port_refused():
    graph = CausalStructuralGraph(stable_frames=2)
    node(graph, 0, 0)
    with pytest.raises(ValueError, match="not observed"):
        graph.depart(99)


def test_truthy_strings_cannot_forge_verification():
    with pytest.raises(ValueError, match="association evidence"):
        AssociationEvidence(0, 0, "False", True, True, True)


def test_sensor_pose_mismatch_does_not_mutate_graph():
    graph = CausalStructuralGraph(stable_frames=2)
    before = graph.snapshot()
    with pytest.raises(ValueError, match="timestamp mismatch"):
        graph.update(StructuralObservation(1, "junction", 1, 0, (PORT,)),
                     PoseEstimate(2, (0, 0, 0), (0, 0, 0)))
    assert graph.snapshot() == before
