import pytest

from mtare_topo.topology.gse_post_commit_consolidation import (
    CommittedEndpointSignature,
    TraversedEdgeEndpoint,
    TraversedEndpointGeometry,
    endpoint_anchor_consensus,
    endpoint_consolidation_mapping,
    executed_branch_witness,
    remap_verified_edges,
    traversed_edge_endpoint,
)


def test_endpoint_token_is_invariant_to_traversal_reversal() -> None:
    assert traversed_edge_endpoint("w:edge_1:d0", 9.0, 10.0) == TraversedEdgeEndpoint("w:edge_1", 1)
    assert traversed_edge_endpoint("w:edge_1:d1", 1.0, 10.0) == TraversedEdgeEndpoint("w:edge_1", 1)
    assert traversed_edge_endpoint("w:edge_1:d0", 1.0, 10.0) == TraversedEdgeEndpoint("w:edge_1", 0)
    assert traversed_edge_endpoint("w:edge_1:d1", 9.0, 10.0) == TraversedEdgeEndpoint("w:edge_1", 0)


def test_opposite_edge_endpoints_never_consolidate() -> None:
    signatures = (
        CommittedEndpointSignature(3, "w", "junction", (TraversedEdgeEndpoint("w:e", 0),)),
        CommittedEndpointSignature(4, "w", "junction", (TraversedEdgeEndpoint("w:e", 1),)),
    )
    mapping, components = endpoint_consolidation_mapping(signatures)
    assert mapping == {3: 3, 4: 4}
    assert components == ()


def test_same_endpoint_consolidates_transitively_and_deterministically() -> None:
    a = TraversedEdgeEndpoint("w:e0", 0)
    b = TraversedEdgeEndpoint("w:e1", 1)
    signatures = (
        CommittedEndpointSignature(9, "w", "junction", (a,)),
        CommittedEndpointSignature(2, "w", "junction", (a, b)),
        CommittedEndpointSignature(7, "w", "junction", (b,)),
    )
    mapping, components = endpoint_consolidation_mapping(signatures)
    assert mapping == {9: 2, 2: 2, 7: 2}
    assert components == ((2, 7, 9),)


def test_world_and_event_boundaries_fail_closed() -> None:
    signatures = (
        CommittedEndpointSignature(1, "w", "junction", (TraversedEdgeEndpoint("w:e", 0),)),
        CommittedEndpointSignature(2, "w", "terminal", (TraversedEdgeEndpoint("w:e", 0),)),
        CommittedEndpointSignature(3, "x", "junction", (TraversedEdgeEndpoint("x:e", 0),)),
    )
    mapping, components = endpoint_consolidation_mapping(signatures)
    assert mapping == {1: 1, 2: 2, 3: 3}
    assert components == ()


def test_verified_edge_remap_deduplicates_and_drops_self_loop() -> None:
    edges = (
        {"id": 0, "from_hypothesis": 1, "to_hypothesis": 2},
        {"id": 1, "from_hypothesis": 3, "to_hypothesis": 2},
        {"id": 2, "from_hypothesis": 1, "to_hypothesis": 3},
    )
    result, self_loops, duplicates = remap_verified_edges(edges, {1: 1, 2: 2, 3: 1})
    assert [(x["from_hypothesis"], x["to_hypothesis"]) for x in result] == [(1, 2)]
    assert self_loops == 1
    assert duplicates == 1


def test_endpoint_arc_contract_fails_closed() -> None:
    with pytest.raises(ValueError):
        traversed_edge_endpoint("w:e:d0", 11.0, 10.0)


def _geometry(edge: str, side: int, xyz: tuple[float, float, float], outward: tuple[float, float, float]) -> TraversedEndpointGeometry:
    return TraversedEndpointGeometry(TraversedEdgeEndpoint(edge, side), xyz, outward)


def test_endpoint_anchor_consensus_uses_route_sampling_bound() -> None:
    values = (
        _geometry("w:e0", 1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        _geometry("w:e1", 0, (1.9, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    accepted, center, maximum = endpoint_anchor_consensus(values, route_sample_spacing_m=1.0)
    assert accepted and center == pytest.approx((0.95, 0.0, 0.0)) and maximum == pytest.approx(1.9)
    rejected, _, _ = endpoint_anchor_consensus((*values[:1], _geometry("w:e1", 0, (2.1, 0.0, 0.0), (0.0, 1.0, 0.0))))
    assert not rejected


def test_opposite_sides_of_one_edge_fail_anchor_consensus() -> None:
    values = (
        _geometry("w:e0", 0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        _geometry("w:e0", 1, (0.1, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    )
    assert not endpoint_anchor_consensus(values)[0]


def test_executed_branch_witness_is_parameter_free() -> None:
    acute = (
        _geometry("w:e0", 0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        _geometry("w:e1", 0, (0.0, 0.0, 0.0), (2 ** -0.5, 2 ** -0.5, 0.0)),
    )
    straight = (
        acute[0], _geometry("w:e1", 0, (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    )
    three = (*straight, _geometry("w:e2", 0, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    assert executed_branch_witness(acute)
    assert not executed_branch_witness(straight)
    assert executed_branch_witness(three)
