import numpy as np
import pytest

from mtare_topo.evaluation.gse_factorized_association_inventory import (
    decision_pair_inventory,
    identity_node_id,
    inbound_profile_identity_counts,
    past_observation_lengths,
)


def test_past_lengths_reset_without_crossing_traversal():
    traversal = np.asarray(["a"] * 7 + ["b"] * 3)
    sequence = np.asarray(list(range(7)) + list(range(3)))
    assert past_observation_lengths(traversal, sequence).tolist() == [1, 2, 3, 4, 5, 5, 5, 1, 2, 3]


def test_past_lengths_reject_repeated_block():
    with pytest.raises(ValueError, match="multiple blocks"):
        past_observation_lengths(np.asarray(["a", "b", "a"]), np.asarray([0, 0, 0]))


def test_identity_node_parser_is_strict():
    assert identity_node_id("world:node:node_0012") == "node_0012"
    with pytest.raises(ValueError):
        identity_node_id("world:turn:x")


def test_inbound_profile_requires_to_node_and_causal_length():
    rows = [
        {"event": "junction", "identity": "w:node:n", "to_node_id": "n", "edge_id": "e0"},
        {"event": "junction", "identity": "w:node:n", "to_node_id": "x", "edge_id": "e1"},
        {"event": "terminal", "identity": "w:node:t", "to_node_id": "t", "edge_id": "e2"},
    ]
    result = inbound_profile_identity_counts(
        rows, np.asarray([5, 5, 3]), np.ones(3, dtype=np.bool_)
    )
    assert result["decision_identities"] == 2
    assert result["identities_with_inbound_profile"] == 1
    assert result["covered_identity_edge_pairs"] == 1


def test_decision_pair_inventory_filters_distance_and_nondecision_rows():
    event = np.asarray(["junction", "junction", "terminal", "corridor"])
    identity = np.asarray(["a", "a", "b", "c"])
    edge = np.asarray(["e0", "e1", "e2", "e3"])
    result = decision_pair_inventory(
        left=np.asarray([0, 0, 0, 0]),
        right=np.asarray([1, 2, 3, 2]),
        label=np.asarray([1, 0, 0, 0]),
        distance_m=np.asarray([2.0, 3.0, 2.0, 20.0]),
        family=np.asarray(["S01"] * 4),
        event=event, identity=identity, edge_id=edge,
    )
    assert result["online_eligible_decision_pairs"] == 2
    assert result["positive_pairs"] == 1
    assert result["positive_different_physical_edge"] == 1
    assert result["negative_same_event_type"] == 0
