"""Exact fixtures: no world payload, teacher generation or model execution."""
import numpy as np
import pytest
from mtare_topo.representation.gse_observed_connection_probe import (
    Surface, BridgeWitness, local_connections,
)


def tile(key, x, y=0., z=0.):
    return Surface(key, np.array([[x, y, z], [x+.5, y, z],
                                  [x, y+.5, z], [x+.5, y+.5, z]]))


def bridge(a, b):
    return BridgeWitness(a.key, b.key, np.concatenate([a.points, b.points]))


def fixture(name):
    if name == "straight":
        surfaces = [tile("a", 0), tile("b", .5), tile("c", 1)]
        pairs = [(0, 1), (1, 2)]
    elif name == "t_junction":
        surfaces = [tile("center", 0), tile("west", -.5), tile("east", .5), tile("north", 0, .5)]
        pairs = [(0, 1), (0, 2), (0, 3)]
    elif name == "nearby_junctions":
        surfaces = [tile("j1", 0), tile("j2", .5), tile("branch1", 0, .5), tile("branch2", .5, -.5)]
        pairs = [(0, 1), (0, 2), (1, 3)]
    elif name == "stacked_crossing":
        surfaces = [tile("low1", 0), tile("low2", .5), tile("high1", 0, 0, 2), tile("high2", 0, .5, 2)]
        pairs = [(0, 1), (2, 3)]
    elif name == "occluded":
        surfaces = [tile("a", 0), tile("b", .5)]
        return surfaces, [BridgeWitness("a", "b", np.empty((0, 3)))], []
    elif name == "ramp":
        surfaces = [tile("a", 0), tile("b", .5)]
        surfaces = [Surface(s.key, s.points @ np.array([[1, 0, .2], [0, 1, 0], [0, 0, 1]])) for s in surfaces]
        pairs = [(0, 1)]
    else:
        raise ValueError(name)
    return surfaces, [bridge(surfaces[a], surfaces[b]) for a, b in pairs], sorted(
        tuple(sorted((surfaces[a].key, surfaces[b].key))) for a, b in pairs)


@pytest.mark.parametrize("name", ["straight", "t_junction", "nearby_junctions", "stacked_crossing", "occluded", "ramp"])
def test_same_backend_evidence_recovery(name):
    surfaces, witnesses, expected = fixture(name)
    baseline = local_connections(surfaces, witnesses)
    primitive = local_connections(surfaces, witnesses, representation="primitives")
    assert baseline == primitive
    assert baseline["local_evidence_links"] == expected
    assert baseline["confirmed_navigation_edges"] == []
    if name == "stacked_crossing":
        assert len(baseline["components"]) == 2
    if name == "occluded":
        assert baseline["unknown_pairs"] == [("a", "b")]


def test_missing_support_never_inferred_from_proximity():
    surfaces, _, _ = fixture("t_junction")
    for method in ("points", "primitives"):
        assert not local_connections(surfaces, [], representation=method)["local_evidence_links"]


def test_same_observation_different_hidden_world_identical():
    surfaces, witnesses, _ = fixture("occluded")
    # Hidden dead end vs hidden connecting tunnel deliberately NEVER forwarded.
    predictions = [local_connections(surfaces, witnesses) for _ in ("dead_end", "connected")]
    assert predictions[0] == predictions[1]


def test_false_vertical_bridge_not_confirmed():
    a, b = tile("a", 0), tile("b", 0, 0, 2)
    for method in ("points", "primitives"):
        output = local_connections([a, b], [bridge(a, b)], representation=method)
        assert not output["local_evidence_links"]
        assert output["unknown_pairs"] == [("a", "b")]


def test_rotation_permutation_and_prefix():
    surfaces, witnesses, _ = fixture("t_junction")
    rotation = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
    for method in ("points", "primitives"):
        before = local_connections(surfaces, witnesses[:1], representation=method)
        after = local_connections(surfaces, witnesses, representation=method)
        assert set(before["local_evidence_links"]) <= set(after["local_evidence_links"])
        rotated = [Surface(s.key, s.points[::-1] @ rotation) for s in surfaces[::-1]]
        rotated_w = [BridgeWitness(w.left, w.right, w.points @ rotation) for w in witnesses[::-1]]
        assert local_connections(rotated, rotated_w, representation=method) == after
        assert local_connections(surfaces, witnesses[:1], representation=method) == before
