import numpy as np
import pytest

from mtare_topo.teacher.gse_factorized_association_teacher import (
    causal_history_row_references,
    causal_history_row_references_unordered,
    choose_hard_negative_identity,
    choose_positive_rows,
    masked_profile_distance,
    objective_geometry_profile,
)


def _row(index, *, traversal="t", sequence=0, identity="w:node:n", edge="e0", from_node="x", to_node="n"):
    return {
        "global_sequence_index": index, "traversal_id": traversal, "sequence_index": sequence,
        "identity": identity, "event": "junction", "edge_id": edge,
        "from_node_id": from_node, "to_node_id": to_node, "geometry_valid": True,
        "width_m": 6.0 + index, "height_m": 4.0, "slope_deg": 1.0,
        "curvature_per_m": .01,
    }


def test_causal_history_references_left_pad_and_reset():
    traversal = np.asarray(["a"] * 3 + ["b"] * 2)
    sequence = np.asarray([0, 1, 2, 0, 1])
    refs, mask = causal_history_row_references(traversal, sequence, maximum=3)
    assert refs[2].tolist() == [0, 1, 2]
    assert refs[3].tolist() == [-1, -1, 3]
    assert mask[4].tolist() == [False, True, True]


def test_unordered_causal_history_retains_caller_rows_and_never_crosses_traversal():
    traversal = np.asarray(["a", "b", "a", "a", "b"])
    sequence = np.asarray([2, 1, 0, 1, 0])
    refs, mask = causal_history_row_references_unordered(traversal, sequence, maximum=3)
    assert refs[0].tolist() == [2, 3, 0]
    assert refs[1].tolist() == [-1, 4, 1]
    assert mask[2].tolist() == [False, False, True]
    with pytest.raises(ValueError, match="consecutive"):
        causal_history_row_references_unordered(np.asarray(["a", "a"]), np.asarray([0, 2]))


def test_objective_profile_masks_invalid_width_height_only():
    rows = [_row(0), _row(1, sequence=1)]
    rows[0]["geometry_valid"] = False
    rows[0]["width_m"] = None
    rows[0]["height_m"] = None
    refs, mask = causal_history_row_references(np.asarray(["t", "t"]), np.asarray([0, 1]))
    feature, valid = objective_geometry_profile(rows, refs, mask, 1)
    assert feature.shape == (8,)
    assert valid.tolist() == [True] * 8
    assert feature[0] == pytest.approx(rows[1]["width_m"] / 30.0)


def test_masked_profile_distance_requires_common_semantics():
    left = np.zeros(8)
    right = np.ones(8)
    mask = np.asarray([False, False, True, True, False, False, True, True])
    assert masked_profile_distance(left, mask, right, mask) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="four common"):
        masked_profile_distance(left, mask & np.asarray([True, True, True, True, False, False, False, False]), right, mask)


def test_positive_prefers_different_inbound_edge():
    rows = [
        _row(0, traversal="a", edge="e0"),
        _row(1, traversal="b", edge="e1"),
        _row(2, traversal="c", edge="e0", from_node="n", to_node="x"),
    ]
    history_mask = np.ones((3, 5), dtype=np.bool_)
    query, positive, kind = choose_positive_rows(rows, [0, 1, 2], history_mask, np.ones(3, dtype=np.uint8))
    assert query == 0
    assert positive == 1
    assert kind == "different_physical_edge"


def test_singleton_terminal_uses_explicit_circular_shift_augmentation():
    row = _row(9)
    row["event"] = "terminal"
    history_mask = np.ones((1, 5), dtype=np.bool_)
    query, positive, kind = choose_positive_rows(
        [row], [0], history_mask, np.ones(1, dtype=np.uint8)
    )
    assert (query, positive) == (0, 0)
    assert kind == "singleton_circular_shift_augmentation"


def test_hard_negative_matches_partition_event_degree_and_ties_identity():
    metadata = {
        "q": {"partition": "fit", "event": "junction", "degree": 3, "profile": np.zeros(8), "profile_mask": np.ones(8, bool)},
        "b": {"partition": "fit", "event": "junction", "degree": 3, "profile": np.ones(8), "profile_mask": np.ones(8, bool)},
        "a": {"partition": "fit", "event": "junction", "degree": 3, "profile": -np.ones(8), "profile_mask": np.ones(8, bool)},
        "x": {"partition": "selection", "event": "junction", "degree": 3, "profile": np.zeros(8), "profile_mask": np.ones(8, bool)},
    }
    identity, distance = choose_hard_negative_identity("q", metadata)
    assert identity == "a"
    assert distance == pytest.approx(1.0)
