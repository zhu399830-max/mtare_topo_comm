from __future__ import annotations

from mtare_topo.data.gse_corrected_teacher_manifest import (
    apply_causal_change_point_labels,
    audit_preserved_sparse_global_indices,
    corrected_identity_summaries,
    corrected_manifest_summary,
    corrected_world_association_pairs,
)


def test_sparse_global_indices_are_preserved_not_renumbered() -> None:
    source = [0, 1, 7, 8, 20]
    audit = audit_preserved_sparse_global_indices(source, source)
    assert audit.to_dict() == {
        "count": 5,
        "minimum": 0,
        "maximum": 20,
        "omitted_internal_index_count": 16,
        "gap_block_count": 2,
        "unique": True,
        "strictly_increasing": True,
        "exact_source_order": True,
    }


def test_sparse_global_index_audit_rejects_renumbering_and_reordering() -> None:
    source = [0, 1, 7, 8]
    renumbered = audit_preserved_sparse_global_indices(source, range(4))
    assert renumbered.unique and renumbered.strictly_increasing
    assert not renumbered.exact_source_order
    reordered = audit_preserved_sparse_global_indices(source, [0, 7, 1, 8])
    assert not reordered.strictly_increasing
    assert not reordered.exact_source_order


def _row(index: int, event: str, identity: str | None) -> dict:
    return {
        "observation_id": f"world:edge_0:d0:s{index:06d}",
        "parent_id": "world",
        "traversal_id": "world:edge_0:d0",
        "edge_id": "edge_0",
        "sequence_index": index,
        "canonical_edge_arc_m": float(index + 4),
        "event": event,
        "identity": identity,
        "width_m": 4.0,
        "height_m": 5.0,
        "slope_deg": 1.0,
        "curvature_per_m": 0.01,
    }


def _label(index: int, identity: str = "world:node:n2") -> dict:
    return {
        "parent_id": "world",
        "traversal_id": "world:edge_0:d0",
        "edge_id": "edge_0",
        "sequence_index": index,
        "identity": identity,
        "identity_kind": "degree_two_endpoint",
    }


def test_old_transition_is_removed_and_causal_label_is_applied() -> None:
    rows = [_row(0, "geometry_transition", "old"), _row(1, "corridor", None)]
    corrected, audit = apply_causal_change_point_labels(rows, {(rows[1]["traversal_id"], 1): _label(1)})
    assert corrected[0]["event"] == "corridor"
    assert corrected[0]["identity"] is None
    assert corrected[1]["event"] == "geometry_transition"
    assert corrected[1]["identity"] == "world:node:n2"
    assert audit[0]["action"] == "applied_geometry_transition"


def test_junction_and_terminal_priority_suppress_causal_label() -> None:
    rows = [_row(0, "junction", "world:node:j"), _row(1, "terminal", "world:node:t")]
    labels = {
        (rows[0]["traversal_id"], 0): _label(0),
        (rows[1]["traversal_id"], 1): _label(1),
    }
    corrected, audit = apply_causal_change_point_labels(rows, labels)
    assert [row["event"] for row in corrected] == ["junction", "terminal"]
    assert [row["identity"] for row in corrected] == ["world:node:j", "world:node:t"]
    assert [row["action"] for row in audit] == [
        "suppressed_by_junction_priority",
        "suppressed_by_terminal_priority",
    ]


def test_transition_priority_over_turn_is_explicit() -> None:
    row = _row(0, "turn", "world:edge_0:turn:000")
    corrected, audit = apply_causal_change_point_labels(
        [row], {(row["traversal_id"], 0): _label(0, "world:edge_0:geometry_change_point:000")}
    )
    assert corrected[0]["event"] == "geometry_transition"
    assert audit[0]["old_event"] == "turn"


def test_missing_or_cross_edge_label_fails_closed() -> None:
    row = _row(0, "corridor", None)
    missing = _label(1)
    try:
        apply_causal_change_point_labels([row], {(row["traversal_id"], 1): missing})
    except RuntimeError as error:
        assert "lack a Teacher observation" in str(error)
    else:
        raise AssertionError("missing label was accepted")

    cross = _label(0)
    cross["edge_id"] = "edge_other"
    try:
        apply_causal_change_point_labels([row], {(row["traversal_id"], 0): cross})
    except ValueError as error:
        assert "physical edge" in str(error)
    else:
        raise AssertionError("cross-edge label was accepted")


def test_corrected_pairs_and_identity_summary_use_new_identity() -> None:
    rows = [_row(0, "corridor", None), _row(1, "corridor", None)]
    labels = {
        (rows[0]["traversal_id"], 0): _label(0),
        (rows[1]["traversal_id"], 1): _label(1),
    }
    corrected, _ = apply_causal_change_point_labels(rows, labels)
    pairs = corrected_world_association_pairs(corrected)
    assert len(pairs) == 2
    assert all(pair["same_identity"] for pair in pairs)
    summaries = corrected_identity_summaries(corrected, {"world:node:n2": "degree_two_endpoint"})
    assert len(summaries) == 1
    assert summaries[0]["observation_count"] == 2
    assert summaries[0]["identity_kind"] == "degree_two_endpoint"
    summary = corrected_manifest_summary(corrected)
    assert summary["event_counts"] == {"geometry_transition": 2}
    assert summary["identity_counts"] == {"geometry_transition": 1}
