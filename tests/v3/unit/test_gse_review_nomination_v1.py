"""Synthetic-only review nomination regression tests."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from mtare_topo.data.gse_review_nomination_v1 import (
    OverlapRows, nominate_interval, project_construction_identity, validate_paired_constructions,
)
from mtare_topo.data.gse_review_source_index_v1 import (
    TraversalIndexInterval, VariantTraversalIndices, VARIANTS,
)
from mtare_topo.data.gse_review_quota_assignment_v1 import assign_review_quotas


PARENT = "S01_synthetic_C01"


def document():
    edges = (("host", "j", "t"), ("branch1", "j", "a"), ("branch2", "j", "b"), ("stacked", "c", "d"))
    primitives, by_node = [], {}
    for edge, a, b in edges:
        ends = [{"primitive_id": "p:" + edge, "endpoint_index": i, "node_id": node}
            for i, node in enumerate((a, b))]
        primitives.append({"primitive_id": "p:" + edge, "source_edge_id": edge, "endpoints": ends})
        for end in ends:
            by_node.setdefault(end["node_id"], []).append(deepcopy(end))
    return {"schema_version": "primitive_relation_realized_construction_v1",
        "base_construction": {"schema_version": "primitive_construction_graph_v1", "primitives": primitives,
            "composition_operations": [{"node_id": n, "degree": len(m), "member_endpoints": m}
                for n, m in by_node.items()]},
        "realized_primitives": [{"primitive_id": row["primitive_id"], "ignored_shape": 2.}
            for row in primitives]}


def interval(q=23, direction=0):
    records = tuple(VariantTraversalIndices(v, PARENT + "__" + v, tuple(range(q)), tuple(range(100, 100 + q)),
        tuple(tuple(range(i, i + 5)) for i in range(q)), tuple(float(i + 4) for i in range(q))) for v in VARIANTS)
    return TraversalIndexInterval(PARENT + ":host:d" + str(direction), records)


def rows(q=23, pair=(0, 3)):
    idx = np.full((q, 32), -1, dtype=np.int32)
    idx[:, :4] = np.arange(4)
    mask = (idx >= 0).astype(np.uint8)
    matrix = np.zeros((q, 32, 32), dtype=np.uint8)
    if pair is not None:
        a, b = pair; matrix[0, a, b] = matrix[0, b, a] = 1
    packed = np.packbits(matrix, axis=2, bitorder="little")
    return OverlapRows(tuple(range(q)), idx, mask, packed)


def run(q=23, direction=0, overlap=None):
    return nominate_interval(parent_id=PARENT, split="fit", interval=interval(q, direction),
        construction=project_construction_identity(document()),
        overlap_by_variant=overlap or {v: rows(q) for v in VARIANTS})


def test_boundary_direction_midpoint_and_no_labels():
    result = run()
    assert result.training_labels_created == 0
    assert result.counts["eligible_windows"] == 3
    junction = [e for e in result.evidence if e["stratum"] == "junction"]
    terminal = [e for e in result.evidence if e["stratum"] == "terminal"]
    assert [(e["target_identity"], e["basis"]) for e in junction] == [("node:j", "first_available_decision")]
    assert [(e["target_identity"], e["basis"]) for e in terminal] == [("node:t", "last_available_decision")]
    assert len([n for n in result.nominations if n.stratum == "corridor"]) == 3
    assert all(e["visible_label"] is None for e in result.evidence)
    reverse = run(direction=1)
    assert [e["basis"] for e in reverse.evidence if e["stratum"] == "junction"] == ["last_available_decision"]
    assert [e["basis"] for e in reverse.evidence if e["stratum"] == "terminal"] == ["first_available_decision"]


def test_alias_does_not_delete_corridor_and_host_edge_counted_once():
    result = run(q=21)
    assert {n.stratum for n in result.nominations} == {"junction", "terminal", "corridor", "alias"}
    assert len({n.clip_id for n in result.nominations}) == 1
    selected = assign_review_quotas(result.nominations)
    assert len(selected.selected) == 1 and not selected.complete


def test_unrelated_remote_overlap_is_not_host_alias():
    result = run(overlap={v: rows(pair=(1, 3)) for v in VARIANTS})
    assert not any(n.stratum == "alias" for n in result.nominations)


def test_one_variant_support_suffices_but_all_variants_required():
    data = {v: rows(pair=None) for v in VARIANTS}
    data[VARIANTS[1]] = rows()
    result = run(overlap=data)
    evidence = [e for e in result.evidence if e["stratum"] == "alias"]
    assert len(evidence) == 1
    assert {w["variant"] for w in evidence[0]["witnesses"]} == {VARIANTS[1]}
    del data[VARIANTS[0]]
    with pytest.raises(ValueError, match="all three"):
        run(overlap=data)


def test_connected_pair_cannot_be_used_as_disconnected_alias():
    with pytest.raises(ValueError, match="contradicts"):
        run(overlap={v: rows(pair=(0, 1)) for v in VARIANTS})


def test_id_projection_permits_shape_changes_not_identity_changes():
    docs = {v: document() for v in VARIANTS}
    docs[VARIANTS[1]]["realized_primitives"][0]["ignored_shape"] = 9.
    assert validate_paired_constructions(docs) == project_construction_identity(document())
    docs[VARIANTS[1]]["base_construction"]["primitives"][0]["source_edge_id"] = "another_edge"
    with pytest.raises(ValueError, match="cross-variant"):
        validate_paired_constructions(docs)


@pytest.mark.parametrize("mutation", [
    lambda d: d["base_construction"]["primitives"][0]["endpoints"].reverse(),
    lambda d: d["base_construction"]["composition_operations"][0].update(degree=2),
    lambda d: d["base_construction"]["composition_operations"][0].update(degree=True),
    lambda d: d["base_construction"]["composition_operations"].pop(),
    lambda d: d["realized_primitives"].reverse(),
    lambda d: d["base_construction"]["primitives"][1].update(source_edge_id="host"),
    lambda d: d["base_construction"]["primitives"][0]["endpoints"][0].update(endpoint_index=False),
    lambda d: d["base_construction"]["composition_operations"][0]["member_endpoints"].append(
        deepcopy(d["base_construction"]["composition_operations"][0]["member_endpoints"][0])),
])
def test_construction_defects_fail(mutation):
    d = document(); mutation(d)
    with pytest.raises(ValueError):
        project_construction_identity(d)


@pytest.mark.parametrize("mutation", [
    lambda r: replace(r, sequence_rows=tuple(reversed(r.sequence_rows))),
    lambda r: replace(r, primitive_index=r.primitive_index.astype(np.int64)),
    lambda r: replace(r, primitive_mask=np.full_like(r.primitive_mask, 2)),
    lambda r: replace(r, primitive_index=np.full_like(r.primitive_index, -2)),
    lambda r: replace(r, primitive_index=np.full_like(r.primitive_index, 4)),
    lambda r: replace(r, disconnected_overlap_packed=np.full_like(r.disconnected_overlap_packed, 255)),
])
def test_overlap_shape_identity_mask_bounds_fail(mutation):
    data = {v: rows() for v in VARIANTS}; data[VARIANTS[0]] = mutation(data[VARIANTS[0]])
    with pytest.raises(ValueError):
        run(overlap=data)


def test_duplicate_or_unsorted_active_slots_fail():
    for replacements in ((0, 0, 2, 3), (3, 2, 1, 0)):
        data = {v: rows(pair=None) for v in VARIANTS}
        data[VARIANTS[0]].primitive_index[:, :4] = replacements
        with pytest.raises(ValueError, match="sorted unique"):
            run(overlap=data)


def test_midpoint_nominations_do_not_require_every_window():
    result = run(q=50)
    # q=50 lower median index=24; window starts4..24 include it.
    assert len([n for n in result.nominations if n.stratum == "corridor"]) == 21


def test_deterministic_read_only():
    data = {v: rows() for v in VARIANTS}
    before = {v: (r.primitive_index.tobytes(), r.primitive_mask.tobytes(), r.disconnected_overlap_packed.tobytes())
        for v, r in data.items()}
    assert run(overlap=data) == run(overlap=dict(reversed(tuple(data.items()))))
    assert before == {v: (r.primitive_index.tobytes(), r.primitive_mask.tobytes(), r.disconnected_overlap_packed.tobytes())
        for v, r in data.items()}


def test_reject_short_interval_and_forbidden_parent():
    with pytest.raises(ValueError, match="eligible"):
        run(q=20)
    with pytest.raises(ValueError, match="permitted"):
        nominate_interval(parent_id="S01_synthetic_C10", split="development", interval=interval(),
            construction=project_construction_identity(document()), overlap_by_variant={v: rows() for v in VARIANTS})


def test_reverse_traversal_not_another_independent_case():
    all_nominations = run(q=21).nominations + run(q=21, direction=1).nominations
    assert len(assign_review_quotas(all_nominations).selected) == 1


@pytest.mark.parametrize("changes", [
    {"source_sequence_ids": (100,) * 23},
    {"source_sequence_ids": ()}, {"frame_rows": ()}, {"decision_arc_m": ()},
    {"frame_rows": ((0, 1, 2, 3, 4),) * 23},
    {"decision_arc_m": (float("nan"),) * 23},
    {"decision_arc_m": (1.,) * 23},
    {"frame_rows": tuple((False, 1, 2, 3, 4) for _ in range(23))},
])
def test_forged_typed_interval_is_not_validation_evidence(changes):
    data = interval()
    bad = replace(data, variants=tuple(replace(v, **changes) for v in data.variants))
    with pytest.raises(ValueError, match="identity/order"):
        nominate_interval(parent_id=PARENT, split="fit", interval=bad,
            construction=project_construction_identity(document()), overlap_by_variant={v: rows() for v in VARIANTS})
