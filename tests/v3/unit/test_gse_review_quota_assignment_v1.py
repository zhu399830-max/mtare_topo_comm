"""Synthetic metadata only; no world, model, teacher or inventory reads."""
from dataclasses import replace
from collections import Counter
from itertools import combinations
import random

import pytest

from mtare_topo.data.gse_review_quota_assignment_v1 import (
    QUOTAS, STRATA, ReviewNomination, _rank, assign_review_quotas,
)


def nomination(parent="p", edge="e", stratum="corridor", target=None,
               split="fit", clip=None):
    return ReviewNomination(parent, split, edge, stratum,
                            target if target is not None else edge,
                            clip if clip is not None else parent + ":" + edge)


def feasible(selected):
    if len({(n.parent, n.host_physical_edge_id) for n in selected}) != len(selected):
        return False
    nodes = [(n.parent, n.target_identity) for n in selected if n.stratum in ("junction", "terminal")]
    if len(set(nodes)) != len(nodes):
        return False
    counts = Counter((n.split, n.stratum) for n in selected)
    return all(v <= QUOTAS[k[0]] for k, v in counts.items())


def objective(selected, ordered):
    counts = Counter(n.parent for n in selected)
    return (len(selected), tuple(sum(v >= level for v in counts.values()) for level in range(1, 33)),
            tuple(n in selected for n in ordered))


def exhaustive(nominations):
    ordered = sorted(nominations, key=_rank)
    candidates = (s for k in range(len(ordered) + 1) for s in combinations(ordered, k))
    return max((s for s in candidates if feasible(s)), key=lambda s: objective(s, ordered))


def test_empty_is_explicit_deficit_not_success():
    result = assign_review_quotas(())
    assert not result.complete and len(result.deficits) == 12
    assert sum(r["missing"] for r in result.deficits) == 48
    assert result.selected == ()


def test_complete_quotas_and_parent_independence_not_claimed():
    rows = tuple(nomination(parent=f"{split}:{i}", edge=f"e:{stratum}", stratum=stratum,
                            target=f"node:{stratum}" if stratum in STRATA[:2] else None, split=split)
                 for split, quota in QUOTAS.items() for i in range(quota) for stratum in STRATA)
    result = assign_review_quotas(rows)
    assert result.complete and len(result.selected) == 48 and result.deficits == ()
    assert result.metadata["source_provenance_verified"] is False
    assert result.metadata["human_labels_created"] == 0
    assert result.metadata["training_eligibility"] is False
    assert feasible(result.selected)


def test_joint_flow_escapes_greedy_shared_host_and_shared_node_trap():
    rows = (
        nomination(edge="a", stratum="junction", target="node:n"),
        nomination(edge="b", stratum="junction", target="node:n"),
        nomination(edge="a", stratum="corridor"),
        nomination(edge="a", stratum="alias"),
        nomination(edge="c", stratum="terminal", target="node:t"),
    )
    result = assign_review_quotas(rows)
    assert len(result.selected) == 3
    assert next(n for n in result.selected if n.stratum == "junction").host_physical_edge_id == "b"
    assert objective(result.selected, sorted(rows, key=_rank)) == objective(exhaustive(rows), sorted(rows, key=_rank))


def test_parent_coverage_precedes_hash_and_then_balances_counts():
    rows = tuple(nomination(parent=p, edge=str(i), split="calibration")
                 for p in ("a", "b", "c") for i in range(4))
    result = assign_review_quotas(rows)
    assert len(result.selected) == 2
    assert len({n.parent for n in result.selected}) == 2
    # Four stratum quotas permit eight total. With three parents and ample
    # alternatives the max-coverage profile gives counts 3,3,2, not 4,3,1.
    rows = tuple(nomination(parent=p, edge=f"{s}:{i}", split="calibration", stratum=s,
                            target=f"node:{s}:{i}" if s in STRATA[:2] else None)
                 for p in ("a", "b", "c") for s in STRATA for i in range(2))
    result = assign_review_quotas(rows)
    assert len(result.selected) == 8
    assert sorted(result.metadata["parent_selected_counts"].values()) == [2, 3, 3]
    assert result.metadata["parent_coverage_profile"][:4] == [3, 3, 2, 0]


@pytest.mark.parametrize("seed", range(16))
def test_exact_objective_matches_bruteforce_small_conflict_networks(seed):
    rng = random.Random(seed)
    rows = set()
    while len(rows) < 10:
        parent = rng.choice(("p0", "p1", "p2"))
        edge = str(rng.randrange(3))
        stratum = rng.choice(STRATA)
        target = f"node:{stratum}:{rng.randrange(2)}" if stratum in STRATA[:2] else None
        rows.add(nomination(parent, edge, stratum, target, "calibration"))
    rows = tuple(rows)
    result = assign_review_quotas(rows)
    ordered = sorted(rows, key=_rank)
    assert objective(result.selected, ordered) == objective(exhaustive(rows), ordered)
    assert result == assign_review_quotas(tuple(reversed(rows)))
    assert feasible(result.selected)


@pytest.mark.parametrize("field,value", [("parent", True), ("target_identity", 1),
    ("split", "train"), ("stratum", "unknown"), ("clip_id", " x"), ("host_physical_edge_id", "")])
def test_invalid_metadata_rejected(field, value):
    with pytest.raises(ValueError):
        assign_review_quotas((replace(nomination(), **{field: value}),))


@pytest.mark.parametrize("case", ("duplicate", "split", "clip_edge", "clip_parent", "node_stratum", "edge_target", "list"))
def test_conflicting_identities_rejected(case):
    first = nomination()
    rows = {
        "duplicate": (first, first),
        "split": (first, nomination(edge="f", split="calibration")),
        "clip_edge": (first, nomination(edge="f", clip=first.clip_id)),
        "clip_parent": (first, nomination(parent="other", clip=first.clip_id)),
        "node_stratum": (nomination(stratum="junction", target="node:n"), nomination(edge="f", stratum="terminal", target="node:n")),
        "edge_target": (replace(first, target_identity="unrelated"),),
        "list": [first],
    }[case]
    with pytest.raises(ValueError):
        assign_review_quotas(rows)


def test_2546_windows_four_nominations_scale_without_payloads():
    rows = tuple(nomination(parent=f"p:{i % 70}", edge=f"e:{i // 3}",
                            stratum=s, target=f"node:{s}:{i // 7}" if s in STRATA[:2] else None,
                            split="fit" if i % 70 < 60 else ("calibration" if i % 70 < 65 else "development"),
                            clip=f"clip:{i}")
                 for i in range(2546) for s in STRATA)
    before = tuple(rows)
    result = assign_review_quotas(rows)
    assert rows == before and len(rows) == 10184
    assert result.complete and len(result.selected) == 48
    assert feasible(result.selected)
    assert result.metadata["parent_coverage_profile"][0] == 42  # 32 fit + 5 cal + 5 dev parents.
