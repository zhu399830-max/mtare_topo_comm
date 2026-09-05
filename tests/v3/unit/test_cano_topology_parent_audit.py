from __future__ import annotations

from copy import deepcopy

from mtare_topo.data.cano_topology_parent_audit import (
    batch_diversity_audit,
    build_arithmetic_candidate_registry,
    graph_wl_hash,
    parent_metrics,
    select_accepted_candidates,
    select_corrective_train_validation_pairs,
    split_audit,
)


def _graph(coordinates: list[list[float]], edges: list[tuple[int, int]], *, grown: int = 1, connector: int = 0) -> dict:
    degrees = [0] * len(coordinates)
    for first, second in edges:
        degrees[first] += 1
        degrees[second] += 1
    return {
        "nodes": [
            {
                "id": f"n{index}",
                "xyz": xyz,
                "degree": degrees[index],
                "node_type": "tunnel_node",
            }
            for index, xyz in enumerate(coordinates)
        ],
        "edges": [
            {"id": f"e{index}", "node_ids": [f"n{first}", f"n{second}"]}
            for index, (first, second) in enumerate(edges)
        ],
        "tunnels": [
            *[{"id": index + 1, "type": "grown"} for index in range(grown)],
            *[{"id": grown + index + 1, "type": "connector"} for index in range(connector)],
        ],
        "statistics": {
            "connected_components": 1,
            "cycle_rank": len(edges) - len(coordinates) + 1,
        },
    }


def _splines(points: list[list[float]], count: int = 1) -> dict:
    return {
        "tunnels": [
            {
                "tunnel_id": index + 1,
                "points": points,
                "distances": [0.0, 400.0],
            }
            for index in range(count)
        ]
    }


def test_wl_hash_ignores_coordinates_but_detects_structure() -> None:
    chain = _graph([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], [(0, 1), (1, 2), (2, 3)])
    moved = deepcopy(chain)
    for index, node in enumerate(moved["nodes"]):
        node["xyz"] = [100.0 + index, -30.0 * index, 8.0]
    star = _graph([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]], [(0, 1), (0, 2), (0, 3)])
    assert graph_wl_hash(chain) == graph_wl_hash(moved)
    assert graph_wl_hash(chain) != graph_wl_hash(star)


def test_parent_metrics_enforce_flat_and_3d_contracts() -> None:
    flat_graph = _graph([[0, 0, 0], [400, 0, 0]], [(0, 1)])
    spline = _splines([[0, 0, 0], [400, 0, 0]])
    flat = parent_metrics(flat_graph, spline, dimensionality="flat", requested_grown=1, requested_connector=0)
    assert flat["passed_geometry_free_contract"]
    assert flat["vertical_span_m"] == 0.0
    failed_3d = parent_metrics(flat_graph, spline, dimensionality="3d", requested_grown=1, requested_connector=0)
    assert not failed_3d["checks"]["three_d_vertical_span"]


def test_parent_metrics_reject_extra_implicit_cycle() -> None:
    graph = _graph(
        [[0, 0, 0], [100, 0, 0], [50, 100, 0]],
        [(0, 1), (1, 2), (2, 0)],
        grown=1,
        connector=0,
    )
    spline = _splines([[0, 0, 0], [400, 0, 0]])
    metrics = parent_metrics(
        graph,
        spline,
        dimensionality="flat",
        requested_grown=1,
        requested_connector=0,
    )
    assert metrics["cycle_rank"] == 1
    assert not metrics["checks"]["exact_connector_cycle_rank"]
    assert not metrics["passed_geometry_free_contract"]


def test_first_ten_valid_selection_and_split_are_parent_disjoint() -> None:
    results = []
    for stratum_index in range(10):
        dimensionality = "3d" if stratum_index % 2 else "flat"
        stratum = f"S{stratum_index:02d}_{dimensionality}_fixture"
        for candidate_index in range(12):
            results.append(
                {
                    "candidate_id": f"{stratum}_C{candidate_index:02d}",
                    "stratum_id": stratum,
                    "topology_seed": 1000 + 100 * stratum_index + candidate_index,
                    "reserved_geometry_seed": 2000 + 100 * stratum_index + candidate_index,
                    "candidate_valid": candidate_index != 0,
                    "metrics": {
                        "canonical_parent_identity": f"identity-{stratum_index}-{candidate_index}",
                        "coordinate_free_wl_hash": f"wl-{stratum_index}-{candidate_index % 5}",
                        "has_degree3_or_more": True,
                            "vertical_span_m": 10.0 if dimensionality == "3d" else 0.0,
                    },
                }
            )
    accepted, strata = select_accepted_candidates(results)
    assert len(accepted) == 100
    assert all(item["retained_count"] == 10 for item in strata.values())
    split = split_audit(accepted)
    assert split["counts"] == {"train": 80, "validation": 10, "development_test": 10}
    assert split["parent_and_identity_sets_pairwise_disjoint"]
    diversity = batch_diversity_audit(accepted, strata)
    assert diversity["passed"]
    assert diversity["unique_coordinate_free_wl_hash_count"] == 50


def _corrective_candidates() -> list[dict]:
    results = []
    for stratum_index in range(1, 11):
        stratum = f"S{stratum_index:02d}_fixture"
        for candidate_index in range(13, 25):
            results.append(
                {
                    "candidate_id": f"{stratum}_C{candidate_index:02d}",
                    "stratum_id": stratum,
                    "topology_seed": 620000 + 100 * stratum_index + candidate_index,
                    "reserved_geometry_seed": 720000 + 100 * stratum_index + candidate_index,
                    "candidate_valid": candidate_index not in {13, 16},
                    "metrics": {
                        "canonical_parent_identity": f"identity-{stratum_index}-{candidate_index}",
                        "coordinate_free_wl_hash": f"wl-{stratum_index}-{candidate_index}",
                    },
                }
            )
    return results


def test_corrective_arithmetic_registry_is_exactly_c13_through_c24() -> None:
    strata = [{"stratum_id": f"S{index:02d}_fixture"} for index in range(1, 11)]
    registry = build_arithmetic_candidate_registry(
        strata,
        first_candidate_index=13,
        candidates_per_stratum=12,
        topology_seed_base=620000,
        geometry_seed_base=720000,
    )
    assert len(registry) == 120
    assert registry[0] == {
        "candidate_id": "S01_fixture_C13",
        "stratum_id": "S01_fixture",
        "candidate_index": 13,
        "topology_seed": 620113,
        "reserved_geometry_seed": 720113,
    }
    assert registry[-1]["candidate_id"] == "S10_fixture_C24"
    assert registry[-1]["topology_seed"] == 621024
    assert registry[-1]["reserved_geometry_seed"] == 721024


def test_corrective_selection_uses_first_two_valid_and_keeps_splits_disjoint() -> None:
    selected, audit = select_corrective_train_validation_pairs(_corrective_candidates())
    assert audit["passed"]
    assert audit["split_counts"] == {"corrective_train": 10, "corrective_validation": 10}
    assert {item["candidate_index"] for item in selected} == {14, 15}
    train = {item["canonical_parent_identity"] for item in selected if item["split"] == "corrective_train"}
    validation = {
        item["canonical_parent_identity"]
        for item in selected
        if item["split"] == "corrective_validation"
    }
    assert train.isdisjoint(validation)


def test_corrective_selection_fails_closed_on_missing_slot_or_insufficient_validity() -> None:
    missing = _corrective_candidates()[1:]
    _, missing_audit = select_corrective_train_validation_pairs(missing)
    assert not missing_audit["passed"]

    insufficient = _corrective_candidates()
    for item in insufficient:
        if item["stratum_id"] == "S10_fixture":
            item["candidate_valid"] = item["candidate_id"].endswith("_C24")
    _, insufficient_audit = select_corrective_train_validation_pairs(insufficient)
    assert not insufficient_audit["passed"]
    assert insufficient_audit["strata"]["S10_fixture"]["selected_candidate_ids"] == [
        "S10_fixture_C24"
    ]
