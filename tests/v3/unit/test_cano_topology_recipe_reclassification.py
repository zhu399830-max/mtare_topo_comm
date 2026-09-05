from __future__ import annotations

from copy import deepcopy

from mtare_topo.data.cano_topology_recipe_reclassification import (
    RECIPE_BY_STRATUM,
    recipe_batch_audit,
    recipe_candidate_audit,
    select_fixed_recipe_parents,
)


def _candidate(
    stratum: str, candidate_index: int, grown: int, connectors: int, cycle_rank: int
) -> dict:
    candidate_id = f"{stratum}_C{candidate_index:02d}"
    operations = [
        {"operation": "grown", "success": True, "parameter_draw_count": 1}
        for _ in range(grown)
    ] + [
        {"operation": "connector", "success": True, "parameter_draw_count": 2}
        for _ in range(connectors)
    ]
    return {
        "candidate_id": candidate_id,
        "stratum_id": stratum,
        "requested_grown_tunnels": grown,
        "requested_connector_tunnels": connectors,
        "topology_seed": 620000 + candidate_index,
        "reserved_geometry_seed": 720000 + candidate_index,
        "generation_succeeded": True,
        "replay_identical": True,
        "operations": operations,
        "failure_reasons": ["exact_connector_cycle_rank"] if cycle_rank != connectors else [],
        "metrics": {
            "checks": {
                "requested_tunnel_counts_match": True,
                "single_connected_component": True,
                "finite_graph": True,
                "finite_splines": True,
                "exact_connector_cycle_rank": cycle_rank == connectors,
                "minimum_centerline_length": True,
                "flat_vertical_span": True,
                "three_d_vertical_span": True,
            },
            "cycle_rank": cycle_rank,
            "terminal_count": 4,
            "degree3_or_more_count": 2,
            "vertical_span_m": 0.0,
            "total_centerline_length_m": 500.0,
            "canonical_parent_identity": f"identity-{candidate_id}",
            "coordinate_free_wl_hash": f"wl-{candidate_id}",
        },
    }


def test_extra_natural_cycle_is_recipe_valid_and_preserved_as_gt() -> None:
    candidate = _candidate("S07_flat_loop_rich", 2, grown=8, connectors=2, cycle_rank=3)
    audit = recipe_candidate_audit(candidate)
    assert audit["recipe_valid"]
    assert audit["recipe_stratum_id"] == "R07_flat_g8_c2"
    assert audit["actual_cycle_rank"] == 3


def test_non_cycle_failure_is_not_silently_accepted() -> None:
    candidate = _candidate("S01_flat_tree_small", 1, grown=4, connectors=0, cycle_rank=0)
    candidate["generation_succeeded"] = False
    candidate["failure_reasons"] = ["requested_tunnel_generation_failed"]
    audit = recipe_candidate_audit(candidate)
    assert not audit["recipe_valid"]
    assert "generation_not_successful" in audit["failure_reasons"]


def test_fixed_first_ten_selection_produces_80_10_10_without_cycle_selection() -> None:
    candidates = {}
    audits = []
    for stratum_index, stratum in enumerate(RECIPE_BY_STRATUM, start=1):
        grown = (4, 4, 4, 4, 6, 6, 8, 8, 10, 10)[stratum_index - 1]
        connectors = (0, 0, 1, 1, 1, 1, 2, 2, 3, 3)[stratum_index - 1]
        for candidate_index in range(1, 13):
            cycle_rank = connectors + (1 if candidate_index in (2, 11) else 0)
            candidate = _candidate(stratum, candidate_index, grown, connectors, cycle_rank)
            candidate["topology_seed"] += 100 * stratum_index
            candidate["reserved_geometry_seed"] += 100 * stratum_index
            candidates[candidate["candidate_id"]] = candidate
            audits.append(recipe_candidate_audit(candidate))
    selected = select_fixed_recipe_parents(audits, candidates)
    batch = recipe_batch_audit(audits, selected)
    assert batch["passed"]
    assert batch["split_counts"] == {"train": 80, "validation": 10, "development_test": 10}
    assert any(item["actual_cycle_rank"] > item["requested_connector_tunnels"] for item in selected)
