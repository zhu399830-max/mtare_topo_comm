"""Read-only recipe reclassification for sealed Cano topology candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


RECIPE_BY_STRATUM = {
    "S01_flat_tree_small": "R01_flat_g4_c0",
    "S02_3d_tree_small": "R02_3d_g4_c0",
    "S03_flat_unicyclic_small": "R03_flat_g4_c1",
    "S04_3d_unicyclic_small": "R04_3d_g4_c1",
    "S05_flat_branch_medium": "R05_flat_g6_c1",
    "S06_3d_branch_medium": "R06_3d_g6_c1",
    "S07_flat_loop_rich": "R07_flat_g8_c2",
    "S08_3d_loop_rich": "R08_3d_g8_c2",
    "S09_flat_complex": "R09_flat_g10_c3",
    "S10_3d_complex": "R10_3d_g10_c3",
}
RECIPE_PARAMETERS = {
    "R01_flat_g4_c0": (4, 0),
    "R02_3d_g4_c0": (4, 0),
    "R03_flat_g4_c1": (4, 1),
    "R04_3d_g4_c1": (4, 1),
    "R05_flat_g6_c1": (6, 1),
    "R06_3d_g6_c1": (6, 1),
    "R07_flat_g8_c2": (8, 2),
    "R08_3d_g8_c2": (8, 2),
    "R09_flat_g10_c3": (10, 3),
    "R10_3d_g10_c3": (10, 3),
}
ALLOWED_HISTORICAL_FAILURE_REASONS = {"exact_connector_cycle_rank"}


def recipe_candidate_audit(candidate: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    stratum_id = str(candidate.get("stratum_id"))
    recipe_id = RECIPE_BY_STRATUM.get(stratum_id)
    if recipe_id is None:
        reasons.append("unknown_recipe_stratum")
    requested_grown = int(candidate.get("requested_grown_tunnels", -1))
    requested_connector = int(candidate.get("requested_connector_tunnels", -1))
    if recipe_id is not None and (requested_grown, requested_connector) != RECIPE_PARAMETERS[recipe_id]:
        reasons.append("recipe_requested_tunnel_count_mismatch")
    operations = list(candidate.get("operations", []))
    if len(operations) != requested_grown + requested_connector:
        reasons.append("requested_operation_count_mismatch")
    if any(operation.get("success") is not True for operation in operations):
        reasons.append("requested_operation_failed")
    if any(
        int(operation.get("parameter_draw_count", 0)) < 1
        or int(operation.get("parameter_draw_count", 0)) > 20
        for operation in operations
    ):
        reasons.append("parameter_draw_count_out_of_contract")
    if candidate.get("generation_succeeded") is not True:
        reasons.append("generation_not_successful")
    if candidate.get("replay_identical") is not True:
        reasons.append("replay_not_identical")

    metrics = candidate.get("metrics") or {}
    checks = metrics.get("checks") or {}
    for key, passed in checks.items():
        if key == "exact_connector_cycle_rank":
            continue
        if passed is not True:
            reasons.append(f"failed_source_check:{key}")
    actual_cycle_rank = int(metrics.get("cycle_rank", -1))
    if actual_cycle_rank < requested_connector:
        reasons.append("cycle_rank_below_requested_connector_count")
    historical_reasons = set(candidate.get("failure_reasons", []))
    unsupported = sorted(historical_reasons - ALLOWED_HISTORICAL_FAILURE_REASONS)
    reasons.extend(f"unsupported_historical_failure:{reason}" for reason in unsupported)
    for required in ("canonical_parent_identity", "coordinate_free_wl_hash"):
        if not metrics.get(required):
            reasons.append(f"missing_metric:{required}")

    return {
        "candidate_id": candidate.get("candidate_id"),
        "source_stratum_id": stratum_id,
        "recipe_stratum_id": recipe_id,
        "recipe_valid": not reasons,
        "failure_reasons": reasons,
        "requested_grown_tunnels": requested_grown,
        "requested_connector_tunnels": requested_connector,
        "actual_cycle_rank": actual_cycle_rank,
        "actual_terminal_count": metrics.get("terminal_count"),
        "actual_degree3_or_more_count": metrics.get("degree3_or_more_count"),
        "vertical_span_m": metrics.get("vertical_span_m"),
        "total_centerline_length_m": metrics.get("total_centerline_length_m"),
        "canonical_parent_identity": metrics.get("canonical_parent_identity"),
        "coordinate_free_wl_hash": metrics.get("coordinate_free_wl_hash"),
        "topology_seed": candidate.get("topology_seed"),
        "reserved_geometry_seed": candidate.get("reserved_geometry_seed"),
    }


def select_fixed_recipe_parents(
    audits: Sequence[Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for audit in audits:
        recipe_id = audit.get("recipe_stratum_id")
        if recipe_id is not None:
            grouped[str(recipe_id)].append(audit)
    if set(grouped) != set(RECIPE_BY_STRATUM.values()):
        raise ValueError("recipe strata are incomplete")

    selected: list[dict[str, Any]] = []
    for recipe_id in sorted(grouped):
        ordered = sorted(grouped[recipe_id], key=lambda item: str(item["candidate_id"]))
        if len(ordered) != 12 or any(item.get("recipe_valid") is not True for item in ordered):
            raise ValueError(f"{recipe_id}: expected twelve recipe-valid candidates")
        for rank, audit in enumerate(ordered[:10], start=1):
            candidate_id = str(audit["candidate_id"])
            source = candidates[candidate_id]
            split = "train" if rank <= 8 else "validation" if rank == 9 else "development_test"
            selected.append(
                {
                    "parent_id": candidate_id,
                    "candidate_id": candidate_id,
                    "recipe_stratum_id": recipe_id,
                    "source_stratum_id": audit["source_stratum_id"],
                    "accepted_rank_in_recipe": rank,
                    "split": split,
                    "topology_seed": source["topology_seed"],
                    "reserved_geometry_seed": source["reserved_geometry_seed"],
                    "requested_grown_tunnels": audit["requested_grown_tunnels"],
                    "requested_connector_tunnels": audit["requested_connector_tunnels"],
                    "actual_cycle_rank": audit["actual_cycle_rank"],
                    "actual_terminal_count": audit["actual_terminal_count"],
                    "actual_degree3_or_more_count": audit["actual_degree3_or_more_count"],
                    "vertical_span_m": audit["vertical_span_m"],
                    "total_centerline_length_m": audit["total_centerline_length_m"],
                    "canonical_parent_identity": audit["canonical_parent_identity"],
                    "coordinate_free_wl_hash": audit["coordinate_free_wl_hash"],
                }
            )
    return selected


def recipe_batch_audit(
    audits: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    identities = [str(item["canonical_parent_identity"]) for item in selected]
    wl_hashes = [str(item["coordinate_free_wl_hash"]) for item in selected]
    split_counts = Counter(str(item["split"]) for item in selected)
    cycle_by_recipe = {
        recipe_id: dict(
            sorted(
                Counter(
                    int(item["actual_cycle_rank"])
                    for item in selected
                    if item["recipe_stratum_id"] == recipe_id
                ).items()
            )
        )
        for recipe_id in sorted(RECIPE_BY_STRATUM.values())
    }
    checks = {
        "all_120_recipe_valid": len(audits) == 120 and all(item["recipe_valid"] for item in audits),
        "exactly_100_selected": len(selected) == 100,
        "split_80_10_10": split_counts == Counter(
            {"train": 80, "validation": 10, "development_test": 10}
        ),
        "all_selected_canonical_unique": len(set(identities)) == 100,
        "all_selected_wl_unique": len(set(wl_hashes)) == 100,
        "all_selected_have_structure_events": all(
            int(item["actual_degree3_or_more_count"]) >= 1 for item in selected
        ),
        "exactly_50_selected_3d": sum("_3d_" in str(item["recipe_stratum_id"]) for item in selected) == 50,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "recipe_valid_candidate_count": sum(bool(item["recipe_valid"]) for item in audits),
        "selected_parent_count": len(selected),
        "split_counts": dict(split_counts),
        "unique_canonical_identity_count": len(set(identities)),
        "unique_coordinate_free_wl_hash_count": len(set(wl_hashes)),
        "cycle_rank_distribution_by_recipe": cycle_by_recipe,
    }


__all__ = [
    "ALLOWED_HISTORICAL_FAILURE_REASONS",
    "RECIPE_BY_STRATUM",
    "RECIPE_PARAMETERS",
    "recipe_batch_audit",
    "recipe_candidate_audit",
    "select_fixed_recipe_parents",
]
