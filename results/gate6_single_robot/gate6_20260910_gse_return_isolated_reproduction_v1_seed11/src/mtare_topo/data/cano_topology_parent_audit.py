"""Pure audit helpers for the Cano 100-topology parent candidate batch."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_arithmetic_candidate_registry(
    strata: Sequence[Mapping[str, Any]],
    *,
    first_candidate_index: int,
    candidates_per_stratum: int,
    topology_seed_base: int,
    geometry_seed_base: int,
) -> list[dict[str, Any]]:
    """Expand one fully bounded, collision-free arithmetic candidate registry."""

    if not strata or first_candidate_index < 1 or candidates_per_stratum < 1:
        raise ValueError("candidate registry bounds must be positive")
    registry = []
    for stratum_index, stratum in enumerate(strata, start=1):
        stratum_id = str(stratum["stratum_id"])
        for candidate_index in range(
            first_candidate_index, first_candidate_index + candidates_per_stratum
        ):
            registry.append(
                {
                    "candidate_id": f"{stratum_id}_C{candidate_index:02d}",
                    "stratum_id": stratum_id,
                    "candidate_index": candidate_index,
                    "topology_seed": int(topology_seed_base) + 100 * stratum_index + candidate_index,
                    "reserved_geometry_seed": int(geometry_seed_base) + 100 * stratum_index + candidate_index,
                }
            )
    for field in ("candidate_id", "topology_seed", "reserved_geometry_seed"):
        values = [item[field] for item in registry]
        if len(values) != len(set(values)):
            raise ValueError(f"candidate registry contains duplicate {field}")
    return registry


def canonical_parent_identity(graph: Mapping[str, Any], splines: Mapping[str, Any]) -> str:
    """Identity of one full coordinate-bearing graph/spline parent."""

    return canonical_json_hash({"graph": graph, "splines": splines})


def graph_wl_hash(graph: Mapping[str, Any], rounds: int = 8) -> str:
    """Coordinate-free Weisfeiler--Lehman hash using node degree/type labels."""

    nodes = graph.get("nodes", [])
    node_ids = [str(item["id"]) for item in nodes]
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in graph.get("edges", []):
        first, second = (str(value) for value in edge["node_ids"])
        adjacency[first].add(second)
        adjacency[second].add(first)
    labels = {
        str(item["id"]): f"d{int(item['degree'])}|{item.get('node_type', 'unknown')}"
        for item in nodes
    }
    for _ in range(rounds):
        updated = {}
        for node_id in sorted(node_ids):
            neighborhood = ",".join(sorted(labels[neighbor] for neighbor in adjacency[node_id]))
            updated[node_id] = hashlib.sha256(
                f"{labels[node_id]}|{neighborhood}".encode("utf-8")
            ).hexdigest()
        labels = updated
    edge_count = len(graph.get("edges", []))
    payload = {
        "node_count": len(node_ids),
        "edge_count": edge_count,
        "labels": sorted(labels.values()),
    }
    return canonical_json_hash(payload)


def parent_metrics(
    graph: Mapping[str, Any],
    splines: Mapping[str, Any],
    *,
    dimensionality: str,
    requested_grown: int,
    requested_connector: int,
) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    coordinates = np.asarray([node["xyz"] for node in nodes], dtype=np.float64)
    finite_graph = bool(coordinates.shape[1:] == (3,) and np.all(np.isfinite(coordinates)))
    vertical_span = float(np.ptp(coordinates[:, 2])) if len(coordinates) else 0.0
    degree3_count = sum(int(node["degree"]) >= 3 for node in nodes)
    terminal_count = sum(int(node["degree"]) == 1 for node in nodes)
    tunnel_types = Counter(str(item["type"]) for item in graph.get("tunnels", []))
    lengths = []
    finite_splines = True
    for tunnel in splines.get("tunnels", []):
        points = np.asarray(tunnel.get("points", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
            finite_splines = False
            continue
        distances = np.asarray(tunnel.get("distances", []), dtype=np.float64).reshape(-1)
        if len(distances) and np.all(np.isfinite(distances)):
            lengths.append(float(distances[-1] - distances[0]))
        elif len(points) > 1:
            lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
        else:
            finite_splines = False
    statistics = graph.get("statistics", {})
    connected_components = int(statistics.get("connected_components", -1))
    cycle_rank = int(statistics.get("cycle_rank", -1))
    total_centerline_length = float(sum(lengths))
    flat_pass = dimensionality != "flat" or vertical_span <= 1e-6
    three_d_pass = dimensionality != "3d" or vertical_span >= 5.0
    checks = {
        "requested_tunnel_counts_match": (
            tunnel_types.get("grown", 0) == requested_grown
            and tunnel_types.get("connector", 0) == requested_connector
        ),
        "single_connected_component": connected_components == 1,
        "finite_graph": finite_graph,
        "finite_splines": finite_splines,
        "exact_connector_cycle_rank": cycle_rank == requested_connector,
        "minimum_centerline_length": total_centerline_length >= 300.0,
        "flat_vertical_span": flat_pass,
        "three_d_vertical_span": three_d_pass,
    }
    return {
        "passed_geometry_free_contract": all(checks.values()),
        "checks": checks,
        "node_count": len(nodes),
        "edge_count": len(graph.get("edges", [])),
        "tunnel_count": len(graph.get("tunnels", [])),
        "grown_tunnel_count": tunnel_types.get("grown", 0),
        "connector_tunnel_count": tunnel_types.get("connector", 0),
        "connected_components": connected_components,
        "cycle_rank": cycle_rank,
        "terminal_count": terminal_count,
        "degree3_or_more_count": degree3_count,
        "has_degree3_or_more": degree3_count > 0,
        "vertical_span_m": vertical_span,
        "total_centerline_length_m": total_centerline_length,
        "canonical_parent_identity": canonical_parent_identity(graph, splines),
        "coordinate_free_wl_hash": graph_wl_hash(graph),
    }


def select_accepted_candidates(
    candidate_results: Sequence[Mapping[str, Any]], retained_per_stratum: int = 10
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in candidate_results:
        grouped[str(result["stratum_id"])].append(result)
    accepted: list[dict[str, Any]] = []
    strata_summary = {}
    for stratum_id in sorted(grouped):
        ordered = sorted(grouped[stratum_id], key=lambda item: str(item["candidate_id"]))
        valid = [item for item in ordered if item.get("candidate_valid") is True]
        retained = valid[:retained_per_stratum]
        for rank, item in enumerate(retained, start=1):
            split = "train" if rank <= 8 else "validation" if rank == 9 else "development_test"
            accepted.append(
                {
                    "candidate_id": item["candidate_id"],
                    "parent_id": item["candidate_id"],
                    "stratum_id": stratum_id,
                    "accepted_rank_in_stratum": rank,
                    "split": split,
                    "topology_seed": item["topology_seed"],
                    "reserved_geometry_seed": item["reserved_geometry_seed"],
                    "canonical_parent_identity": item["metrics"]["canonical_parent_identity"],
                    "coordinate_free_wl_hash": item["metrics"]["coordinate_free_wl_hash"],
                    "metrics": item["metrics"],
                }
            )
        strata_summary[stratum_id] = {
            "candidate_count": len(ordered),
            "valid_count": len(valid),
            "invalid_count": len(ordered) - len(valid),
            "retained_count": len(retained),
            "retained_candidate_ids": [item["candidate_id"] for item in retained],
            "passed_minimum_valid_count": len(valid) >= retained_per_stratum,
        }
    return accepted, strata_summary


def select_corrective_train_validation_pairs(
    candidate_results: Sequence[Mapping[str, Any]],
    *,
    first_candidate_index: int = 13,
    candidates_per_stratum: int = 12,
    expected_strata: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the first two valid candidates in each frozen corrective stratum.

    Candidate slots must be exactly C13--C24 (by default), rather than an
    open-ended result-dependent stream.  The first valid candidate is the
    corrective-train world and the second is corrective-validation.  The
    entire selection fails if a stratum has fewer than two valid candidates or
    if any selected coordinate-bearing identity is repeated.
    """

    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    expected_indices = set(range(first_candidate_index, first_candidate_index + candidates_per_stratum))
    for result in candidate_results:
        candidate_id = str(result["candidate_id"])
        match = re.search(r"_C(\d+)$", candidate_id)
        if match is None:
            raise ValueError(f"candidate ID lacks numeric suffix: {candidate_id}")
        grouped[str(result["stratum_id"])].append((int(match.group(1)), result))
    if len(grouped) != expected_strata:
        raise ValueError(f"expected {expected_strata} strata, received {len(grouped)}")

    selected: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for stratum_id in sorted(grouped):
        ordered = sorted(grouped[stratum_id], key=lambda item: item[0])
        observed_indices = {index for index, _ in ordered}
        exact_registry = len(ordered) == candidates_per_stratum and observed_indices == expected_indices
        valid = [(index, item) for index, item in ordered if item.get("candidate_valid") is True]
        retained = valid[:2] if exact_registry else []
        for role, (candidate_index, item) in zip(
            ("corrective_train", "corrective_validation"), retained
        ):
            selected.append(
                {
                    "candidate_id": str(item["candidate_id"]),
                    "parent_id": str(item["candidate_id"]),
                    "candidate_index": candidate_index,
                    "stratum_id": stratum_id,
                    "split": role,
                    "topology_seed": int(item["topology_seed"]),
                    "reserved_geometry_seed": int(item["reserved_geometry_seed"]),
                    "canonical_parent_identity": str(item["metrics"]["canonical_parent_identity"]),
                    "coordinate_free_wl_hash": str(item["metrics"]["coordinate_free_wl_hash"]),
                }
            )
        summaries[stratum_id] = {
            "candidate_count": len(ordered),
            "valid_count": len(valid),
            "exact_frozen_registry": exact_registry,
            "selected_candidate_ids": [str(item["candidate_id"]) for _, item in retained],
            "passed": exact_registry and len(retained) == 2,
        }

    identities = [item["canonical_parent_identity"] for item in selected]
    parent_ids = [item["parent_id"] for item in selected]
    split_counts = Counter(item["split"] for item in selected)
    audit = {
        "passed": (
            all(item["passed"] for item in summaries.values())
            and len(selected) == 2 * expected_strata
            and len(set(identities)) == len(identities)
            and len(set(parent_ids)) == len(parent_ids)
            and split_counts == Counter(
                {"corrective_train": expected_strata, "corrective_validation": expected_strata}
            )
        ),
        "strata": summaries,
        "selected_count": len(selected),
        "split_counts": dict(split_counts),
        "unique_parent_count": len(set(parent_ids)),
        "unique_identity_count": len(set(identities)),
    }
    return selected, audit


def split_audit(accepted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_sets = {
        split: {str(item["parent_id"]) for item in accepted if item["split"] == split}
        for split in ("train", "validation", "development_test")
    }
    identity_sets = {
        split: {
            str(item["canonical_parent_identity"])
            for item in accepted
            if item["split"] == split
        }
        for split in split_sets
    }
    pairwise_disjoint = all(
        not (split_sets[first] & split_sets[second])
        and not (identity_sets[first] & identity_sets[second])
        for index, first in enumerate(split_sets)
        for second in list(split_sets)[index + 1 :]
    )
    return {
        "counts": {key: len(value) for key, value in split_sets.items()},
        "parent_and_identity_sets_pairwise_disjoint": pairwise_disjoint,
        "parent_id_count": len(set().union(*split_sets.values())),
        "canonical_identity_count": len(set().union(*identity_sets.values())),
    }


def batch_diversity_audit(
    accepted: Sequence[Mapping[str, Any]], strata_summary: Mapping[str, Any]
) -> dict[str, Any]:
    identities = [str(item["canonical_parent_identity"]) for item in accepted]
    wl_hashes = [str(item["coordinate_free_wl_hash"]) for item in accepted]
    wl_per_stratum = {
        stratum_id: len(
            {
                str(item["coordinate_free_wl_hash"])
                for item in accepted
                if item["stratum_id"] == stratum_id
            }
        )
        for stratum_id in strata_summary
    }
    structural_event_parents = sum(
        bool(item["metrics"]["has_degree3_or_more"]) for item in accepted
    )
    retained_3d = [item for item in accepted if "_3d_" in str(item["stratum_id"])]
    checks = {
        "all_strata_have_ten": all(
            item["passed_minimum_valid_count"] and item["retained_count"] == 10
            for item in strata_summary.values()
        ),
        "exactly_100_accepted": len(accepted) == 100,
        "all_canonical_identities_unique": len(set(identities)) == 100,
        "at_least_40_wl_hashes": len(set(wl_hashes)) >= 40,
        "at_least_3_wl_hashes_per_stratum": all(value >= 3 for value in wl_per_stratum.values()),
        "at_least_70_structural_event_parents": structural_event_parents >= 70,
        "all_50_3d_parents_have_vertical_span": (
            len(retained_3d) == 50
            and all(item["metrics"]["vertical_span_m"] >= 5.0 for item in retained_3d)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "accepted_parent_count": len(accepted),
        "unique_canonical_identity_count": len(set(identities)),
        "unique_coordinate_free_wl_hash_count": len(set(wl_hashes)),
        "unique_wl_hash_count_per_stratum": wl_per_stratum,
        "structural_event_parent_count": structural_event_parents,
        "retained_3d_parent_count": len(retained_3d),
    }


__all__ = [
    "batch_diversity_audit",
    "build_arithmetic_candidate_registry",
    "canonical_json_hash",
    "canonical_parent_identity",
    "graph_wl_hash",
    "parent_metrics",
    "select_accepted_candidates",
    "select_corrective_train_validation_pairs",
    "split_audit",
]
