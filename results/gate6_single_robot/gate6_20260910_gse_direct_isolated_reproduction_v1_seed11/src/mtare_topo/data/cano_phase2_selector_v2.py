"""Spatially balanced, zero-ray selector for the Phase-2 Cano dataset."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.data.cano_phase2_dataset import ROLE_ORDER


MAXIMUM_COVERAGE_RADIUS_M = 10.0


def length_proportional_tunnel_quotas(
    candidates: Sequence[Mapping[str, Any]], selected_count: int
) -> dict[int, int]:
    """Allocate exact selected counts by candidate-lattice arc length."""

    capacities = Counter(int(item["tunnel_id"]) for item in candidates)
    if not capacities or selected_count < len(capacities) or selected_count > len(candidates):
        raise ValueError("selected_count cannot provide a valid per-tunnel allocation")
    remaining = selected_count - len(capacities)
    residual_capacities = {key: value - 1 for key, value in capacities.items()}
    total_residual = sum(residual_capacities.values())
    raw = {
        key: (remaining * residual_capacities[key] / total_residual if total_residual else 0.0)
        for key in sorted(capacities)
    }
    quota = {key: 1 + int(math.floor(raw[key])) for key in sorted(capacities)}
    remainder = selected_count - sum(quota.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - math.floor(raw[key])), key))
    for key in order[:remainder]:
        quota[key] += 1
    if sum(quota.values()) != selected_count or any(quota[key] > capacities[key] for key in quota):
        raise ValueError("length-proportional quota construction drifted")
    return quota


def _event_constraints(candidates: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, list[int]]]:
    result: list[tuple[str, str, list[int]]] = []
    for role, key in (("junction", "junction_event_ids"), ("terminal", "terminal_event_ids")):
        event_indices: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(candidates):
            if item["primary_role"] != role:
                continue
            for event_id in item.get(key, []):
                event_indices[str(event_id)].append(index)
        for event_id in sorted(event_indices):
            result.append((role, event_id, event_indices[event_id]))
    return result


def select_spatially_balanced_clusters(
    candidates: Sequence[Mapping[str, Any]],
    role_quota: Mapping[str, int],
    maximum_coverage_radius_m: float = MAXIMUM_COVERAGE_RADIUS_M,
    *,
    tunnel_quota: Mapping[int, int] | None = None,
    constraint_reference: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve exact quotas and reference-lattice coverage over eligible candidates.

    ``candidates`` are selectable records. ``constraint_reference`` may also
    contain ineligible records, freezing the original event/spatial lattice.
    """

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    records = [dict(item) for item in candidates]
    reference = [dict(item) for item in (constraint_reference or candidates)]
    if not records or not reference:
        raise ValueError("selector requires non-empty candidate and reference sets")
    selected_count = sum(int(role_quota[role]) for role in ROLE_ORDER)
    resolved_tunnel_quota = (
        {int(key): int(value) for key, value in tunnel_quota.items()}
        if tunnel_quota is not None
        else length_proportional_tunnel_quotas(reference, selected_count)
    )
    tunnel_indices: dict[int, list[int]] = defaultdict(list)
    role_indices: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        tunnel_indices[int(item["tunnel_id"])].append(index)
        role_indices[str(item["primary_role"])].append(index)

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    def add(indices: Sequence[int], low: float, high: float) -> None:
        nonlocal row
        row_indices.extend([row] * len(indices)); column_indices.extend(indices); values.extend([1.0] * len(indices))
        lower.append(low); upper.append(high); row += 1

    for role in ROLE_ORDER:
        add(role_indices[role], float(role_quota[role]), float(role_quota[role]))
    for tunnel_id in sorted(resolved_tunnel_quota):
        indices = tunnel_indices[tunnel_id]
        if len(indices) < resolved_tunnel_quota[tunnel_id]:
            raise RuntimeError(f"tunnel {tunnel_id} has {len(indices)} eligible candidates for quota {resolved_tunnel_quota[tunnel_id]}")
        add(indices, float(resolved_tunnel_quota[tunnel_id]), float(resolved_tunnel_quota[tunnel_id]))
    selectable_id_to_index = {str(item["cluster_id"]): index for index, item in enumerate(records)}
    events = _event_constraints(reference)
    for role, event_id, reference_indices in events:
        indices = [selectable_id_to_index[str(reference[index]["cluster_id"])] for index in reference_indices if str(reference[index]["cluster_id"]) in selectable_id_to_index]
        if not indices:
            raise RuntimeError(f"event {role}:{event_id} has no eligible candidate")
        add(indices, 1.0, float("inf"))
    reference_by_tunnel: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in reference:
        reference_by_tunnel[int(item["tunnel_id"])].append(item)
    for tunnel_id in sorted(reference_by_tunnel):
        indices = tunnel_indices[tunnel_id]
        arcs = np.asarray([float(records[index]["center_arc_m"]) for index in indices])
        for reference_item in reference_by_tunnel[tunnel_id]:
            arc = float(reference_item["center_arc_m"])
            covering = [indices[offset] for offset in np.flatnonzero(np.abs(arcs - arc) <= maximum_coverage_radius_m + 1e-9)]
            if not covering:
                raise RuntimeError(f"tunnel {tunnel_id} arc {arc:.6f} has no eligible candidate within {maximum_coverage_radius_m}m")
            add(covering, 1.0, float("inf"))

    matrix = coo_matrix((values, (row_indices, column_indices)), shape=(row, len(records))).tocsr()
    objective = np.zeros(len(records), dtype=np.float64)
    for tunnel_id in sorted(tunnel_indices):
        indices = tunnel_indices[tunnel_id]; quota = resolved_tunnel_quota[tunnel_id]
        arcs = np.asarray([float(records[index]["center_arc_m"]) for index in indices])
        low, high = float(np.min(arcs)), float(np.max(arcs))
        ideals = np.linspace(low, high, quota) if quota > 1 else np.asarray([(low + high) / 2.0])
        scale = max(high - low, 1.0)
        objective[indices] = np.min(np.abs(arcs[:, None] - ideals[None, :]), axis=1) / scale + np.arange(len(indices)) * 1e-10
    result = milp(
        objective,
        integrality=np.ones(len(records), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"presolve": True, "time_limit": 30.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"selector V2 MILP infeasible: status={result.status}, message={result.message}")
    selected_indices = np.flatnonzero(np.asarray(result.x) > 0.5).tolist()
    selected = [records[index] for index in selected_indices]
    selected.sort(key=lambda item: str(item["cluster_id"]))
    audit = selector_v2_audit(reference, selected, role_quota, resolved_tunnel_quota, maximum_coverage_radius_m)
    audit["eligible_candidate_clusters"] = len(records)
    if not audit["passed"]:
        raise RuntimeError(f"selector V2 post-solve audit failed: {audit}")
    return selected, audit


def selector_v2_audit(
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    role_quota: Mapping[str, int],
    tunnel_quota: Mapping[int, int],
    maximum_coverage_radius_m: float = MAXIMUM_COVERAGE_RADIUS_M,
) -> dict[str, Any]:
    selected_ids = [str(item["cluster_id"]) for item in selected]
    selected_by_tunnel: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in selected:
        selected_by_tunnel[int(item["tunnel_id"])].append(item)
    coverage: dict[str, float] = {}
    for item in candidates:
        tunnel_id = int(item["tunnel_id"])
        nearest = min(
            (abs(float(item["center_arc_m"]) - float(other["center_arc_m"])) for other in selected_by_tunnel[tunnel_id]),
            default=float("inf"),
        )
        key = str(tunnel_id); coverage[key] = max(coverage.get(key, 0.0), nearest)
    events = _event_constraints(candidates)
    event_failures = []
    selected_set = set(selected_ids)
    for role, event_id, indices in events:
        if not any(str(candidates[index]["cluster_id"]) in selected_set for index in indices):
            event_failures.append({"role": role, "event_id": event_id})
    observed_roles = Counter(str(item["primary_role"]) for item in selected)
    observed_tunnels = Counter(int(item["tunnel_id"]) for item in selected)
    maximum = max(coverage.values(), default=float("inf"))
    passed = bool(
        len(selected_ids) == len(set(selected_ids)) == sum(int(role_quota[role]) for role in ROLE_ORDER)
        and all(observed_roles[role] == int(role_quota[role]) for role in ROLE_ORDER)
        and dict(sorted(observed_tunnels.items())) == dict(sorted(tunnel_quota.items()))
        and not event_failures
        and all(value > 0 for value in observed_tunnels.values())
        and maximum <= maximum_coverage_radius_m + 1e-9
    )
    return {
        "passed": passed,
        "candidate_clusters": len(candidates),
        "selected_clusters": len(selected),
        "selected_ids_unique": len(selected_ids) == len(set(selected_ids)),
        "role_quota": {role:int(role_quota[role]) for role in ROLE_ORDER},
        "observed_role_counts": dict(observed_roles),
        "tunnel_quota": {str(key):value for key,value in sorted(tunnel_quota.items())},
        "observed_tunnel_counts": {str(key):value for key,value in sorted(observed_tunnels.items())},
        "all_tunnels_represented": set(observed_tunnels) == {int(item["tunnel_id"]) for item in candidates},
        "event_constraint_count": len(events),
        "event_failures": event_failures,
        "coverage_radius_m_per_tunnel": coverage,
        "maximum_candidate_to_selected_same_tunnel_arc_distance_m": maximum,
        "maximum_allowed_coverage_radius_m": maximum_coverage_radius_m,
    }
