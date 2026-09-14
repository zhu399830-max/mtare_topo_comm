"""Pure deterministic contracts for the five-topology Cano sensor pilot."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.data.cano_sensor_smoke import (
    LIDAR_HEIGHT_ABOVE_FLOOR_M,
    interpolate_polyline,
    structural_label,
)


PARENT_REGISTRY = (
    {"parent_id": "P01_straight_turn", "topology_seed": 101, "geometry_seed": 10101},
    {"parent_id": "P02_branch_deadend", "topology_seed": 211, "geometry_seed": 21101},
    {"parent_id": "P03_loop_bottleneck", "topology_seed": 307, "geometry_seed": 30701},
    {"parent_id": "P04_chamber_multiexit", "topology_seed": 401, "geometry_seed": 40101},
    {"parent_id": "P05_slope_multiheight", "topology_seed": 503, "geometry_seed": 50301},
)
ANCHORS_PER_PARENT = 50
VIEW_OFFSETS_DEG = (-30.0, 0.0, 30.0)
MINIMUM_ANCHOR_SPACING_M = 5.0
EVENT_OFFSET_M = 6.0
ANCHOR_CANDIDATE_STEP_M = 0.5
MAXIMUM_SAME_TUNNEL_COVERAGE_RADIUS_M = 7.5


def canonical_document_hash(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def spline_arrays(document: Mapping[str, Any]) -> dict[int, np.ndarray]:
    return {
        int(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in document["tunnels"]
    }


def _polyline_distances(points: np.ndarray) -> np.ndarray:
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )


def _nearest_spline_location(
    query: np.ndarray, splines: Mapping[int, np.ndarray], allowed: Sequence[int]
) -> tuple[int, float, np.ndarray, np.ndarray]:
    best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
    for tunnel_id in sorted(int(value) for value in allowed):
        points = splines[tunnel_id]
        cumulative = _polyline_distances(points)
        segments = points[1:] - points[:-1]
        lengths_sq = np.sum(segments * segments, axis=1)
        ratios = np.sum((query - points[:-1]) * segments, axis=1) / np.where(
            lengths_sq > 1e-15, lengths_sq, 1.0
        )
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = points[:-1] + ratios[:, None] * segments
        errors = np.linalg.norm(projections - query, axis=1)
        index = int(np.argmin(errors))
        distance = float(
            cumulative[index] + ratios[index] * math.sqrt(float(lengths_sq[index]))
        )
        tangent = segments[index] / math.sqrt(float(lengths_sq[index]))
        candidate = (
            float(errors[index]),
            tunnel_id,
            distance,
            projections[index],
            tangent,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        raise ValueError("no spline location is available")
    return best[1], best[2], best[3], best[4]


def _candidate_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    point = record["axis_xyz_m"]
    return (
        int(record["priority"]),
        int(record["source_tunnel_id"]),
        float(record["arc_distance_m"]),
        tuple(round(float(value), 8) for value in point),
    )


def _append_unique(
    records: list[dict[str, Any]], candidate: dict[str, Any], minimum_spacing_m: float
) -> bool:
    point = np.asarray(candidate["axis_xyz_m"], dtype=np.float64)
    if any(
        np.linalg.norm(point - np.asarray(old["axis_xyz_m"], dtype=np.float64))
        < minimum_spacing_m - 1e-8
        for old in records
    ):
        return False
    records.append(candidate)
    return True


def _structural_event_candidates(
    graph: Mapping[str, Any], splines: Mapping[int, np.ndarray]
) -> list[dict[str, Any]]:
    """Return deterministic terminal, junction-offset and strongest-turn events."""

    events: list[dict[str, Any]] = []
    node_records = sorted(
        (node for node in graph["nodes"] if int(node["degree"]) != 2),
        key=lambda item: item["id"],
    )
    for node in node_records:
        center = np.asarray(node["xyz"], dtype=np.float64)
        tunnel_id, distance, _, _ = _nearest_spline_location(
            center, splines, node["incident_tunnel_ids"]
        )
        total = float(_polyline_distances(splines[tunnel_id])[-1])
        direction = 1.0 if distance <= total / 2.0 else -1.0
        for offset_index, offset in enumerate((0.0, EVENT_OFFSET_M)):
            target = float(np.clip(distance + direction * offset, 0.0, total))
            selected_point, selected_tangent = interpolate_polyline(
                splines[tunnel_id], target
            )
            selected_tangent = selected_tangent * direction
            events.append(
                {
                    "priority": 0,
                    "source": (
                        "structural_event_center"
                        if offset_index == 0
                        else "structural_event_offset"
                    ),
                    "source_node_id": node["id"],
                    "source_tunnel_id": tunnel_id,
                    "arc_distance_m": target,
                    "axis_xyz_m": selected_point.tolist(),
                    "tangent_xyz": selected_tangent.tolist(),
                }
            )

    # A degree-2 bend is a geometric event even when it is not a graph node.
    for tunnel_id, points in sorted(splines.items()):
        segments = np.diff(points, axis=0)
        horizontal = np.linalg.norm(segments[:, :2], axis=1)
        valid = horizontal > 1e-9
        if np.count_nonzero(valid) < 2:
            continue
        headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
        changes = np.abs(np.diff(headings))
        index = int(np.argmax(changes))
        if float(changes[index]) < math.radians(3.0):
            continue
        point_index = index + 1
        tangent = segments[min(point_index, len(segments) - 1)]
        tangent /= np.linalg.norm(tangent)
        cumulative = _polyline_distances(points)
        events.append(
            {
                "priority": 0,
                "source": "maximum_horizontal_turn_event",
                "source_node_id": None,
                "source_tunnel_id": tunnel_id,
                "arc_distance_m": float(cumulative[point_index]),
                "axis_xyz_m": points[point_index].astype(float).tolist(),
                "tangent_xyz": tangent.astype(float).tolist(),
            }
        )
    return events


def _balanced_arc_candidates(
    splines: Mapping[int, np.ndarray], candidate_step_m: float
) -> dict[int, list[dict[str, Any]]]:
    """Interpolate a knot-density-independent candidate lattice per tunnel."""

    if candidate_step_m <= 0.0:
        raise ValueError("candidate_step_m must be positive")
    result: dict[int, list[dict[str, Any]]] = {}
    for tunnel_id, points in sorted(splines.items()):
        total = float(_polyline_distances(points)[-1])
        distances = np.arange(0.0, total + 1e-9, candidate_step_m)
        if len(distances) == 0 or not np.isclose(distances[-1], total):
            distances = np.append(distances, total)
        records = []
        for distance in distances:
            point, tangent = interpolate_polyline(points, float(distance))
            records.append(
                {
                    "priority": 1,
                    "source": "balanced_arc_fill",
                    "source_node_id": None,
                    "source_tunnel_id": tunnel_id,
                    "arc_distance_m": float(distance),
                    "axis_xyz_m": point.tolist(),
                    "tangent_xyz": tangent.tolist(),
                }
            )
        result[tunnel_id] = records
    return result


def _length_proportional_fill_quotas(
    splines: Mapping[int, np.ndarray], fill_count: int
) -> dict[int, int]:
    """Allocate non-event anchors by arc length using largest remainders."""

    if fill_count < 0:
        raise ValueError("fill_count must be non-negative")
    lengths = {
        tunnel_id: float(_polyline_distances(points)[-1])
        for tunnel_id, points in sorted(splines.items())
    }
    total_length = sum(lengths.values())
    if total_length <= 0.0:
        raise ValueError("total spline length must be positive")
    raw = {
        tunnel_id: fill_count * length / total_length
        for tunnel_id, length in lengths.items()
    }
    quotas = {tunnel_id: int(math.floor(value)) for tunnel_id, value in raw.items()}
    remainder = fill_count - sum(quotas.values())
    order = sorted(
        raw,
        key=lambda tunnel_id: (-(raw[tunnel_id] - quotas[tunnel_id]), tunnel_id),
    )
    for tunnel_id in order[:remainder]:
        quotas[tunnel_id] += 1
    return quotas


def _coverage_constrained_arc_fill(
    splines: Mapping[int, np.ndarray],
    mandatory: Sequence[Mapping[str, Any]],
    fill_count: int,
    minimum_spacing_m: float,
    maximum_coverage_radius_m: float,
    candidate_step_m: float,
) -> list[dict[str, Any]]:
    """Solve exact arc-length quotas, global spacing and full-spline coverage."""

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    quotas = _length_proportional_fill_quotas(splines, fill_count)
    mandatory_points = np.asarray(
        [item["axis_xyz_m"] for item in mandatory], dtype=np.float64
    )
    candidates: list[dict[str, Any]] = []
    candidates_by_tunnel = _balanced_arc_candidates(splines, candidate_step_m)
    for tunnel_id in sorted(candidates_by_tunnel):
        for source in candidates_by_tunnel[tunnel_id]:
            point = np.asarray(source["axis_xyz_m"], dtype=np.float64)
            if len(mandatory_points) and float(
                np.min(np.linalg.norm(mandatory_points - point, axis=1))
            ) < minimum_spacing_m - 1e-8:
                continue
            candidates.append({**source, "source": "arc_length_coverage_fill"})
    if not candidates and fill_count:
        raise ValueError("no coverage-fill candidates remain after event spacing")

    points = np.asarray(
        [item["axis_xyz_m"] for item in candidates], dtype=np.float64
    )
    tunnel_ids = np.asarray(
        [int(item["source_tunnel_id"]) for item in candidates], dtype=np.int64
    )
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    # Exact fill quotas make sampling density proportional to tunnel arc length.
    for tunnel_id in sorted(splines):
        indices = np.flatnonzero(tunnel_ids == tunnel_id)
        row_indices.extend([row] * len(indices))
        column_indices.extend(indices.tolist())
        values.extend([1.0] * len(indices))
        lower.append(float(quotas[tunnel_id]))
        upper.append(float(quotas[tunnel_id]))
        row += 1

    # No two selected candidates may violate global Euclidean spacing.
    for first in range(len(candidates)):
        conflicts = np.flatnonzero(
            np.linalg.norm(points[first + 1 :] - points[first], axis=1)
            < minimum_spacing_m - 1e-8
        ) + first + 1
        for second in conflicts:
            row_indices.extend((row, row))
            column_indices.extend((first, int(second)))
            values.extend((1.0, 1.0))
            lower.append(float("-inf"))
            upper.append(1.0)
            row += 1

    # Every 0.5 m arc sample must be covered by an event or same-tunnel fill.
    for tunnel_id, records in sorted(candidates_by_tunnel.items()):
        tunnel_candidate_indices = np.flatnonzero(tunnel_ids == tunnel_id)
        for record in records:
            sample = np.asarray(record["axis_xyz_m"], dtype=np.float64)
            if len(mandatory_points) and float(
                np.min(np.linalg.norm(mandatory_points - sample, axis=1))
            ) <= maximum_coverage_radius_m:
                continue
            covering = tunnel_candidate_indices[
                np.linalg.norm(points[tunnel_candidate_indices] - sample, axis=1)
                <= maximum_coverage_radius_m
            ]
            if not len(covering):
                raise ValueError(
                    f"tunnel {tunnel_id} has an arc sample without a feasible "
                    f"{maximum_coverage_radius_m} m coverage candidate"
                )
            row_indices.extend([row] * len(covering))
            column_indices.extend(covering.tolist())
            values.extend([1.0] * len(covering))
            lower.append(1.0)
            upper.append(float("inf"))
            row += 1

    constraint = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(row, len(candidates)),
    ).tocsr()
    objective = np.zeros(len(candidates), dtype=np.float64)
    for tunnel_id, spline in sorted(splines.items()):
        indices = np.flatnonzero(tunnel_ids == tunnel_id)
        quota = quotas[tunnel_id]
        if quota == 0:
            continue
        length = float(_polyline_distances(spline)[-1])
        ideals = (np.arange(quota, dtype=np.float64) + 0.5) * length / quota
        arc = np.asarray(
            [float(candidates[index]["arc_distance_m"]) for index in indices]
        )
        objective[indices] = (
            np.min(np.abs(arc[:, None] - ideals[None, :]), axis=1)
            / max(length, 1.0)
            + np.arange(len(indices), dtype=np.float64) * 1e-9
        )
    result = milp(
        objective,
        integrality=np.ones(len(candidates), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            constraint,
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        ),
        options={"presolve": True, "time_limit": 30.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise ValueError(
            "coverage-constrained anchor allocation is infeasible: "
            f"status={result.status}, message={result.message}, quotas={quotas}"
        )
    selected = [
        candidates[index]
        for index in np.flatnonzero(np.asarray(result.x) > 0.5)
    ]
    observed_quotas = {
        tunnel_id: sum(
            int(item["source_tunnel_id"]) == tunnel_id for item in selected
        )
        for tunnel_id in sorted(splines)
    }
    if len(selected) != fill_count or observed_quotas != quotas:
        raise ValueError(
            f"coverage solver count drift: {len(selected)}, {observed_quotas}, {quotas}"
        )
    return selected


def select_canonical_anchors(
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    fta_distance_m: float | None,
    count: int = ANCHORS_PER_PARENT,
    minimum_spacing_m: float = MINIMUM_ANCHOR_SPACING_M,
    candidate_step_m: float = ANCHOR_CANDIDATE_STEP_M,
) -> list[dict[str, Any]]:
    """Select events plus exact arc-length-density, full-coverage samples."""

    splines = spline_arrays(spline_document)
    mandatory: list[dict[str, Any]] = []
    event_candidates = _structural_event_candidates(graph, splines)
    for event in event_candidates:
        _append_unique(
            mandatory,
            event,
            minimum_spacing_m,
        )

    fill_count = count - len(mandatory)
    selected = list(mandatory) + _coverage_constrained_arc_fill(
        splines,
        mandatory,
        fill_count,
        minimum_spacing_m,
        MAXIMUM_SAME_TUNNEL_COVERAGE_RADIUS_M,
        candidate_step_m,
    )

    selected = selected[:count]
    points = np.asarray([item["axis_xyz_m"] for item in selected], dtype=np.float64)
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    pairwise += np.eye(len(points)) * 1e9
    if float(np.min(pairwise)) < minimum_spacing_m - 1e-6:
        raise ValueError("selected anchor spacing contract failed")

    anchors = []
    for index, record in enumerate(selected):
        axis_xyz = np.asarray(record["axis_xyz_m"], dtype=np.float64)
        tangent = np.asarray(record["tangent_xyz"], dtype=np.float64)
        base_yaw = float(math.degrees(math.atan2(tangent[1], tangent[0])) % 360.0)
        world_label = structural_label(axis_xyz, base_yaw, splines)
        branch_count = int(world_label["branch_count"])
        role = "terminal" if branch_count <= 1 else "corridor" if branch_count == 2 else "junction"
        anchor = {
            **record,
            "anchor_id": f"anchor_{index:03d}",
            "base_yaw_deg": base_yaw,
            "objective_structure": {
                "branch_count": branch_count,
                "topological_role": role,
                "vertical_state": "slope" if abs(float(tangent[2])) >= 0.08 else "flat",
            },
        }
        if fta_distance_m is not None:
            sensor_xyz = axis_xyz.copy()
            sensor_xyz[2] += float(fta_distance_m) + LIDAR_HEIGHT_ABOVE_FLOOR_M
            anchor["sensor_xyz_m"] = sensor_xyz.tolist()
        anchors.append(anchor)
    return anchors


def anchor_selection_audit(
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    requested_count: int = ANCHORS_PER_PARENT,
    minimum_spacing_m: float = MINIMUM_ANCHOR_SPACING_M,
    maximum_coverage_radius_m: float = MAXIMUM_SAME_TUNNEL_COVERAGE_RADIUS_M,
    candidate_step_m: float = ANCHOR_CANDIDATE_STEP_M,
) -> dict[str, Any]:
    """Audit count, spacing, events, arc-length quotas and full coverage."""

    splines = spline_arrays(spline_document)
    points = np.asarray([item["axis_xyz_m"] for item in anchors], dtype=np.float64)
    if len(points) >= 2:
        pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        pairwise += np.eye(len(points)) * 1e9
        minimum_pairwise = float(np.min(pairwise))
    else:
        minimum_pairwise = float("inf")
    tunnel_counts = {
        str(tunnel_id): sum(
            int(item["source_tunnel_id"]) == tunnel_id for item in anchors
        )
        for tunnel_id in sorted(splines)
    }
    count_values = list(tunnel_counts.values())
    tunnel_count_spread = max(count_values) - min(count_values) if count_values else 0
    events = _structural_event_candidates(graph, splines)
    event_distances = []
    for event in events:
        event_point = np.asarray(event["axis_xyz_m"], dtype=np.float64)
        distance = (
            float(np.min(np.linalg.norm(points - event_point, axis=1)))
            if len(points)
            else float("inf")
        )
        event_distances.append(distance)
    maximum_event_distance = max(event_distances, default=0.0)
    all_tunnels_covered = all(value > 0 for value in count_values)
    mandatory_count = sum(int(item["priority"]) == 0 for item in anchors)
    expected_fill_quotas = _length_proportional_fill_quotas(
        splines, requested_count - mandatory_count
    )
    observed_fill_quotas = {
        tunnel_id: sum(
            int(item["priority"]) == 1
            and int(item["source_tunnel_id"]) == tunnel_id
            for item in anchors
        )
        for tunnel_id in sorted(splines)
    }
    coverage_by_tunnel: dict[str, float] = {}
    lattice = _balanced_arc_candidates(splines, candidate_step_m)
    event_points = np.asarray(
        [item["axis_xyz_m"] for item in anchors if int(item["priority"]) == 0],
        dtype=np.float64,
    )
    for tunnel_id, records in sorted(lattice.items()):
        own_fill = np.asarray(
            [
                item["axis_xyz_m"]
                for item in anchors
                if int(item["priority"]) == 1
                and int(item["source_tunnel_id"]) == tunnel_id
            ],
            dtype=np.float64,
        )
        coverage_points = (
            np.vstack((event_points, own_fill))
            if len(event_points) and len(own_fill)
            else event_points
            if len(event_points)
            else own_fill
        )
        samples = np.asarray(
            [record["axis_xyz_m"] for record in records], dtype=np.float64
        )
        coverage_by_tunnel[str(tunnel_id)] = (
            float(
                np.max(
                    np.min(
                        np.linalg.norm(
                            samples[:, None, :] - coverage_points[None, :, :], axis=2
                        ),
                        axis=1,
                    )
                )
            )
            if len(coverage_points)
            else float("inf")
        )
    maximum_observed_coverage = max(coverage_by_tunnel.values(), default=0.0)
    fill_quotas_match = observed_fill_quotas == expected_fill_quotas
    passed = bool(
        len(anchors) == requested_count
        and minimum_pairwise >= minimum_spacing_m - 1e-6
        and maximum_event_distance <= minimum_spacing_m + 1e-6
        and all_tunnels_covered
        and fill_quotas_match
        and maximum_observed_coverage <= maximum_coverage_radius_m + 1e-6
    )
    return {
        "passed": passed,
        "requested_anchor_count": requested_count,
        "selected_anchor_count": len(anchors),
        "minimum_pairwise_distance_m": minimum_pairwise,
        "required_minimum_spacing_m": minimum_spacing_m,
        "structural_event_candidate_count": len(events),
        "maximum_structural_event_to_anchor_distance_m": maximum_event_distance,
        "structural_event_coverage_radius_m": minimum_spacing_m,
        "selected_per_tunnel": tunnel_counts,
        "all_tunnels_covered": all_tunnels_covered,
        "tunnel_count_spread": tunnel_count_spread,
        "expected_arc_length_fill_quota_per_tunnel": {
            str(key): value for key, value in expected_fill_quotas.items()
        },
        "observed_fill_quota_per_tunnel": {
            str(key): value for key, value in observed_fill_quotas.items()
        },
        "fill_quotas_match": fill_quotas_match,
        "coverage_radius_m_per_tunnel": coverage_by_tunnel,
        "maximum_same_tunnel_or_shared_event_coverage_radius_m": maximum_observed_coverage,
        "maximum_allowed_coverage_radius_m": maximum_coverage_radius_m,
    }


def graph_family_metrics(
    parent_id: str,
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
) -> dict[str, Any]:
    degrees = [int(node["degree"]) for node in graph["nodes"]]
    points = np.asarray([node["xyz"] for node in graph["nodes"]], dtype=np.float64)
    statistics = graph["statistics"]
    splines = spline_arrays(spline_document)
    heading_change = 0.0
    finite_slope = True
    for spline in splines.values():
        segments = np.diff(spline, axis=0)
        horizontal = np.linalg.norm(segments[:, :2], axis=1)
        valid = horizontal > 1e-9
        slopes = np.divide(segments[:, 2], horizontal, out=np.zeros_like(horizontal), where=valid)
        finite_slope = finite_slope and bool(np.all(np.isfinite(slopes)))
        headings = np.unwrap(np.arctan2(segments[valid, 1], segments[valid, 0]))
        if len(headings) > 1:
            heading_change += float(np.sum(np.abs(np.diff(headings))))
    common = {
        "connected_components": int(statistics["connected_components"]),
        "cycle_rank": int(statistics["cycle_rank"]),
        "maximum_degree": max(degrees),
        "terminal_count": sum(value == 1 for value in degrees),
        "degree3_or_more_count": sum(value >= 3 for value in degrees),
        "vertical_span_m": float(np.ptp(points[:, 2])),
        "cumulative_horizontal_heading_change_rad": heading_change,
        "finite_spline_slope": finite_slope,
    }
    if parent_id == "P01_straight_turn":
        passed = common["connected_components"] == 1 and common["cycle_rank"] == 0 and common["maximum_degree"] <= 2 and common["terminal_count"] == 2 and heading_change > 0.2
    elif parent_id == "P02_branch_deadend":
        passed = common["connected_components"] == 1 and common["cycle_rank"] == 0 and common["maximum_degree"] >= 3 and common["terminal_count"] >= 3
    elif parent_id == "P03_loop_bottleneck":
        passed = common["connected_components"] == 1 and common["cycle_rank"] >= 1 and common["maximum_degree"] >= 3
    elif parent_id == "P04_chamber_multiexit":
        passed = common["connected_components"] == 1 and common["cycle_rank"] == 0 and common["maximum_degree"] >= 4 and common["terminal_count"] >= 4
    elif parent_id == "P05_slope_multiheight":
        passed = common["connected_components"] == 1 and common["vertical_span_m"] >= 15.0 and finite_slope and common["maximum_degree"] >= 3
    else:
        raise ValueError(f"unknown topology parent: {parent_id}")
    return {**common, "passed": bool(passed)}


def match_headings(
    predicted_deg: Sequence[float], truth_deg: Sequence[float], tolerance_deg: float = 20.0
) -> dict[str, Any]:
    remaining = list(float(value) for value in truth_deg)
    errors = []
    for predicted in sorted(float(value) for value in predicted_deg):
        if not remaining:
            break
        index, error = min(
            enumerate(remaining),
            key=lambda item: abs((predicted - item[1] + 180.0) % 360.0 - 180.0),
        )
        angular_error = abs((predicted - remaining[index] + 180.0) % 360.0 - 180.0)
        if angular_error <= tolerance_deg:
            errors.append(float(angular_error))
            remaining.pop(index)
    return {
        "matched": len(errors),
        "predicted": len(predicted_deg),
        "truth": len(truth_deg),
        "angular_errors_deg": errors,
    }
