"""Pure contracts for local floor-following pose feasibility audits."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def contiguous_feasible_intervals(records: Sequence[Mapping[str, Any]], step_m: float) -> list[dict[str, Any]]:
    feasible = [dict(record) for record in records if record["feasible"]]
    if not feasible:
        return []
    feasible.sort(key=lambda item: float(item["z_offset_m"]))
    groups: list[list[dict[str, Any]]] = [[feasible[0]]]
    for record in feasible[1:]:
        gap = float(record["z_offset_m"]) - float(groups[-1][-1]["z_offset_m"])
        if abs(gap - step_m) <= 1e-8:
            groups[-1].append(record)
        else:
            groups.append([record])
    result = []
    for index, group in enumerate(groups):
        recommended = min(group, key=lambda item: (abs(float(item["z_offset_m"])), float(item["z_offset_m"])))
        result.append({
            "interval_index": index,
            "minimum_z_offset_m": float(group[0]["z_offset_m"]),
            "maximum_z_offset_m": float(group[-1]["z_offset_m"]),
            "candidate_count": len(group),
            "recommended_z_offset_m": float(recommended["z_offset_m"]),
            "recommended_horizontal_clearance_m": float(recommended["horizontal_clearance_m"]),
            "recommended_downward_distance_m": float(recommended["downward_distance_m"]),
            "recommended_upward_distance_m": float(recommended["upward_distance_m"]),
        })
    return result


def physical_groups(records: Sequence[Mapping[str, Any]], radius_m: float) -> list[list[int]]:
    """Deterministically group route failures by nearby axis xyz."""

    import numpy as np

    points = [np.asarray(record["axis_xyz_m"], dtype=float) for record in records]
    remaining = set(range(len(points))); groups = []
    while remaining:
        root = min(remaining); group = {root}; changed = True
        while changed:
            changed = False
            for candidate in sorted(remaining - group):
                if any(float(np.linalg.norm(points[candidate] - points[item])) <= radius_m for item in group):
                    group.add(candidate); changed = True
        groups.append(sorted(group)); remaining -= group
    return groups


def feasible_grid_components(records: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Return deterministic four-neighbour components of feasible grid cells."""

    cell_to_record = {
        (int(record["x_index"]), int(record["y_index"])): index
        for index, record in enumerate(records)
        if bool(record["feasible"])
    }
    remaining = set(cell_to_record)
    components: list[list[int]] = []
    while remaining:
        root = min(remaining)
        frontier = [root]
        cells = {root}
        remaining.remove(root)
        while frontier:
            x_index, y_index = frontier.pop(0)
            for neighbour in (
                (x_index - 1, y_index),
                (x_index + 1, y_index),
                (x_index, y_index - 1),
                (x_index, y_index + 1),
            ):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    cells.add(neighbour)
                    frontier.append(neighbour)
        components.append(sorted(cell_to_record[cell] for cell in cells))
    components.sort(key=lambda component: (-len(component), component[0]))
    return components


def nearest_common_feasible_offset(
    candidates_by_frame: Mapping[int, Mapping[tuple[float, float], bool]],
    frame_indices: Sequence[int],
) -> tuple[float, float]:
    """Select the nearest deterministic x/y cell feasible for every observation."""

    import math

    common = {
        offset for offset, feasible in candidates_by_frame[int(frame_indices[0])].items()
        if feasible
    }
    for frame_index in frame_indices[1:]:
        common &= {
            offset for offset, feasible in candidates_by_frame[int(frame_index)].items()
            if feasible
        }
    if not common:
        raise ValueError("no common feasible lateral offset")
    return min(common, key=lambda offset: (math.hypot(*offset), offset[0], offset[1]))


def compact_support_lateral_field(
    points_xyz: Any,
    anchor_points_xyz: Any,
    anchor_offsets_xy: Any,
    radius_m: float,
) -> Any:
    """Blend spatial anchors with a C1 compact kernel; return zero outside support."""

    import numpy as np

    points = np.asarray(points_xyz, dtype=np.float64)
    anchors = np.asarray(anchor_points_xyz, dtype=np.float64)
    offsets = np.asarray(anchor_offsets_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("points and anchors must have shape [N,3]")
    if offsets.shape != (len(anchors), 2) or radius_m <= 0:
        raise ValueError("anchor offsets/radius mismatch")
    distances = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=2)
    scaled = distances / float(radius_m)
    weights = np.where(scaled < 1.0, (1.0 - scaled * scaled) ** 2, 0.0)
    totals = weights.sum(axis=1, keepdims=True)
    return np.divide(weights @ offsets, totals, out=np.zeros((len(points), 2)), where=totals > 0)


def common_boolean_grid(records_by_observation: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Intersect matching x/y grids while preserving deterministic cell identity."""

    if not records_by_observation:
        return []
    indexed = []
    for records in records_by_observation:
        indexed.append({(int(item["x_index"]), int(item["y_index"])): item for item in records})
    keys = set(indexed[0])
    for mapping in indexed[1:]:
        keys &= set(mapping)
    result = []
    for key in sorted(keys):
        first = indexed[0][key]
        result.append({
            "x_index": key[0], "y_index": key[1],
            "x_offset_m": float(first["x_offset_m"]),
            "y_offset_m": float(first["y_offset_m"]),
            "feasible": all(bool(mapping[key]["feasible"]) for mapping in indexed),
        })
    return result
