"""Read-only native-mesh teacher distribution audit for GSE-Graph."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.data.gse_sequence_inventory import deduplicated_frame_arcs_and_sequence_indices
from mtare_topo.data.gse_teacher_inventory import oriented_edge_polyline
from mtare_topo.teacher.gse_geometry_teacher import (
    GSETeacherConfig,
    PolylineGeometrySampler,
    classify_structural_event,
)
from mtare_topo.teacher.gse_mesh_geometry_teacher import (
    BatchRaycastFunction,
    MeshGeometryTeacherConfig,
    apply_mesh_cross_section,
    geometry_transition_mask,
    measure_mesh_geometry_batch,
    sensor_origin_from_axis,
)


def _finite_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("geometry summary requires nonempty finite values")
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def world_mesh_teacher_inventory(
    *,
    parent_id: str,
    split: str,
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    geometry_parameters: Mapping[str, Any],
    cast_distances: BatchRaycastFunction,
    teacher_config: GSETeacherConfig | None = None,
    mesh_config: MeshGeometryTeacherConfig | None = None,
    spacing_m: float = 1.0,
    history_frames: int = 5,
) -> dict[str, Any]:
    """Measure every directed-traversal frame in one allowed parent."""

    if split not in {"train", "validation"} or str(parent_id).endswith("_C10"):
        raise ValueError("mesh teacher inventory permits only train/validation parents")
    teacher_cfg = teacher_config or GSETeacherConfig()
    mesh_cfg = mesh_config or MeshGeometryTeacherConfig()
    node_by_id = {str(node["id"]): node for node in graph["nodes"]}
    nodes = {node_id: np.asarray(node["xyz"], dtype=np.float64) for node_id, node in node_by_id.items()}
    splines = {
        str(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in spline_document["tunnels"]
    }
    radii = {
        str(item["tunnel_id"]): float(item["radius_m"])
        for item in geometry_parameters["tunnels"]
    }
    fta_distance_m = float(geometry_parameters["fta_distance_m"])
    if not math.isfinite(fta_distance_m):
        raise ValueError("fta_distance_m must be finite")

    event_counts: Counter[str] = Counter()
    invalid_geometry_event_counts: Counter[str] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    slopes: list[float] = []
    curvatures: list[float] = []
    complete_frame_count = 0
    incomplete_frame_count = 0
    sequence_count = 0
    transition_candidate_count = 0
    directed_traversal_count = 0

    for edge in sorted(graph["edges"], key=lambda item: str(item["id"])):
        for reverse in (False, True):
            points, tunnel_id, from_node_id, to_node_id = oriented_edge_polyline(
                edge=edge,
                nodes=nodes,
                splines=splines,
                reverse=reverse,
            )
            sampler = PolylineGeometrySampler(points)
            frame_arcs, anchor_arcs, _ = deduplicated_frame_arcs_and_sequence_indices(
                sampler.length_m,
                spacing_m=spacing_m,
                history_frames=history_frames,
            )
            directed_traversal_count += 1
            if len(frame_arcs) == 0:
                continue
            spline_targets = [
                sampler.target(
                    float(arc),
                    tunnel_radius_m=radii[tunnel_id],
                    span_m=teacher_cfg.local_geometry_span_m,
                )
                for arc in frame_arcs
            ]
            axes = np.asarray([sampler.interpolate(float(arc)) for arc in frame_arcs])
            origins = np.asarray(
                [
                    sensor_origin_from_axis(
                        axis,
                        fta_distance_m=fta_distance_m,
                        sensor_height_above_floor_m=mesh_cfg.sensor_height_above_floor_m,
                    )
                    for axis in axes
                ]
            )
            tangents = np.asarray([target.axis for target in spline_targets])
            measurements, complete = measure_mesh_geometry_batch(
                origins_xyz_m=origins,
                tangents_xyz=tangents,
                cast_distances=cast_distances,
                config=mesh_cfg,
            )
            width_profile = np.asarray(
                [measurement.width_m if measurement is not None else np.nan for measurement in measurements]
            )
            height_profile = np.asarray(
                [measurement.height_m if measurement is not None else np.nan for measurement in measurements]
            )
            transition = geometry_transition_mask(
                width_profile,
                height_profile,
                spacing_m=spacing_m,
                valid_mask=complete,
                config=mesh_cfg,
            )
            complete_frame_count += int(np.sum(complete))
            incomplete_frame_count += int(np.sum(~complete))
            transition_candidate_count += int(np.sum(transition[history_frames - 1 :]))
            for local_index, anchor in enumerate(anchor_arcs):
                frame_index = local_index + history_frames - 1
                sequence_count += 1
                measurement = measurements[frame_index]
                geometry = (
                    apply_mesh_cross_section(spline_targets[frame_index], measurement)
                    if measurement is not None
                    else spline_targets[frame_index]
                )
                canonical_arc = float(anchor if not reverse else sampler.length_m - anchor)
                transition_identity = (
                    f"{parent_id}:{edge['id']}:mesh_transition:{canonical_arc:.1f}"
                    if transition[frame_index]
                    else None
                )
                event, _ = classify_structural_event(
                    traversal_arc_m=float(anchor),
                    traversal_length_m=sampler.length_m,
                    from_node=node_by_id[from_node_id],
                    to_node=node_by_id[to_node_id],
                    local_geometry=geometry,
                    geometry_transition=bool(transition[frame_index]),
                    geometry_transition_identity=transition_identity,
                    config=teacher_cfg,
                )
                event_counts[event.value] += 1
                if measurement is None:
                    invalid_geometry_event_counts[event.value] += 1
                    continue
                widths.append(geometry.width_m)
                heights.append(geometry.height_m)
                slopes.append(geometry.slope_deg)
                curvatures.append(geometry.curvature_per_m)

    labelled_sequence_count = int(sum(event_counts.values()))
    return {
        "parent_id": str(parent_id),
        "split": str(split),
        "edge_count": len(graph["edges"]),
        "directed_traversal_count": directed_traversal_count,
        "sequence_count": sequence_count,
        "labelled_sequence_count": labelled_sequence_count,
        "unlabelled_sequence_count": sequence_count - labelled_sequence_count,
        "continuous_geometry_target_count": len(widths),
        "missing_continuous_geometry_target_count": sequence_count - len(widths),
        "missing_continuous_geometry_by_event": {
            name: int(invalid_geometry_event_counts.get(name, 0))
            for name in ("corridor", "junction", "terminal", "turn", "geometry_transition")
        },
        "complete_mesh_frame_count": complete_frame_count,
        "incomplete_mesh_frame_count": incomplete_frame_count,
        "transition_candidate_count": transition_candidate_count,
        "event_counts": {
            name: int(event_counts.get(name, 0))
            for name in ("corridor", "junction", "terminal", "turn", "geometry_transition")
        },
        "geometry": {
            "width_m": _finite_summary(widths),
            "height_m": _finite_summary(heights),
            "slope_deg": _finite_summary(slopes),
            "curvature_per_m": _finite_summary(curvatures),
        },
        "teacher_config": teacher_cfg.to_dict(),
        "mesh_teacher_config": mesh_cfg.to_dict(),
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }


__all__ = ["world_mesh_teacher_inventory"]
