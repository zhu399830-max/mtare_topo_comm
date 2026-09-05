"""Read-only event/geometry distribution proof for the GSE Data Card."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.data.gse_sequence_inventory import causal_anchor_arcs
from mtare_topo.teacher.gse_geometry_teacher import (
    GSETeacherConfig,
    PolylineGeometrySampler,
    classify_structural_event,
)
from mtare_topo.topology.continuous_trajectory import polyline_between, project_to_polyline


def oriented_edge_polyline(
    *,
    edge: Mapping[str, Any],
    nodes: Mapping[str, np.ndarray],
    splines: Mapping[str, np.ndarray],
    reverse: bool,
) -> tuple[np.ndarray, str, str, str]:
    node_ids = tuple(str(value) for value in edge.get("node_ids", ()))
    tunnel_ids = tuple(str(value) for value in edge.get("tunnel_ids", ()))
    if len(node_ids) != 2 or len(tunnel_ids) != 1 or tunnel_ids[0] not in splines:
        raise ValueError(f"invalid edge contract: {edge.get('id')}")
    tunnel_id = tunnel_ids[0]
    first_projection = project_to_polyline(nodes[node_ids[0]], splines[tunnel_id])
    second_projection = project_to_polyline(nodes[node_ids[1]], splines[tunnel_id])
    spline_part = polyline_between(splines[tunnel_id], first_projection.arc_m, second_projection.arc_m)
    points = np.vstack(
        (
            nodes[node_ids[0]],
            first_projection.xyz_m,
            spline_part,
            second_projection.xyz_m,
            nodes[node_ids[1]],
        )
    )
    keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12))
    points = points[keep]
    if reverse:
        points = points[::-1]
        node_ids = (node_ids[1], node_ids[0])
    return points, tunnel_id, node_ids[0], node_ids[1]


def world_teacher_inventory(
    *,
    parent_id: str,
    split: str,
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    geometry_parameters: Mapping[str, Any],
    config: GSETeacherConfig | None = None,
    spacing_m: float = 1.0,
    history_frames: int = 5,
) -> dict[str, Any]:
    if split not in {"train", "validation"} or str(parent_id).endswith("_C10"):
        raise ValueError("teacher inventory permits only train/validation parents")
    cfg = config or GSETeacherConfig()
    node_by_id = {str(node["id"]): node for node in graph["nodes"]}
    nodes = {node_id: np.asarray(node["xyz"], dtype=np.float64) for node_id, node in node_by_id.items()}
    splines = {str(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64) for item in spline_document["tunnels"]}
    radii = {str(item["tunnel_id"]): float(item["radius_m"]) for item in geometry_parameters["tunnels"]}
    event_counts: Counter[str] = Counter()
    identity_sets: dict[str, set[str]] = {"junction": set(), "terminal": set(), "geometry_transition": set()}
    widths: list[float] = []
    heights: list[float] = []
    slopes: list[float] = []
    curvatures: list[float] = []
    directed_traversals = 0
    sequence_count = 0
    unique_frame_count = 0
    referenced_frame_count = 0
    directed_length_m = 0.0
    for edge in sorted(graph["edges"], key=lambda item: str(item["id"])):
        for reverse in (False, True):
            points, tunnel_id, from_node_id, to_node_id = oriented_edge_polyline(
                edge=edge, nodes=nodes, splines=splines, reverse=reverse
            )
            sampler = PolylineGeometrySampler(points)
            anchors = causal_anchor_arcs(sampler.length_m, spacing_m=spacing_m, history_frames=history_frames)
            directed_traversals += 1
            directed_length_m += sampler.length_m
            sequence_count += len(anchors)
            if len(anchors):
                unique_frame_count += len(anchors) + history_frames - 1
                referenced_frame_count += len(anchors) * history_frames
            for anchor in anchors:
                geometry = sampler.target(anchor, tunnel_radius_m=radii[tunnel_id], span_m=cfg.local_geometry_span_m)
                event, identity = classify_structural_event(
                    traversal_arc_m=float(anchor),
                    traversal_length_m=sampler.length_m,
                    from_node=node_by_id[from_node_id],
                    to_node=node_by_id[to_node_id],
                    local_geometry=geometry,
                    tunnel_radius_by_id=radii,
                    config=cfg,
                )
                event_counts[event.value] += 1
                if identity is not None and event.value in identity_sets:
                    identity_sets[event.value].add(identity)
                widths.append(geometry.width_m)
                heights.append(geometry.height_m)
                slopes.append(geometry.slope_deg)
                curvatures.append(geometry.curvature_per_m)
    if sequence_count != sum(event_counts.values()):
        raise RuntimeError("every GSE sequence must have exactly one primary event")

    def summary(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if len(array) == 0 or not np.all(np.isfinite(array)):
            raise RuntimeError("teacher geometry summary is empty or non-finite")
        return {
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
        }

    return {
        "parent_id": str(parent_id),
        "split": str(split),
        "edge_count": len(graph["edges"]),
        "directed_traversal_count": directed_traversals,
        "directed_length_m": float(directed_length_m),
        "sequence_count": int(sequence_count),
        "unique_frame_count": int(unique_frame_count),
        "referenced_frame_count": int(referenced_frame_count),
        "event_counts": {name: int(event_counts.get(name, 0)) for name in ("corridor", "junction", "terminal", "turn", "geometry_transition")},
        "event_identity_counts": {name: len(values) for name, values in identity_sets.items()},
        "geometry": {
            "width_m": summary(widths),
            "height_m": summary(heights),
            "slope_deg": summary(slopes),
            "curvature_per_m": summary(curvatures),
        },
    }


def teacher_inventory_from_registry(
    *,
    registry_path: Path,
    mesh_root: Path,
    config: GSETeacherConfig | None = None,
) -> dict[str, Any]:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    row_schema = registry["sampling_contract"]["row_schema"]
    rows = [dict(zip(row_schema, row, strict=True)) if isinstance(row, list) else dict(row) for row in registry["rows"]]
    rows = [row for row in rows if row["split"] in {"train", "validation"}]
    if len(rows) != 90 or any(str(row["parent_id"]).endswith("_C10") for row in rows):
        raise ValueError("expected exactly 90 train/validation registry rows and zero C10")
    worlds: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["parent_id"])):
        primary = Path(mesh_root) / str(row["parent_id"]) / "primary"
        worlds.append(
            world_teacher_inventory(
                parent_id=str(row["parent_id"]),
                split=str(row["split"]),
                graph=json.loads((primary / "graph.json").read_text(encoding="utf-8")),
                spline_document=json.loads((primary / "splines.json").read_text(encoding="utf-8")),
                geometry_parameters=json.loads((primary / "geometry_parameters.json").read_text(encoding="utf-8")),
                config=config,
            )
        )
    split_summaries: dict[str, Any] = {}
    for split in ("train", "validation"):
        selected = [world for world in worlds if world["split"] == split]
        split_summaries[split] = {
            "world_count": len(selected),
            "edge_count": sum(world["edge_count"] for world in selected),
            "directed_traversal_count": sum(world["directed_traversal_count"] for world in selected),
            "directed_length_m": sum(world["directed_length_m"] for world in selected),
            "sequence_count": sum(world["sequence_count"] for world in selected),
            "unique_frame_count": sum(world["unique_frame_count"] for world in selected),
            "referenced_frame_count": sum(world["referenced_frame_count"] for world in selected),
            "event_counts": {
                name: sum(world["event_counts"][name] for world in selected)
                for name in ("corridor", "junction", "terminal", "turn", "geometry_transition")
            },
        }
    return {
        "schema_version": "gse_teacher_inventory_v1",
        "teacher_config": (config or GSETeacherConfig()).to_dict(),
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "split_summaries": split_summaries,
        "worlds": worlds,
    }


__all__ = ["oriented_edge_polyline", "teacher_inventory_from_registry", "world_teacher_inventory"]
