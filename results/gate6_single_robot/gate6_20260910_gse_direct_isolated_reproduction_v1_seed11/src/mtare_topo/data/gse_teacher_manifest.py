"""Deterministic geometry-semantic manifest and association-pair construction."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping

import numpy as np

from mtare_topo.data.gse_sequence_inventory import (
    deduplicated_frame_arcs_and_sequence_indices,
    enumerate_directed_traversals,
)
from mtare_topo.data.gse_teacher_inventory import oriented_edge_polyline
from mtare_topo.semantics.geometric_semantics import StructuralEvent
from mtare_topo.teacher.gse_association_teacher import (
    AssociationTeacherExample,
    EdgeEventSample,
    cluster_edge_event_identities,
    deterministic_association_pairs,
    node_event_identity,
)
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


EVENT_NAMES = tuple(event.value for event in StructuralEvent)


def _association_signature(
    *,
    width_m: float | None,
    height_m: float | None,
    slope_deg: float,
    curvature_per_m: float,
    maximum_ray_distance_m: float,
) -> tuple[float, float, float, float, float]:
    """Return finite dimensionless geometry values without inventing missing labels."""

    valid = width_m is not None and height_m is not None
    scale = 2.0 * float(maximum_ray_distance_m)
    signature = (
        float(width_m) / scale if valid else 0.0,
        float(height_m) / scale if valid else 0.0,
        abs(float(slope_deg)) / 90.0,
        float(curvature_per_m) / math.pi,
        1.0 if valid else 0.0,
    )
    if not all(math.isfinite(value) for value in signature):
        raise ValueError("association signature must be finite")
    return signature


def world_teacher_manifest(
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
    hard_negatives_per_anchor: int = 1,
) -> dict[str, Any]:
    """Build one world's traversal, event-identity and association-pair manifest."""

    if split not in {"train", "validation"} or str(parent_id).endswith("_C10"):
        raise ValueError("teacher manifest permits only train/validation parents")
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

    traversal_inventory = enumerate_directed_traversals(
        parent_id=parent_id,
        split=split,
        graph=graph,
        spline_document=spline_document,
        spacing_m=spacing_m,
        history_frames=history_frames,
    )
    traversal_by_id = {record.traversal_id: record for record in traversal_inventory}
    observations: list[dict[str, Any]] = []
    identity_summaries: list[dict[str, Any]] = []

    for edge in sorted(graph["edges"], key=lambda item: str(item["id"])):
        edge_id = str(edge["id"])
        edge_observations: list[dict[str, Any]] = []
        edge_samples: dict[StructuralEvent, list[EdgeEventSample]] = {
            StructuralEvent.TURN: [],
            StructuralEvent.GEOMETRY_TRANSITION: [],
        }
        for direction_index, reverse in enumerate((False, True)):
            points, tunnel_id, from_node_id, to_node_id = oriented_edge_polyline(
                edge=edge,
                nodes=nodes,
                splines=splines,
                reverse=reverse,
            )
            traversal_id = f"{parent_id}:{edge_id}:d{direction_index}"
            inventory = traversal_by_id[traversal_id]
            sampler = PolylineGeometrySampler(points)
            if not math.isclose(sampler.length_m, inventory.length_m, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError("oriented teacher and traversal inventory lengths disagree")
            frame_arcs, anchor_arcs, references = deduplicated_frame_arcs_and_sequence_indices(
                sampler.length_m,
                spacing_m=spacing_m,
                history_frames=history_frames,
            )
            if len(anchor_arcs) != inventory.sequence_count or len(frame_arcs) != inventory.unique_frame_count:
                raise RuntimeError("teacher manifest and sequence inventory counts disagree")
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
            for sequence_index, anchor in enumerate(anchor_arcs):
                frame_index = sequence_index + history_frames - 1
                measurement = measurements[frame_index]
                geometry = (
                    apply_mesh_cross_section(spline_targets[frame_index], measurement)
                    if measurement is not None
                    else spline_targets[frame_index]
                )
                event, local_identity = classify_structural_event(
                    traversal_arc_m=float(anchor),
                    traversal_length_m=sampler.length_m,
                    from_node=node_by_id[from_node_id],
                    to_node=node_by_id[to_node_id],
                    local_geometry=geometry,
                    geometry_transition=bool(transition[frame_index]),
                    geometry_transition_identity=None,
                    config=teacher_cfg,
                )
                canonical_arc = float(anchor if not reverse else sampler.length_m - anchor)
                observation_id = f"{traversal_id}:s{sequence_index:06d}"
                identity = (
                    node_event_identity(parent_id, str(local_identity))
                    if event in {StructuralEvent.JUNCTION, StructuralEvent.TERMINAL}
                    else None
                )
                record = {
                    "observation_id": observation_id,
                    "parent_id": parent_id,
                    "split": split,
                    "traversal_id": traversal_id,
                    "edge_id": edge_id,
                    "from_node_id": from_node_id,
                    "to_node_id": to_node_id,
                    "sequence_index": sequence_index,
                    "frame_index": frame_index,
                    "traversal_arc_m": float(anchor),
                    "canonical_edge_arc_m": canonical_arc,
                    "local_frame_references": [int(value) for value in references[sequence_index]],
                    "event": event.value,
                    "identity": identity,
                    "geometry_valid": measurement is not None,
                    "width_m": float(geometry.width_m) if measurement is not None else None,
                    "height_m": float(geometry.height_m) if measurement is not None else None,
                    "slope_deg": float(geometry.slope_deg),
                    "curvature_per_m": float(geometry.curvature_per_m),
                }
                edge_observations.append(record)
                if event in edge_samples:
                    edge_samples[event].append(
                        EdgeEventSample(
                            traversal_id=traversal_id,
                            frame_index=frame_index,
                            canonical_edge_arc_m=canonical_arc,
                        )
                    )
        for event, samples in edge_samples.items():
            mapping, summaries = cluster_edge_event_identities(
                parent_id=parent_id,
                edge_id=edge_id,
                event=event,
                samples=samples,
            )
            identity_summaries.extend(summary.to_dict() for summary in summaries)
            for record in edge_observations:
                if record["event"] == event.value:
                    record["identity"] = mapping[(record["traversal_id"], record["frame_index"])]
        observations.extend(edge_observations)

    event_counts = Counter(record["event"] for record in observations)
    structural = [record for record in observations if record["event"] != StructuralEvent.CORRIDOR.value]
    if any(record["identity"] is None for record in structural):
        raise RuntimeError("structural observation lacks association identity")
    examples = [
        AssociationTeacherExample(
            observation_id=record["observation_id"],
            parent_id=parent_id,
            traversal_id=record["traversal_id"],
            identity=str(record["identity"]),
            event=record["event"],
            geometry_signature=_association_signature(
                width_m=record["width_m"],
                height_m=record["height_m"],
                slope_deg=record["slope_deg"],
                curvature_per_m=record["curvature_per_m"],
                maximum_ray_distance_m=mesh_cfg.maximum_ray_distance_m,
            ),
        )
        for record in structural
    ]
    pairs = deterministic_association_pairs(
        examples,
        hard_negatives_per_anchor=hard_negatives_per_anchor,
    )
    pair_counts = Counter(pair.pair_kind for pair in pairs)
    positive_anchors = {pair.anchor_observation_id for pair in pairs if pair.same_identity}
    negative_anchors = {pair.anchor_observation_id for pair in pairs if not pair.same_identity}
    identities = Counter(str(record["identity"]) for record in structural)
    if len({record["observation_id"] for record in observations}) != len(observations):
        raise RuntimeError("teacher observation identities are not unique")
    if sum(event_counts.values()) != sum(record.sequence_count for record in traversal_inventory):
        raise RuntimeError("teacher observations do not cover every sequence")

    return {
        "parent_id": parent_id,
        "split": split,
        "traversals": [record.to_dict() for record in traversal_inventory],
        "observations": observations,
        "association_pairs": [pair.to_dict() for pair in pairs],
        "edge_event_identities": identity_summaries,
        "summary": {
            "edge_count": len(graph["edges"]),
            "directed_traversal_count": len(traversal_inventory),
            "zero_sequence_traversal_count": sum(
                record.sequence_count == 0 for record in traversal_inventory
            ),
            "sequence_count": len(observations),
            "unique_frame_count": sum(record.unique_frame_count for record in traversal_inventory),
            "referenced_frame_count": sum(record.referenced_frame_count for record in traversal_inventory),
            "event_counts": {name: int(event_counts[name]) for name in EVENT_NAMES},
            "structural_observation_count": len(structural),
            "association_identity_count": len(identities),
            "association_pair_counts": dict(sorted(pair_counts.items())),
            "positive_anchor_count": len(positive_anchors),
            "hard_negative_anchor_count": len(negative_anchors),
            "geometry_valid_structural_observation_count": sum(
                bool(record["geometry_valid"]) for record in structural
            ),
        },
        "teacher_config": teacher_cfg.to_dict(),
        "mesh_teacher_config": mesh_cfg.to_dict(),
        "hard_negatives_per_anchor": hard_negatives_per_anchor,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }


__all__ = ["EVENT_NAMES", "world_teacher_manifest"]
