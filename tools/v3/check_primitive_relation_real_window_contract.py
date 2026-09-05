#!/usr/bin/env python3
"""Exercise the P1 provenance/codebook/visible-target contract on real C01 windows."""

from __future__ import annotations

import json
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.gse_sensor_export import world_unique_frame_poses
from mtare_topo.data.primitive_relation_dataset import (
    PrimitiveMembershipCodebook,
    visible_primitive_window_targets,
)
from mtare_topo.governance import load_json
from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster, mesh_swept_superellipse
from mtare_topo.teacher.geometry_variant_contract import GeometryRealization, realize_construction
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


WORLD = "S01_flat_tree_small_C01"
MESH_ROOT = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
TRAVERSALS = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/traversal_manifest.jsonl"


def _traversals():
    rows = []
    with TRAVERSALS.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["parent_id"] == WORLD:
                rows.append(row)
    return rows


def _select_windows(poses, graph):
    junctions = np.asarray([row["xyz"] for row in graph["nodes"] if int(row["degree"]) >= 3], dtype=float)
    terminals = np.asarray([row["xyz"] for row in graph["nodes"] if int(row["degree"]) == 1], dtype=float)
    valid = np.flatnonzero(poses.local_frame_indices >= 4)
    junction_distance = np.min(np.linalg.norm(poses.axis_xyz_m[valid, None] - junctions[None], axis=2), axis=1)
    terminal_distance = np.min(np.linalg.norm(poses.axis_xyz_m[valid, None] - terminals[None], axis=2), axis=1)
    event_distance = np.minimum(junction_distance, terminal_distance)
    selected = {
        "junction": int(valid[np.argmin(junction_distance)]),
        "terminal": int(valid[np.argmin(terminal_distance)]),
        "interior": int(valid[np.argmax(event_distance)]),
    }
    windows = {}
    for role, current in selected.items():
        indices = np.arange(current - 4, current + 1)
        if len(set(poses.traversal_ids[index] for index in indices)) != 1:
            raise RuntimeError(f"{role} window crosses a traversal")
        windows[role] = indices
    return windows


def main():
    primary = MESH_ROOT / WORLD / "primary"
    graph = load_json(primary / "graph.json"); splines = load_json(primary / "splines.json"); geometry = load_json(primary / "geometry_parameters.json")
    construction = build_primitive_construction_graph(graph, splines, geometry, endpoint_attachment_mode="free_space_overlap", node_degree_source="edge_incidence")
    primitives = realize_construction(WORLD, construction, GeometryRealization.C1_MIXED)
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=.025)
    raycaster = CSGMeshProvenanceRaycaster(
        [mesh_swept_superellipse(value, axial_spacing_m=.05, angular_segments=64) for value in primitives],
        operand_signed_distances=field.operand_signed_distances_sparse,
    )
    poses = world_unique_frame_poses(parent_id=WORLD, traversal_manifest=_traversals(), graph=graph, spline_document=splines, geometry_parameters=geometry)
    windows = _select_windows(poses, graph)
    local = lidar_local_directions().reshape(-1, 3)
    codebook = PrimitiveMembershipCodebook(field.primitive_ids)
    rows = []
    for role, indices in windows.items():
        hits_by_frame = []
        codes_by_frame = []
        for index in indices:
            origin = poses.sensor_xyz_m[index]
            origins = np.broadcast_to(origin, (len(local), 3))
            directions = world_directions(local, float(poses.yaw_deg[index]))
            initial = field.operand_signed_distances_sparse(origins) <= 0
            hits = raycaster.ray_exit_hits(origins, directions, initial, maximum_m=50.)
            hits_by_frame.append(hits); codes_by_frame.append(codebook.encode(hits))
        codes = np.stack(codes_by_frame)
        target = visible_primitive_window_targets(
            field=field, hits_by_frame=hits_by_frame,
            current_sensor_xyz_m=poses.sensor_xyz_m[indices[-1]],
            current_yaw_deg=float(poses.yaw_deg[indices[-1]]),
            maximum_slots=len(primitives),
        )
        decoded = codebook.decode(codes)
        if len(decoded) != codes.size:
            raise RuntimeError("codebook round trip count drift")
        cardinality = codebook.cardinality(codes)
        rows.append({
            "role": role,
            "traversal_id": poses.traversal_ids[indices[-1]],
            "current_local_frame_index": int(poses.local_frame_indices[indices[-1]]),
            "rays": int(codes.size),
            "qualified_hits": int(np.sum(cardinality > 0)),
            "ambiguous_hits": int(np.sum(cardinality > 1)),
            "maximum_source_cardinality": int(np.max(cardinality)),
            "visible_primitive_slots": int(np.sum(target.mask)),
            "visible_primitive_indices": target.primitive_index[target.mask.astype(bool)].tolist(),
            "minimum_support_rays": int(np.min(target.support_ray_count[target.mask.astype(bool)])),
            "maximum_support_rays": int(np.max(target.support_ray_count)),
        })
    maximum_slots = max(row["visible_primitive_slots"] for row in rows)
    result = {
        "status": "PASS_PRIMITIVE_RELATION_REAL_WINDOW_CONTRACT" if maximum_slots <= 8 else "FAIL_PRIMITIVE_RELATION_SLOT_CAPACITY_GT8",
        "world": WORLD, "realization": GeometryRealization.C1_MIXED.value,
        "windows": rows, "total_rays": int(sum(row["rays"] for row in rows)),
        "codebook_entries": len(codebook.source_sets),
        "maximum_codebook_membership": max(len(value) for value in codebook.source_sets),
        "maximum_visible_primitive_slots": maximum_slots,
        "frozen_slot_capacity": 8,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
