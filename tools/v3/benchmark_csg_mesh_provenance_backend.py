#!/usr/bin/env python3
"""Deterministic C01 development benchmark for the CSG provenance backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import (
    LIDAR_HEIGHT_ABOVE_FLOOR_M,
    interpolate_polyline,
    lidar_local_directions,
    world_directions,
)
from mtare_topo.governance import load_json
from mtare_topo.teacher.csg_mesh_provenance import (
    CSGMeshProvenanceRaycaster,
    mesh_swept_superellipse,
)
from mtare_topo.teacher.geometry_variant_contract import (
    GeometryRealization,
    realize_construction,
)
from mtare_topo.teacher.primitive_construction_supervisor import (
    build_primitive_construction_graph,
)
from mtare_topo.teacher.swept_superellipse_field import (
    SweptSuperellipseProvenanceField,
)


WORLD = "S01_flat_tree_small_C01"
MESH_ROOT = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
MANIFEST = PROJECT_ROOT / "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0/artifacts/manifest.jsonl"


def _poses(primary: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    selected: dict[str, list[dict]] = {value: [] for value in ("interior", "junction", "terminal")}
    with MANIFEST.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            role = row.get("primary_role")
            if row.get("parent_id") == WORLD and role in selected and len(selected[role]) < 2:
                selected[role].append(row)
    rows = [row for role in ("interior", "junction", "terminal") for row in selected[role]]
    if len(rows) != 6:
        raise RuntimeError("C01 does not provide two frozen poses for every role")
    splines = {int(value["tunnel_id"]): np.asarray(value["points"], dtype=np.float64) for value in load_json(primary / "splines.json")["tunnels"]}
    fta = float(load_json(primary / "geometry_parameters.json")["fta_distance_m"])
    origins = []
    for row in rows:
        axis, _ = interpolate_polyline(splines[int(row["tunnel_id"])], float(row["arc_m"]))
        axis[2] += fta + LIDAR_HEIGHT_ABOVE_FLOOR_M
        origins.append(axis)
    return np.asarray(origins), np.asarray([float(value["yaw_deg"]) for value in rows]), rows


def _hit_sha256(hits) -> str:
    rows = [
        None if value is None else {
            "distance_hex": float(value.distance_m).hex(),
            "source_primitive_ids": list(value.source_primitive_ids),
            "provenance_unique": bool(value.provenance_unique),
        }
        for value in hits
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rays-per-pose", type=int, default=256)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rays_per_pose <= 0 or args.rays_per_pose > 11520:
        raise ValueError("rays-per-pose must be in [1,11520]")
    primary = MESH_ROOT / WORLD / "primary"
    graph = load_json(primary / "graph.json")
    splines = load_json(primary / "splines.json")
    geometry = load_json(primary / "geometry_parameters.json")
    construction = build_primitive_construction_graph(
        graph, splines, geometry,
        endpoint_attachment_mode="free_space_overlap",
        node_degree_source="edge_incidence",
    )
    primitives = realize_construction(WORLD, construction, GeometryRealization.C1_MIXED)
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=.025)
    started = time.perf_counter()
    meshes = [mesh_swept_superellipse(value, axial_spacing_m=.05, angular_segments=64) for value in primitives]
    mesh_seconds = time.perf_counter() - started
    query = field.operand_signed_distances_sparse if args.sparse else field.operand_signed_distances
    raycaster = CSGMeshProvenanceRaycaster(meshes, operand_signed_distances=query)
    origins, yaws, rows = _poses(primary)
    local = lidar_local_directions().reshape(-1, 3)
    indices = np.floor(np.arange(args.rays_per_pose) * len(local) / args.rays_per_pose).astype(np.int64)
    all_origins = np.concatenate([np.broadcast_to(origin, (len(indices), 3)) for origin in origins])
    all_directions = np.concatenate([world_directions(local[indices], yaw) for yaw in yaws])
    initial = field.operand_signed_distances_sparse(all_origins) <= 0
    started = time.perf_counter()
    csg = raycaster.ray_exit_hits(all_origins, all_directions, initial, maximum_m=50.)
    csg_seconds = time.perf_counter() - started
    started = time.perf_counter()
    implicit = field.ray_exit_hits(all_origins, all_directions, maximum_m=50.)
    implicit_seconds = time.perf_counter() - started
    csg_valid = np.asarray([value is not None for value in csg])
    implicit_valid = np.asarray([value is not None for value in implicit])
    both = csg_valid & implicit_valid
    errors = np.asarray([abs(csg[i].distance_m - implicit[i].distance_m) for i in np.flatnonzero(both)])
    identity = np.asarray([set(csg[i].source_primitive_ids) == set(implicit[i].source_primitive_ids) for i in np.flatnonzero(both)])
    summary = {
        "world": WORLD,
        "realization": GeometryRealization.C1_MIXED.value,
        "primitive_count": len(primitives),
        "mesh_vertices": int(sum(len(value.vertices_xyz_m) for value in meshes)),
        "mesh_triangles": int(sum(len(value.triangle_vertex_indices) for value in meshes)),
        "mesh_construction_seconds": mesh_seconds,
        "poses": [{"frame_id": row["frame_id"], "role": row["primary_role"]} for row in rows],
        "rays": len(all_origins),
        "operand_query": "sparse_aabb" if args.sparse else "full",
        "csg_seconds": csg_seconds,
        "implicit_seconds": implicit_seconds,
        "speedup": implicit_seconds / csg_seconds,
        "valid_agreement": float(np.mean(csg_valid == implicit_valid)),
        "csg_qualified_coverage": float(np.mean(csg_valid)),
        "implicit_coverage": float(np.mean(implicit_valid)),
        "range_mae_m": float(np.mean(errors)),
        "range_p95_m": float(np.quantile(errors, .95)),
        "range_p99_m": float(np.quantile(errors, .99)),
        "range_max_m": float(np.max(errors)),
        "range_gt_0p05_fraction": float(np.mean(errors > .05)),
        "identity_agreement": float(np.mean(identity)),
        "csg_hit_sha256": _hit_sha256(csg),
        "implicit_hit_sha256": _hit_sha256(implicit),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
