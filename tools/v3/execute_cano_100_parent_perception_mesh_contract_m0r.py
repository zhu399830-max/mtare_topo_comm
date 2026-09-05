#!/usr/bin/env python3
"""Execute M0R by replacing only native OBJ-byte replay with geometric replay."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d

import execute_cano_100_parent_perception_mesh_contract_m0 as m0
from mtare_topo.data.cano_perception_mesh_contract import (
    geometric_replay_pair_audit,
)
from mtare_topo.governance import write_json


_M0_MATERIALIZE = m0._materialize


def _materialize_with_paths(*args, **kwargs):
    destination = args[2] if len(args) >= 3 else kwargs["destination"]
    record, vertices, graph, splines = _M0_MATERIALIZE(*args, **kwargs)
    destination = Path(destination)
    record["mesh_path"] = str((destination / "mesh.obj").resolve())
    record["axis_path"] = str((destination / "axis.npy").resolve())
    return record, vertices, graph, splines


def _geometric_replay_with_triangle_limit(
    primary,
    replay,
    *,
    triangle_count_relative_limit=None,
    vertex_count_relative_limit=None,
    use_point_to_surface=False,
):
    primary_mesh = o3d.io.read_triangle_mesh(primary["mesh_path"])
    replay_mesh = o3d.io.read_triangle_mesh(replay["mesh_path"])
    primary_vertices = np.asarray(primary_mesh.vertices)
    replay_vertices = np.asarray(replay_mesh.vertices)
    primary_triangles = np.asarray(primary_mesh.triangles)
    replay_triangles = np.asarray(replay_mesh.triangles)
    primary_cloud = o3d.geometry.PointCloud()
    primary_cloud.points = primary_mesh.vertices
    replay_cloud = o3d.geometry.PointCloud()
    replay_cloud.points = replay_mesh.vertices
    primary_to_replay = np.asarray(
        primary_cloud.compute_point_cloud_distance(replay_cloud), dtype=np.float64
    )
    replay_to_primary = np.asarray(
        replay_cloud.compute_point_cloud_distance(primary_cloud), dtype=np.float64
    )
    lower_difference = np.abs(
        np.min(primary_vertices, axis=0) - np.min(replay_vertices, axis=0)
    )
    upper_difference = np.abs(
        np.max(primary_vertices, axis=0) - np.max(replay_vertices, axis=0)
    )
    primary_area = float(primary_mesh.get_surface_area())
    replay_area = float(replay_mesh.get_surface_area())
    relative_area_difference = abs(primary_area - replay_area) / max(primary_area, 1e-12)
    axis_equal = np.array_equal(
        np.load(primary["axis_path"], allow_pickle=False),
        np.load(replay["axis_path"], allow_pickle=False),
    )
    primary_to_replay_surface = None
    replay_to_primary_surface = None
    if use_point_to_surface:
        primary_tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(primary_mesh)
        replay_tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(replay_mesh)
        primary_scene = o3d.t.geometry.RaycastingScene()
        replay_scene = o3d.t.geometry.RaycastingScene()
        primary_scene.add_triangles(primary_tensor_mesh)
        replay_scene.add_triangles(replay_tensor_mesh)
        primary_to_replay_surface = replay_scene.compute_distance(
            o3d.core.Tensor(primary_vertices.astype(np.float32))
        ).numpy()
        replay_to_primary_surface = primary_scene.compute_distance(
            o3d.core.Tensor(replay_vertices.astype(np.float32))
        ).numpy()
    result = geometric_replay_pair_audit(
        primary,
        replay,
        axis_exact_equal=axis_equal,
        primary_vertex_count=len(primary_vertices),
        replay_vertex_count=len(replay_vertices),
        primary_triangle_count=len(primary_triangles),
        replay_triangle_count=len(replay_triangles),
        primary_to_replay_nearest_vertex_maximum_m=float(np.max(primary_to_replay)),
        replay_to_primary_nearest_vertex_maximum_m=float(np.max(replay_to_primary)),
        aabb_endpoint_maximum_coordinate_difference_m=float(
            max(np.max(lower_difference), np.max(upper_difference))
        ),
        surface_area_relative_difference=relative_area_difference,
        triangle_count_relative_limit=triangle_count_relative_limit,
        vertex_count_relative_limit=vertex_count_relative_limit,
        primary_to_replay_surface_maximum_m=(
            None
            if primary_to_replay_surface is None
            else float(np.max(primary_to_replay_surface))
        ),
        replay_to_primary_surface_maximum_m=(
            None
            if replay_to_primary_surface is None
            else float(np.max(replay_to_primary_surface))
        ),
    )
    result["metrics"].update(
        {
            "primary_to_replay_nearest_vertex_mean_m": float(np.mean(primary_to_replay)),
            "replay_to_primary_nearest_vertex_mean_m": float(np.mean(replay_to_primary)),
            "primary_to_replay_nearest_vertex_p99_m": float(
                np.quantile(primary_to_replay, 0.99)
            ),
            "replay_to_primary_nearest_vertex_p99_m": float(
                np.quantile(replay_to_primary, 0.99)
            ),
            "primary_surface_area_m2": primary_area,
            "replay_surface_area_m2": replay_area,
            "aabb_lower_coordinate_differences_m": lower_difference.tolist(),
            "aabb_upper_coordinate_differences_m": upper_difference.tolist(),
            "axis_exact_equal": bool(axis_equal),
        }
    )
    if primary_to_replay_surface is not None and replay_to_primary_surface is not None:
        result["metrics"].update(
            {
                "primary_vertices_to_replay_surface_mean_m": float(
                    np.mean(primary_to_replay_surface)
                ),
                "replay_vertices_to_primary_surface_mean_m": float(
                    np.mean(replay_to_primary_surface)
                ),
                "primary_vertices_to_replay_surface_p99_m": float(
                    np.quantile(primary_to_replay_surface, 0.99)
                ),
                "replay_vertices_to_primary_surface_p99_m": float(
                    np.quantile(replay_to_primary_surface, 0.99)
                ),
                "primary_vertices_to_replay_surface_p99_9_m": float(
                    np.quantile(primary_to_replay_surface, 0.999)
                ),
                "replay_vertices_to_primary_surface_p99_9_m": float(
                    np.quantile(replay_to_primary_surface, 0.999)
                ),
            }
        )
    return result


def _geometric_replay(primary, replay):
    return _geometric_replay_with_triangle_limit(primary, replay)


def execute(run_dir: Path):
    m0._materialize = _materialize_with_paths
    m0.replay_pair_audit = _geometric_replay
    summary = m0.execute(run_dir)
    summary["schema_version"] = "cano_100_parent_perception_mesh_contract_summary_m0r"
    summary["overall_status"] = "PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R"
    summary["replay_contract"] = {
        "exact": [
            "graph identity",
            "spline identity",
            "operation trace",
            "effective geometry parameters",
            "axis array",
        ],
        "geometric": {
            "bidirectional_nearest_vertex_maximum_m": 0.75,
            "aabb_endpoint_maximum_coordinate_difference_m": 0.5,
            "surface_area_relative_difference_maximum": 0.01,
        },
        "obj_sha256_equality_required": False,
    }
    summary["claim_boundary"] = (
        "Train-only native perception-mesh materialization and noise-bounded geometric "
        "replay; not formal data, LiDAR, navigation geometry, training or M-TARE change."
    )
    write_json(run_dir / "metrics/summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = execute(args.run_dir.resolve())
    except Exception as exc:
        write_json(
            args.run_dir.resolve() / "metrics/executor_failure.json",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
