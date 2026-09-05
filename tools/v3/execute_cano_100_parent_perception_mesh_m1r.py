#!/usr/bin/env python3
"""Materialize and uniformly sanitize all 100 Cano perception assets."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d

import execute_cano_100_parent_perception_mesh_contract_m0 as m0
from mtare_topo.data.cano_perception_mesh_contract import (
    immutable_full_parent_asset_batch_audit,
    mesh_array_audit,
    zero_area_triangle_sanitation,
)
from mtare_topo.governance import load_json, write_json


EXPECTED_SCOPE = {
    "topology_parents_selected": 100,
    "topology_reconstructions": 100,
    "native_materializations": 100,
    "sanitized_primary_assets": 100,
    "replay_meshes": 0,
    "train_complete_previews": 80,
    "validation_complete_previews": 0,
    "development_test_complete_previews": 0,
    "validation_or_development_test_parents_read": 20,
    "anchors": 0,
    "lidar_observations": 0,
    "teacher_labels": 0,
    "formal_dataset_samples": 0,
    "training_samples": 0,
    "models": 0,
    "trajectories": 0,
    "gazebo_runs": 0,
    "isaac_runs": 0,
    "mtare_changes": 0,
}


def _filter_obj_face_lines(mesh_path: Path, keep: np.ndarray) -> None:
    """Remove selected OBJ face lines while preserving every other byte."""

    keep = np.asarray(keep, dtype=bool).reshape(-1)
    lines = mesh_path.read_text(encoding="utf-8").splitlines(keepends=True)
    face_index = 0
    output = []
    for line in lines:
        if line.lstrip().startswith("f "):
            if face_index >= len(keep):
                raise RuntimeError("OBJ contains more face lines than parsed triangles")
            if keep[face_index]:
                output.append(line)
            face_index += 1
        else:
            output.append(line)
    if face_index != len(keep):
        raise RuntimeError(f"OBJ face-line count mismatch: {face_index} != {len(keep)}")
    mesh_path.write_text("".join(output), encoding="utf-8")


def _sanitize_materialization(
    destination: Path,
    primary: dict,
    graph: dict,
    splines: dict,
) -> tuple[dict, np.ndarray]:
    mesh_path = destination / "mesh.obj"
    raw_hash = primary["mesh_sha256"]
    raw_bytes = mesh_path.stat().st_size
    raw_mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    raw_vertices = np.asarray(raw_mesh.vertices).copy()
    raw_triangles = np.asarray(raw_mesh.triangles).copy()
    raw_area = float(raw_mesh.get_surface_area())
    keep, sanitation = zero_area_triangle_sanitation(raw_vertices, raw_triangles)

    if sanitation["removed_triangle_count"] > 0:
        _filter_obj_face_lines(mesh_path, keep)

    final_mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    vertices = np.asarray(final_mesh.vertices)
    triangles = np.asarray(final_mesh.triangles)
    _, component_counts, component_areas = final_mesh.cluster_connected_triangles()
    axis = np.load(destination / "axis.npy", allow_pickle=False)
    source_tunnel_ids = [int(item["id"]) for item in graph["tunnels"]]
    audit = mesh_array_audit(
        vertices,
        triangles,
        list(component_counts),
        axis,
        source_tunnel_ids,
        m0._source_spline_points(splines),
    )
    audit["edge_manifold_allow_boundary"] = bool(final_mesh.is_edge_manifold(True))
    audit["edge_manifold_no_boundary"] = bool(final_mesh.is_edge_manifold(False))
    audit["vertex_manifold"] = bool(final_mesh.is_vertex_manifold())
    audit["watertight"] = bool(final_mesh.is_watertight())
    audit["orientable"] = bool(final_mesh.is_orientable())
    audit["component_surface_areas_m2"] = [float(value) for value in component_areas]

    final_area = float(final_mesh.get_surface_area())
    area_change = raw_area - final_area
    vertex_coordinate_maximum_difference = (
        float(np.max(np.abs(raw_vertices - vertices)))
        if raw_vertices.shape == vertices.shape
        else None
    )
    sanitation.update(
        {
            "raw_obj_sha256": raw_hash,
            "raw_obj_bytes": raw_bytes,
            "final_obj_sha256": m0._sha256(mesh_path),
            "final_obj_bytes": mesh_path.stat().st_size,
            "surface_area_before_m2": raw_area,
            "surface_area_after_m2": final_area,
            "surface_area_change_m2": area_change,
            "vertex_coordinate_maximum_difference_m": vertex_coordinate_maximum_difference,
            "checks": {
                "triangle_count_conserved": len(raw_triangles) - sanitation["removed_triangle_count"] == len(triangles),
                "vertex_count_and_coordinates_exactly_preserved": vertex_coordinate_maximum_difference == 0.0,
                "removed_surface_area_within_predicate_bound": abs(area_change) <= sanitation["removed_surface_area_sum_m2"] + 1e-9,
                "final_zero_degenerate_triangles": audit["degenerate_triangle_count"] == 0,
                "final_individual_mesh_contract_passed": audit["passed"] is True,
            },
        }
    )
    sanitation["passed"] = all(sanitation["checks"].values())
    write_json(destination / "sanitation.json", sanitation)
    write_json(destination / "mesh_audit.json", audit)
    primary.update(
        {
            "raw_native_mesh_sha256": raw_hash,
            "mesh_sha256": sanitation["final_obj_sha256"],
            "mesh_bytes": sanitation["final_obj_bytes"],
            "mesh_audit": audit,
            "sanitation": sanitation,
        }
    )
    write_json(destination / "materialization.json", primary)
    if sanitation["passed"] is not True:
        raise RuntimeError(f"{primary['parent_id']}: sanitation or final mesh quality failed")
    return primary, vertices


def execute(run_dir: Path) -> dict:
    started = time.monotonic()
    source_before = m0._source_precheck()
    parents = load_json(m0.SOURCE_V2R / "artifacts/accepted_parent_manifest.json")["parents"]
    strata = m0._stratum_registry()
    meshes_root = run_dir / "artifacts/meshes"
    maps_root = run_dir / "previews/train_complete_maps"
    meshes_root.mkdir(parents=True, exist_ok=False)
    maps_root.mkdir(parents=True, exist_ok=False)
    records = []

    for parent in parents:
        parent_id = parent["parent_id"]
        root = meshes_root / parent_id
        root.mkdir()
        primary, _, graph, splines = m0._materialize(
            parent,
            strata[parent["source_stratum_id"]],
            root / "primary",
            role="primary",
        )
        primary, final_vertices = _sanitize_materialization(root / "primary", primary, graph, splines)
        record = {
            "parent_id": parent_id,
            "recipe_stratum_id": parent["recipe_stratum_id"],
            "split": parent["split"],
            "source_parent": parent,
            "primary": primary,
            "immutable_asset": {"mesh_sha256": primary["mesh_sha256"], "remeshing_substitution_forbidden": True},
        }
        write_json(run_dir / "metrics" / f"{parent_id}.json", record)
        records.append(record)
        if parent["split"] == "train":
            m0._render_complete_train_map(
                parent,
                final_vertices,
                graph,
                splines,
                maps_root / f"{parent_id}_complete_xy_xz.png",
                stage_label="M1R",
            )

    batch = immutable_full_parent_asset_batch_audit(records, parents)
    sanitation_passed = all(item["primary"]["sanitation"]["passed"] is True for item in records)
    source_after = m0._source_precheck()
    passed = batch["passed"] and sanitation_passed
    status = "PASS_CANO_100_PARENT_PERCEPTION_MESH_M1R" if passed else "FAIL_CANO_100_PARENT_PERCEPTION_MESH_M1R"
    summary = {
        "schema_version": "cano_100_parent_perception_mesh_summary_m1r",
        "overall_status": status,
        "batch_audit": batch,
        "sanitation_batch": {
            "passed": sanitation_passed,
            "assets": len(records),
            "assets_with_removed_triangles": sum(item["primary"]["sanitation"]["removed_triangle_count"] > 0 for item in records),
            "removed_triangle_count": sum(item["primary"]["sanitation"]["removed_triangle_count"] for item in records),
            "removed_surface_area_sum_m2": sum(item["primary"]["sanitation"]["removed_surface_area_sum_m2"] for item in records),
        },
        "scope": EXPECTED_SCOPE,
        "parents": records,
        "source_before": source_before,
        "source_after": source_after,
        "duration_seconds": time.monotonic() - started,
        "asset_policy": "Downstream must read sealed M1R sanitized OBJ; remeshing substitution forbidden.",
        "split_policy": "Only 80 train parents are rendered with M1R provenance; validation and development-test receive automatic checks only.",
        "claim_boundary": "Sanitized immutable perception assets only; zero LiDAR, labels, formal data, training, simulator or M-TARE change.",
    }
    write_json(run_dir / "artifacts/mesh_manifest.json", {"asset_policy": summary["asset_policy"], "split_policy": summary["split_policy"], "parents": records})
    write_json(run_dir / "previews/provenance.json", {"stage_label": "M1R TRAIN ONLY", "rendered_split": "train_only", "displayed_parent_ids": [item["parent_id"] for item in records if item["split"] == "train"], "train_rendered": 80, "validation_rendered": 0, "development_test_rendered": 0, "units": "meters"})
    write_json(run_dir / "metrics/summary.json", summary)
    if not passed:
        raise RuntimeError("M1R batch audit failed")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = execute(args.run_dir.resolve())
    except Exception as exc:
        write_json(args.run_dir.resolve() / "metrics/executor_failure.json", {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
