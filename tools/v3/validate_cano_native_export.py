#!/usr/bin/env python3
"""Validate one original Cano native export without repairing its outputs."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _rounded_axis_conflicts(axis: np.ndarray) -> dict[str, float | int]:
    tunnel_rows = axis[np.isclose(axis[:, 7], 1.0)]
    coordinate_ids: dict[tuple[float, float, float], set[int]] = {}
    for row in tunnel_rows:
        key = tuple(np.round(row[:3], decimals=6).tolist())
        coordinate_ids.setdefault(key, set()).add(int(round(row[8])))
    conflict_coordinates = {
        key for key, tunnel_ids in coordinate_ids.items() if len(tunnel_ids) > 1
    }
    conflict_rows = sum(
        tuple(np.round(row[:3], decimals=6).tolist()) in conflict_coordinates
        for row in tunnel_rows
    )
    return {
        "tunnel_axis_rows": int(len(tunnel_rows)),
        "unique_tunnel_axis_coordinates": int(len(coordinate_ids)),
        "cross_tunnel_label_conflict_coordinates": int(len(conflict_coordinates)),
        "cross_tunnel_label_conflict_rows": int(conflict_rows),
        "cross_tunnel_label_conflict_row_fraction": (
            float(conflict_rows / len(tunnel_rows)) if len(tunnel_rows) else math.nan
        ),
    }


def _render_preview(
    vertices: np.ndarray,
    axis: np.ndarray,
    output: Path,
    run_id: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(vertices) > 100_000:
        indices = np.linspace(0, len(vertices) - 1, 100_000, dtype=int)
        shown_vertices = vertices[indices]
    else:
        shown_vertices = vertices
    tunnel_axis = axis[np.isclose(axis[:, 7], 1.0)]
    intersection_axis = axis[np.isclose(axis[:, 7], 2.0)]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    projections = ((0, 1, "X–Y"), (0, 2, "X–Z"), (1, 2, "Y–Z"))
    for panel, (first, second, label) in zip(axes, projections):
        panel.scatter(
            shown_vertices[:, first],
            shown_vertices[:, second],
            s=0.15,
            c="#777777",
            alpha=0.15,
            rasterized=True,
            label="mesh vertices",
        )
        if len(tunnel_axis):
            panel.scatter(
                tunnel_axis[:, first],
                tunnel_axis[:, second],
                s=2.0,
                c="#1565c0",
                alpha=0.45,
                rasterized=True,
                label="axis flag=tunnel",
            )
        if len(intersection_axis):
            panel.scatter(
                intersection_axis[:, first],
                intersection_axis[:, second],
                s=3.0,
                c="#d32f2f",
                alpha=0.55,
                rasterized=True,
                label="axis flag=intersection",
            )
        panel.set_title(label)
        panel.set_xlabel("XYZ"[first] + " (source units, expected m)")
        panel.set_ylabel("XYZ"[second] + " (source units, expected m)")
        panel.set_aspect("equal", adjustable="box")
        panel.grid(alpha=0.2)
    axes[0].legend(loc="best", markerscale=3)
    figure.suptitle(
        f"{run_id} | original Cano native mesh and axis overlay\n"
        "Diagnostic only: does not establish topology-label validity",
        fontsize=11,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-dir", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    world = args.world_dir.resolve()
    required = {
        name: world / name
        for name in ("mesh.obj", "axis.txt", "fta_dist.txt", "model.sdf")
    }
    files_present = {name: path.is_file() for name, path in required.items()}
    result: dict[str, object] = {
        "schema_version": "cano_native_export_validation_v1",
        "run_id": args.run_id,
        "world_dir": str(world),
        "files_present": files_present,
        "requested_environment_count": 1,
        "formal_dataset_world_count": 0,
    }
    if not all(files_present.values()):
        result.update(
            overall_status="FAIL_MISSING_FILES",
            acceptance={"native_files": "FAIL"},
        )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    mesh = o3d.io.read_triangle_mesh(str(required["mesh.obj"]))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    mesh_readable = (
        vertices.ndim == 2
        and vertices.shape[1:] == (3,)
        and len(vertices) > 0
        and triangles.ndim == 2
        and triangles.shape[1:] == (3,)
        and len(triangles) > 0
        and np.isfinite(vertices).all()
    )
    if mesh_readable:
        triangle_components, component_counts, _ = mesh.cluster_connected_triangles()
        component_count = len(component_counts)
    else:
        triangle_components = []
        component_count = 0

    axis = np.loadtxt(required["axis.txt"])
    if axis.ndim == 1:
        axis = axis.reshape(1, -1)
    axis_schema_valid = (
        axis.ndim == 2
        and axis.shape[1] == 9
        and len(axis) > 0
        and np.isfinite(axis).all()
    )
    conflict_metrics = _rounded_axis_conflicts(axis) if axis_schema_valid else {}
    unique_tunnel_ids = (
        int(len(np.unique(axis[np.isclose(axis[:, 7], 1.0), 8])))
        if axis_schema_valid
        else 0
    )
    conflict_count = int(
        conflict_metrics.get("cross_tunnel_label_conflict_coordinates", 0)
    )
    axis_tunnel_gt_status = (
        "PASS"
        if axis_schema_valid and unique_tunnel_ids >= 2 and conflict_count == 0
        else "FAIL"
        if axis_schema_valid and unique_tunnel_ids >= 2 and conflict_count > 0
        else "UNVERIFIED_INSUFFICIENT_TUNNELS"
    )

    fta_values = np.atleast_1d(np.loadtxt(required["fta_dist.txt"]))
    fta_valid = len(fta_values) == 1 and np.isfinite(fta_values).all()
    sdf_root = ET.parse(required["model.sdf"]).getroot()
    sdf_uris = [element.text for element in sdf_root.findall(".//uri")]
    sdf_valid = len(sdf_uris) == 2 and all(
        Path(uri).resolve() == required["mesh.obj"].resolve() for uri in sdf_uris
    )

    acceptance = {
        "native_files": "PASS",
        "mesh_readable_nonempty": _status(mesh_readable),
        "axis_schema": _status(axis_schema_valid),
        "axis_per_tunnel_ground_truth": axis_tunnel_gt_status,
        "sdf_mesh_reference": _status(sdf_valid),
        "fta_scalar": _status(fta_valid),
        "topology_graph_export": "FAIL_NOT_EXPORTED",
        "generator_return_success_audit": "FAIL_NOT_EXPORTED",
        "seed_replay": "FAIL_NOT_SUPPORTED_BY_ORIGINAL_ENTRYPOINT",
    }
    overall = (
        "PARTIAL_NATIVE_MESH_PASS_GT_FAIL"
        if mesh_readable and sdf_valid and fta_valid
        else "FAIL_NATIVE_EXPORT"
    )
    result.update(
        overall_status=overall,
        acceptance=acceptance,
        mesh={
            "vertices": int(len(vertices)),
            "triangles": int(len(triangles)),
            "triangle_components": int(component_count),
            "extent_min": vertices.min(axis=0).tolist() if len(vertices) else None,
            "extent_max": vertices.max(axis=0).tolist() if len(vertices) else None,
            "edge_manifold_allow_boundary": bool(mesh.is_edge_manifold(True)),
            "edge_manifold_no_boundary": bool(mesh.is_edge_manifold(False)),
            "vertex_manifold": bool(mesh.is_vertex_manifold()),
            "watertight": bool(mesh.is_watertight()),
            "orientable": bool(mesh.is_orientable()),
            "self_intersecting": bool(mesh.is_self_intersecting()),
        },
        axis={
            "rows": int(len(axis)),
            "columns": int(axis.shape[1]) if axis.ndim == 2 else None,
            "unique_tunnel_ids_on_tunnel_rows": unique_tunnel_ids,
            **conflict_metrics,
        },
        fta_distance=float(fta_values[0]) if fta_valid else None,
        sdf_uris=sdf_uris,
        claim_boundary=(
            "This validation can establish only original native file emission and mesh "
            "readability. It cannot recover the unexported generator return values or "
            "topology graph, and it rejects axis.txt as per-tunnel GT when duplicated "
            "coordinates carry conflicting tunnel IDs."
        ),
    )
    _render_preview(vertices, axis, args.preview, args.run_id)
    result["preview"] = {
        "path": str(args.preview.resolve()),
        "method": "Deterministic full-export diagnostic projections with at most 100000 evenly sampled mesh vertices and all axis rows",
        "supports": "Visual audit of exported mesh extent and axis alignment",
        "does_not_support": "Topology-label validity, graph-mesh consistency, navigation feasibility, Isaac rendering, or dataset quality",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if overall.startswith("PARTIAL_NATIVE_MESH_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
