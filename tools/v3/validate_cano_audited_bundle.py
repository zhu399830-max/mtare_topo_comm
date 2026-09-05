#!/usr/bin/env python3
"""Strictly validate one read-only Cano adapter bundle and render its full extent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


REQUIRED_FILES = (
    "mesh.obj",
    "graph.json",
    "splines.json",
    "axis.txt",
    "world_info.json",
    "generation_audit.json",
    "metadata.json",
    "fta_dist.txt",
    "model.sdf",
)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar containers without changing computed values."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_conflicts(axis: np.ndarray) -> dict[str, Any]:
    interior = axis[np.isclose(axis[:, 7], 1.0)]
    coordinate_ids: dict[tuple[float, float, float], set[int]] = {}
    rows_per_tunnel: dict[int, int] = {}
    for row in interior:
        key = tuple(np.round(row[:3], 6).tolist())
        tunnel_id = int(round(row[8]))
        coordinate_ids.setdefault(key, set()).add(tunnel_id)
        rows_per_tunnel[tunnel_id] = rows_per_tunnel.get(tunnel_id, 0) + 1
    conflict_coordinates = {
        key for key, tunnel_ids in coordinate_ids.items() if len(tunnel_ids) > 1
    }
    conflict_rows = sum(
        tuple(np.round(row[:3], 6).tolist()) in conflict_coordinates
        for row in interior
    )
    return {
        "interior_rows": int(len(interior)),
        "unique_interior_coordinates": int(len(coordinate_ids)),
        "cross_tunnel_conflict_coordinates": int(len(conflict_coordinates)),
        "cross_tunnel_conflict_rows": int(conflict_rows),
        "cross_tunnel_conflict_fraction": (
            float(conflict_rows / len(interior)) if len(interior) else math.nan
        ),
        "interior_rows_per_tunnel": {
            str(key): value for key, value in sorted(rows_per_tunnel.items())
        },
    }


def _graph_audit(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    tunnels = graph.get("tunnels", [])
    node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    unique_nodes = len(node_ids) == len(set(node_ids)) == len(nodes)
    adjacency = {node_id: set() for node_id in node_ids}
    references_valid = True
    edge_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or not isinstance(edge.get("node_ids"), list):
            references_valid = False
            continue
        pair = edge["node_ids"]
        if len(pair) != 2 or pair[0] not in adjacency or pair[1] not in adjacency:
            references_valid = False
            continue
        canonical = tuple(sorted((pair[0], pair[1])))
        if canonical in edge_pairs or pair[0] == pair[1]:
            references_valid = False
            continue
        edge_pairs.add(canonical)
        adjacency[pair[0]].add(pair[1])
        adjacency[pair[1]].add(pair[0])
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            unseen = adjacency[current] & remaining
            remaining.difference_update(unseen)
            stack.extend(unseen)
    cycle_rank = len(edge_pairs) - len(nodes) + components if nodes else 0
    tunnel_ids = [
        tunnel.get("id") for tunnel in tunnels if isinstance(tunnel, dict)
    ]
    tunnel_nodes_valid = all(
        isinstance(tunnel, dict)
        and isinstance(tunnel.get("node_ids"), list)
        and len(tunnel["node_ids"]) >= 2
        and all(node_id in adjacency for node_id in tunnel["node_ids"])
        for tunnel in tunnels
    )
    type_counts: dict[str, int] = {}
    for tunnel in tunnels:
        if isinstance(tunnel, dict):
            tunnel_type = str(tunnel.get("type"))
            type_counts[tunnel_type] = type_counts.get(tunnel_type, 0) + 1
    statistics = graph.get("statistics", {})
    statistics_match = statistics == {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "tunnel_count": len(tunnels),
        "intersection_count": len(graph.get("intersections", [])),
        "connected_components": components,
        "cycle_rank": cycle_rank,
    }
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "tunnel_count": len(tunnels),
        "tunnel_ids": sorted(tunnel_ids),
        "tunnel_type_counts": type_counts,
        "connected_components": components,
        "cycle_rank": cycle_rank,
        "unique_node_ids": unique_nodes,
        "edge_references_valid": references_valid,
        "tunnel_node_references_valid": tunnel_nodes_valid,
        "statistics_match": statistics_match,
        "pass": bool(
            nodes
            and unique_nodes
            and references_valid
            and tunnel_nodes_valid
            and components == 1
            and cycle_rank == 1
            and len(tunnels) == 4
            and type_counts == {"connector": 1, "grown": 3}
            and statistics_match
        ),
    }


def _spline_audit(graph: dict[str, Any], splines: dict[str, Any]) -> dict[str, Any]:
    node_xyz = {
        node["id"]: np.asarray(node["xyz"], dtype=float) for node in graph["nodes"]
    }
    graph_tunnels = {int(tunnel["id"]): tunnel for tunnel in graph["tunnels"]}
    records = splines.get("tunnels", [])
    endpoint_errors = []
    finite = True
    nonempty = True
    spline_ids = []
    total_points = 0
    for record in records:
        tunnel_id = int(record["tunnel_id"])
        spline_ids.append(tunnel_id)
        points = np.asarray(record.get("points", []), dtype=float)
        total_points += len(points)
        finite = finite and points.ndim == 2 and points.shape[1:] == (3,) and np.isfinite(points).all()
        nonempty = nonempty and len(points) >= 2
        graph_tunnel = graph_tunnels.get(tunnel_id)
        if graph_tunnel is None or len(points) < 2:
            endpoint_errors.append(math.inf)
            continue
        expected = graph_tunnel["endpoint_node_ids"]
        endpoint_errors.extend(
            [
                float(np.linalg.norm(points[0] - node_xyz[expected[0]])),
                float(np.linalg.norm(points[-1] - node_xyz[expected[1]])),
            ]
        )
    max_error = max(endpoint_errors, default=math.inf)
    return {
        "spline_count": len(records),
        "total_points": total_points,
        "tunnel_ids": sorted(spline_ids),
        "finite": finite,
        "nonempty": nonempty,
        "max_endpoint_error": max_error,
        "pass": bool(
            len(records) == len(graph_tunnels) == 4
            and sorted(spline_ids) == sorted(graph_tunnels)
            and finite
            and nonempty
            and max_error <= 1e-6
        ),
    }


def _render_preview(
    vertices: np.ndarray,
    splines: dict[str, Any],
    output: Path,
    run_id: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shown_vertices = vertices
    if len(vertices) > 100_000:
        indices = np.linspace(0, len(vertices) - 1, 100_000, dtype=int)
        shown_vertices = vertices[indices]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    projections = ((0, 1, "X–Y"), (0, 2, "X–Z"), (1, 2, "Y–Z"))
    colors = plt.get_cmap("tab10")
    for panel, (first, second, label) in zip(axes, projections):
        panel.scatter(
            shown_vertices[:, first],
            shown_vertices[:, second],
            s=0.15,
            c="#777777",
            alpha=0.14,
            rasterized=True,
            label="mesh vertices",
        )
        for index, record in enumerate(splines["tunnels"]):
            points = np.asarray(record["points"], dtype=float)
            panel.plot(
                points[:, first],
                points[:, second],
                linewidth=1.4,
                color=colors(index % 10),
                label=f"tunnel {record['tunnel_id']} ({record['type']})" if panel is axes[0] else None,
            )
        panel.set_title(label)
        panel.set_xlabel("XYZ"[first] + " (source units, expected m)")
        panel.set_ylabel("XYZ"[second] + " (source units, expected m)")
        panel.set_aspect("equal", adjustable="box")
        panel.grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=7, markerscale=3)
    figure.suptitle(
        f"{run_id} | fixed-seed audited adapter: actual mesh + per-tunnel spline\n"
        "Acceptance diagnostic; not an Isaac rendering or navigation proof",
        fontsize=11,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    presence = {name: (bundle / name).is_file() for name in REQUIRED_FILES}
    if not all(presence.values()):
        result = {
            "schema_version": "cano_audited_bundle_validation_v1",
            "run_id": args.run_id,
            "overall_status": "FAIL_ADAPTER_MISSING_FILES",
            "files_present": presence,
            "formal_dataset_world_count": 0,
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    graph = _load_json(bundle / "graph.json")
    splines = _load_json(bundle / "splines.json")
    metadata = _load_json(bundle / "metadata.json")
    generation = _load_json(bundle / "generation_audit.json")
    world_info = _load_json(bundle / "world_info.json")
    graph_metrics = _graph_audit(graph)
    spline_metrics = _spline_audit(graph, splines)

    axis = np.loadtxt(bundle / "axis.txt")
    if axis.ndim == 1:
        axis = axis.reshape(1, -1)
    axis_schema = bool(
        axis.ndim == 2
        and axis.shape[1] == 9
        and len(axis)
        and np.isfinite(axis).all()
    )
    axis_metrics = _axis_conflicts(axis) if axis_schema else {}
    graph_tunnel_ids = set(graph_metrics["tunnel_ids"])
    axis_tunnel_ids = set(int(round(value)) for value in axis[:, 8]) if axis_schema else set()
    axis_pass = bool(
        axis_schema
        and axis_metrics["cross_tunnel_conflict_coordinates"] == 0
        and axis_tunnel_ids == graph_tunnel_ids
        and all(
            int(axis_metrics["interior_rows_per_tunnel"].get(str(tunnel_id), 0)) > 0
            for tunnel_id in graph_tunnel_ids
        )
    )

    mesh = o3d.io.read_triangle_mesh(str(bundle / "mesh.obj"))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    mesh_readable = bool(
        vertices.ndim == 2
        and vertices.shape[1:] == (3,)
        and len(vertices)
        and triangles.ndim == 2
        and triangles.shape[1:] == (3,)
        and len(triangles)
        and np.isfinite(vertices).all()
    )
    if mesh_readable:
        _, component_counts, component_areas = mesh.cluster_connected_triangles()
        component_counts_list = [int(value) for value in component_counts]
        component_areas_list = [float(value) for value in component_areas]
    else:
        component_counts_list = []
        component_areas_list = []
    mesh_metrics = {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "components": len(component_counts_list),
        "component_triangle_counts_desc": sorted(component_counts_list, reverse=True),
        "component_areas_desc": sorted(component_areas_list, reverse=True),
        "edge_manifold_allow_boundary": bool(mesh.is_edge_manifold(True)),
        "edge_manifold_no_boundary": bool(mesh.is_edge_manifold(False)),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "watertight": bool(mesh.is_watertight()),
        "orientable": bool(mesh.is_orientable()),
        "self_intersecting": bool(mesh.is_self_intersecting()),
        "extent_min": vertices.min(axis=0).tolist() if len(vertices) else None,
        "extent_max": vertices.max(axis=0).tolist() if len(vertices) else None,
    }
    mesh_pass = bool(
        mesh_readable
        and mesh_metrics["components"] == 1
        and mesh_metrics["edge_manifold_no_boundary"]
        and mesh_metrics["vertex_manifold"]
        and mesh_metrics["watertight"]
        and mesh_metrics["orientable"]
        and not mesh_metrics["self_intersecting"]
    )

    expected_hashes = metadata.get("files", {})
    hash_checks = {
        name: _sha256(bundle / name) == value.get("sha256")
        for name, value in expected_hashes.items()
        if name != "metadata.json" and (bundle / name).is_file()
    }
    provenance_pass = bool(
        metadata.get("formal_dataset_world_count") == 0
        and metadata.get("source_unchanged") is True
        and metadata.get("seed") == 0
        and metadata.get("random_controls", {}).get("numpy_random_seed") == 0
        and metadata.get("random_controls", {}).get("python_random_seed") == 0
        and metadata.get("random_controls", {}).get("python_hash_seed") == "0"
        and metadata.get("adapter", {}).get("upstream_core_modified") is False
        and hash_checks
        and all(hash_checks.values())
    )
    calls = generation.get("calls", [])
    generation_pass = bool(
        generation.get("overall_status") == "PASS_ALL_REQUESTED_TUNNELS_CONFIRMED"
        and generation.get("requested") == {"grown": 3, "connector": 1}
        and generation.get("succeeded") == {"grown": 3, "connector": 1}
        and len(calls) == 4
        and all(call.get("success") is True and call.get("return_schema") == "tuple_v1" for call in calls)
    )
    world_info_pass = bool(
        isinstance(world_info.get("tunnels_info"), list)
        and len(world_info["tunnels_info"]) == 4
        and isinstance(world_info.get("intersections_info"), list)
    )
    fta = np.atleast_1d(np.loadtxt(bundle / "fta_dist.txt"))
    fta_pass = bool(len(fta) == 1 and np.isfinite(fta).all())
    sdf_uris = [item.text for item in ET.parse(bundle / "model.sdf").findall(".//uri")]
    sdf_pass = bool(
        len(sdf_uris) == 2
        and all(Path(uri).resolve() == (bundle / "mesh.obj").resolve() for uri in sdf_uris)
    )

    contract_pass = bool(
        graph_metrics["pass"]
        and spline_metrics["pass"]
        and axis_pass
        and generation_pass
        and provenance_pass
        and world_info_pass
        and fta_pass
        and sdf_pass
    )
    if contract_pass and mesh_pass:
        overall = "PASS_ADAPTER_SINGLE_WORLD_SMOKE"
    elif contract_pass:
        overall = "FAIL_ADAPTER_MESH_GATE"
    else:
        overall = "FAIL_ADAPTER_CONTRACT"

    _render_preview(vertices, splines, args.preview, args.run_id)
    result = {
        "schema_version": "cano_audited_bundle_validation_v1",
        "run_id": args.run_id,
        "overall_status": overall,
        "files_present": presence,
        "generation_return_audit": {"pass": generation_pass, "calls": calls},
        "graph": graph_metrics,
        "splines": spline_metrics,
        "axis": {"schema_pass": axis_schema, "pass": axis_pass, **axis_metrics},
        "mesh": {"readable": mesh_readable, "strict_pass": mesh_pass, **mesh_metrics},
        "world_info_pass": world_info_pass,
        "provenance": {"pass": provenance_pass, "file_hash_checks": hash_checks},
        "fta_pass": fta_pass,
        "sdf": {"pass": sdf_pass, "uris": sdf_uris},
        "formal_dataset_world_count": 0,
        "preview": {
            "path": str(args.preview.resolve()),
            "supports": "Complete actual mesh extent and per-tunnel spline alignment audit",
            "does_not_support": "Isaac rendering, dynamic navigation, LiDAR, labels, dataset sufficiency, or learning readiness",
        },
        "claim_boundary": (
            "PASS would authorize only a later deterministic-replay specification. This "
            "one-world smoke never creates a formal dataset world and does not authorize "
            "batch generation, Isaac/LiDAR, labels, or training."
        ),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if overall == "PASS_ADAPTER_SINGLE_WORLD_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
