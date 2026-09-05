#!/usr/bin/env python3
"""Execute the approved train-only Cano perception-mesh M0 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from execute_cano_100_topology_parent_candidate_audit_v1 import (
    _reset,
    _spline_records,
    _stable_graph,
)
from execute_cano_100_topology_parent_candidate_audit_v2 import (
    _attempt_requested_tunnel,
)
from generate_cano_audited_bundle import _per_tunnel_axis, _source_precheck
from mtare_topo.data.cano_perception_mesh_contract import (
    batch_mesh_audit,
    canonical_json_hash,
    mesh_array_audit,
    replay_pair_audit,
    select_train_sentinels,
)
from mtare_topo.data.cano_topology_parent_audit import canonical_parent_identity
from mtare_topo.governance import load_json, write_json
from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenParams,
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
)
from subt_proc_gen.tunnel import TunnelNetwork, TunnelNetworkParams


SOURCE_V2R = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0"
)
SOURCE_V2 = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0"
)
V2_PROPOSAL = PROJECT_ROOT / (
    "configs/v3/gate0/"
    "cano_100_topology_parent_candidate_audit_v2_bounded_resampling.proposal.json"
)
EXPECTED_SCOPE = {
    "topology_parents_selected": 10,
    "same_seed_topology_reconstructions": 20,
    "native_perception_meshes": 20,
    "primary_meshes": 10,
    "replay_meshes": 10,
    "train_complete_previews": 10,
    "validation_or_development_test_parents_read": 0,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratum_registry() -> dict[str, dict[str, Any]]:
    proposal = load_json(V2_PROPOSAL)
    return {
        str(item["stratum_id"]): dict(item)
        for item in proposal["frozen_candidate_scope"]["strata"]
    }


def reconstruct_parent_network(
    parent: dict[str, Any], stratum: dict[str, Any]
) -> tuple[TunnelNetwork, dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[Any, str]]:
    """Reconstruct one frozen V2 parent and retain the network for meshing."""

    _reset(int(parent["topology_seed"]))
    flat = stratum["dimensionality"] == "flat"
    params = TunnelNetworkParams.from_defaults()
    params.collision_distance = 10.0
    params.min_distance_between_intersections = 30.0
    params.min_intersection_angle = np.deg2rad(30.0)
    params.max_inclination = np.deg2rad(30.0)
    params.flat = flat
    network = TunnelNetwork(params=params)
    operations: list[dict[str, Any]] = []
    for operation, requested_count in (
        ("grown", int(stratum["grown_tunnels"])),
        ("connector", int(stratum["connector_tunnels"])),
    ):
        for index in range(requested_count):
            outcome = _attempt_requested_tunnel(
                network, operation=operation, index=index, flat=flat
            )
            operations.append(outcome)
            if not outcome["success"]:
                raise RuntimeError(
                    f"{parent['parent_id']}: sealed topology reconstruction failed at "
                    f"{operation}[{index}]"
                )
    graph, node_ids = _stable_graph(network)
    splines = _spline_records(network, node_ids)
    return network, graph, splines, operations, node_ids


def _perlin_record(params: Any) -> dict[str, Any]:
    return {
        "res": int(params.res),
        "octaves": int(params.octaves),
        "persistence": float(params.persistence),
        "lacunarity": int(params.lacunarity),
    }


def _effective_geometry_parameters(
    generator: TunnelNetworkMeshGenerator,
    node_ids: dict[Any, str],
    *,
    geometry_seed: int,
    fta_distance_m: float,
) -> dict[str, Any]:
    tunnels = []
    for tunnel in sorted(generator.tunnels, key=lambda item: item.tunnel_id):
        params = generator.ptcl_params_of_tunnel(tunnel)
        tunnels.append(
            {
                "tunnel_id": int(tunnel.tunnel_id),
                "dist_between_circles_m": float(params.dist_between_circles),
                "n_points_per_circle": int(params.n_points_per_circle),
                "radius_m": float(params.radius),
                "noise_multiplier": float(params.noise_multiplier),
                "perlin": _perlin_record(params.perlin_params),
            }
        )
    intersections = []
    for node in sorted(generator.intersections, key=lambda item: node_ids[item]):
        params = generator.params_of_intersection(node)
        intersections.append(
            {
                "node_id": node_ids[node],
                "radius_m": float(params.radius),
                "pointcloud_type": params.ptcl_type.name,
                "points_per_square_meter": int(params.points_per_sm),
                "noise_multiplier": float(params.noise_multiplier),
                "perlin": _perlin_record(params.perlin_params),
            }
        )
    mesh_params = generator._meshing_params
    return {
        "geometry_seed": int(geometry_seed),
        "fta_distance_m": float(fta_distance_m),
        "pointcloud_strategy": generator._ptcl_gen_params.strategy.name,
        "tunnels": tunnels,
        "intersections": intersections,
        "mesh": {
            "poisson_depth": int(mesh_params.poisson_depth),
            "simplification_voxel_size_m": float(mesh_params.simplification_voxel_size),
            "voxelization_voxel_size_m": float(mesh_params.voxelization_voxel_size),
            "floor_smoothing_iterations": int(mesh_params.floor_smoothing_iter),
            "floor_smoothing_radius_m": float(mesh_params.floor_smoothing_r),
        },
    }


def _source_spline_points(splines: dict[str, Any]) -> np.ndarray:
    return np.vstack(
        [np.asarray(item["points"], dtype=np.float64) for item in splines["tunnels"]]
    )


def _materialize(
    parent: dict[str, Any],
    stratum: dict[str, Any],
    destination: Path,
    *,
    role: str,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=False)
    network, graph, splines, operations, node_ids = reconstruct_parent_network(parent, stratum)
    source_graph = load_json(PROJECT_ROOT / parent["source_graph"])
    source_splines = load_json(PROJECT_ROOT / parent["source_splines"])
    graph_identity = canonical_json_hash(graph)
    spline_identity = canonical_json_hash(splines)
    source_graph_identity = canonical_json_hash(source_graph)
    source_spline_identity = canonical_json_hash(source_splines)
    parent_identity = canonical_parent_identity(graph, splines)
    source_metric = load_json(SOURCE_V2 / "metrics" / f"{parent['parent_id']}.json")
    identity_checks = {
        "graph_matches_sealed_source": graph_identity == source_graph_identity,
        "splines_match_sealed_source": spline_identity == source_spline_identity,
        "parent_identity_matches_v2r": parent_identity == parent["canonical_parent_identity"],
        "operation_trace_matches_sealed_source": (
            canonical_json_hash(operations) == canonical_json_hash(source_metric["operations"])
        ),
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"{parent['parent_id']}/{role}: topology identity mismatch {identity_checks}")

    geometry_seed = int(parent["reserved_geometry_seed"])
    np.random.seed(geometry_seed)
    random.seed(geometry_seed)
    fta_distance = float(np.random.uniform(-2.0, -1.0))
    pointcloud_params = TunnelNetworkPtClGenParams.random()
    mesh_params = TunnelNetworkMeshGenParams.from_defaults()
    mesh_params.fta_distance = fta_distance
    generator = TunnelNetworkMeshGenerator(
        network, ptcl_gen_params=pointcloud_params, meshing_params=mesh_params
    )
    print(f"[{parent['parent_id']}/{role}] compute_all start", flush=True)
    generator.compute_all()
    mesh_path = destination / "mesh.obj"
    generator.save_mesh(str(mesh_path))
    axis = _per_tunnel_axis(generator)
    np.save(destination / "axis.npy", axis, allow_pickle=False)
    write_json(destination / "graph.json", graph)
    write_json(destination / "splines.json", splines)

    effective = _effective_geometry_parameters(
        generator, node_ids, geometry_seed=geometry_seed, fta_distance_m=fta_distance
    )
    write_json(destination / "geometry_parameters.json", effective)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    _, component_counts, component_areas = mesh.cluster_connected_triangles()
    source_tunnel_ids = [int(item["id"]) for item in graph["tunnels"]]
    mesh_audit = mesh_array_audit(
        vertices,
        triangles,
        list(component_counts),
        axis,
        source_tunnel_ids,
        _source_spline_points(splines),
    )
    mesh_audit["edge_manifold_allow_boundary"] = bool(mesh.is_edge_manifold(True))
    mesh_audit["edge_manifold_no_boundary"] = bool(mesh.is_edge_manifold(False))
    mesh_audit["vertex_manifold"] = bool(mesh.is_vertex_manifold())
    mesh_audit["watertight"] = bool(mesh.is_watertight())
    mesh_audit["orientable"] = bool(mesh.is_orientable())
    mesh_audit["component_surface_areas_m2"] = [float(value) for value in component_areas]
    write_json(destination / "mesh_audit.json", mesh_audit)
    duration = time.monotonic() - started
    record = {
        "parent_id": parent["parent_id"],
        "role": role,
        "split": parent["split"],
        "topology_seed": int(parent["topology_seed"]),
        "geometry_seed": geometry_seed,
        "identity_checks": identity_checks,
        "graph_identity": graph_identity,
        "spline_identity": spline_identity,
        "parent_identity": parent_identity,
        "operation_trace_sha256": canonical_json_hash(operations),
        "effective_geometry_parameter_sha256": canonical_json_hash(effective),
        "mesh_sha256": _sha256(mesh_path),
        "mesh_bytes": mesh_path.stat().st_size,
        "mesh_audit": mesh_audit,
        "duration_seconds": duration,
    }
    write_json(destination / "materialization.json", record)
    print(
        f"[{parent['parent_id']}/{role}] done triangles={len(triangles)} "
        f"bytes={record['mesh_bytes']} duration={duration:.2f}s passed={mesh_audit['passed']}",
        flush=True,
    )
    return record, vertices, graph, splines


def _render_complete_train_map(
    parent: dict[str, Any],
    vertices: np.ndarray,
    graph: dict[str, Any],
    splines: dict[str, Any],
    destination: Path,
    *,
    stage_label: str = "M0",
) -> None:
    limit = min(len(vertices), 50000)
    indices = np.linspace(0, len(vertices) - 1, limit, dtype=np.int64)
    sampled = vertices[indices]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].scatter(sampled[:, 0], sampled[:, 1], s=0.15, c="#8c8c8c", alpha=0.25)
    axes[1].scatter(sampled[:, 0], sampled[:, 2], s=0.15, c="#8c8c8c", alpha=0.25)
    for tunnel in splines["tunnels"]:
        points = np.asarray(tunnel["points"], dtype=np.float64)
        color = "#d95f02" if tunnel["type"] == "connector" else "#0072b2"
        axes[0].plot(points[:, 0], points[:, 1], color=color, linewidth=1.2)
        axes[1].plot(points[:, 0], points[:, 2], color=color, linewidth=1.2)
    nodes = np.asarray([item["xyz"] for item in graph["nodes"]], dtype=np.float64)
    degree = np.asarray([item["degree"] for item in graph["nodes"]])
    event = degree >= 3
    axes[0].scatter(nodes[:, 0], nodes[:, 1], s=8, c="#111111", zorder=3)
    axes[1].scatter(nodes[:, 0], nodes[:, 2], s=8, c="#111111", zorder=3)
    axes[0].scatter(nodes[event, 0], nodes[event, 1], s=35, c="#cc0000", marker="x", zorder=4)
    axes[1].scatter(nodes[event, 0], nodes[event, 2], s=35, c="#cc0000", marker="x", zorder=4)
    axes[0].set_title("Complete X-Y | gray mesh sample, blue grown, orange connector")
    axes[1].set_title("Complete X-Z | red x = degree >= 3 event")
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[1].set_ylabel("z [m]")
    fig.suptitle(
        f"{stage_label} TRAIN ONLY | {parent['parent_id']} | {parent['recipe_stratum_id']} | "
        f"topology_seed={parent['topology_seed']} geometry_seed={parent['reserved_geometry_seed']}"
    )
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def execute(run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    source_before = _source_precheck()
    parents_document = load_json(SOURCE_V2R / "artifacts/accepted_parent_manifest.json")
    sentinels = select_train_sentinels(parents_document["parents"])
    strata = _stratum_registry()
    meshes_root = run_dir / "artifacts/meshes"
    maps_root = run_dir / "previews/train_complete_maps"
    meshes_root.mkdir(parents=True, exist_ok=False)
    maps_root.mkdir(parents=True, exist_ok=False)
    parent_records = []
    for parent in sentinels:
        stratum = strata[parent["source_stratum_id"]]
        parent_root = meshes_root / parent["parent_id"]
        parent_root.mkdir()
        primary, vertices, graph, splines = _materialize(
            parent, stratum, parent_root / "primary", role="primary"
        )
        replay, _, _, _ = _materialize(
            parent, stratum, parent_root / "replay", role="replay"
        )
        replay_audit = replay_pair_audit(primary, replay)
        record = {
            "parent_id": parent["parent_id"],
            "recipe_stratum_id": parent["recipe_stratum_id"],
            "split": parent["split"],
            "source_parent": parent,
            "primary": primary,
            "replay": replay,
            "replay_audit": replay_audit,
        }
        write_json(run_dir / "metrics" / f"{parent['parent_id']}.json", record)
        parent_records.append(record)
        _render_complete_train_map(
            parent,
            vertices,
            graph,
            splines,
            maps_root / f"{parent['parent_id']}_complete_xy_xz.png",
        )
        if not replay_audit["passed"]:
            raise RuntimeError(f"{parent['parent_id']}: exact geometry replay failed: {replay_audit}")

    batch = batch_mesh_audit(parent_records)
    source_after = _source_precheck()
    scope = dict(EXPECTED_SCOPE)
    summary = {
        "schema_version": "cano_100_parent_perception_mesh_contract_summary_m0",
        "overall_status": (
            "PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0"
            if batch["passed"]
            else "FAIL_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0"
        ),
        "batch_audit": batch,
        "scope": scope,
        "parents": parent_records,
        "source_before": source_before,
        "source_after": source_after,
        "duration_seconds": time.monotonic() - started,
        "claim_boundary": (
            "Train-only native perception-mesh materialization/replay contract; not a "
            "formal dataset, LiDAR test, navigation mesh, learned model or M-TARE change."
        ),
    }
    write_json(run_dir / "artifacts/mesh_manifest.json", {"parents": parent_records})
    write_json(
        run_dir / "previews/provenance.json",
        {
            "split": "train_only",
            "selection": "accepted_rank_in_recipe=1 fixed before meshing",
            "displayed_parent_ids": [item["parent_id"] for item in sentinels],
            "validation_or_development_test_rendered": 0,
            "hypothesis": "Native Cano meshes remain complete and aligned with sealed graph/splines across ten random recipe sentinels.",
            "units": "meters",
        },
    )
    write_json(run_dir / "metrics/summary.json", summary)
    if not batch["passed"]:
        raise RuntimeError(f"M0 batch audit failed: {batch}")
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
