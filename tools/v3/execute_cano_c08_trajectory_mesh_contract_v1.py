#!/usr/bin/env python3
"""Execute the approved zero-ray C08 spline/connector mesh contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from mtare_topo.topology.continuous_trajectory import build_spline_route, resample_route


RUN_ID = "gate4_20260813_cano_c08_trajectory_mesh_contract_v1_seed0"
SOURCE = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
WORLDS = ("S01_flat_tree_small_C08", "S06_3d_branch_medium_C08", "S10_3d_complex_C08")
MAX_CONNECTOR_M = 0.5
CONNECTOR_STEP_M = 0.05
MINIMUM_CLEARANCE_M = 0.8
FRAME_SPACING_M = 2.0


def _scene(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"empty mesh: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _connector_samples(connector: dict) -> np.ndarray:
    start = np.asarray(connector["node_xyz_m"], dtype=np.float64)
    end = np.asarray(connector["projection_xyz_m"], dtype=np.float64)
    length = float(connector["length_m"])
    count = max(2, int(math.ceil(length / CONNECTOR_STEP_M)) + 1)
    return np.linspace(start, end, count)


def _save_world_preview(world: str, route: np.ndarray, connectors: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, pair, labels in (
        (axes[0], (0, 1), ("X (m)", "Y (m)")),
        (axes[1], (0, 2), ("X (m)", "Z (m)")),
    ):
        axis.plot(route[:, pair[0]], route[:, pair[1]], color="#295f8a", linewidth=0.65, label="spline route")
        for connector in connectors:
            a = connector["node_xyz_m"]; b = connector["projection_xyz_m"]
            axis.plot([a[pair[0]], b[pair[0]]], [a[pair[1]], b[pair[1]]], color="#d1495b", linewidth=1.1)
        axis.set_xlabel(labels[0]); axis.set_ylabel(labels[1]); axis.set_aspect("equal", adjustable="datalim"); axis.grid(alpha=0.2)
    figure.suptitle(f"Gate 4 C08 trajectory geometry contract — {world}\nblue=spline traversal; red=explicit graph-node connector; units=m")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _save_anomaly_preview(connectors: list[dict], output: Path) -> None:
    selected = [item for item in connectors if item["node_id"] == "node_0015"]
    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    seen = set()
    for item in selected:
        key = (item["tunnel_id"], tuple(item["node_xyz_m"]), tuple(item["projection_xyz_m"]))
        if key in seen:
            continue
        seen.add(key)
        a=np.asarray(item["node_xyz_m"]); b=np.asarray(item["projection_xyz_m"])
        axis.plot([a[0],b[0]],[a[1],b[1]],marker="o",label=f"tunnel {item['tunnel_id']}: {item['length_m']:.3f} m")
    axis.set_xlabel("X (m)");axis.set_ylabel("Y (m)");axis.set_aspect("equal",adjustable="datalim");axis.grid(alpha=.25);axis.legend()
    axis.set_title("S01 node_0015 explicit intersection connectors\nnot to scale with full world; units=m")
    figure.savefig(output,dpi=180);plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID:
        raise RuntimeError("unexpected run directory")
    all_metrics=[]; anomaly_connectors=[]
    for world in WORLDS:
        source=SOURCE/world/"primary"; graph=load_json(source/"graph.json"); splines=load_json(source/"splines.json")
        route,traversals,connectors=build_spline_route(graph,splines,MAX_CONNECTOR_M)
        scene=_scene(source/"mesh.obj")
        clearance_values=[]
        for connector in connectors:
            samples=_connector_samples(connector)
            distances=scene.compute_distance(o3d.core.Tensor(samples.astype(np.float32))).numpy().astype(float)
            connector["sample_count"]=len(samples);connector["minimum_mesh_surface_distance_m"]=float(np.min(distances))
            clearance_values.extend(distances.tolist())
        samples,tangents,distances=resample_route(route,FRAME_SPACING_M)
        np.savez_compressed(run_dir/f"artifacts/{world}_trajectory.npz",xyz_m=samples,tangent_world=tangents,route_arc_m=distances)
        write_json(run_dir/f"artifacts/{world}_traversals.json",traversals)
        write_json(run_dir/f"artifacts/{world}_connectors.json",connectors)
        _save_world_preview(world,route,connectors,run_dir/f"previews/{world}_complete_trajectory_xy_xz.png")
        direction_counts={}
        edge_endpoints={str(item['id']):tuple(map(str,item['node_ids'])) for item in graph['edges']}
        for item in traversals:
            key=f"{item['edge_id']}:{item['from_node']}->{item['to_node']}";direction_counts[key]=direction_counts.get(key,0)+1
        bidirectional=all(direction_counts.get(f"{eid}:{a}->{b}")==1 and direction_counts.get(f"{eid}:{b}->{a}")==1 for eid,(a,b) in edge_endpoints.items())
        route_length=float(sum(item["length_m"] for item in traversals))
        metrics={
            "world":world,"graph_node_count":len(graph["nodes"]),"graph_edge_count":len(graph["edges"]),
            "traversal_count":len(traversals),"all_edges_once_per_direction":bidirectional,"route_length_m":route_length,
            "frame_spacing_m":FRAME_SPACING_M,"frame_count":len(samples),"maximum_connector_m":max(item["length_m"] for item in connectors),
            "minimum_connector_mesh_surface_distance_m":min(clearance_values),"connector_sample_count":len(clearance_values),
            "connector_limit_m":MAX_CONNECTOR_M,"minimum_clearance_limit_m":MINIMUM_CLEARANCE_M,
            "passed":bool(bidirectional and len(traversals)==2*len(graph['edges']) and max(item['length_m'] for item in connectors)<=MAX_CONNECTOR_M+1e-9 and min(clearance_values)>=MINIMUM_CLEARANCE_M),
        }
        write_json(run_dir/f"metrics/{world}.json",metrics);all_metrics.append(metrics)
        if world.startswith("S01_"): anomaly_connectors=connectors
    _save_anomaly_preview(anomaly_connectors,run_dir/"previews/S01_node_0015_connector_detail.png")
    summary={
        "schema_version":"cano_c08_trajectory_mesh_contract_summary_v1",
        "overall_status":"PASS_CANO_C08_TRAJECTORY_MESH_CONTRACT_V1" if all(item['passed'] for item in all_metrics) else "FAIL_CANO_C08_TRAJECTORY_MESH_CONTRACT_V1",
        "world_count":len(all_metrics),"trajectory_count":len(all_metrics),"graph_edge_count":sum(item['graph_edge_count'] for item in all_metrics),
        "traversal_count":sum(item['traversal_count'] for item in all_metrics),"route_length_m":sum(item['route_length_m'] for item in all_metrics),
        "frame_count":sum(item['frame_count'] for item in all_metrics),"maximum_connector_m":max(item['maximum_connector_m'] for item in all_metrics),
        "minimum_connector_mesh_surface_distance_m":min(item['minimum_connector_mesh_surface_distance_m'] for item in all_metrics),
        "ray_count":0,"inference_count":0,"graph_runtime_updates":0,"c09_reads":0,"c10_reads":0,"mtare_changes":0,
        "claim_boundary":"Zero-ray perception-mesh trajectory geometry contract only; not dynamic-navigation collision qualification and not topology replay performance.",
        "worlds":all_metrics,
    }
    write_json(run_dir/"metrics/summary.json",summary)
    print(json.dumps(summary,indent=2))
    return 0 if summary["overall_status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
