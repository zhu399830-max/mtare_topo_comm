#!/usr/bin/env python3
"""Sweep and freeze C08 causal graph parameters without future/test data."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components
from mtare_topo.governance import load_json, write_json
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig, CausalTopometricGraphV2, role_from_branch_count


WORLDS = ("S01_flat_tree_small_C08", "S06_3d_branch_medium_C08", "S10_3d_complex_C08")
METHODS = ("b0", "m1d_seed0", "m1d_seed1", "m1d_seed2", "oracle")
MESH_RUN = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
MATCH_RADIUS_M = 12.0
SCORE_WEIGHTS = {
    "exit_f1": .30, "connectivity": .15, "verified_edge_correctness": .15,
    "terminal_reachability": .15, "path_distance_fidelity": .15,
    "node_redundancy_quality": .05, "temporal_churn_quality": .05,
}


def configurations() -> list[CausalGraphConfig]:
    return [CausalGraphConfig(*values, distance_anchor_interval_m=20.0) for values in itertools.product(
        (1, 2, 3), (4.0, 6.0, 8.0), (3.0, 4.0, 6.0),
        (20.0, 25.0, 35.0), (25.0, 35.0, 45.0),
    )]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def method_stream(run_dir: Path, world: str, method: str) -> dict[str, Any]:
    shard = zarr.open_group(str(run_dir / f"artifacts/sensor/{world}.zarr"), mode="r")
    manifest = load_manifest(run_dir / f"artifacts/sensor/{world}_frames.jsonl")
    frames = len(manifest)
    headings: list[list[float]] = []
    if method == "oracle":
        headings = [item["headings_robot_deg"] for item in manifest]
        counts = np.asarray([item["branch_count"] for item in manifest], dtype=np.int64)
        roles = np.eye(3, dtype=np.float32)[np.asarray(shard["role_index"])]
        confidence = np.ones(frames, dtype=np.float32)
        embeddings = None
    elif method == "b0":
        headings = [item["b0_headings_robot_deg"] for item in manifest]
        counts = np.asarray([item["b0_branch_count"] for item in manifest], dtype=np.int64)
        role_indices = np.asarray([("interior", "junction", "terminal").index(role_from_branch_count(int(value))) for value in counts])
        roles = np.eye(3, dtype=np.float32)[role_indices]
        confidence = np.ones(frames, dtype=np.float32)
        embeddings = None
    else:
        seed = int(method[-1])
        values = np.load(run_dir / f"artifacts/inference/m1d_seed{seed}_{world}.npz")
        logits = values["direction_logits"]
        headings = [decode_direction_components(row, .5) for row in logits]
        counts = np.argmax(values["count_probabilities"], axis=1).astype(np.int64) + 1
        roles = values["role_probabilities"].astype(np.float32)
        confidence = np.max(roles, axis=1).astype(np.float32)
        embeddings = values["z_role"].astype(np.float32)
    if not (len(headings) == len(counts) == len(roles) == frames):
        raise RuntimeError(f"{world}/{method}: semantic stream shape mismatch")
    return {
        "shard": shard, "manifest": manifest, "headings": headings, "counts": counts,
        "roles": roles, "confidence": confidence, "embeddings": embeddings,
    }


def build_graph(config: CausalGraphConfig, stream: dict[str, Any]) -> CausalTopometricGraphV2:
    graph = CausalTopometricGraphV2(config)
    shard = stream["shard"]
    for index, record in enumerate(stream["manifest"]):
        embedding = None if stream["embeddings"] is None else stream["embeddings"][index]
        graph.update(
            frame_index=index, route_arc_m=float(shard["route_arc_m"][index]),
            xyz_m=shard["axis_xyz_m"][index], yaw_deg=float(shard["yaw_deg"][index]),
            headings_robot_deg=stream["headings"][index],
            role_probabilities=stream["roles"][index], branch_count=int(stream["counts"][index]),
            confidence=float(stream["confidence"][index]), z_role=embedding,
            evaluator_gt_edge_id=record["edge_id"],
        )
    return graph


def components(snapshot: dict[str, Any]) -> list[set[int]]:
    adjacency = {int(node["id"]): set() for node in snapshot["nodes"]}
    for edge in snapshot["edges"]:
        adjacency[int(edge["from"])].add(int(edge["to"]))
        adjacency[int(edge["to"])].add(int(edge["from"]))
    result = []
    remaining = set(adjacency)
    while remaining:
        root = min(remaining); seen = {root}; stack = [root]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in seen:
                    seen.add(neighbour); stack.append(neighbour)
        result.append(seen); remaining -= seen
    return result


def event_nodes(gt_graph: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for node in gt_graph["nodes"]:
        degree = int(node["degree"])
        if degree == 1 or degree >= 3:
            result.append({
                "id": str(node["id"]), "xyz_m": np.asarray(node["xyz"], dtype=np.float64),
                "role": "terminal" if degree == 1 else "junction",
            })
    return result


def greedy_matches(predicted: list[dict[str, Any]], truth: list[dict[str, Any]]) -> list[tuple[int, int, float]]:
    candidates = []
    for pred_index, pred in enumerate(predicted):
        for truth_index, target in enumerate(truth):
            if pred["role"] != target["role"]:
                continue
            distance = float(np.linalg.norm(np.asarray(pred["xyz_m"]) - target["xyz_m"]))
            if distance <= MATCH_RADIUS_M:
                candidates.append((distance, pred_index, truth_index))
    used_pred, used_truth, result = set(), set(), []
    for distance, pred_index, truth_index in sorted(candidates):
        if pred_index not in used_pred and truth_index not in used_truth:
            used_pred.add(pred_index); used_truth.add(truth_index)
            result.append((pred_index, truth_index, distance))
    return result


def evaluate(snapshot: dict[str, Any], gt_graph: dict[str, Any], route_length_m: float) -> dict[str, float]:
    predicted = [node for node in snapshot["nodes"] if node["node_kind"] == "structural"]
    truth = event_nodes(gt_graph)
    matches = greedy_matches(predicted, truth)
    precision = len(matches) / max(len(predicted), 1)
    recall = len(matches) / max(len(truth), 1)
    exit_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    truth_terminal = sum(item["role"] == "terminal" for item in truth)
    matched_terminal = sum(truth[truth_index]["role"] == "terminal" for _, truth_index, _ in matches)
    groups = components(snapshot)
    connectivity = max((len(group) for group in groups), default=0) / max(snapshot["node_count"], 1)
    traversals = [item for edge in snapshot["edges"] for item in edge["traversals"]]
    verified = sum(bool(item["evaluator_gt_edge_ids"]) and item["length_m"] > 0 for item in traversals) / max(len(traversals), 1)
    represented = sum(float(item["length_m"]) for item in traversals)
    path_fidelity = max(0.0, 1.0 - min(abs(represented - route_length_m) / max(route_length_m, 1e-12), 1.0))
    redundancy_quality = min(len(predicted), len(truth)) / max(len(predicted), len(truth), 1)
    created_per_100m = snapshot["node_count"] * 100.0 / max(route_length_m, 1e-12)
    churn_quality = max(0.0, 1.0 - min(created_per_100m / 10.0, 1.0))
    values = {
        "exit_precision": precision, "exit_recall": recall, "exit_f1": exit_f1,
        "connectivity": connectivity, "verified_edge_correctness": verified,
        "terminal_reachability": matched_terminal / max(truth_terminal, 1),
        "path_distance_fidelity": path_fidelity,
        "node_redundancy_quality": redundancy_quality,
        "temporal_churn_quality": churn_quality,
        "predicted_structural_nodes": float(len(predicted)), "truth_event_nodes": float(len(truth)),
        "matched_event_nodes": float(len(matches)), "graph_nodes": float(snapshot["node_count"]),
        "graph_edges": float(snapshot["edge_count"]), "represented_route_m": represented,
        "route_length_m": route_length_m, "created_nodes_per_100m": created_per_100m,
    }
    values["composite_score"] = sum(SCORE_WEIGHTS[key] * values[key] for key in SCORE_WEIGHTS)
    return values


def parameter_id(config: CausalGraphConfig) -> str:
    return f"sf{config.stable_frames}_tr{config.minimum_event_travel_m:g}_lr{config.loop_merge_radius_m:g}_hh{config.branch_heading_merge_deg:g}_te{config.turn_event_deg:g}_da20"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); started = time.monotonic()
    configs = configurations()
    if len(configs) != 243:
        raise RuntimeError("frozen grid must contain 243 configurations")
    streams = {(method, world): method_stream(run_dir, world, method) for method in METHODS for world in WORLDS}
    gt_graphs = {world: load_json(MESH_RUN / f"artifacts/meshes/{world}/primary/graph.json") for world in WORLDS}
    rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs, 1):
        pid = parameter_id(config)
        for method in METHODS:
            for world in WORLDS:
                graph = build_graph(config, streams[(method, world)])
                snapshot = graph.snapshot()
                route_length = float(streams[(method, world)]["shard"]["route_arc_m"][-1])
                metrics = evaluate(snapshot, gt_graphs[world], route_length)
                rows.append({"parameter_id": pid, **config.to_dict(), "method": method, "world": world, **metrics})
                if method == "oracle" and (metrics["verified_edge_correctness"] != 1.0 or metrics["connectivity"] != 1.0):
                    raise RuntimeError(f"oracle graph integrity failed for {pid}/{world}: {metrics}")
        if config_index % 10 == 0 or config_index == len(configs):
            print(json.dumps({"configuration": config_index, "of": len(configs), "rows": len(rows)}), flush=True)

    fieldnames = list(rows[0])
    with (run_dir / "metrics/parameter_sweep.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    m1d_methods = {"m1d_seed0", "m1d_seed1", "m1d_seed2"}
    aggregate = []
    for config in configs:
        pid = parameter_id(config)
        selected_rows = [row for row in rows if row["parameter_id"] == pid and row["method"] in m1d_methods]
        aggregate.append({"parameter_id": pid, "mean_m1d_composite_score": float(np.mean([row["composite_score"] for row in selected_rows])), **config.to_dict()})
    aggregate.sort(key=lambda item: (-item["mean_m1d_composite_score"], item["parameter_id"]))
    selected = aggregate[0]
    selected_config = CausalGraphConfig(
        stable_frames=int(selected["stable_frames"]),
        minimum_event_travel_m=float(selected["minimum_event_travel_m"]),
        loop_merge_radius_m=float(selected["loop_merge_radius_m"]),
        branch_heading_merge_deg=float(selected["branch_heading_merge_deg"]),
        turn_event_deg=float(selected["turn_event_deg"]),
        distance_anchor_interval_m=float(selected["distance_anchor_interval_m"]),
    )
    write_json(run_dir / "artifacts/frozen_c08_graph_parameters.json", {
        "schema_version": "cano_c08_frozen_graph_parameters_v1",
        "selection_source": "mean composite over M1D seeds 0/1/2 and all three C08 worlds only",
        "selection_score": selected["mean_m1d_composite_score"],
        "parameter_id": selected["parameter_id"], "config": selected_config.to_dict(),
        "event_match_radius_m": MATCH_RADIUS_M, "score_weights": SCORE_WEIGHTS,
    })
    selected_metrics = []
    graph_root = run_dir / "artifacts/selected_graphs"; graph_root.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        for world in WORLDS:
            graph = build_graph(selected_config, streams[(method, world)])
            snapshot = graph.snapshot()
            write_json(graph_root / f"{method}_{world}_graph.json", snapshot)
            write_json(graph_root / f"{method}_{world}_decision_trace.json", {"records": graph.decision_trace})
            route_length = float(streams[(method, world)]["shard"]["route_arc_m"][-1])
            selected_metrics.append({"method": method, "world": world, **evaluate(snapshot, gt_graphs[world], route_length)})
    method_summary = {}
    for method in METHODS:
        subset = [item for item in selected_metrics if item["method"] == method]
        method_summary[method] = {key: float(np.mean([item[key] for item in subset])) for key in SCORE_WEIGHTS | {"composite_score": 0}}
    summary = {
        "schema_version": "cano_c08_causal_graph_stage_v1",
        "overall_status": "PASS_C08_CAUSAL_GRAPH_SWEEP_AND_FREEZE_V1",
        "grid_configurations": len(configs), "grid_graph_replays": len(rows),
        "selected_parameter_id": selected["parameter_id"], "selected_config": selected_config.to_dict(),
        "selection_score": selected["mean_m1d_composite_score"],
        "score_weights": SCORE_WEIGHTS, "event_match_radius_m": MATCH_RADIUS_M,
        "selected_metrics": selected_metrics, "method_mean_metrics": method_summary,
        "oracle_integrity_passed_for_all_grid_rows": True,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(run_dir / "metrics/graph_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
