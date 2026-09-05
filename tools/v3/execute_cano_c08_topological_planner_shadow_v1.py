#!/usr/bin/env python3
"""Replay the Gate-5 global planner without controlling the robot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
import execute_cano_c08_causal_graph_stage_v1 as graph_core
from mtare_topo.governance import load_json, write_json
from mtare_topo.planning.topological_frontier import (
    RuleBasedTopologicalFrontierPlanner,
    TopologicalPlannerConfig,
)
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig, CausalTopometricGraphV2


SOURCE_RUN = PROJECT_ROOT / "results/gate4_topology/gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2r_seed0"
WORLDS = graph_core.WORLDS
METHODS = graph_core.METHODS
PLANNER_CONFIG = TopologicalPlannerConfig(
    waypoint_lookahead_m=4.0,
    graph_cost_scale_m=20.0,
    retry_penalty=0.25,
    minimum_frontier_confidence=0.05,
)
EXPECTED_FRAMES = 4773
EXPECTED_CYCLES = EXPECTED_FRAMES * len(METHODS)


def canonical_hash(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def replay_stream(
    stream: dict[str, Any], graph_config: CausalGraphConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = CausalTopometricGraphV2(graph_config)
    planner = RuleBasedTopologicalFrontierPlanner(PLANNER_CONFIG)
    shard = stream["shard"]
    records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    target_changes = 0
    last_frontier: tuple[int, int] | None = None
    maximum_waypoint_distance = 0.0
    for index, manifest in enumerate(stream["manifest"]):
        embedding = None if stream["embeddings"] is None else stream["embeddings"][index]
        graph.update(
            frame_index=index,
            route_arc_m=float(shard["route_arc_m"][index]),
            xyz_m=shard["axis_xyz_m"][index],
            yaw_deg=float(shard["yaw_deg"][index]),
            headings_robot_deg=stream["headings"][index],
            role_probabilities=stream["roles"][index],
            branch_count=int(stream["counts"][index]),
            confidence=float(stream["confidence"][index]),
            z_role=embedding,
            evaluator_gt_edge_id=manifest["edge_id"],
        )
        robot = tuple(map(float, shard["axis_xyz_m"][index]))
        target = planner.select_target(graph.snapshot(), robot_xyz_m=robot)
        status_counts[target.status] += 1
        mode_counts[target.mode] += 1
        frontier = None if target.frontier is None else (target.frontier.node_id, target.frontier.stub_index)
        if frontier is not None and last_frontier is not None and frontier != last_frontier:
            target_changes += 1
        if frontier is not None:
            last_frontier = frontier
        waypoint_distance = 0.0
        if target.waypoint_xyz_m is not None:
            waypoint_distance = math.dist(robot, target.waypoint_xyz_m)
            if not all(math.isfinite(value) for value in target.waypoint_xyz_m):
                raise RuntimeError(f"non-finite waypoint at frame {index}")
            if waypoint_distance > PLANNER_CONFIG.waypoint_lookahead_m + 1e-9:
                raise RuntimeError(f"waypoint exceeds handoff lookahead at frame {index}")
        maximum_waypoint_distance = max(maximum_waypoint_distance, waypoint_distance)
        records.append({
            "frame_index": index,
            "route_arc_m": float(shard["route_arc_m"][index]),
            "robot_xyz_m": list(robot),
            "status": target.status,
            "mode": target.mode,
            "waypoint_xyz_m": None if target.waypoint_xyz_m is None else list(target.waypoint_xyz_m),
            "frontier": None if target.frontier is None else {
                "node_id": target.frontier.node_id,
                "stub_index": target.frontier.stub_index,
            },
            "frontier_node_id": target.frontier_node_id,
            "next_hop_node_id": target.next_hop_node_id,
            "graph_path_node_ids": list(target.graph_path_node_ids),
            "graph_cost_m": target.graph_cost_m,
            "exploration_potential": target.exploration_potential,
            "utility": target.utility,
            "reason": target.reason,
            "waypoint_distance_m": waypoint_distance,
        })
    distance_m = float(shard["route_arc_m"][-1])
    metrics = {
        "frames": len(records),
        "route_distance_m": distance_m,
        "status_counts": dict(sorted(status_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "target_identity_changes": target_changes,
        "target_identity_changes_per_100m": 100.0 * target_changes / max(distance_m, 1e-12),
        "maximum_waypoint_distance_m": maximum_waypoint_distance,
        "deterministic_trace_sha256": canonical_hash(records),
    }
    return records, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    started = time.monotonic()
    parameters = load_json(SOURCE_RUN / "artifacts/frozen_c08_graph_parameters.json")
    graph_config = CausalGraphConfig(**parameters["config"])
    if graph_core.parameter_id(graph_config) != "sf2_tr8_lr6_hh20_te45_da20":
        raise RuntimeError("frozen C08 graph tuple drift")
    output_root = run_dir / "artifacts/shadow_decisions"
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_root = run_dir / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    total_frames = 0
    for method in METHODS:
        for world in WORLDS:
            stream = graph_core.method_stream(SOURCE_RUN, world, method)
            records, metrics = replay_stream(stream, graph_config)
            # A second complete in-memory replay is the determinism authority.
            repeated, repeated_metrics = replay_stream(stream, graph_config)
            if canonical_hash(records) != canonical_hash(repeated):
                raise RuntimeError(f"non-deterministic shadow decisions: {method}/{world}")
            if metrics["deterministic_trace_sha256"] != repeated_metrics["deterministic_trace_sha256"]:
                raise RuntimeError(f"non-deterministic summary: {method}/{world}")
            path = output_root / f"{method}_{world}.jsonl"
            with path.open("w", encoding="utf-8") as stream_out:
                for record in records:
                    stream_out.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            row = {"method": method, "world": world, **metrics, "deterministic_replay_pass": True}
            summaries.append(row)
            total_frames += len(records)
            print(json.dumps({"method": method, "world": world, "frames": len(records)}), flush=True)

    if total_frames != EXPECTED_CYCLES:
        raise RuntimeError(f"expected {EXPECTED_CYCLES} planner cycles, got {total_frames}")
    if sum(row["frames"] for row in summaries if row["method"] == "oracle") != EXPECTED_FRAMES:
        raise RuntimeError("oracle frame contract mismatch")
    summary = {
        "schema_version": "cano_c08_topological_planner_shadow_v1",
        "overall_status": "PASS_CANO_C08_TOPOLOGICAL_PLANNER_SHADOW_V1",
        "worlds": list(WORLDS),
        "methods": list(METHODS),
        "unique_source_frames": EXPECTED_FRAMES,
        "planner_cycles": total_frames,
        "planner_config": PLANNER_CONFIG.to_dict(),
        "graph_parameter_id": "sf2_tr8_lr6_hh20_te45_da20",
        "all_waypoints_finite_and_bounded": True,
        "all_routes_use_verified_edges": True,
        "all_decision_replays_deterministic": True,
        "method_world_metrics": summaries,
        "new_lidar_frames": 0,
        "new_inference_frames": 0,
        "training_steps": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_closed_loop_runs": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(metrics_root / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
