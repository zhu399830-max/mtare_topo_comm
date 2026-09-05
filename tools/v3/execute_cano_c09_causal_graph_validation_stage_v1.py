#!/usr/bin/env python3
"""Validate the single C08-frozen graph tuple on all C09 worlds."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import execute_cano_c08_causal_graph_stage_v1 as core
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig


WORLDS = tuple(f"S{index:02d}_{name}_C09" for index, name in enumerate((
    "flat_tree_small", "3d_tree_small", "flat_unicyclic_small", "3d_unicyclic_small",
    "flat_branch_medium", "3d_branch_medium", "flat_loop_rich", "3d_loop_rich",
    "flat_complex", "3d_complex",
), 1))
FROZEN_PARAMETERS = PROJECT_ROOT / "results/gate4_topology/gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2r_seed0/artifacts/frozen_c08_graph_parameters.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); started = time.monotonic()
    core.WORLDS = WORLDS
    frozen = load_json(FROZEN_PARAMETERS)
    config = CausalGraphConfig(**frozen["config"])
    if core.parameter_id(config) != "sf2_tr8_lr6_hh20_te45_da20":
        raise RuntimeError("frozen parameter identity mismatch")
    if frozen["score_weights"] != core.SCORE_WEIGHTS or float(frozen["event_match_radius_m"]) != core.MATCH_RADIUS_M:
        raise RuntimeError("frozen evaluation contract mismatch")
    streams = {(method, world): core.method_stream(run_dir, world, method)
               for method in core.METHODS for world in WORLDS}
    gt_graphs = {world: load_json(core.MESH_RUN / f"artifacts/meshes/{world}/primary/graph.json") for world in WORLDS}
    rows, graph_root = [], run_dir / "artifacts/selected_graphs"
    graph_root.mkdir(parents=True, exist_ok=True)
    for method in core.METHODS:
        for world in WORLDS:
            graph = core.build_graph(config, streams[(method, world)])
            snapshot = graph.snapshot()
            route_length = float(streams[(method, world)]["shard"]["route_arc_m"][-1])
            metrics = core.evaluate(snapshot, gt_graphs[world], route_length)
            if method == "oracle" and (metrics["verified_edge_correctness"] != 1.0 or metrics["connectivity"] != 1.0):
                raise RuntimeError(f"oracle graph integrity failed for {world}: {metrics}")
            rows.append({"parameter_id": frozen["parameter_id"], **config.to_dict(),
                         "method": method, "world": world, **metrics})
            write_json(graph_root / f"{method}_{world}_graph.json", snapshot)
            write_json(graph_root / f"{method}_{world}_decision_trace.json", {"records": graph.decision_trace})
            print(json.dumps({"method": method, "world": world, "replay": len(rows), "of": 50}), flush=True)
    with (run_dir / "metrics/c09_validation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    method_summary = {}
    for method in core.METHODS:
        subset = [item for item in rows if item["method"] == method]
        method_summary[method] = {key: float(np.mean([item[key] for item in subset]))
                                  for key in core.SCORE_WEIGHTS | {"composite_score": 0}}
    summary = {"schema_version": "cano_c09_causal_graph_validation_stage_v1",
               "overall_status": "PASS_C09_CAUSAL_GRAPH_VALIDATION_V1",
               "parameter_configurations": 1, "graph_replays": len(rows),
               "frozen_parameter_id": frozen["parameter_id"], "frozen_config": config.to_dict(),
               "selection_performed_on_c09": False, "score_weights": core.SCORE_WEIGHTS,
               "event_match_radius_m": core.MATCH_RADIUS_M, "validation_metrics": rows,
               "method_mean_metrics": method_summary, "oracle_integrity_passed": True,
               "c08_parameter_source_read": True, "c10_worlds_read": 0, "mtare_worlds_read": 0,
               "duration_seconds": time.monotonic() - started}
    write_json(run_dir / "metrics/graph_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
