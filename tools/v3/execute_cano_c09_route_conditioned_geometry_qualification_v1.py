#!/usr/bin/env python3
"""Materialize exact C09 trajectories and run the frozen route-conditioned geometry qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT
import execute_cano_c08_hybrid_implicit_qualification_v3 as core
from mtare_topo.governance import load_json, write_json
from mtare_topo.topology.continuous_trajectory import build_spline_route, resample_route


RUN_ID = "gate4_20260817_cano_c09_route_conditioned_geometry_qualification_v1_seed0"
PASS_STATUS = "PASS_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1"
FAIL_STATUS = "FAIL_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1"
SCHEMA_VERSION = "cano_c09_route_conditioned_geometry_qualification_v1"
PREVIEW_TITLE = "C09 route-conditioned geometry qualification V1"
MANIFEST = PROJECT_ROOT / "configs/v3/gate4/cano_c09_readonly_continuous_contract_manifest_v1.json"
ASSETS = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
MAX_CONNECTOR_M = 0.5
FRAME_SPACING_M = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_trajectories(run_dir: Path) -> tuple[tuple[str, int], ...]:
    """Build the exact doubled-edge trajectories frozen by the read-only audit."""
    manifest = load_json(MANIFEST)
    expected_worlds = manifest["worlds"]
    metrics = []
    suite: list[tuple[str, int]] = []
    for expected in expected_worlds:
        world = str(expected["world"])
        root = ASSETS / world / "primary"
        for filename, key in (
            ("graph.json", "graph_sha256"), ("splines.json", "splines_sha256"),
            ("mesh.obj", "mesh_sha256"), ("geometry_parameters.json", "geometry_sha256"),
        ):
            actual = sha256(root / filename)
            if actual != expected[key]:
                raise RuntimeError(f"C09 frozen input drift: {world}/{filename}: {actual}")
        graph = load_json(root / "graph.json")
        splines = load_json(root / "splines.json")
        route, traversals, connectors = build_spline_route(graph, splines, MAX_CONNECTOR_M)
        samples, tangents, route_arc = resample_route(route, FRAME_SPACING_M)
        route_length = float(sum(float(item["length_m"]) for item in traversals))
        if (
            len(graph["edges"]) != int(expected["edges"])
            or len(traversals) != int(expected["traversals"])
            or len(samples) != int(expected["frames"])
            or not math.isclose(route_length, float(expected["route_length_m"]), rel_tol=0.0, abs_tol=1e-9)
            or max(float(item["length_m"]) for item in connectors) > MAX_CONNECTOR_M + 1e-9
        ):
            raise RuntimeError(f"C09 trajectory contract drift: {world}")
        np.savez_compressed(
            run_dir / f"artifacts/{world}_trajectory.npz",
            xyz_m=samples, tangent_world=tangents, route_arc_m=route_arc,
        )
        write_json(run_dir / f"artifacts/{world}_traversals.json", traversals)
        write_json(run_dir / f"artifacts/{world}_connectors.json", connectors)
        metric = {
            "world": world, "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
            "directed_traversals": len(traversals), "route_length_m": route_length,
            "frames": len(samples), "maximum_connector_m": max(float(item["length_m"]) for item in connectors),
            "all_edges_once_per_direction": len(traversals) == 2 * len(graph["edges"]),
        }
        metrics.append(metric)
        suite.append((world, len(samples)))
    totals = {
        "schema_version": "cano_c09_materialized_trajectory_contract_v1",
        "overall_status": "PASS_CANO_C09_MATERIALIZED_TRAJECTORY_CONTRACT_V1",
        "worlds": len(metrics), "edges": sum(item["edges"] for item in metrics),
        "directed_traversals": sum(item["directed_traversals"] for item in metrics),
        "route_length_m": sum(item["route_length_m"] for item in metrics),
        "frames": sum(item["frames"] for item in metrics), "world_metrics": metrics,
    }
    frozen = manifest["totals"]
    if (
        totals["worlds"] != frozen["worlds"] or totals["edges"] != frozen["edges"]
        or totals["directed_traversals"] != frozen["directed_traversals"]
        or totals["frames"] != frozen["frames"]
        or not math.isclose(totals["route_length_m"], frozen["route_length_m"], rel_tol=0.0, abs_tol=1e-9)
    ):
        raise RuntimeError("aggregate C09 trajectory contract drift")
    write_json(run_dir / "metrics/trajectory_contract.json", totals)
    print(json.dumps(totals), flush=True)
    return tuple(suite)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID:
        raise RuntimeError("unexpected C09 qualification run directory")
    suite = materialize_trajectories(run_dir)
    core.RUN_ID = RUN_ID
    core.WORLDS = suite
    core.ASSETS = ASSETS
    core.TRAJECTORIES = run_dir / "artifacts"
    core.FORMAL_PASS_STATUS = PASS_STATUS
    core.FORMAL_FAIL_STATUS = FAIL_STATUS
    core.FORMAL_SCHEMA_VERSION = SCHEMA_VERSION
    core.FORMAL_WORLD_COUNT = 10
    core.FORMAL_C09_WORLDS_READ = 10
    core.FORMAL_CLAIM_BOUNDARY = "C09 route-conditioned geometry qualification only; no LiDAR, model inference, graph replay, parameter selection, C10 or M-TARE."
    core.FORMAL_PREVIEW_TITLE = PREVIEW_TITLE
    sys.argv = [sys.argv[0], "--run-dir", str(run_dir)]
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
