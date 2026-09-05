#!/usr/bin/env python3
"""Run and seal the immutable P0 construction-supervision feasibility proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.primitive_construction_supervisor import (
    audit_archived_supervision,
    build_primitive_construction_graph,
)


RUN_ID = "gate3_20260830_primitive_construction_supervision_feasibility_v1_seed0"
PASS = "PASS_PRIMITIVE_CONSTRUCTION_SUPERVISION_FEASIBILITY_V1"
FAIL = "FAIL_PRIMITIVE_CONSTRUCTION_SUPERVISION_FEASIBILITY_V1"
MESH = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes/S01_flat_tree_small_C01/primary"
ZARR = PROJECT_ROOT / "results/gate1_data/gate1_20260812_cano_phase2_supervised_range_dataset_v2_seed0/artifacts/dataset/train/S01_flat_tree_small_C01.zarr"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _plot(construction, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.5, 6.5))
    for primitive in construction.primitives:
        points = primitive.centerline_xyz_m
        axis.plot(points[:, 0], points[:, 1], linewidth=1.2, color="#3973ac")
    for composition in construction.compositions:
        point = composition.member_endpoints[0].xyz_m
        if composition.degree >= 3:
            axis.scatter(point[0], point[1], s=55, color="#c43c39", zorder=3)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("C01 edge-level swept primitives; red = endpoint composition degree >= 3")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("P0 feasibility proof executes exactly once")
    started = time.monotonic()
    overall, error = FAIL, None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit":
            raise RuntimeError("P0 spec scope mismatch")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("P0 authorization is not bound")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen tool drift: {record['path']}")
        graph = load_json(MESH / "graph.json")
        splines = load_json(MESH / "splines.json")
        geometry = load_json(MESH / "geometry_parameters.json")
        arrays = sorted(path.name for path in ZARR.iterdir() if path.is_dir() and (path / ".zarray").is_file())
        shapes = {name: load_json(ZARR / name / ".zarray")["shape"] for name in arrays}
        construction = build_primitive_construction_graph(graph, splines, geometry)
        audit = audit_archived_supervision(
            construction, graph, mesh_obj_path=MESH / "mesh.obj", lidar_array_names=arrays
        )
        write_json(run_dir / "artifacts/primitive_construction_graph.json", construction.as_dict())
        write_json(run_dir / "metrics/archived_supervision_audit.json", audit.as_dict())
        write_json(run_dir / "metrics/lidar_schema_audit.json", {
            "world": "S01_flat_tree_small_C01", "array_shapes": shapes,
            "fixed_readonly_rows": [4, 7, 25, 89, 90, 216],
            "raw_frames": shapes.get("range_m", [0])[0],
        })
        _plot(construction, run_dir / "previews/primitive_construction_graph")
        overall = PASS if audit.construction_supervision_complete else FAIL
        summary = {
            "schema_version": "primitive_construction_supervision_feasibility_v1",
            "overall_status": overall,
            "scientific_pass": overall == PASS,
            "worlds_read": 1,
            "raw_frames_available": shapes.get("range_m", [0])[0],
            "sampled_frame_rows": 6,
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "c07_c10_worlds_read": 0,
            "graph_replays": 0,
            "construction": audit.as_dict(),
            "decision": (
                "ARCHIVED_ASSETS_SUFFICIENT_FOR_TRAINING_TEACHER"
                if overall == PASS else
                "ADD_PRE_POISSON_PRIMITIVE_PROVENANCE_AND_RAY_HIT_IDENTITY_BEFORE_DATA_GENERATION"
            ),
            "duration_seconds": time.monotonic() - started,
            "error": None,
        }
        write_json(run_dir / "metrics/summary.json", summary)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
    })
    count = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "evidence_files": count}, indent=2))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
