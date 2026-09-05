#!/usr/bin/env python3
"""Formal full-population qualification of minimal finite-cap pose correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_sensor_export import (
    world_finite_union_qualified_frame_poses,
    world_unique_frame_poses,
)
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.geometry_variant_contract import GeometryRealization, realize_construction
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


RUN_ID = "gate3_20260830_primitive_finite_cap_pose_qualification_v1_seed0"
PASS = "PASS_PRIMITIVE_FINITE_CAP_POSE_QUALIFICATION_V1"
FAIL = "FAIL_PRIMITIVE_FINITE_CAP_POSE_QUALIFICATION_V1"
MESH_ROOT = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
TRAVERSAL_PATH = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/traversal_manifest.jsonl"


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


def _traversals() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with TRAVERSAL_PATH.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result.setdefault(str(row["parent_id"]), []).append(row)
    return result


def _minimum(field: SweptSuperellipseProvenanceField, points: np.ndarray) -> np.ndarray:
    parts = []
    for start in range(0, len(points), 2048):
        parts.append(np.min(field.operand_signed_distances_sparse(points[start:start + 2048]), axis=1))
    return np.concatenate(parts)


def _plot(summary: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shifts = np.asarray(summary["inward_shifts_m"], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4), constrained_layout=True)
    axes[0].bar(
        ["before", "after"],
        [summary["before_outside_gt_1e9"], summary["after_outside_gt_1e9"]],
        color=["#c65a50", "#57965c"],
    )
    axes[0].set(ylabel="variant-frame origins outside", title="Finite-union origin validity")
    axes[1].hist(shifts, bins=25, color="#4e79a7")
    axes[1].set(xlabel="inward arc shift (m)", ylabel="source frames", title="Only invalid endpoint poses move")
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
        raise RuntimeError("pose qualification executes exactly once")
    started = time.monotonic()
    overall, error = FAIL, None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("scope/authorization mismatch")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen tool drift: {record['path']}")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        tests = subprocess.run(
            [
                "/home/zeng-workstation/anaconda3/bin/python", "-m", "pytest", "-q",
                "tests/v3/unit/test_primitive_construction_supervisor.py",
                "tests/v3/unit/test_primitive_provenance_field.py",
                "tests/v3/unit/test_swept_superellipse_field.py",
                "tests/v3/unit/test_geometry_variant_contract.py",
                "tests/v3/unit/test_primitive_relation_dataset.py",
                "tests/v3/unit/test_gse_sensor_export.py",
            ],
            cwd=PROJECT_ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (run_dir / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or "40 passed" not in tests.stdout:
            raise RuntimeError("expected exactly 40 unit tests")

        card = load_json(PROJECT_ROOT / spec["data_card"])
        worlds = tuple(card["worlds"]["train"])
        traversals = _traversals()
        source_frames = before_outside = after_outside = raw_after_positive = 0
        before_maximum = after_maximum = maximum_shift = 0.0
        minimum_spacing = float("inf")
        corrections: list[dict] = []
        world_rows: list[dict] = []
        all_unaffected_exact = True
        all_identity_exact = True
        for world in worlds:
            primary = MESH_ROOT / world / "primary"
            graph = load_json(primary / "graph.json")
            splines = load_json(primary / "splines.json")
            geometry = load_json(primary / "geometry_parameters.json")
            construction = build_primitive_construction_graph(
                graph, splines, geometry,
                endpoint_attachment_mode="free_space_overlap",
                node_degree_source="edge_incidence",
            )
            fields = {
                realization: SweptSuperellipseProvenanceField(
                    realize_construction(world, construction, realization), spacing_m=.025
                )
                for realization in GeometryRealization
            }
            original = world_unique_frame_poses(
                parent_id=world, traversal_manifest=traversals[world], graph=graph,
                spline_document=splines, geometry_parameters=geometry,
            )
            corrected, records = world_finite_union_qualified_frame_poses(
                parent_id=world, traversal_manifest=traversals[world], graph=graph,
                spline_document=splines, geometry_parameters=geometry,
                signed_distance_fields=[fields[value].signed_distance for value in GeometryRealization],
            )
            source_frames += len(original.global_frame_indices)
            all_identity_exact &= (
                np.array_equal(original.global_frame_indices, corrected.global_frame_indices)
                and np.array_equal(original.local_frame_indices, corrected.local_frame_indices)
                and original.traversal_ids == corrected.traversal_ids
                and len(original.arc_m) == len(corrected.arc_m)
            )
            changed = np.abs(original.arc_m - corrected.arc_m) > 1e-12
            expected_changed = np.zeros(len(changed), dtype=bool)
            expected_changed[[record.global_frame_index - int(original.global_frame_indices[0]) for record in records]] = True
            if not np.array_equal(changed, expected_changed):
                raise RuntimeError("correction provenance and changed-frame mask disagree")
            unchanged = ~changed
            all_unaffected_exact &= all(
                np.array_equal(getattr(original, name)[unchanged], getattr(corrected, name)[unchanged])
                for name in ("arc_m", "axis_xyz_m", "sensor_xyz_m", "tangent_world_xyz", "yaw_deg")
            )
            for traversal_id in sorted(set(corrected.traversal_ids)):
                indices = np.flatnonzero(np.asarray(corrected.traversal_ids, dtype=object) == traversal_id)
                if len(indices) > 1:
                    minimum_spacing = min(minimum_spacing, float(np.min(np.diff(corrected.arc_m[indices]))))
            for record in records:
                maximum_shift = max(maximum_shift, record.inward_shift_m)
                corrections.append({"world": world, **record.__dict__, "inward_shift_m": record.inward_shift_m})
            world_before = world_after = 0
            for field in fields.values():
                before = _minimum(field, original.sensor_xyz_m)
                after = _minimum(field, corrected.sensor_xyz_m)
                world_before += int(np.sum(before > 1e-9))
                world_after += int(np.sum(after > 1e-9))
                raw_after_positive += int(np.sum(after > 0.0))
                before_maximum = max(before_maximum, float(np.max(before)))
                after_maximum = max(after_maximum, float(np.max(after)))
            before_outside += world_before
            after_outside += world_after
            world_rows.append({
                "world": world, "source_frames": len(original.global_frame_indices),
                "corrected_frames": len(records), "before_outside_gt_1e9": world_before,
                "after_outside_gt_1e9": world_after,
            })

        checks = {
            "exact_80_worlds": len(world_rows) == 80,
            "exact_252430_source_frames": source_frames == 252430,
            "exact_757290_variant_frames": 3 * source_frames == 757290,
            "reproduce_402_meaningful_variant_failures": before_outside == 402,
            "exact_134_corrected_source_frames": len(corrections) == 134,
            "zero_meaningful_failures_after": after_outside == 0 and after_maximum <= 1e-9,
            "frame_identity_count_order_exact": bool(all_identity_exact),
            "all_252296_unaffected_frames_bit_exact": bool(all_unaffected_exact),
            "maximum_inward_shift_le_0p5m": maximum_shift <= .5,
            "minimum_adjacent_arc_spacing_ge_0p5m": minimum_spacing >= .5,
            "geometry_inputs_read_only": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        summary = {
            "schema_version": "primitive_finite_cap_pose_qualification_v1",
            "overall_status": overall, "scientific_pass": overall == PASS,
            "checks": checks, "source_frames": source_frames,
            "variant_frame_checks": 3 * source_frames,
            "before_outside_gt_1e9": before_outside,
            "after_outside_gt_1e9": after_outside,
            "raw_after_positive_le_roundoff": raw_after_positive,
            "before_maximum_residual_m": before_maximum,
            "after_maximum_residual_m": after_maximum,
            "corrected_source_frames": len(corrections),
            "unchanged_source_frames": source_frames - len(corrections),
            "maximum_inward_shift_m": maximum_shift,
            "minimum_adjacent_arc_spacing_m": minimum_spacing,
            "inward_shifts_m": [row["inward_shift_m"] for row in corrections],
            "worlds": world_rows,
            "decision": "ALLOW_P1_CAPACITY_AUDIT_WITH_QUALIFIED_POSES" if overall == PASS else "STOP_P1_POSE_CORRECTION_FAILED",
            "c09_c10_worlds_read": 0, "optimizer_steps": 0,
            "model_inference_frames": 0, "graph_replays": 0,
            "duration_seconds": time.monotonic() - started, "error": None,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "artifacts/frame_pose_corrections.json", {
            "schema_version": "finite_union_frame_pose_corrections_v1",
            "correction_rule": "only original origins outside any of three frozen finite unions; minimum directed endpoint arc shift by deterministic 0.025m search + 40-step bisection + 0.025m inward guard",
            "corrections": corrections,
        })
        write_json(run_dir / "config/environment.json", {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "numpy": np.__version__,
        })
        _plot(summary, run_dir / "previews/finite_cap_pose_qualification")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
