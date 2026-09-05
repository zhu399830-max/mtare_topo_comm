#!/usr/bin/env python3
"""Formal V1R fit-only primitive-slot audit using qualified endpoint poses."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.gse_sensor_export import world_finite_union_qualified_frame_poses
from mtare_topo.data.primitive_relation_dataset import PrimitiveMembershipCodebook, visible_primitive_window_targets
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster, mesh_swept_superellipse
from mtare_topo.teacher.geometry_variant_contract import GeometryRealization, realize_construction
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


RUN_ID = "gate3_20260830_primitive_slot_capacity_audit_v1r_seed0"
PASS = "PASS_PRIMITIVE_SLOT_CAPACITY_AUDIT_V1R"
FAIL = "FAIL_PRIMITIVE_SLOT_CAPACITY_AUDIT_V1R"
MESH_ROOT = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
TRAVERSAL_PATH = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/traversal_manifest.jsonl"
RAYS_PER_WINDOW = 5 * 16 * 720
CANDIDATE_CAPACITIES = (8, 16, 32)


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


def _load_traversals() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with TRAVERSAL_PATH.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line); result.setdefault(str(row["parent_id"]), []).append(row)
    return result


def _select_windows(poses, graph):
    degree = {str(node["id"]): 0 for node in graph["nodes"]}
    for edge in graph["edges"]:
        for node_id in edge["node_ids"]:
            degree[str(node_id)] += 1
    node_xyz = {str(node["id"]): np.asarray(node["xyz"], dtype=np.float64) for node in graph["nodes"]}
    junctions = np.asarray([node_xyz[key] for key, value in degree.items() if value >= 3])
    terminals = np.asarray([node_xyz[key] for key, value in degree.items() if value == 1])
    if not len(junctions) or not len(terminals):
        raise RuntimeError("role audit requires at least one junction and terminal")
    valid = np.flatnonzero(poses.local_frame_indices >= 4)
    junction_distance = np.min(np.linalg.norm(poses.axis_xyz_m[valid, None] - junctions[None], axis=2), axis=1)
    terminal_distance = np.min(np.linalg.norm(poses.axis_xyz_m[valid, None] - terminals[None], axis=2), axis=1)
    event_distance = np.minimum(junction_distance, terminal_distance)
    selected = {
        "junction": int(valid[np.argmin(junction_distance)]),
        "terminal": int(valid[np.argmin(terminal_distance)]),
        "interior": int(valid[np.argmax(event_distance)]),
    }
    result = {}
    for role, current in selected.items():
        indices = np.arange(current - 4, current + 1, dtype=np.int64)
        if len(set(poses.traversal_ids[index] for index in indices)) != 1:
            raise RuntimeError("selected causal window crosses a traversal")
        result[role] = indices
    return result


def _audit_task(task: tuple[str, str, list[dict]]) -> dict:
    world, realization_value, traversal_rows = task
    primary = MESH_ROOT / world / "primary"
    graph = load_json(primary / "graph.json"); splines = load_json(primary / "splines.json"); geometry = load_json(primary / "geometry_parameters.json")
    construction = build_primitive_construction_graph(graph, splines, geometry, endpoint_attachment_mode="free_space_overlap", node_degree_source="edge_incidence")
    realization = GeometryRealization(realization_value)
    primitives_by_realization = {
        value: realize_construction(world, construction, value) for value in GeometryRealization
    }
    fields = {
        value: SweptSuperellipseProvenanceField(primitives_by_realization[value], spacing_m=.025)
        for value in GeometryRealization
    }
    primitives = primitives_by_realization[realization]
    field = fields[realization]
    raycaster = CSGMeshProvenanceRaycaster(
        [mesh_swept_superellipse(value, axial_spacing_m=.05, angular_segments=64) for value in primitives],
        operand_signed_distances=field.operand_signed_distances_sparse,
    )
    poses, corrections = world_finite_union_qualified_frame_poses(
        parent_id=world, traversal_manifest=traversal_rows, graph=graph,
        spline_document=splines, geometry_parameters=geometry,
        signed_distance_fields=[fields[value].signed_distance for value in GeometryRealization],
    )
    if np.any(field.signed_distance(poses.sensor_xyz_m) > 1e-9):
        raise RuntimeError("qualified pose remains outside finite union")
    windows = _select_windows(poses, graph); local = lidar_local_directions().reshape(-1, 3)
    unique_indices = sorted(set(int(index) for values in windows.values() for index in values))
    hits_by_index = {}; codebook = PrimitiveMembershipCodebook(field.primitive_ids)
    total_qualified = total_ambiguous = 0; maximum_source_cardinality = 0
    for index in unique_indices:
        origins = np.broadcast_to(poses.sensor_xyz_m[index], (len(local), 3))
        directions = world_directions(local, float(poses.yaw_deg[index]))
        inside = field.operand_signed_distances_sparse(origins) <= 1e-9
        hits = raycaster.ray_exit_hits(origins, directions, inside, maximum_m=50.)
        codes = codebook.encode(hits); cardinality = codebook.cardinality(codes)
        total_qualified += int(np.sum(cardinality > 0)); total_ambiguous += int(np.sum(cardinality > 1))
        maximum_source_cardinality = max(maximum_source_cardinality, int(np.max(cardinality)))
        if codebook.decode(codes) != tuple(codebook.source_sets[int(value)] for value in codes):
            raise RuntimeError("provenance codebook round trip failed")
        hits_by_index[index] = hits
    window_rows = []
    for role, indices in windows.items():
        hits_by_frame = [hits_by_index[int(index)] for index in indices]
        target = visible_primitive_window_targets(
            field=field, hits_by_frame=hits_by_frame,
            current_sensor_xyz_m=poses.sensor_xyz_m[indices[-1]], current_yaw_deg=float(poses.yaw_deg[indices[-1]]),
            maximum_slots=len(primitives),
        )
        active = target.mask.astype(bool)
        window_rows.append({
            "role": role, "visible_primitive_slots": int(np.sum(active)),
            "minimum_support_rays": int(np.min(target.support_ray_count[active])),
            "maximum_support_rays": int(np.max(target.support_ray_count[active])),
            "traversal_id": poses.traversal_ids[indices[-1]],
            "current_local_frame_index": int(poses.local_frame_indices[indices[-1]]),
        })
    split = "fit" if world.endswith(tuple(f"_C{i:02d}" for i in range(1, 7))) else "c07" if world.endswith("_C07") else "c08"
    return {
        "world": world, "split": split, "realization": realization.value,
        "primitive_count": len(primitives), "unique_rendered_frames": len(unique_indices),
        "corrected_source_frames": len(corrections),
        "rendered_rays": len(unique_indices) * 16 * 720,
        "qualified_hits": total_qualified, "ambiguous_hits": total_ambiguous,
        "maximum_source_cardinality": maximum_source_cardinality,
        "codebook_entries": len(codebook.source_sets), "windows": window_rows,
    }


def _plot(summary: dict, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = summary["window_distribution"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for split, color in (("fit", "#4e79a7"), ("c07", "#f28e2b"), ("c08", "#59a14f")):
        values = [row["visible_primitive_slots"] for row in rows if row["split"] == split]
        bins = np.arange(0, max(values) + 2) - .5
        axes[0].hist(values, bins=bins, histtype="step", linewidth=2, label=split.upper(), color=color)
    axes[0].axvline(8, color="#b33", linestyle="--", label="old capacity 8")
    axes[0].axvline(summary["selected_capacity"], color="#333", linestyle=":", label=f"selected {summary['selected_capacity']}")
    axes[0].set(xlabel="visible primitives / five-frame window", ylabel="windows", title="Target cardinality"); axes[0].legend(frameon=False)
    roles = ("interior", "junction", "terminal")
    axes[1].boxplot([[row["visible_primitive_slots"] for row in rows if row["role"] == role] for role in roles], tick_labels=roles)
    axes[1].axhline(8, color="#b33", linestyle="--"); axes[1].set(ylabel="visible primitives", title="Role-stratified capacity")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("capacity audit executes exactly once")
    started = time.monotonic(); overall, error = FAIL, None
    try:
        if args.workers != 4 or spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("user_authorization", {}).get("status") != "APPROVED": raise RuntimeError("scope/worker/authorization mismatch")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        environment = os.environ.copy(); environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        tests = subprocess.run(
            ["/home/zeng-workstation/anaconda3/bin/python", "-m", "pytest", "-q",
             "tests/v3/unit/test_primitive_construction_supervisor.py", "tests/v3/unit/test_primitive_provenance_field.py",
             "tests/v3/unit/test_swept_superellipse_field.py", "tests/v3/unit/test_geometry_variant_contract.py",
             "tests/v3/unit/test_primitive_relation_dataset.py", "tests/v3/unit/test_gse_sensor_export.py"],
            cwd=PROJECT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (run_dir / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or "40 passed" not in tests.stdout: raise RuntimeError("expected exactly 40 unit tests")
        analytic = subprocess.run(
            [sys.executable, "tools/v3/check_csg_mesh_provenance_contract.py"], cwd=PROJECT_ROOT, env=environment,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (run_dir / "logs/analytic_contract.log").write_text(analytic.stdout, encoding="utf-8")
        if analytic.returncode or "PASS_CSG_MESH_PROVENANCE_ANALYTIC_CONTRACT" not in analytic.stdout: raise RuntimeError("analytic CSG regression")
        card = load_json(PROJECT_ROOT / spec["data_card"]); worlds = tuple(card["worlds"]["train"])
        if len(worlds) != 80 or any(world.endswith(("_C09", "_C10")) for world in worlds): raise RuntimeError("expected exact C01-C08 worlds")
        traversal_groups = _load_traversals()
        tasks = [(world, realization.value, traversal_groups[world]) for world in worlds for realization in GeometryRealization]
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            task_rows = list(pool.map(_audit_task, tasks, chunksize=1))
        window_rows = [{"world": task["world"], "split": task["split"], "realization": task["realization"], **window} for task in task_rows for window in task["windows"]]
        fit_values = [row["visible_primitive_slots"] for row in window_rows if row["split"] == "fit"]
        fit_max = max(fit_values); required = int(math.ceil(1.25 * fit_max))
        selected = next((value for value in CANDIDATE_CAPACITIES if value >= required), None)
        c07_max = max(row["visible_primitive_slots"] for row in window_rows if row["split"] == "c07")
        c08_max = max(row["visible_primitive_slots"] for row in window_rows if row["split"] == "c08")
        counts = {str(value): sum(row["visible_primitive_slots"] > value for row in window_rows) for value in CANDIDATE_CAPACITIES}
        checks = {
            "exact_40_unit_tests": True,
            "analytic_csg_regression_pass": True,
            "exact_80_worlds_240_realizations": len(task_rows) == 240,
            "exact_720_role_windows": len(window_rows) == 720,
            "exact_role_balance": all(sum(row["role"] == role for row in window_rows) == 240 for role in ("interior", "junction", "terminal")),
            "exact_split_windows_540_90_90": [sum(row["split"] == split for row in window_rows) for split in ("fit", "c07", "c08")] == [540, 90, 90],
            "old_capacity_8_empirically_invalid": counts["8"] > 0,
            "selected_capacity_exists_le_32": selected is not None,
            "c07_transfer_fits_unchanged": selected is not None and c07_max <= selected,
            "c08_transfer_fits_unchanged": selected is not None and c08_max <= selected,
            "all_codebooks_uint16_safe": max(row["codebook_entries"] for row in task_rows) < np.iinfo(np.uint16).max,
            "all_source_memberships_nonempty_bounded": 1 <= max(row["maximum_source_cardinality"] for row in task_rows) <= 255,
            "exact_402_paired_pose_corrections": sum(row["corrected_source_frames"] for row in task_rows) == 402,
        }
        overall = PASS if all(checks.values()) else FAIL
        summary = {
            "schema_version": "primitive_slot_capacity_audit_v1r", "overall_status": overall, "scientific_pass": overall == PASS,
            "selection_rule": "smallest of 8/16/32 >= ceil(1.25 * C01-C06 maximum); C07/C08 may only pass/fail unchanged",
            "fit_maximum_visible_primitives": fit_max, "headroom_required_slots": required, "selected_capacity": selected,
            "c07_maximum_visible_primitives": c07_max, "c08_maximum_visible_primitives": c08_max,
            "overflow_window_counts": counts, "maximum_source_membership": max(row["maximum_source_cardinality"] for row in task_rows),
            "task_count": len(task_rows), "window_count": len(window_rows), "rendered_rays": sum(row["rendered_rays"] for row in task_rows),
            "checks": checks, "decision": "FREEZE_SELECTED_FIXED_CAPACITY_FOR_P1" if overall == PASS else "STOP_FIXED_SLOT_DECODER_AND_DESIGN_VARIABLE_SET",
            "window_distribution": window_rows, "c09_c10_worlds_read": 0, "optimizer_steps": 0, "model_inference_frames": 0, "graph_replays": 0,
            "duration_seconds": time.monotonic() - started, "error": None,
        }
        write_json(run_dir / "artifacts/task_results.json", {"tasks": task_rows}); write_json(run_dir / "metrics/summary.json", summary)
        (run_dir / "logs/task_progress.log").write_text("\n".join(json.dumps({"world": row["world"], "realization": row["realization"], "rendered_rays": row["rendered_rays"], "slots": {item["role"]: item["visible_primitive_slots"] for item in row["windows"]}}, sort_keys=True) for row in task_rows) + "\n", encoding="utf-8")
        write_json(run_dir / "config/environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__, "open3d": __import__("open3d").__version__, "workers": args.workers, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS")})
        _plot(summary, run_dir / "previews/primitive_slot_capacity")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8"); write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__": raise SystemExit(main())
