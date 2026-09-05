#!/usr/bin/env python3
"""Formal P1a streaming export of paired LiDAR and primitive provenance."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from numcodecs import Blosc
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_sensor_export import FramePoseCorrection, world_frame_poses_with_corrections
from mtare_topo.data.primitive_relation_dataset import PrimitiveMembershipCodebook
from mtare_topo.data.primitive_relation_sensor_export import render_primitive_sensor_frame
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster, mesh_swept_superellipse
from mtare_topo.teacher.geometry_variant_contract import GeometryRealization, realize_construction
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


RUN_ID = "gate3_20260830_primitive_relation_p1a_sensor_provenance_export_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_P1A_SENSOR_PROVENANCE_EXPORT_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_P1A_SENSOR_PROVENANCE_EXPORT_V1"
MESH_ROOT = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
TRAVERSAL_PATH = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/traversal_manifest.jsonl"
CORRECTION_PATH = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_finite_cap_pose_qualification_v1_seed0/artifacts/frame_pose_corrections.json"
RAYS_PER_FRAME = 16 * 720


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256(); files = sorted(item for item in path.rglob("*") if item.is_file()); total = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode(); file_digest = bytes.fromhex(_sha(item))
        digest.update(len(relative).to_bytes(4, "little")); digest.update(relative); digest.update(file_digest)
        total += item.stat().st_size
    return digest.hexdigest(), len(files), total


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


def _partition(world: str) -> str:
    if world.endswith(tuple(f"_C{index:02d}" for index in range(1, 7))): return "fit"
    if world.endswith("_C07"): return "c07"
    if world.endswith("_C08"): return "c08"
    raise ValueError("P1a permits only C01-C08")


def _write_array(group, name: str, values: np.ndarray, *, first_chunk: int, compressor) -> None:
    data = np.asarray(values); chunk = max(1, min(first_chunk, len(data)))
    group.create_dataset(name, data=data, chunks=(chunk, *data.shape[1:]), compressor=compressor)


def _export_task(task: dict) -> dict:
    started = time.monotonic(); world = str(task["world"]); realization = GeometryRealization(str(task["realization"])); output_root = Path(task["output_root"])
    primary = MESH_ROOT / world / "primary"; graph = load_json(primary / "graph.json"); splines = load_json(primary / "splines.json"); geometry = load_json(primary / "geometry_parameters.json")
    construction = build_primitive_construction_graph(graph, splines, geometry, endpoint_attachment_mode="free_space_overlap", node_degree_source="edge_incidence")
    primitives = realize_construction(world, construction, realization); field = SweptSuperellipseProvenanceField(primitives, spacing_m=.025)
    raycaster = CSGMeshProvenanceRaycaster(
        [mesh_swept_superellipse(value, axial_spacing_m=.05, angular_segments=64) for value in primitives],
        operand_signed_distances=field.operand_signed_distances_sparse,
    )
    corrections = tuple(FramePoseCorrection(**row) for row in task["corrections"])
    poses = world_frame_poses_with_corrections(
        parent_id=world, traversal_manifest=task["traversals"], graph=graph,
        spline_document=splines, geometry_parameters=geometry, corrections=corrections,
    )
    origin_residual = field.signed_distance(poses.sensor_xyz_m)
    if np.any(origin_residual > 1e-9): raise RuntimeError("sealed corrected origin lies outside task geometry")
    task_id = f"{world}__{realization.value}"; partition = _partition(world)
    shard_path = output_root / "dataset" / partition / f"{task_id}.zarr"
    codebook_path = output_root / "codebooks" / partition / f"{task_id}.json"
    construction_path = output_root / "constructions" / partition / f"{task_id}.json"
    if shard_path.exists() or codebook_path.exists() or construction_path.exists(): raise RuntimeError("P1a task output already exists")
    shard_path.parent.mkdir(parents=True, exist_ok=True); codebook_path.parent.mkdir(parents=True, exist_ok=True); construction_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE); frame_count = len(poses.global_frame_indices)
    group = zarr.open_group(str(shard_path), mode="w")
    group.attrs.update({
        "schema_version": "primitive_relation_p1a_sensor_shard_v1", "parent_id": world,
        "partition": partition, "geometry_realization": realization.value,
        "sensor_shape": [16, 720], "maximum_range_m": 50.0,
        "primitive_membership_semantics": "world-local uint16 code retaining every qualified union-exit source primitive",
        "pose_correction_count": len(corrections), "student_pose_input_forbidden": True,
    })
    range_array = group.create_dataset("range_m", shape=(frame_count, 16, 720), chunks=(16, 16, 720), dtype="f4", compressor=compressor)
    valid_array = group.create_dataset("valid_mask", shape=(frame_count, 16, 720), chunks=(32, 16, 720), dtype="u1", compressor=compressor)
    code_array = group.create_dataset("primitive_membership_code", shape=(frame_count, 16, 720), chunks=(16, 16, 720), dtype="u2", compressor=compressor)
    traversal_id_to_index = {value: index for index, value in enumerate(dict.fromkeys(poses.traversal_ids))}
    frame_arrays = {
        "global_frame_index": poses.global_frame_indices, "local_frame_index": poses.local_frame_indices,
        "traversal_index": np.asarray([traversal_id_to_index[value] for value in poses.traversal_ids], dtype=np.int32),
        "route_arc_m": poses.arc_m, "axis_xyz_m": poses.axis_xyz_m,
        "sensor_xyz_m": poses.sensor_xyz_m, "tangent_world_xyz": poses.tangent_world_xyz,
        "yaw_deg": poses.yaw_deg,
    }
    for name, values in frame_arrays.items(): _write_array(group, name, values, first_chunk=4096, compressor=compressor)
    group.attrs["traversal_ids"] = list(traversal_id_to_index)
    codebook = PrimitiveMembershipCodebook(field.primitive_ids); valid_returns = ambiguous_rays = 0; first_digest = None
    for start in range(0, frame_count, 16):
        stop = min(start + 16, frame_count); batch_range = np.empty((stop-start, 16, 720), dtype=np.float32); batch_valid = np.empty((stop-start, 16, 720), dtype=np.uint8); batch_code = np.empty((stop-start, 16, 720), dtype=np.uint16)
        for local, frame_index in enumerate(range(start, stop)):
            rendered = render_primitive_sensor_frame(
                raycaster=raycaster, field=field, codebook=codebook,
                sensor_xyz_m=poses.sensor_xyz_m[frame_index], yaw_deg=float(poses.yaw_deg[frame_index]),
            )
            batch_range[local] = rendered.range_m; batch_valid[local] = rendered.valid_mask; batch_code[local] = rendered.primitive_membership_code
            valid_returns += int(np.sum(rendered.valid_mask)); ambiguous_rays += rendered.ambiguous_ray_count
        range_array[start:stop] = batch_range; valid_array[start:stop] = batch_valid; code_array[start:stop] = batch_code
        if start == 0:
            digest = hashlib.sha256(); digest.update(batch_range[0].tobytes()); digest.update(batch_valid[0].tobytes()); digest.update(batch_code[0].tobytes()); first_digest = digest.hexdigest()
    entries_before = len(codebook.source_sets)
    repeated = render_primitive_sensor_frame(
        raycaster=raycaster, field=field, codebook=codebook,
        sensor_xyz_m=poses.sensor_xyz_m[0], yaw_deg=float(poses.yaw_deg[0]),
    )
    repeated_digest = hashlib.sha256(repeated.range_m.tobytes() + repeated.valid_mask.tobytes() + repeated.primitive_membership_code.tobytes()).hexdigest()
    if repeated_digest != first_digest or len(codebook.source_sets) != entries_before: raise RuntimeError("first-frame deterministic replay drift")
    write_json(codebook_path, {**codebook.as_dict(), "parent_id": world, "geometry_realization": realization.value})
    write_json(construction_path, {
        "schema_version": "primitive_relation_realized_construction_v1", "parent_id": world,
        "geometry_realization": realization.value, "base_construction": construction.as_dict(),
        "realized_primitives": [value.as_dict() for value in primitives],
    })
    reopened = zarr.open_group(str(shard_path), mode="r")
    if reopened["range_m"].shape != (frame_count, 16, 720) or int(np.sum(reopened["valid_mask"][:])) != valid_returns: raise RuntimeError("written shard verification failed")
    tree_sha, file_count, shard_bytes = _tree_hash(shard_path)
    maximum_membership = max(len(value) for value in codebook.source_sets)
    return {
        "task_id": task_id, "world": world, "partition": partition, "realization": realization.value,
        "frames": frame_count, "rays": frame_count * RAYS_PER_FRAME,
        "valid_returns": valid_returns, "ambiguous_rays": ambiguous_rays,
        "primitive_count": len(primitives), "codebook_entries": len(codebook.source_sets),
        "maximum_source_membership": maximum_membership, "pose_corrections": len(corrections),
        "maximum_origin_residual_m": float(np.max(origin_residual)),
        "first_frame_digest": first_digest, "deterministic_replay": repeated_digest == first_digest,
        "shard_tree_sha256": tree_sha, "shard_files": file_count, "shard_bytes": shard_bytes,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "duration_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); parser.add_argument("--workers", type=int, default=28); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("P1a export executes exactly once")
    started = time.monotonic(); overall, error = FAIL, None
    try:
        if args.workers != 28 or spec.get("gate") != 3 or spec.get("operation") != "data_export" or spec.get("user_authorization", {}).get("status") != "APPROVED": raise RuntimeError("scope/worker/authorization mismatch")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        environment = os.environ.copy(); environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        test_paths = [
            "tests/v3/unit/test_governance.py",
            "tests/v3/unit/test_primitive_construction_supervisor.py", "tests/v3/unit/test_primitive_provenance_field.py",
            "tests/v3/unit/test_swept_superellipse_field.py", "tests/v3/unit/test_geometry_variant_contract.py",
            "tests/v3/unit/test_primitive_relation_dataset.py", "tests/v3/unit/test_gse_sensor_export.py",
            "tests/v3/unit/test_primitive_relation_sensor_export.py",
        ]
        tests = subprocess.run(["/home/zeng-workstation/anaconda3/bin/python", "-m", "pytest", "-q", *test_paths], cwd=PROJECT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (run_dir / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or "67 passed" not in tests.stdout: raise RuntimeError("expected exactly 67 unit tests")
        analytic = subprocess.run([sys.executable, "tools/v3/check_csg_mesh_provenance_contract.py"], cwd=PROJECT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (run_dir / "logs/analytic_contract.log").write_text(analytic.stdout, encoding="utf-8")
        if analytic.returncode or "PASS_CSG_MESH_PROVENANCE_ANALYTIC_CONTRACT" not in analytic.stdout: raise RuntimeError("analytic CSG regression")
        card = load_json(PROJECT_ROOT / spec["data_card"]); worlds = tuple(card["worlds"]["train"])
        if len(worlds) != 80 or any(value.endswith(("_C09", "_C10")) for value in worlds): raise RuntimeError("expected exact C01-C08 population")
        traversal_groups = _load_traversals(); correction_document = load_json(CORRECTION_PATH)
        correction_groups: dict[str, list[dict]] = {world: [] for world in worlds}
        for row in correction_document["corrections"]: correction_groups[str(row["world"])].append({key: row[key] for key in ("global_frame_index", "traversal_id", "local_frame_index", "original_arc_m", "corrected_arc_m")})
        output_root = run_dir / "artifacts"; tasks = [
            {"world": world, "realization": realization.value, "traversals": traversal_groups[world], "corrections": correction_groups[world], "output_root": str(output_root)}
            for world in worlds for realization in GeometryRealization
        ]
        progress_path = run_dir / "logs/task_progress.jsonl"; rows = []
        context = mp.get_context("spawn")
        with progress_path.open("w", encoding="utf-8") as progress, ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            future_to_task = {pool.submit(_export_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                row = future.result(); rows.append(row); progress.write(json.dumps(row, sort_keys=True) + "\n"); progress.flush()
                print(json.dumps({"completed": len(rows), "of": len(tasks), "task": row["task_id"], "frames": row["frames"], "mib": row["shard_bytes"] / 1024**2}, sort_keys=True), flush=True)
        rows.sort(key=lambda row: row["task_id"])
        total_bytes = sum(row["shard_bytes"] for row in rows)
        partition_tasks = {partition: sum(row["partition"] == partition for row in rows) for partition in ("fit", "c07", "c08")}
        checks = {
            "exact_80_worlds_240_tasks": len(rows) == 240 and len({row["world"] for row in rows}) == 80,
            "exact_partition_tasks_180_30_30": partition_tasks == {"fit": 180, "c07": 30, "c08": 30},
            "exact_757290_frames": sum(row["frames"] for row in rows) == 757290,
            "exact_8723980800_rays": sum(row["rays"] for row in rows) == 8723980800,
            "exact_24117_primitives": sum(row["primitive_count"] for row in rows) == 24117,
            "exact_402_paired_pose_corrections": sum(row["pose_corrections"] for row in rows) == 402,
            "all_origins_inside_1e9": max(row["maximum_origin_residual_m"] for row in rows) <= 1e-9,
            "all_deterministic_replays": all(row["deterministic_replay"] for row in rows),
            "all_provenance_nonempty_uint16_safe": all(1 < row["codebook_entries"] < np.iinfo(np.uint16).max for row in rows),
            "source_membership_uint8_bounded_and_ambiguous_preserved": 1 <= max(row["maximum_source_membership"] for row in rows) <= 255 and sum(row["ambiguous_rays"] for row in rows) > 0,
            "all_shards_nonempty_hashed": all(row["shard_bytes"] > 0 and len(row["shard_tree_sha256"]) == 64 for row in rows),
            "result_bytes_le_20gib": total_bytes <= 20 * 1024**3,
        }
        overall = PASS if all(checks.values()) else FAIL
        write_json(run_dir / "artifacts/task_manifest.json", {"schema_version": "primitive_relation_p1a_task_manifest_v1", "tasks": rows})
        shutil.copy2(TRAVERSAL_PATH, run_dir / "artifacts/traversal_manifest.jsonl"); shutil.copy2(CORRECTION_PATH, run_dir / "artifacts/frame_pose_corrections.json")
        summary = {
            "schema_version": "primitive_relation_p1a_sensor_provenance_export_v1", "overall_status": overall,
            "scientific_pass": overall == PASS, "checks": checks, "task_count": len(rows),
            "partition_tasks": partition_tasks, "frames": sum(row["frames"] for row in rows),
            "planned_sequences": 564378, "rays": sum(row["rays"] for row in rows),
            "valid_returns": sum(row["valid_returns"] for row in rows), "ambiguous_rays": sum(row["ambiguous_rays"] for row in rows),
            "primitives": sum(row["primitive_count"] for row in rows), "paired_pose_corrections": sum(row["pose_corrections"] for row in rows),
            "maximum_source_membership": max(row["maximum_source_membership"] for row in rows),
            "maximum_task_peak_rss_kib": max(row["peak_rss_kib"] for row in rows),
            "dataset_bytes": total_bytes, "workers": args.workers,
            "decision": "ALLOW_P1B_RELATION_TEACHER_MATERIALIZATION" if overall == PASS else "STOP_P1A_EXPORT_FAILED",
            "c09_c10_worlds_read": 0, "optimizer_steps": 0, "model_inference_frames": 0, "graph_replays": 0,
            "duration_seconds": time.monotonic() - started, "error": None,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "config/environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__, "zarr": zarr.__version__, "workers": args.workers, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS")})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8"); write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__": raise SystemExit(main())
