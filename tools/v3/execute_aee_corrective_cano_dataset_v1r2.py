#!/usr/bin/env python3
"""Materialize the eligibility-first 10,000-frame corrective Cano set."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import zarr
from numcodecs import Blosc

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_corrective_dataset import (
    CANO_FRAMES_PER_WORLD,
    audit_corrective_sensor_shard,
    build_corrective_cano_cluster_eligibility,
    select_qualified_corrective_cano_clusters,
)
from mtare_topo.data.aee_sensor_operator_parity import (
    AEE_GAZEBO_RAW_MAX_RANGE_M,
    aee_gazebo_lidar_directions_sensor,
    resample_aee_gazebo_organized_azimuth,
)
from mtare_topo.data.cano_phase2_dataset import (
    candidate_clusters,
    evaluate_frame,
    frame_contract,
    spline_arrays,
)
from mtare_topo.data.cano_phase2_evidence_audit import compare_scan_replay
from mtare_topo.data.cano_sensor_smoke import MAX_RANGE_M, NEAR_RANGE_M, world_directions
from mtare_topo.governance import load_json, write_json


SOURCE_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260822_aee_corrective_perception_mesh_v1r4_seed20260821"
SOURCE_MESHES = SOURCE_RUN / "artifacts/meshes"
REPLAY_THRESHOLD_M = 1e-6
RAW_NEAR_RANGE_M = 0.1


def scene_from_mesh(path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty corrective mesh: {path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def cast_raw_source(
    scene: o3d.t.geometry.RaycastingScene,
    origin: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origins = np.broadcast_to(np.asarray(origin, dtype=np.float32), directions.shape)
    rays = np.concatenate((origins, np.asarray(directions, dtype=np.float32)), axis=-1)
    hit = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"].numpy().reshape(16, 350)
    raw_valid = np.isfinite(hit) & (hit > RAW_NEAR_RANGE_M) & (hit < AEE_GAZEBO_RAW_MAX_RANGE_M)
    raw_range = np.where(raw_valid, hit, AEE_GAZEBO_RAW_MAX_RANGE_M).astype(np.float32)
    source_valid = np.isfinite(hit) & (hit >= NEAR_RANGE_M) & (hit <= MAX_RANGE_M)
    source_range = np.where(source_valid, hit, MAX_RANGE_M).astype(np.float32)
    model_range, model_valid = resample_aee_gazebo_organized_azimuth(source_range, source_valid)
    return raw_range, raw_valid.astype(np.uint8), model_range, model_valid


def pose7(frame: dict[str, Any]) -> np.ndarray:
    half = np.radians(float(frame["yaw_deg"])) / 2.0
    return np.asarray(
        [*frame["sensor_xyz_m"], 0.0, 0.0, np.sin(half), np.cos(half)],
        dtype=np.float64,
    )


def create_shard(path: Path) -> zarr.Group:
    group = zarr.open_group(str(path), mode="w")
    codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    for name, shape, chunks, dtype in (
        ("raw_range_m", (500, 16, 350), (10, 16, 350), "f4"),
        ("raw_valid_mask", (500, 16, 350), (20, 16, 350), "u1"),
        ("range_m", (500, 16, 720), (10, 16, 720), "f4"),
        ("valid_mask", (500, 16, 720), (20, 16, 720), "u1"),
        ("exit_target", (500, 720), (100, 720), "f4"),
        ("pose_world_sensor", (500, 7), (500, 7), "f8"),
        ("teacher_axis_xyz_m", (500, 3), (500, 3), "f8"),
        ("branch_count", (500,), (500,), "u1"),
    ):
        group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, compressor=codec)
    return group


def plot_train_preview(samples: list[dict[str, Any]], path: Path, parent_id: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for axis, sample in zip(axes.ravel(), samples, strict=True):
        values = np.ma.masked_where(sample["valid"] == 0, sample["range"])
        image = axis.imshow(values, aspect="auto", origin="lower", vmin=0.3, vmax=50.0, cmap="viridis")
        for heading in sample["headings"]:
            axis.axvline((float(heading) % 360.0) / 0.5, color="white", linewidth=1)
        axis.set_title(f"{sample['role']} | branches={len(sample['headings'])}", fontsize=8)
        axis.set_xlabel("azimuth bin")
        axis.set_ylabel("ring")
    fig.colorbar(image, ax=axes, shrink=0.55, label="range m; grey=no return")
    fig.suptitle(f"Gate 2 corrective TRAIN ONLY — {parent_id}")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    started = time.monotonic()
    dataset_root = run / "artifacts/cano_dataset"
    preview_root = run / "previews/cano_train"
    dataset_root.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)
    (run / "metrics/cano").mkdir(parents=True, exist_ok=True)
    manifest_path = run / "artifacts/cano_manifest.jsonl"
    cluster_path = run / "artifacts/cano_clusters.jsonl"
    eligibility_path = run / "artifacts/cano_eligibility.jsonl"
    frame_stream = manifest_path.open("x", encoding="utf-8")
    cluster_stream = cluster_path.open("x", encoding="utf-8")
    eligibility_stream = eligibility_path.open("x", encoding="utf-8")

    mesh_manifest = load_json(SOURCE_RUN / "artifacts/mesh_manifest.json")
    assets = mesh_manifest["assets"]
    if len(assets) != 20:
        raise RuntimeError("corrective mesh source must contain exactly 20 worlds")
    totals = Counter()
    world_metrics = []
    source_directions = aee_gazebo_lidar_directions_sensor()
    for world_index, asset in enumerate(assets):
        parent_id = str(asset["parent_id"])
        split = str(asset["split"])
        source = SOURCE_MESHES / parent_id / "primary"
        graph = load_json(source / "graph.json")
        splines_document = load_json(source / "splines.json")
        splines = spline_arrays(splines_document)
        fta = float(load_json(source / "geometry_parameters.json")["fta_distance_m"])
        candidates = candidate_clusters(parent_id, graph, splines_document)
        scene_a = scene_from_mesh(source / "mesh.obj")
        scene_b = scene_from_mesh(source / "mesh.obj")
        qualifications = []
        for candidate in candidates:
            frame_results = []
            for frame in frame_contract(candidate, splines_document, fta):
                teacher = evaluate_frame(scene_a, frame, splines, o3d)
                frame_results.append(
                    {
                        "frame_index": frame["frame_index"],
                        "eligible": teacher["eligible"],
                        "teacher_eligible": teacher["teacher_eligible"],
                        "native_clearance_passed": teacher["native_clearance_passed"],
                        "minimum_horizontal_clearance_m": teacher["minimum_horizontal_clearance_m"],
                        "branch_los": teacher["branch_los"],
                    }
                )
            qualification = build_corrective_cano_cluster_eligibility(
                str(candidate["cluster_id"]), frame_results
            )
            qualifications.append(qualification)
            eligibility_stream.write(
                json.dumps(
                    {"parent_id": parent_id, "split": split, **qualification.to_dict()},
                    separators=(",", ":"),
                )
                + "\n"
            )
        selected, selector_audit = select_qualified_corrective_cano_clusters(candidates, qualifications)
        replayed, replay_audit = select_qualified_corrective_cano_clusters(
            list(reversed(candidates)), list(reversed(qualifications))
        )
        if selector_audit != replay_audit or [item["cluster_id"] for item in selected] != [item["cluster_id"] for item in replayed]:
            raise RuntimeError(f"{parent_id}: eligibility-first selector deterministic replay failed")
        qualification_by_id = {item.cluster_id: item for item in qualifications}
        shard = create_shard(dataset_root / split / f"{parent_id}.zarr")
        shard.attrs.update(
            {
                "schema_version": "aee_corrective_cano_shard_v1",
                "parent_id": parent_id,
                "split": split,
                "source_mesh_sha256": asset["mesh_sha256"],
                "source_angles": "closed_-pi_plus_pi_350_duplicate_endpoint",
                "student_fields": ["range_m", "valid_mask"],
                "teacher_fields": ["exit_target"],
                "forbidden_student_fields": ["pose_world_sensor", "teacher_axis_xyz_m", "branch_count"],
                "selection_protocol": "objective_geometry_eligibility_then_topology",
            }
        )
        frame_index = 0
        roles = Counter()
        branches = Counter()
        replay_failures = 0
        teacher_failures = 0
        preview_samples: list[dict[str, Any]] = []
        for cluster in selected:
            roles[str(cluster["primary_role"])] += 1
            cluster_stream.write(json.dumps({**cluster, "split": split}, separators=(",", ":")) + "\n")
            for frame in frame_contract(cluster, splines_document, fta):
                directions = world_directions(source_directions, float(frame["yaw_deg"]))
                raw_range, raw_valid, model_range, model_valid = cast_raw_source(
                    scene_a, np.asarray(frame["sensor_xyz_m"]), directions
                )
                raw_range_b, raw_valid_b, model_range_b, model_valid_b = cast_raw_source(
                    scene_b, np.asarray(frame["sensor_xyz_m"]), directions
                )
                replay = compare_scan_replay(
                    model_range,
                    model_valid,
                    model_range,
                    model_valid,
                    model_range_b,
                    model_valid_b,
                    REPLAY_THRESHOLD_M,
                )
                raw_replay_exact = bool(
                    np.array_equal(raw_range, raw_range_b) and np.array_equal(raw_valid, raw_valid_b)
                )
                replay_passed = bool(replay["passed"] and raw_replay_exact)
                replay_failures += int(not replay_passed)
                teacher = evaluate_frame(scene_a, frame, splines, o3d)
                qualified_frame = qualification_by_id[str(cluster["cluster_id"])].frames[int(frame["frame_index"])]
                qualification_replay_passed = bool(
                    qualified_frame.eligible == bool(teacher["eligible"])
                    and qualified_frame.teacher_eligible == bool(teacher["teacher_eligible"])
                    and qualified_frame.native_clearance_passed == bool(teacher["native_clearance_passed"])
                    and qualified_frame.minimum_horizontal_clearance_m
                    == float(teacher["minimum_horizontal_clearance_m"])
                    and qualified_frame.branch_los_passed
                    == bool(teacher["branch_los"] and all(teacher["branch_los"]))
                )
                teacher_passed = bool(teacher["eligible"])
                teacher_failures += int(not teacher_passed)
                if not replay_passed or not teacher_passed or not qualification_replay_passed:
                    raise RuntimeError(
                        f"{parent_id}:{cluster['cluster_id']}:{frame['frame_index']} failed replay/teacher/qualification contract"
                    )
                shard["raw_range_m"][frame_index] = raw_range
                shard["raw_valid_mask"][frame_index] = raw_valid
                shard["range_m"][frame_index] = model_range
                shard["valid_mask"][frame_index] = model_valid
                shard["exit_target"][frame_index] = teacher["exit_target"]
                shard["pose_world_sensor"][frame_index] = pose7(frame)
                shard["teacher_axis_xyz_m"][frame_index] = frame["axis_xyz_m"]
                shard["branch_count"][frame_index] = teacher["branch_count"]
                branches[int(teacher["branch_count"])] += 1
                frame_id = f"{cluster['cluster_id']}_f{frame['frame_index']}"
                record = {
                    "frame_id": frame_id,
                    "parent_id": parent_id,
                    "split": split,
                    "cluster_id": cluster["cluster_id"],
                    "tunnel_id": cluster["tunnel_id"],
                    "primary_role": cluster["primary_role"],
                    "near_junction": cluster["near_junction"],
                    "near_terminal": cluster["near_terminal"],
                    "frame_in_cluster": frame["frame_index"],
                    "arc_m": frame["arc_m"],
                    "yaw_deg": frame["yaw_deg"],
                    "branch_count": teacher["branch_count"],
                    "headings_robot_deg": teacher["headings_robot_deg"],
                    "minimum_horizontal_clearance_m": teacher["minimum_horizontal_clearance_m"],
                    "branch_los": teacher["branch_los"],
                    "raw_replay_exact": raw_replay_exact,
                    "model_replay": replay,
                    "qualification_replay_exact": qualification_replay_passed,
                    "zarr_row": frame_index,
                }
                frame_stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                if split == "corrective_train" and frame["frame_index"] == 2 and len(preview_samples) < 10:
                    preview_samples.append(
                        {
                            "range": model_range,
                            "valid": model_valid,
                            "headings": teacher["headings_robot_deg"],
                            "role": cluster["primary_role"],
                        }
                    )
                frame_index += 1
        if frame_index != CANO_FRAMES_PER_WORLD:
            raise RuntimeError(f"{parent_id}: expected 500 frames, observed {frame_index}")
        sensor_audit = audit_corrective_sensor_shard(
            {
                "raw_range_m": shard["raw_range_m"][:],
                "raw_valid_mask": shard["raw_valid_mask"][:],
                "range_m": shard["range_m"][:],
                "valid_mask": shard["valid_mask"][:],
                "sensor_xyz_m": shard["pose_world_sensor"][:, :3],
                "sensor_orientation_xyzw": shard["pose_world_sensor"][:, 3:],
                "yaw_deg": np.asarray(
                    [record["yaw_deg"] for cluster in selected for record in frame_contract(cluster, splines_document, fta)],
                    dtype=np.float64,
                ),
                "frame_id": np.asarray(
                    [f"{cluster['cluster_id']}_f{index}" for cluster in selected for index in range(5)]
                ),
            },
            CANO_FRAMES_PER_WORLD,
        )
        if not sensor_audit["passed"]:
            raise RuntimeError(f"{parent_id}: final corrective sensor audit failed: {sensor_audit}")
        if split == "corrective_train":
            if len(preview_samples) != 10:
                raise RuntimeError(f"{parent_id}: incomplete train preview")
            plot_train_preview(preview_samples, preview_root / f"{parent_id}.png", parent_id)
        elif preview_samples:
            raise RuntimeError(f"{parent_id}: validation preview leakage")
        metric = {
            "schema_version": "aee_corrective_cano_world_metric_v1",
            "parent_id": parent_id,
            "split": split,
            "candidate_clusters": len(candidates),
            "eligible_clusters": selector_audit["eligible_clusters"],
            "ineligible_clusters": selector_audit["ineligible_clusters"],
            "ineligible_frames": selector_audit["ineligible_frames"],
            "selected_clusters": len(selected),
            "frames": frame_index,
            "role_counts": dict(roles),
            "branch_counts": dict(branches),
            "selector_audit": selector_audit,
            "sensor_audit": sensor_audit,
            "raw_replay_failures": replay_failures,
            "teacher_failures": teacher_failures,
        }
        write_json(run / f"metrics/cano/{parent_id}.json", metric)
        world_metrics.append(metric)
        totals.update(
            {
                "worlds": 1,
                "candidates": len(candidates),
                "eligible_clusters": selector_audit["eligible_clusters"],
                "ineligible_clusters": selector_audit["ineligible_clusters"],
                "ineligible_frames": selector_audit["ineligible_frames"],
                "clusters": len(selected),
                "frames": frame_index,
            }
        )
        print(json.dumps({"world": parent_id, "index": world_index + 1, "of": 20, "frames": frame_index}), flush=True)
    frame_stream.close()
    cluster_stream.close()
    eligibility_stream.close()
    expected_totals = Counter(
        {
            "frames": 10_000,
            "clusters": 2_000,
            "candidates": 6_058,
            "eligible_clusters": 6_043,
            "ineligible_clusters": 15,
            "ineligible_frames": 21,
            "worlds": 20,
        }
    )
    if totals != expected_totals:
        raise RuntimeError(f"aggregate corrective Cano counts drift: {dict(totals)}")
    summary = {
        "schema_version": "aee_corrective_cano_dataset_summary_v1",
        "overall_status": "PASS_AEE_CORRECTIVE_CANO_DATASET_V1R2",
        "worlds": 20,
        "train_worlds": 10,
        "validation_worlds": 10,
        "clusters": 2_000,
        "selected_clusters": 2_000,
        "candidate_clusters": 6_058,
        "candidate_frames_qualified": 30_290,
        "eligible_clusters": 6_043,
        "eligible_frames": 30_215,
        "ineligible_clusters": 15,
        "ineligible_frames": 21,
        "frames": 10_000,
        "teacher_labels": 10_000,
        "raw_source_rays": 56_000_000,
        "dual_scene_raw_source_rays": 112_000_000,
        "eligibility_teacher_rays": 348_940_800,
        "train_previews": 10,
        "validation_previews": 0,
        "training_steps": 0,
        "models": 0,
        "c09_reads": 0,
        "c10_reads": 0,
        "mtare_changes": 0,
        "duration_seconds": time.monotonic() - started,
        "world_metrics": world_metrics,
    }
    write_json(run / "metrics/cano_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
