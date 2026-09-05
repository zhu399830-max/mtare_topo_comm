#!/usr/bin/env python3
"""Execute and seal the identity-preserving Teacher-ray provenance corrective."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions, MAX_RANGE_M
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.primitive_provenance_field import PrimitiveProvenanceField


RUN_ID = "gate3_20260830_primitive_provenance_corrective_v1_seed0"
PASS = "PASS_PRIMITIVE_PROVENANCE_CORRECTIVE_V1"
FAIL = "FAIL_PRIMITIVE_PROVENANCE_CORRECTIVE_V1"
MESH = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes/S01_flat_tree_small_C01/primary"
TEST_PYTHON = Path("/home/zeng-workstation/anaconda3/bin/python")
FIELD_SPACING_M = 0.05
FIXED_PRIMITIVE_INDICES = (0, 1, 2, 3, 4, 5)


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


def _sensor_rays(construction, fta_distance_m: float):
    local = lidar_local_directions()
    origins, directions, poses = [], [], []
    for primitive_index in FIXED_PRIMITIVE_INDICES:
        primitive = construction.primitives[primitive_index]
        points = primitive.centerline_xyz_m
        middle = len(points) // 2
        origin = points[middle].copy()
        origin[2] += float(fta_distance_m) + 1.0
        tangent = points[min(middle + 1, len(points) - 1)] - points[max(middle - 1, 0)]
        yaw_deg = float(np.degrees(np.arctan2(tangent[1], tangent[0])) % 360.0)
        world = world_directions(local, yaw_deg)
        origins.append(np.broadcast_to(origin, world.shape))
        directions.append(world)
        poses.append({
            "pose_index": len(poses), "primitive_id": primitive.primitive_id,
            "origin_xyz_m": origin.astype(float).tolist(), "yaw_deg": yaw_deg,
        })
    return np.stack(origins), np.stack(directions), poses


def _plot(range_m, valid, source_index, cardinality, primitive_ids, destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), constrained_layout=True)
    shown_range = np.where(valid[0] > 0, range_m[0], np.nan)
    image0 = axes[0].imshow(shown_range, aspect="auto", cmap="viridis", vmin=0, vmax=MAX_RANGE_M)
    axes[0].set_title("C01 identity-preserving Teacher range (pose 0)")
    axes[0].set_ylabel("LiDAR elevation row")
    fig.colorbar(image0, ax=axes[0], label="range (m)")
    shown_source = np.where(valid[0] > 0, source_index[0], -1)
    image1 = axes[1].imshow(shown_source, aspect="auto", cmap="tab20")
    ambiguous = cardinality[0] > 1
    yy, xx = np.nonzero(ambiguous)
    axes[1].scatter(xx, yy, s=2, c="white", marker=".", label="multi-source masked")
    axes[1].set_title(f"Primitive hit identity; {len(primitive_ids)} construction operands")
    axes[1].set_xlabel("azimuth column (0.5 deg)")
    axes[1].set_ylabel("LiDAR elevation row")
    if len(xx): axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(image1, ax=axes[1], label="primitive index")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("primitive provenance corrective executes exactly once")
    started = time.monotonic(); overall, error = FAIL, None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("corrective scope/authorization mismatch")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        with (run_dir / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            tests = subprocess.run([str(TEST_PYTHON), "-m", "pytest", "-q", "tests/v3/unit/test_primitive_construction_supervisor.py", "tests/v3/unit/test_primitive_provenance_field.py"], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False, timeout=300)
        if tests.returncode: raise RuntimeError("primitive provenance tests failed")
        graph = load_json(MESH / "graph.json"); splines = load_json(MESH / "splines.json"); geometry = load_json(MESH / "geometry_parameters.json")
        construction = build_primitive_construction_graph(graph, splines, geometry)
        field = PrimitiveProvenanceField(construction, spacing_m=FIELD_SPACING_M)
        origins, directions, poses = _sensor_rays(construction, float(geometry["fta_distance_m"]))
        flat_origins = origins.reshape(-1, 3); flat_directions = directions.reshape(-1, 3)
        first = field.ray_exit_hits(flat_origins, flat_directions, maximum_m=MAX_RANGE_M)
        second = field.ray_exit_hits(flat_origins, flat_directions, maximum_m=MAX_RANGE_M)
        if first != second: raise RuntimeError("Teacher ray provenance is nondeterministic")
        shape = origins.shape[:-1]
        range_m = np.full(shape, MAX_RANGE_M, dtype=np.float32); valid = np.zeros(shape, dtype=np.uint8)
        source_index = np.full(shape, -1, dtype=np.int16); cardinality = np.zeros(shape, dtype=np.uint8)
        id_to_index = {value: index for index, value in enumerate(field.primitive_ids)}
        source_membership = np.zeros(shape + (len(field.primitive_ids),), dtype=np.uint8)
        for index, hit in enumerate(first):
            if hit is None: continue
            multi = np.unravel_index(index, shape); range_m[multi] = hit.distance_m; valid[multi] = 1
            cardinality[multi] = len(hit.source_primitive_ids)
            source_membership[multi + ([id_to_index[value] for value in hit.source_primitive_ids],)] = 1
            if len(hit.source_primitive_ids) == 1:
                source_index[multi] = id_to_index[hit.source_primitive_ids[0]]
        unique = (valid > 0) & (cardinality == 1); ambiguous = cardinality > 1
        np.savez_compressed(
            run_dir / "artifacts/teacher_rays.npz",
            range_m=range_m,
            valid_mask=valid,
            primary_primitive_index=source_index,
            source_cardinality=cardinality,
            source_membership=source_membership,
            primitive_supervision_valid=unique.astype(np.uint8),
        )
        write_json(run_dir / "artifacts/primitive_identity.json", {"primitive_ids": list(field.primitive_ids), "poses": poses, "field_spacing_m": FIELD_SPACING_M})
        _plot(range_m, valid, source_index, cardinality, field.primitive_ids, run_dir / "previews/primitive_provenance_teacher")
        ray_count = int(np.prod(shape)); hit_count = int(valid.sum()); unique_count = int(unique.sum()); ambiguous_count = int(ambiguous.sum())
        checks = {
            "ten_unit_tests_pass": True, "ray_count_exact": ray_count == 6 * 16 * 720,
            "all_hit_sources_nonempty": bool(np.all(cardinality[valid > 0] >= 1)),
            "all_unique_hits_have_valid_index": bool(np.all(source_index[unique] >= 0)),
            "source_membership_matches_cardinality": bool(np.array_equal(source_membership.sum(axis=-1), cardinality)),
            "ambiguous_primary_identity_rejected": bool(np.all(source_index[ambiguous] == -1)),
            "ambiguous_hits_masked": bool(np.all(~unique[ambiguous])),
            "deterministic_repeat": True, "all_ranges_finite_bounded": bool(np.isfinite(range_m).all() and range_m.min() >= 0 and range_m.max() <= MAX_RANGE_M),
        }
        overall = PASS if all(checks.values()) else FAIL
        write_json(run_dir / "metrics/summary.json", {"schema_version": "primitive_provenance_corrective_v1", "overall_status": overall, "scientific_pass": overall == PASS, "checks": checks, "worlds_read": 1, "poses": 6, "rays": ray_count, "hits": hit_count, "unique_primitive_hits": unique_count, "ambiguous_hits_masked": ambiguous_count, "misses_at_50m": ray_count-hit_count, "source_primitives_observed": int(len(np.unique(source_index[source_index >= 0]))), "optimizer_steps": 0, "model_inference_frames": 0, "c07_c10_worlds_read": 0, "graph_replays": 0, "duration_seconds": time.monotonic()-started, "error": None})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__": raise SystemExit(main())
