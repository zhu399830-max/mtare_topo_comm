#!/usr/bin/env python3
"""Export sealed unique LiDAR frames and complete masked GSE teachers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Iterator

from numcodecs import Blosc
import numpy as np
import open3d as o3d
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_dataset_export import pack_world_teacher_targets
from mtare_topo.data.gse_sensor_export import cast_unique_frame_ranges, world_unique_frame_poses
from mtare_topo.governance import load_json, write_json


EXPECTED = {
    "train": {"worlds": 80, "sequences": 188126, "frames": 252430, "references": 940630},
    "validation": {"worlds": 10, "sequences": 24462, "frames": 32678, "references": 122310},
}
EXPECTED_CANDIDATES = 448338
EXPECTED_VISIBLE_TOKENS = 448279
EXPECTED_VISIBLE_WIDTH_VALID = 438968
EXPECTED_ASSOCIATION_IDENTITIES = 5087
EXPECTED_EXIT_IDENTITIES = 17616
EXPECTED_ASSOCIATION_PAIRS = 119390
RAYS_PER_FRAME = 16 * 720


def _scene(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty mesh: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _batch_cast(scene: o3d.t.geometry.RaycastingScene):
    def cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        expanded = np.broadcast_to(origins[:, None, None, :], directions.shape)
        rays = np.concatenate((expanded, directions), axis=3).astype(np.float32, copy=False)
        return (
            scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"]
            .numpy()
            .reshape(directions.shape[:-1])
        )

    return cast


def _grouped_jsonl(path: Path) -> Iterator[tuple[str, list[dict]]]:
    current_parent: str | None = None
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            parent_id = str(row["parent_id"])
            if current_parent is not None and parent_id != current_parent:
                yield current_parent, rows
                rows = []
            current_parent = parent_id
            rows.append(row)
    if current_parent is not None:
        yield current_parent, rows


def _read_grouped(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[str(row["parent_id"])].append(row)
    return dict(result)


def _write_row(stream, row: dict) -> None:
    stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _tree_hash(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    total = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        payload = item.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(hashlib.sha256(payload).digest())
        total += len(payload)
    return digest.hexdigest(), len(files), total


def _identity_maps(
    observation_path: Path,
    exit_path: Path,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    association: set[str] = set()
    observation_to_sequence: dict[str, int] = {}
    with observation_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            observation_id = str(row["observation_id"])
            if observation_id in observation_to_sequence:
                raise RuntimeError("duplicate upstream observation identity")
            observation_to_sequence[observation_id] = int(row["global_sequence_index"])
            if row.get("identity") is not None:
                association.add(str(row["identity"]))
    exits: set[str] = set()
    with exit_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            exits.add(str(json.loads(line)["token"]["identity"]))
    return (
        {identity: index for index, identity in enumerate(sorted(association))},
        {identity: index for index, identity in enumerate(sorted(exits))},
        observation_to_sequence,
    )


def _write_array(group, name: str, values: np.ndarray, *, first_chunk: int, compressor) -> None:
    data = np.asarray(values)
    chunk = max(1, min(int(first_chunk), len(data)))
    chunks = (chunk, *data.shape[1:])
    group.create_dataset(name, data=data, chunks=chunks, compressor=compressor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest-run", required=True, type=Path)
    parser.add_argument("--exit-audit-run", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_artifacts = args.manifest_run.resolve() / "artifacts"
    exit_artifacts = args.exit_audit_run.resolve() / "artifacts"
    mesh_root = args.mesh_root.resolve()
    started = time.monotonic()
    dataset_root = run_dir / "artifacts/dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    observation_path = manifest_artifacts / "teacher_observations.jsonl"
    exit_path = exit_artifacts / "exit_token_audit.jsonl"
    association_map, exit_map, observation_to_sequence = _identity_maps(
        observation_path,
        exit_path,
    )
    if len(association_map) != EXPECTED_ASSOCIATION_IDENTITIES or len(exit_map) != EXPECTED_EXIT_IDENTITIES:
        raise RuntimeError("teacher identity map cardinality drift")
    write_json(run_dir / "artifacts/association_identity_map.json", association_map)
    write_json(run_dir / "artifacts/exit_identity_map.json", exit_map)
    traversals = _read_grouped(manifest_artifacts / "traversal_manifest.jsonl")

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    totals: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
    global_totals: Counter[str] = Counter()
    shard_rows: list[dict] = []
    sequence_path = run_dir / "artifacts/sequence_manifest.jsonl"
    frame_path = run_dir / "artifacts/frame_manifest.jsonl"
    world_path = run_dir / "artifacts/world_dataset_summary.jsonl"
    with sequence_path.open("w", encoding="utf-8") as sequence_stream, frame_path.open(
        "w", encoding="utf-8"
    ) as frame_stream, world_path.open("w", encoding="utf-8") as world_stream:
        observation_groups = _grouped_jsonl(observation_path)
        exit_groups = _grouped_jsonl(exit_path)
        for world_index, (observation_group, exit_group) in enumerate(
            zip(observation_groups, exit_groups, strict=True), start=1
        ):
            observation_parent, observations = observation_group
            exit_parent, exit_rows = exit_group
            if observation_parent != exit_parent or observation_parent not in traversals:
                raise RuntimeError("upstream world group alignment drift")
            parent_id = observation_parent
            split_values = {str(row["split"]) for row in observations}
            if len(split_values) != 1:
                raise RuntimeError("one world crosses data splits")
            split = split_values.pop()
            if split not in EXPECTED or parent_id.endswith("_C10"):
                raise RuntimeError("forbidden split entered GSE dataset export")
            primary = mesh_root / parent_id / "primary"
            graph = load_json(primary / "graph.json")
            splines = load_json(primary / "splines.json")
            geometry = load_json(primary / "geometry_parameters.json")
            poses = world_unique_frame_poses(
                parent_id=parent_id,
                traversal_manifest=traversals[parent_id],
                graph=graph,
                spline_document=splines,
                geometry_parameters=geometry,
            )
            ranges, valid = cast_unique_frame_ranges(
                poses,
                cast_hit_distances=_batch_cast(_scene(primary / "mesh.obj")),
                batch_frames=32,
            )
            packed = pack_world_teacher_targets(
                parent_id=parent_id,
                observations=observations,
                exit_token_rows=exit_rows,
                poses=poses,
                association_identity_to_index=association_map,
                exit_identity_to_index=exit_map,
            )
            shard_path = dataset_root / split / f"{parent_id}.zarr"
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            group = zarr.open_group(str(shard_path), mode="w")
            group.attrs.update(
                {
                    "schema_version": "gse_deduplicated_world_v1",
                    "parent_id": parent_id,
                    "split": split,
                    "student_input": "five referenced range_m/valid_mask frames only",
                    "pose_teacher_only": True,
                    "maximum_exit_tokens": 6,
                }
            )
            _write_array(group, "range_m", ranges, first_chunk=16, compressor=compressor)
            _write_array(group, "valid_mask", valid, first_chunk=32, compressor=compressor)
            frame_arrays = {
                "global_frame_index": poses.global_frame_indices,
                "local_frame_index": poses.local_frame_indices,
                "route_arc_m": poses.arc_m,
                "axis_xyz_m": poses.axis_xyz_m,
                "sensor_xyz_m": poses.sensor_xyz_m,
                "tangent_world_xyz": poses.tangent_world_xyz,
                "yaw_deg": poses.yaw_deg,
            }
            for name, values in frame_arrays.items():
                _write_array(group, name, values, first_chunk=4096, compressor=compressor)
            for name, values in packed.arrays.items():
                _write_array(
                    group,
                    name,
                    values,
                    first_chunk=1024 if name.startswith("exit_") else 4096,
                    compressor=compressor,
                )
            for frame_row, global_index in enumerate(poses.global_frame_indices):
                _write_row(
                    frame_stream,
                    {
                        "parent_id": parent_id,
                        "split": split,
                        "world_frame_row": frame_row,
                        "global_frame_index": int(global_index),
                        "traversal_id": poses.traversal_ids[frame_row],
                        "local_frame_index": int(poses.local_frame_indices[frame_row]),
                        "route_arc_m": float(poses.arc_m[frame_row]),
                    },
                )
            for row in packed.sequence_manifest:
                _write_row(sequence_stream, row)
            tree_sha, file_count, shard_bytes = _tree_hash(shard_path)
            frame_count = len(poses.global_frame_indices)
            sequence_count = len(observations)
            reference_count = sequence_count * 5
            visible_count = int(np.sum(packed.arrays["exit_mask"]))
            visible_width_count = int(np.sum(packed.arrays["exit_width_valid_mask"]))
            valid_return_count = int(np.sum(valid))
            uncompressed_bytes = int(ranges.nbytes + valid.nbytes + sum(value.nbytes for value in frame_arrays.values()) + sum(value.nbytes for value in packed.arrays.values()))
            world_summary = {
                "parent_id": parent_id,
                "split": split,
                "frame_count": frame_count,
                "sequence_count": sequence_count,
                "reference_count": reference_count,
                "candidate_count": len(exit_rows),
                "visible_token_count": visible_count,
                "visible_width_valid_count": visible_width_count,
                "valid_return_count": valid_return_count,
                "full_scan_ray_count": frame_count * RAYS_PER_FRAME,
                "uncompressed_array_bytes": uncompressed_bytes,
                "shard_bytes": shard_bytes,
                "shard_file_count": file_count,
                "shard_tree_sha256": tree_sha,
            }
            _write_row(world_stream, world_summary)
            shard_rows.append(world_summary)
            totals[split].update(
                {
                    "worlds": 1,
                    "frames": frame_count,
                    "sequences": sequence_count,
                    "references": reference_count,
                }
            )
            global_totals.update(
                {
                    "candidates": len(exit_rows),
                    "visible_tokens": visible_count,
                    "visible_width_valid": visible_width_count,
                    "valid_returns": valid_return_count,
                    "full_scan_rays": frame_count * RAYS_PER_FRAME,
                    "uncompressed_array_bytes": uncompressed_bytes,
                    "shard_bytes": shard_bytes,
                    "shard_files": file_count,
                }
            )
            print(
                json.dumps(
                    {
                        "world": parent_id,
                        "index": world_index,
                        "of": 90,
                        "frames": frame_count,
                        "sequences": sequence_count,
                        "shard_mib": shard_bytes / 1024**2,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    pair_count = 0
    pair_keys: set[tuple[int, int, str]] = set()
    pair_path = run_dir / "artifacts/association_pairs_numeric.jsonl"
    with (manifest_artifacts / "association_pairs.jsonl").open("r", encoding="utf-8") as source, pair_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            pair = json.loads(line)
            anchor_id = str(pair["anchor_observation_id"])
            paired_id = str(pair["paired_observation_id"])
            if anchor_id not in observation_to_sequence or paired_id not in observation_to_sequence:
                raise RuntimeError("association pair references an unknown observation")
            row = {
                **pair,
                "anchor_global_sequence_index": observation_to_sequence[anchor_id],
                "paired_global_sequence_index": observation_to_sequence[paired_id],
            }
            key = (
                row["anchor_global_sequence_index"],
                row["paired_global_sequence_index"],
                str(row["pair_kind"]),
            )
            pair_count += 1
            pair_keys.add(key)
            _write_row(destination, row)

    write_json(
        run_dir / "artifacts/shard_manifest.json",
        {
            "schema_version": "gse_shard_manifest_v1",
            "shards": shard_rows,
            "compressor": {"name": "blosc", "cname": "zstd", "clevel": 5, "shuffle": "bitshuffle"},
        },
    )
    checks = {
        "exact_split_counts": all(dict(totals[split]) == expected for split, expected in EXPECTED.items()),
        "exact_world_count": sum(values["worlds"] for values in totals.values()) == 90,
        "exact_candidate_count": global_totals["candidates"] == EXPECTED_CANDIDATES,
        "exact_visible_token_count": global_totals["visible_tokens"] == EXPECTED_VISIBLE_TOKENS,
        "exact_visible_width_valid_count": global_totals["visible_width_valid"] == EXPECTED_VISIBLE_WIDTH_VALID,
        "exact_full_scan_ray_count": global_totals["full_scan_rays"] == 3284444160,
        "exact_association_identity_count": len(association_map) == EXPECTED_ASSOCIATION_IDENTITIES,
        "exact_exit_identity_count": len(exit_map) == EXPECTED_EXIT_IDENTITIES,
        "exact_association_pair_count": pair_count == EXPECTED_ASSOCIATION_PAIRS,
        "all_numeric_pair_keys_unique": pair_count == len(pair_keys),
        "all_shards_nonempty_and_hashed": len(shard_rows) == 90
        and all(row["shard_bytes"] > 0 and len(row["shard_tree_sha256"]) == 64 for row in shard_rows),
        "zero_strict_test_mtare_model_training": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": "gse_deduplicated_dataset_export_v1",
        "overall_status": "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
        if passed
        else "FAIL_GSE_DEDUPLICATED_DATASET_EXPORT_V1",
        "split_totals": {split: dict(values) for split, values in totals.items()},
        "global_totals": dict(global_totals),
        "association_identity_count": len(association_map),
        "exit_identity_count": len(exit_map),
        "association_pair_count": pair_count,
        "checks": checks,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "model_inference_frames": 0,
        "training_samples_consumed": 0,
        "optimizer_steps": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("GSE deduplicated dataset export checks failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
