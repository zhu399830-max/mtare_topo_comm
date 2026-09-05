#!/usr/bin/env python3
"""Read-only 6,000-frame proof for the oriented-DAE AEE teacher.

This command creates no teacher labels and performs no model inference.  It
audits the sealed V1R3 sensor identities, uses the Gazebo DAE only for upward
support, retains the frozen preview PLY for obstacles, and prints one JSON
proof record after all ten trajectories pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

from mtare_topo.data.aee_domain_adaptation import (
    EFFECTIVE_FRAMES_PER_TRAJECTORY,
    audit_sensor_shard,
    enumerate_aee_domain_trajectories,
    sha256,
)
from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components
from mtare_topo.oracle.layered_gt_map import LayeredGTMapConfig, LayeredGTMapOracle
from mtare_topo.oracle.oriented_dae_support import load_gazebo_collision_mesh_binding


EXPECTED_SOURCE_SEAL_SHA256 = "95b5d7c29183040bffbbad0490e0182ac6e8050ae272c28fa5083a02d435485f"
EXPECTED_SOURCE_MANIFEST_SHA256 = "7de0091cb13fc88166233f3f9d2a99b13e395df27657b276880001adc3f38a04"


def verify_source_seal(project_root: Path, source_run: Path) -> int:
    seal_path = source_run / "artifacts/evidence_sha256.txt"
    if sha256(seal_path) != EXPECTED_SOURCE_SEAL_SHA256:
        raise RuntimeError("source V1R3 seal identity drift")
    source_prefix = source_run.relative_to(project_root).as_posix() + "/"
    entries = 0
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        lowered = relative.lower()
        if "_c09" in lowered or "_c10" in lowered:
            raise RuntimeError(f"forbidden strict-test entry in source seal: {relative}")
        if not relative.startswith(source_prefix):
            raise RuntimeError(f"source seal entry escapes V1R3 run: {relative}")
        if sha256(project_root / relative) != expected:
            raise RuntimeError(f"source seal entry drift: {relative}")
        entries += 1
    if entries != 112:
        raise RuntimeError(f"source seal entry count drift: {entries}")
    return entries


def parse_world_binding(value: str) -> tuple[str, Path, str, Path, str]:
    parts = value.split("=", 4)
    if len(parts) != 5 or parts[0] not in {"tunnel", "garage"}:
        raise argparse.ArgumentTypeError("world binding must be WORLD=PLY=PLY_SHA256=DAE=DAE_SHA256")
    return parts[0], Path(parts[1]), parts[2], Path(parts[3]), parts[4]


def parse_gazebo_binding(value: str) -> tuple[str, Path, str, Path, str, str, str]:
    parts = value.split("=", 6)
    if len(parts) != 7 or parts[0] not in {"tunnel", "garage"}:
        raise argparse.ArgumentTypeError(
            "Gazebo binding must be WORLD=WORLD_FILE=WORLD_SHA256=MODEL_SDF=SDF_SHA256=INCLUDE_URI=MESH_URI"
        )
    return parts[0], Path(parts[1]), parts[2], Path(parts[3]), parts[4], parts[5], parts[6]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--world", required=True, action="append", type=parse_world_binding)
    parser.add_argument("--gazebo-binding", required=True, action="append", type=parse_gazebo_binding)
    args = parser.parse_args()

    started = time.monotonic()
    project_root = args.project_root.resolve()
    source_run = args.source_run.resolve()
    source_run.relative_to(project_root)
    seal_entries = verify_source_seal(project_root, source_run)
    manifest_path = source_run / "artifacts/data_manifest.json"
    if sha256(manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source V1R3 manifest identity drift")
    records = json.loads(manifest_path.read_text(encoding="utf-8")).get("records")
    schedule = enumerate_aee_domain_trajectories()
    if not isinstance(records, list) or len(records) != 10:
        raise RuntimeError("source manifest must contain exactly ten trajectories")

    bindings = {
        world: {"ply": ply, "ply_sha256": ply_hash, "dae": dae, "dae_sha256": dae_hash}
        for world, ply, ply_hash, dae, dae_hash in args.world
    }
    if set(bindings) != {"tunnel", "garage"} or len(args.world) != 2:
        raise RuntimeError("exactly one tunnel and one garage binding are required")
    for world, binding in bindings.items():
        for kind in ("ply", "dae"):
            if sha256(binding[kind]) != binding[f"{kind}_sha256"]:
                raise RuntimeError(f"{world} {kind.upper()} identity drift")
    gazebo_bindings = {
        world: {
            "world_file": world_file,
            "world_sha256": world_hash,
            "model_sdf": model_sdf,
            "model_sdf_sha256": model_hash,
            "include_uri": include_uri,
            "mesh_uri": mesh_uri,
        }
        for world, world_file, world_hash, model_sdf, model_hash, include_uri, mesh_uri
        in args.gazebo_binding
    }
    if set(gazebo_bindings) != {"tunnel", "garage"} or len(args.gazebo_binding) != 2:
        raise RuntimeError("exactly one tunnel and one garage Gazebo binding are required")
    for world, binding in gazebo_bindings.items():
        if sha256(binding["world_file"]) != binding["world_sha256"]:
            raise RuntimeError(f"{world} Gazebo world identity drift")
        if sha256(binding["model_sdf"]) != binding["model_sdf_sha256"]:
            raise RuntimeError(f"{world} Gazebo model.sdf identity drift")

    all_frame_ids: set[str] = set()
    trajectory_proofs: list[dict[str, object]] = []
    world_provenance: dict[str, object] = {}
    split_samples = Counter()
    total_queries = 0
    for world in ("tunnel", "garage"):
        binding = bindings[world]
        gazebo = gazebo_bindings[world]
        coordinate_binding = load_gazebo_collision_mesh_binding(
            gazebo["world_file"],
            gazebo["model_sdf"],
            gazebo["include_uri"],
            gazebo["mesh_uri"],
        )
        oracle = LayeredGTMapOracle.from_ply_and_dae(
            binding["ply"],
            binding["dae"],
            LayeredGTMapConfig(),
            world_from_mesh=coordinate_binding.world_from_mesh,
        )
        world_provenance[world] = {
            "ply_sha256": binding["ply_sha256"],
            "dae_sha256": binding["dae_sha256"],
            "world_sha256": gazebo["world_sha256"],
            "model_sdf_sha256": gazebo["model_sdf_sha256"],
            "gazebo_coordinate_binding": coordinate_binding.provenance(),
            **oracle.support_provenance(),
        }
        for record, expected in zip(records, schedule):
            if record["world"] != world:
                continue
            identity = (
                record.get("trajectory_id"),
                record.get("world"),
                record.get("split"),
                record.get("environment_seed"),
            )
            expected_identity = (
                expected.trajectory_id,
                expected.world,
                expected.split,
                expected.environment_seed,
            )
            if identity != expected_identity:
                raise RuntimeError(f"trajectory schedule drift: {identity}")
            shard_path = source_run / record["sensor_shard"]
            if sha256(shard_path) != record["sensor_shard_sha256"]:
                raise RuntimeError(f"sensor shard identity drift: {expected.trajectory_id}")
            with np.load(shard_path, allow_pickle=False) as archive:
                sensor = {key: np.asarray(archive[key]) for key in archive.files}
            audit = audit_sensor_shard(sensor)
            if not audit["passed"]:
                raise RuntimeError(f"sensor shard contract failed: {expected.trajectory_id}: {audit}")
            frame_ids = sensor["frame_id"].astype(str)
            if not np.all(np.char.startswith(frame_ids, f"{expected.trajectory_id}:")):
                raise RuntimeError(f"frame trajectory prefix drift: {expected.trajectory_id}")
            duplicates = all_frame_ids.intersection(frame_ids.tolist())
            if duplicates:
                raise RuntimeError(f"duplicate global frame identity: {expected.trajectory_id}")

            exit_histogram: Counter[int] = Counter()
            role_histogram: Counter[int] = Counter()
            support_values: list[float] = []
            multilayer_frames = 0
            multilayer_cells = 0
            support_nodes = 0
            reachable_support_nodes = 0
            maximum_reachable_layers = 0
            for sample_index, (xyz, yaw) in enumerate(zip(sensor["sensor_xyz_m"], sensor["yaw_deg"])):
                try:
                    prediction = oracle.predict(xyz, float(yaw))
                except Exception as exc:
                    raise RuntimeError(
                        f"teacher query failed: {expected.trajectory_id} sample {sample_index} "
                        f"frame {frame_ids[sample_index]} pose {np.asarray(xyz).tolist()}: {exc}"
                    ) from exc
                headings = decode_direction_components(prediction.direction_logits, 0.5)
                if prediction.exit_count < 1 or prediction.exit_count > 6:
                    raise RuntimeError(
                        f"empty/out-of-range teacher: {expected.trajectory_id} sample {sample_index}"
                    )
                if len(headings) != prediction.exit_count:
                    raise RuntimeError(
                        f"direction/count inconsistency: {expected.trajectory_id} sample {sample_index}"
                    )
                support_z = float(prediction.evidence.selected_center_support_z_m)
                if not np.isfinite(support_z):
                    raise RuntimeError(
                        f"non-finite support: {expected.trajectory_id} sample {sample_index}"
                    )
                exit_histogram[prediction.exit_count] += 1
                role_histogram[int(np.argmax(prediction.role_probabilities))] += 1
                support_values.append(support_z)
                evidence = prediction.evidence
                multilayer_frames += int(evidence.multilayer_cell_count > 0)
                multilayer_cells += evidence.multilayer_cell_count
                support_nodes += evidence.support_node_count
                reachable_support_nodes += evidence.reachable_support_node_count
                maximum_reachable_layers = max(
                    maximum_reachable_layers, evidence.maximum_reachable_layers_per_cell
                )
                if (sample_index + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "trajectory_id": expected.trajectory_id,
                                "sample_progress": f"{sample_index + 1}/600",
                                "aggregate_queries": total_queries + sample_index + 1,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            samples = len(frame_ids)
            if samples != EFFECTIVE_FRAMES_PER_TRAJECTORY:
                raise RuntimeError(f"effective frame count drift: {expected.trajectory_id}")
            all_frame_ids.update(frame_ids.tolist())
            split_samples[expected.split] += samples
            total_queries += samples
            proof = {
                "trajectory_id": expected.trajectory_id,
                "world": expected.world,
                "split": expected.split,
                "environment_seed": expected.environment_seed,
                "samples": samples,
                "sensor_shard_sha256": record["sensor_shard_sha256"],
                "first_frame_id": frame_ids[0],
                "last_frame_id": frame_ids[-1],
                "exit_count_histogram": {str(k): v for k, v in sorted(exit_histogram.items())},
                "role_histogram": {str(k): v for k, v in sorted(role_histogram.items())},
                "support_z_min_m": min(support_values),
                "support_z_max_m": max(support_values),
                "multilayer_frames": multilayer_frames,
                "multilayer_cells": multilayer_cells,
                "support_nodes": support_nodes,
                "reachable_support_nodes": reachable_support_nodes,
                "maximum_reachable_layers_per_cell": maximum_reachable_layers,
            }
            trajectory_proofs.append(proof)
            print(
                json.dumps(
                    {
                        "progress": f"{len(trajectory_proofs)}/10",
                        "trajectory_id": expected.trajectory_id,
                        "queries": total_queries,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        del oracle

    if total_queries != 6000 or len(all_frame_ids) != 6000:
        raise RuntimeError("aggregate teacher query/frame identity drift")
    if split_samples != Counter({"train": 3000, "validation": 3000}):
        raise RuntimeError(f"split count drift: {dict(split_samples)}")
    result = {
        "schema_version": "aee_dae_support_teacher_readonly_proof_v1",
        "status": "PASS_AEE_DAE_SUPPORT_TEACHER_READONLY_PROOF_V1",
        "method": "GAZEBO_WORLD_SDF_DAE_UPWARD_SUPPORT_PLUS_FROZEN_PLY_OBSTACLES",
        "zero_area_policy": "SKIP_EXACT_CROSS_NORM_ZERO_ONLY_NO_AREA_THRESHOLD",
        "source_sensor_run": str(source_run.relative_to(project_root)),
        "source_seal_sha256": EXPECTED_SOURCE_SEAL_SHA256,
        "source_seal_entries": seal_entries,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "trajectories": 10,
        "raw_frames": 30000,
        "effective_frames": total_queries,
        "unique_frame_ids": len(all_frame_ids),
        "split_samples": dict(split_samples),
        "world_provenance": world_provenance,
        "trajectory_proofs": trajectory_proofs,
        "teacher_shards_written": 0,
        "training_steps": 0,
        "model_inference_frames": 0,
        "c09_reads": 0,
        "c10_reads": 0,
        "duration_seconds": time.monotonic() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
