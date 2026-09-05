#!/usr/bin/env python3
"""Generate one AEE teacher shard with DAE support and PLY obstacles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mtare_topo.data.aee_domain_adaptation import audit_sensor_shard, sha256
from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components
from mtare_topo.oracle.layered_gt_map import LayeredGTMapConfig, LayeredGTMapOracle
from mtare_topo.oracle.oriented_dae_support import load_gazebo_collision_mesh_binding


STATUS_PASS = "PASS_AEE_DAE_MULTILAYER_OBJECTIVE_TEACHER_SHARD_V1R"


def compact_support_provenance(oracle: LayeredGTMapOracle) -> dict[str, object]:
    provenance = oracle.support_provenance()
    names = provenance.pop("support_surface_names", [])
    encoded = json.dumps(names, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    provenance["support_surface_names_sha256"] = hashlib.sha256(encoded).hexdigest()
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-shard", required=True, type=Path)
    parser.add_argument("--sensor-shard-sha256", required=True)
    parser.add_argument("--obstacle-map", required=True, type=Path)
    parser.add_argument("--obstacle-map-sha256", required=True)
    parser.add_argument("--support-mesh", required=True, type=Path)
    parser.add_argument("--support-mesh-sha256", required=True)
    parser.add_argument("--world-file", required=True, type=Path)
    parser.add_argument("--world-file-sha256", required=True)
    parser.add_argument("--model-sdf", required=True, type=Path)
    parser.add_argument("--model-sdf-sha256", required=True)
    parser.add_argument("--include-uri", required=True)
    parser.add_argument("--mesh-uri", required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    inputs = {
        "sensor shard": (args.sensor_shard.resolve(), args.sensor_shard_sha256),
        "obstacle map": (args.obstacle_map.resolve(), args.obstacle_map_sha256),
        "support mesh": (args.support_mesh.resolve(), args.support_mesh_sha256),
        "world file": (args.world_file.resolve(), args.world_file_sha256),
        "model SDF": (args.model_sdf.resolve(), args.model_sdf_sha256),
    }
    for label, (path, expected_hash) in inputs.items():
        if sha256(path) != expected_hash:
            raise RuntimeError(f"{label} identity drift")

    with np.load(inputs["sensor shard"][0], allow_pickle=False) as source:
        sensor = {key: np.asarray(source[key]) for key in source.files}
    audit = audit_sensor_shard(sensor)
    if not audit["passed"]:
        raise RuntimeError(f"sensor shard audit failed: {audit}")
    frame_ids = np.asarray(sensor["frame_id"]).astype(str)
    if not np.all(np.char.startswith(frame_ids, f"{args.trajectory_id}:")):
        raise RuntimeError("trajectory identity does not match sensor frame IDs")

    binding = load_gazebo_collision_mesh_binding(
        inputs["world file"][0],
        inputs["model SDF"][0],
        args.include_uri,
        args.mesh_uri,
    )
    oracle = LayeredGTMapOracle.from_ply_and_dae(
        inputs["obstacle map"][0],
        inputs["support mesh"][0],
        LayeredGTMapConfig(),
        world_from_mesh=binding.world_from_mesh,
    )

    direction: list[np.ndarray] = []
    count: list[int] = []
    role: list[int] = []
    exit_counts: list[int] = []
    heading_counts: list[int] = []
    support_z: list[float] = []
    multilayer_frames = 0
    maximum_reachable_layers = 0
    for index, (xyz, yaw) in enumerate(zip(sensor["sensor_xyz_m"], sensor["yaw_deg"])):
        prediction = oracle.predict(xyz, float(yaw))
        headings = decode_direction_components(prediction.direction_logits, 0.5)
        if prediction.exit_count < 1 or prediction.exit_count > 6 or len(headings) != prediction.exit_count:
            raise RuntimeError(f"nonunique or empty teacher at sample {index}")
        evidence = prediction.evidence
        direction.append((prediction.direction_logits > 0.0).astype(np.uint8))
        count.append(prediction.exit_count - 1)
        role.append(int(np.argmax(prediction.role_probabilities)))
        exit_counts.append(prediction.exit_count)
        heading_counts.append(len(headings))
        support_z.append(float(evidence.selected_center_support_z_m))
        multilayer_frames += int(evidence.multilayer_cell_count > 0)
        maximum_reachable_layers = max(
            maximum_reachable_layers, evidence.maximum_reachable_layers_per_cell
        )

    payload = {
        "direction_target": np.stack(direction).astype(np.uint8),
        "count_target": np.asarray(count, dtype=np.int64),
        "role_target": np.asarray(role, dtype=np.int64),
        "exit_count": np.asarray(exit_counts, dtype=np.int64),
        "heading_count": np.asarray(heading_counts, dtype=np.int64),
        "support_z_m": np.asarray(support_z, dtype=np.float64),
        "frame_id": sensor["frame_id"],
        "raw_frame_index": sensor["raw_frame_index"],
    }
    output = args.output.resolve()
    summary_path = output.with_suffix(".summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("teacher output is immutable and already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "schema_version": "aee_dae_multilayer_objective_teacher_shard_summary_v1r",
        "status": STATUS_PASS,
        "trajectory_id": args.trajectory_id,
        "samples": len(direction),
        "teacher_queries": len(direction),
        "nonempty_samples": sum(value > 0 for value in exit_counts),
        "exit_count_histogram": {str(value): exit_counts.count(value) for value in sorted(set(exit_counts))},
        "role_histogram": {str(value): role.count(value) for value in sorted(set(role))},
        "direction_heading_identity": bool(np.array_equal(payload["exit_count"], payload["heading_count"])),
        "sensor_shard_sha256": args.sensor_shard_sha256,
        "complete_map_sha256": args.obstacle_map_sha256,
        "obstacle_map_sha256": args.obstacle_map_sha256,
        "support_mesh_sha256": args.support_mesh_sha256,
        "world_file_sha256": args.world_file_sha256,
        "model_sdf_sha256": args.model_sdf_sha256,
        "gazebo_coordinate_binding": binding.provenance(),
        "support_provenance": compact_support_provenance(oracle),
        "multilayer_frames": multilayer_frames,
        "maximum_reachable_layers_per_cell": maximum_reachable_layers,
        "zero_area_policy": "SKIP_EXACT_CROSS_NORM_ZERO_ONLY_NO_AREA_THRESHOLD",
        "teacher_shard_sha256": sha256(output),
        "teacher_shard_bytes": output.stat().st_size,
        "training_steps": 0,
        "model_inference_frames": 0,
        "c09_reads": 0,
        "c10_reads": 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
