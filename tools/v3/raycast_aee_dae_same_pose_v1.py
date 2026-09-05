#!/usr/bin/env python3
"""Generate the 64 frozen same-pose ideal DAE scans in the Open3D sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_sensor_operator_parity import (
    build_open3d_scene,
    cast_ideal_range,
    evenly_spaced_indices,
)
from mtare_topo.oracle.oriented_dae_support import load_collada_triangle_mesh


SOURCE = PROJECT_ROOT / "results/gate2_representation/gate2_20260820_aee_domain_sensor_export_v1r3_seed20260820"
TRANSLATION = {"tunnel": (-22.0, 92.5, 0.0), "garage": (0.0, 0.0, 0.0)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((SOURCE / "artifacts/data_manifest.json").read_text())
    selected = evenly_spaced_indices(3000, 32)
    output: dict[str, np.ndarray] = {}
    records = []
    for world in ("tunnel", "garage"):
        world_records = [item for item in manifest["records"] if item["world"] == world]
        if len(world_records) != 5 or any(int(item["effective_frames"]) != 600 for item in world_records):
            raise RuntimeError(f"{world} five-trajectory contract drift")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = TRANSLATION[world]
        mesh = load_collada_triangle_mesh(args.assets_dir / f"{world}.dae", transform)
        scene = build_open3d_scene(mesh.vertices_xyz_m, mesh.triangle_vertex_indices, o3d)
        ranges, valid = [], []
        for global_index in selected:
            trajectory_index, row = divmod(int(global_index), 600)
            item = world_records[trajectory_index]
            sensor_path = SOURCE / item["sensor_shard"]
            with np.load(sensor_path, allow_pickle=False) as sensor:
                ideal_range, ideal_valid = cast_ideal_range(
                    scene, sensor["sensor_xyz_m"][row], sensor["sensor_orientation_xyzw"][row], o3d
                )
                frame_id = str(sensor["frame_id"][row])
                raw_index = int(sensor["raw_frame_index"][row])
            ranges.append(ideal_range)
            valid.append(ideal_valid)
            records.append(
                {
                    "world": world,
                    "global_effective_index": int(global_index),
                    "trajectory_id": item["trajectory_id"],
                    "trajectory_row": row,
                    "raw_frame_index": raw_index,
                    "frame_id": frame_id,
                }
            )
        output[f"{world}_range_m"] = np.stack(ranges)
        output[f"{world}_valid_mask"] = np.stack(valid)
        output[f"{world}_selected_global_index"] = selected
        output[f"{world}_mesh_counts"] = np.asarray(
            [len(mesh.vertices_xyz_m), mesh.triangle_count, mesh.source_triangle_count, len(mesh.skipped_zero_area_source_indices)],
            dtype=np.int64,
        )
    output["record_json"] = np.asarray([json.dumps(item, sort_keys=True) for item in records], dtype="U512")
    if len(records) != 64 or len({item["frame_id"] for item in records}) != 64:
        raise RuntimeError("64-frame identity contract failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(json.dumps({"status": "PASS_AEE_DAE_SAME_POSE_RAYCAST_V1", "frames": 64, "open3d": o3d.__version__}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

