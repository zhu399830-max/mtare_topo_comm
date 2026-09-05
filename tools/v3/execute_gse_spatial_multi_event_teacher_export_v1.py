#!/usr/bin/env python3
"""Export the sealed C01-C08 native-LOS spatial event set Teacher."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numcodecs import Blosc
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from execute_gse_spatial_multi_event_teacher_feasibility_v1 import (
    LOS_MARGIN_M,
    MAX_EVENT_RANGE_M,
    _cast_pairs,
    _event_nodes,
    _family,
    _partition,
    _scene,
    _sha256,
    _tree_hash,
    _world_suffix,
)
from mtare_topo.governance import load_json, write_json


PASS = "PASS_GSE_SPATIAL_MULTI_EVENT_TEACHER_EXPORT_V1"
FAIL = "FAIL_GSE_SPATIAL_MULTI_EVENT_TEACHER_EXPORT_V1"
MAXIMUM_EVENT_TOKENS = 16
EXPECTED_WORLDS = 80
EXPECTED_OBSERVATIONS = 188_126
EXPECTED_IDENTITIES = 1_076
EXPECTED_VISIBLE_TOKENS = 133_055
EXPECTED_EVENT_TOKEN_COUNTS = {"terminal": 30_789, "junction": 102_266}
EVENT_TYPE_TO_INDEX = {"terminal": 0, "junction": 1}


def _write_array(group, name: str, values: np.ndarray, *, first_chunk: int, compressor) -> None:
    data = np.asarray(values)
    chunk = max(1, min(int(first_chunk), len(data)))
    group.create_dataset(
        name,
        data=data,
        chunks=(chunk, *data.shape[1:]),
        compressor=compressor,
    )


def _json_line(stream, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _load_world_summary(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            parent_id = str(row["parent_id"])
            if parent_id in result:
                raise RuntimeError(f"duplicate feasibility world: {parent_id}")
            result[parent_id] = row
    if len(result) != EXPECTED_WORLDS:
        raise RuntimeError("feasibility world summary population drift")
    return result


def _event_identity_inventory(
    worlds: list[str], mesh_root: Path
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    nodes_by_world: dict[str, list[dict[str, Any]]] = {}
    for parent_id in worlds:
        primary = mesh_root / parent_id / "primary"
        geometry = load_json(primary / "geometry_parameters.json")
        nodes = _event_nodes(
            parent_id,
            load_json(primary / "graph.json"),
            float(geometry["fta_distance_m"]),
        )
        nodes_by_world[parent_id] = nodes
        for node in nodes:
            rows.append(
                {
                    "identity": str(node["identity"]),
                    "parent_id": parent_id,
                    "partition": _partition(parent_id),
                    "family": _family(parent_id),
                    "node_id": str(node["node_id"]),
                    "event_type": str(node["event_type"]),
                    "degree": int(node["degree"]),
                }
            )
    rows.sort(key=lambda row: row["identity"])
    if len(rows) != EXPECTED_IDENTITIES or len({row["identity"] for row in rows}) != len(rows):
        raise RuntimeError("spatial event identity inventory drift")
    identity_to_index = {str(row["identity"]): index for index, row in enumerate(rows)}
    for index, row in enumerate(rows):
        row["identity_index"] = index
    return rows, identity_to_index, nodes_by_world


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--shard-manifest", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--mesh-manifest", required=True, type=Path)
    parser.add_argument("--feasibility-world-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher_root = output / "teacher"
    teacher_root.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.resolve()
    mesh_root = args.mesh_root.resolve()
    shard_document = load_json(args.shard_manifest.resolve())
    mesh_document = load_json(args.mesh_manifest.resolve())
    feasibility = _load_world_summary(args.feasibility_world_summary.resolve())
    shard_rows = sorted(
        (dict(row) for row in shard_document["shards"] if row["split"] == "train"),
        key=lambda row: str(row["parent_id"]),
    )
    worlds = [str(row["parent_id"]) for row in shard_rows]
    if (
        len(worlds) != EXPECTED_WORLDS
        or set(worlds) != set(feasibility)
        or any(not 1 <= _world_suffix(parent_id) <= 8 for parent_id in worlds)
    ):
        raise RuntimeError("C01-C08 export world isolation drift")
    mesh_rows = {str(row["parent_id"]): row for row in mesh_document["parents"]}
    identity_rows, identity_to_index, nodes_by_world = _event_identity_inventory(worlds, mesh_root)
    write_json(
        output / "event_identity_map.json",
        {
            "schema_version": "gse_spatial_event_identity_map_v1",
            "teacher_only": True,
            "student_input_forbidden": True,
            "rows": identity_rows,
        },
    )

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    partition_totals = {"fit": Counter(), "selection": Counter()}
    event_type_counts: Counter[str] = Counter()
    global_cardinality_histogram: Counter[int] = Counter()
    manifest_rows: list[dict[str, Any]] = []
    plot_relative: dict[str, list[np.ndarray]] = {"terminal": [], "junction": []}

    manifest_path = output / "teacher_shard_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_stream:
        for world_index, shard_record in enumerate(shard_rows, start=1):
            parent_id = str(shard_record["parent_id"])
            partition = _partition(parent_id)
            source_shard = dataset_root / "train" / f"{parent_id}.zarr"
            source_tree_sha, source_files, source_bytes = _tree_hash(source_shard)
            if (
                source_tree_sha != str(shard_record["shard_tree_sha256"])
                or source_files != int(shard_record["shard_file_count"])
                or source_bytes != int(shard_record["shard_bytes"])
            ):
                raise RuntimeError(f"source dataset shard drift: {parent_id}")
            primary = mesh_root / parent_id / "primary"
            mesh_path = primary / "mesh.obj"
            mesh_sha = _sha256(mesh_path)
            if mesh_sha != str(mesh_rows[parent_id]["primary"]["mesh_sha256"]):
                raise RuntimeError(f"native perception mesh drift: {parent_id}")

            group = zarr.open_group(str(source_shard), mode="r")
            references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
            global_indices = np.asarray(group["global_sequence_index"][:], dtype=np.int64)
            sensors_all = np.asarray(group["sensor_xyz_m"][:], dtype=np.float64)
            yaws_all = np.asarray(group["yaw_deg"][:], dtype=np.float64)
            if references.ndim != 2 or references.shape[1] != 5 or len(references) != len(global_indices):
                raise RuntimeError(f"causal reference drift: {parent_id}")
            current_frames = references[:, -1]
            sensors = sensors_all[current_frames]
            yaws = yaws_all[current_frames]
            if not np.all(np.isfinite(sensors)) or not np.all(np.isfinite(yaws)):
                raise RuntimeError(f"nonfinite causal pose: {parent_id}")

            nodes = nodes_by_world[parent_id]
            targets = np.stack([node["target"] for node in nodes])
            node_identity_indices = np.asarray(
                [identity_to_index[str(node["identity"])] for node in nodes], dtype=np.int32
            )
            node_type_indices = np.asarray(
                [EVENT_TYPE_TO_INDEX[str(node["event_type"])] for node in nodes], dtype=np.int8
            )
            observation_count = len(sensors)
            event_type_index = np.full((observation_count, MAXIMUM_EVENT_TOKENS), -1, dtype=np.int8)
            event_relative_xyz_m = np.zeros((observation_count, MAXIMUM_EVENT_TOKENS, 3), dtype=np.float32)
            event_distance_m = np.zeros((observation_count, MAXIMUM_EVENT_TOKENS), dtype=np.float32)
            event_identity_index = np.full((observation_count, MAXIMUM_EVENT_TOKENS), -1, dtype=np.int32)
            event_mask = np.zeros((observation_count, MAXIMUM_EVENT_TOKENS), dtype=np.uint8)
            set_cardinality = np.zeros(observation_count, dtype=np.uint8)

            candidate_total = 0
            visible_total = 0
            blocked_total = 0
            scene = _scene(mesh_path)
            for obs_start in range(0, observation_count, 4096):
                obs_stop = min(obs_start + 4096, observation_count)
                delta = targets[None, :, :] - sensors[obs_start:obs_stop, None, :]
                distances = np.linalg.norm(delta, axis=2)
                obs_local, node_index = np.nonzero(distances <= MAX_EVENT_RANGE_M + 1e-9)
                if len(node_index) == 0:
                    continue
                obs_index = obs_local + obs_start
                pair_distance = distances[obs_local, node_index]
                directions = delta[obs_local, node_index].copy()
                nonzero = pair_distance > 1e-8
                directions[nonzero] /= pair_distance[nonzero, None]
                directions[~nonzero] = np.asarray((1.0, 0.0, 0.0))
                hits = _cast_pairs(scene, sensors[obs_index], directions)
                if np.any(np.isnan(hits)) or np.any(hits < 0.0):
                    raise RuntimeError(f"native mesh raycast result drift: {parent_id}")
                visible = (~np.isfinite(hits)) | (hits >= pair_distance - LOS_MARGIN_M)
                visible_obs = obs_index[visible]
                visible_nodes = node_index[visible]
                visible_distances = pair_distance[visible]
                visible_identity = node_identity_indices[visible_nodes]
                order = np.lexsort((visible_identity, visible_distances, visible_obs))
                visible_obs = visible_obs[order]
                visible_nodes = visible_nodes[order]
                visible_distances = visible_distances[order]

                counts = np.bincount(visible_obs, minlength=observation_count)
                if int(np.max(counts, initial=0)) > MAXIMUM_EVENT_TOKENS:
                    raise RuntimeError(f"spatial event Teacher truncation would occur: {parent_id}")
                positions = np.empty(len(visible_obs), dtype=np.int64)
                previous = -1
                position = 0
                for row_index, obs in enumerate(visible_obs):
                    obs_value = int(obs)
                    if obs_value != previous:
                        previous = obs_value
                        position = 0
                    positions[row_index] = position
                    position += 1
                type_values = node_type_indices[visible_nodes]
                identity_values = node_identity_indices[visible_nodes]
                target_values = targets[visible_nodes]
                yaw_radians = np.radians(yaws[visible_obs])
                world_delta = target_values - sensors[visible_obs]
                cosine = np.cos(yaw_radians)
                sine = np.sin(yaw_radians)
                relative = np.stack(
                    (
                        cosine * world_delta[:, 0] + sine * world_delta[:, 1],
                        -sine * world_delta[:, 0] + cosine * world_delta[:, 1],
                        world_delta[:, 2],
                    ),
                    axis=1,
                )
                event_type_index[visible_obs, positions] = type_values
                event_relative_xyz_m[visible_obs, positions] = relative.astype(np.float32)
                event_distance_m[visible_obs, positions] = visible_distances.astype(np.float32)
                event_identity_index[visible_obs, positions] = identity_values
                event_mask[visible_obs, positions] = 1
                set_cardinality += counts.astype(np.uint8)
                for name, index in EVENT_TYPE_TO_INDEX.items():
                    selected = type_values == index
                    event_type_counts[name] += int(np.sum(selected))
                    if np.any(selected):
                        plot_relative[name].append(relative[selected].astype(np.float32))
                candidate_total += len(node_index)
                visible_total += int(np.sum(visible))
                blocked_total += int(np.sum(~visible))

            if not np.array_equal(set_cardinality, event_mask.sum(axis=1, dtype=np.uint8)):
                raise RuntimeError(f"spatial event cardinality/mask drift: {parent_id}")
            for obs_index in range(observation_count):
                active = event_identity_index[obs_index, event_mask[obs_index].astype(bool)]
                if len(active) != len(set(int(value) for value in active)):
                    raise RuntimeError(f"duplicate event identity in one set: {parent_id}:{obs_index}")
            histogram = Counter(int(value) for value in set_cardinality)
            expected = feasibility[parent_id]
            if (
                candidate_total != int(expected["candidate_pairs"])
                or visible_total != int(expected["visible_pairs"])
                or blocked_total != int(expected["blocked_pairs"])
                or {str(key): value for key, value in sorted(histogram.items())}
                != dict(expected["cardinality_histogram"])
            ):
                raise RuntimeError(f"feasibility/export semantic replay drift: {parent_id}")

            target_shard = teacher_root / partition / f"{parent_id}.zarr"
            target_shard.parent.mkdir(parents=True, exist_ok=True)
            target_group = zarr.open_group(str(target_shard), mode="w")
            target_group.attrs.update(
                {
                    "schema_version": "gse_spatial_multi_event_teacher_v1",
                    "parent_id": parent_id,
                    "partition": partition,
                    "teacher_only": True,
                    "student_input": "none; join global_sequence_index to existing five-frame LiDAR shard",
                    "event_type_index": {"terminal": 0, "junction": 1, "padding": -1},
                    "maximum_event_tokens": MAXIMUM_EVENT_TOKENS,
                    "maximum_event_range_m": MAX_EVENT_RANGE_M,
                    "los_margin_m": LOS_MARGIN_M,
                    "relative_frame": "forward-left-up at fifth causal frame",
                    "truncation": "forbidden",
                }
            )
            arrays = {
                "global_sequence_index": global_indices,
                "event_type_index": event_type_index,
                "event_relative_xyz_m": event_relative_xyz_m,
                "event_distance_m": event_distance_m,
                "event_identity_index": event_identity_index,
                "event_mask": event_mask,
                "set_cardinality": set_cardinality,
            }
            for name, values in arrays.items():
                _write_array(target_group, name, values, first_chunk=2048, compressor=compressor)
            target_tree_sha, target_files, target_bytes = _tree_hash(target_shard)
            manifest_row = {
                "parent_id": parent_id,
                "partition": partition,
                "observations": observation_count,
                "event_identities": len(nodes),
                "candidate_pairs": candidate_total,
                "visible_tokens": visible_total,
                "blocked_pairs": blocked_total,
                "maximum_set_cardinality": int(np.max(set_cardinality, initial=0)),
                "cardinality_histogram": {str(key): value for key, value in sorted(histogram.items())},
                "source_dataset_tree_sha256": source_tree_sha,
                "native_mesh_sha256": mesh_sha,
                "teacher_shard_tree_sha256": target_tree_sha,
                "teacher_shard_file_count": target_files,
                "teacher_shard_bytes": target_bytes,
            }
            manifest_rows.append(manifest_row)
            _json_line(manifest_stream, manifest_row)
            partition_totals[partition].update(
                {
                    "worlds": 1,
                    "observations": observation_count,
                    "candidate_pairs": candidate_total,
                    "visible_tokens": visible_total,
                    "blocked_pairs": blocked_total,
                    "zero_event_observations": histogram[0],
                    "multi_event_observations": sum(value for key, value in histogram.items() if key >= 2),
                    "teacher_shard_bytes": target_bytes,
                }
            )
            global_cardinality_histogram.update(histogram)
            print(
                json.dumps(
                    {
                        "world": parent_id,
                        "index": world_index,
                        "of": EXPECTED_WORLDS,
                        "observations": observation_count,
                        "visible_tokens": visible_total,
                        "max_tokens": int(np.max(set_cardinality, initial=0)),
                        "shard_bytes": target_bytes,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    total_observations = sum(values["observations"] for values in partition_totals.values())
    total_visible = sum(values["visible_tokens"] for values in partition_totals.values())
    maximum_cardinality = max(int(row["maximum_set_cardinality"]) for row in manifest_rows)
    global_ids = []
    for row in manifest_rows:
        shard = teacher_root / str(row["partition"]) / f"{row['parent_id']}.zarr"
        global_ids.append(np.asarray(zarr.open_group(str(shard), mode="r")["global_sequence_index"][:]))
    concatenated_ids = np.concatenate(global_ids)
    unique_global_ids = len(np.unique(concatenated_ids))

    checks = {
        "exact_80_worlds": len(manifest_rows) == EXPECTED_WORLDS,
        "exact_fit_selection_worlds_60_20": (
            partition_totals["fit"]["worlds"] == 60 and partition_totals["selection"]["worlds"] == 20
        ),
        "exact_188126_observations": total_observations == EXPECTED_OBSERVATIONS,
        "all_global_sequence_indices_unique": unique_global_ids == EXPECTED_OBSERVATIONS,
        "exact_1076_teacher_only_identities": len(identity_rows) == EXPECTED_IDENTITIES,
        "exact_133055_visible_tokens": total_visible == EXPECTED_VISIBLE_TOKENS,
        "exact_terminal_junction_token_counts": dict(event_type_counts) == EXPECTED_EVENT_TOKEN_COUNTS,
        "zero_truncation_maximum_at_most_16": maximum_cardinality <= MAXIMUM_EVENT_TOKENS,
        "zero_event_rows_retained": global_cardinality_histogram[0] == 81_069,
        "feasibility_world_replay_exact": len(manifest_rows) == len(feasibility),
        "zero_training_inference_normalization_threshold_selection": True,
        "zero_C09_C10_MTARE_reads": True,
    }
    checks["all_passed"] = all(checks.values())
    status = PASS if checks["all_passed"] else FAIL

    plot_source: dict[str, Any] = {
        "schema_version": "gse_spatial_multi_event_teacher_export_figure_source_v1",
        "cardinality_histogram": {str(key): value for key, value in sorted(global_cardinality_histogram.items())},
        "event_type_counts": dict(event_type_counts),
        "relative_histograms": {},
    }
    edges = np.linspace(-50.0, 50.0, 81)
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    bins = np.asarray(sorted(global_cardinality_histogram))
    values = np.asarray([global_cardinality_histogram[int(key)] for key in bins]) / EXPECTED_OBSERVATIONS * 100.0
    axes[0].bar(bins, values, color="#2d6fa3")
    axes[0].set_xlabel("Visible event tokens")
    axes[0].set_ylabel("Observations (%)")
    axes[0].set_title("Lossless 16-slot set export")
    axes[0].set_xticks(bins)
    for axis, name, color_map in zip(axes[1:], ("terminal", "junction"), ("Oranges", "Blues"), strict=True):
        points = np.concatenate(plot_relative[name], axis=0)
        histogram, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=(edges, edges))
        plot_source["relative_histograms"][name] = {
            "forward_edges_m": x_edges.tolist(),
            "left_edges_m": y_edges.tolist(),
            "counts": histogram.astype(int).tolist(),
        }
        image = axis.pcolormesh(x_edges, y_edges, np.log1p(histogram.T), cmap=color_map, shading="auto")
        axis.set_aspect("equal")
        axis.set_xlabel("Forward (m)")
        axis.set_ylabel("Left (m)")
        axis.set_title(f"{name.capitalize()} targets ({len(points):,})")
        figure.colorbar(image, ax=axis, label="log(1 + count)")
    write_json(output / "figure_source.json", plot_source)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_spatial_multi_event_teacher_export.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(figure)

    summary = {
        "schema_version": "gse_spatial_multi_event_teacher_export_v1",
        "status": status,
        "population": {
            "worlds": len(manifest_rows),
            "fit_worlds": partition_totals["fit"]["worlds"],
            "selection_worlds": partition_totals["selection"]["worlds"],
            "observations": total_observations,
            "unique_global_sequence_indices": unique_global_ids,
            "event_identities": len(identity_rows),
            "visible_event_tokens": total_visible,
            "event_type_counts": dict(event_type_counts),
            "zero_event_observations": global_cardinality_histogram[0],
            "multi_event_observations": sum(
                value for key, value in global_cardinality_histogram.items() if key >= 2
            ),
            "maximum_set_cardinality": maximum_cardinality,
        },
        "partition_totals": {name: dict(values) for name, values in partition_totals.items()},
        "cardinality_histogram": {str(key): value for key, value in sorted(global_cardinality_histogram.items())},
        "target_contract": {
            "maximum_event_tokens": MAXIMUM_EVENT_TOKENS,
            "type_indices": {"terminal": 0, "junction": 1, "padding": -1},
            "relative_frame": "forward-left-up at fifth causal frame",
            "teacher_only_identity": True,
            "student_inputs_exported": False,
            "zero_event_rows_retained": True,
            "truncation": "forbidden",
        },
        "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "model_updates": 0,
        "normalization_steps": 0,
        "threshold_selection_steps": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if status == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
