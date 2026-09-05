#!/usr/bin/env python3
"""Audit whether C01-C08 can support a spatial multi-event Teacher.

This is deliberately a zero-training feasibility proof.  It reads only the
sealed C01-C08 training shards for causal poses, the frozen TNG graphs and the
native perception meshes.  It does not export a reusable training target.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.gse_spatial_multi_event_teacher import (
    LOS_MARGIN_M,
    MAX_EVENT_RANGE_M,
    event_type_for_degree,
    has_opposite_headings,
    robot_relative_xyz,
)


PASS = "PASS_GSE_SPATIAL_MULTI_EVENT_TEACHER_FEASIBILITY_V1"
FAIL = "FAIL_GSE_SPATIAL_MULTI_EVENT_TEACHER_FEASIBILITY_V1"
EXPECTED_WORLDS = 80
EXPECTED_OBSERVATIONS = 188_126
MAXIMUM_TOKEN_CAPACITY = 16
MINIMUM_OPPOSITE_HEADING_COVERAGE = 0.90
MINIMUM_TARGET_SEPARATION_M = LOS_MARGIN_M
RAY_BATCH_SIZE = 500_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        payload_digest = hashlib.sha256()
        size = 0
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                payload_digest.update(block)
                size += len(block)
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(payload_digest.digest())
        total_bytes += size
    return digest.hexdigest(), len(files), total_bytes


def _world_suffix(parent_id: str) -> int:
    try:
        return int(parent_id.rsplit("_C", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"invalid development parent identity: {parent_id}") from exc


def _partition(parent_id: str) -> str:
    suffix = _world_suffix(parent_id)
    if 1 <= suffix <= 6:
        return "fit"
    if 7 <= suffix <= 8:
        return "selection"
    raise RuntimeError(f"forbidden world entered multi-event proof: {parent_id}")


def _family(parent_id: str) -> str:
    return parent_id.rsplit("_C", 1)[0]


def _scene(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty native perception mesh: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _cast_pairs(
    scene: o3d.t.geometry.RaycastingScene,
    origins: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("pair-ray shape drift")
    result = np.empty(len(origins), dtype=np.float64)
    for start in range(0, len(origins), RAY_BATCH_SIZE):
        stop = min(start + RAY_BATCH_SIZE, len(origins))
        rays = np.concatenate((origins[start:stop], directions[start:stop]), axis=1)
        hits = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32, copy=False)))["t_hit"].numpy()
        result[start:stop] = np.asarray(hits, dtype=np.float64)
    return result


def _json_line(stream, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _load_json_object_array(path: Path) -> list[dict[str, Any]]:
    """Load an evidence JSON array without weakening governance.load_json."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected a JSON array of objects: {path}")
    return [dict(row) for row in payload]


def _event_nodes(parent_id: str, graph: dict[str, Any], fta_distance_m: float) -> list[dict[str, Any]]:
    raw_nodes = list(graph.get("nodes", ()))
    node_ids = [str(node.get("id", "")) for node in raw_nodes]
    if not raw_nodes or any(not value for value in node_ids) or len(set(node_ids)) != len(node_ids):
        raise RuntimeError(f"graph node identity drift: {parent_id}")
    result: list[dict[str, Any]] = []
    for node in raw_nodes:
        event_type = event_type_for_degree(int(node["degree"]))
        if event_type is None:
            continue
        axis = np.asarray(node["xyz"], dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise RuntimeError(f"graph event coordinate drift: {parent_id}:{node['id']}")
        target = axis.copy()
        target[2] += float(fta_distance_m) + 1.0
        result.append(
            {
                "identity": f"{parent_id}:node:{node['id']}",
                "node_id": str(node["id"]),
                "event_type": event_type,
                "degree": int(node["degree"]),
                "axis": axis,
                "target": target,
            }
        )
    if not result:
        raise RuntimeError(f"world has no structural event nodes: {parent_id}")
    targets = np.stack([row["target"] for row in result])
    if len(targets) > 1:
        delta = targets[:, None, :] - targets[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        distances[np.diag_indices_from(distances)] = np.inf
        minimum = float(np.min(distances))
        if minimum < MINIMUM_TARGET_SEPARATION_M - 1e-9:
            raise RuntimeError(
                f"ambiguous objective event targets in {parent_id}: {minimum:.6f} m"
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--shard-manifest", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--mesh-manifest", required=True, type=Path)
    parser.add_argument("--low-observability", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.resolve()
    mesh_root = args.mesh_root.resolve()
    shard_document = load_json(args.shard_manifest.resolve())
    mesh_document = load_json(args.mesh_manifest.resolve())
    low_rows = _load_json_object_array(args.low_observability.resolve())
    if len(low_rows) != 4:
        raise RuntimeError("rare endpoint observation population drift")
    rare_by_global = {int(row["global_sequence_index"]): row for row in low_rows}
    if len(rare_by_global) != 4:
        raise RuntimeError("rare endpoint global identity drift")

    train_shards = sorted(
        (dict(row) for row in shard_document["shards"] if row["split"] == "train"),
        key=lambda row: str(row["parent_id"]),
    )
    if (
        len(train_shards) != EXPECTED_WORLDS
        or any(not 1 <= _world_suffix(str(row["parent_id"])) <= 8 for row in train_shards)
        or any(str(row["parent_id"]).endswith(("_C09", "_C10")) for row in train_shards)
    ):
        raise RuntimeError("C01-C08 shard isolation drift")
    mesh_rows = {str(row["parent_id"]): row for row in mesh_document["parents"]}
    if any(str(row["parent_id"]) not in mesh_rows for row in train_shards):
        raise RuntimeError("native mesh manifest coverage drift")

    partition_totals: dict[str, Counter[str]] = {"fit": Counter(), "selection": Counter()}
    partition_histograms: dict[str, Counter[int]] = {"fit": Counter(), "selection": Counter()}
    family_partition_covisibility: Counter[tuple[str, str]] = Counter()
    event_stats: dict[str, dict[str, Any]] = {}
    rare_evidence: dict[int, dict[str, Any]] = {}
    world_count = 0
    observation_count = 0
    candidate_pair_count = 0
    visible_pair_count = 0
    blocked_pair_count = 0
    maximum_cardinality = 0
    duplicate_set_identity_count = 0
    nonfinite_target_count = 0
    source_rows: list[dict[str, Any]] = []

    world_summary_path = output / "world_feasibility_summary.jsonl"
    with world_summary_path.open("w", encoding="utf-8") as world_stream:
        for world_index, shard_row in enumerate(train_shards, start=1):
            parent_id = str(shard_row["parent_id"])
            partition = _partition(parent_id)
            family = _family(parent_id)
            shard_path = dataset_root / "train" / f"{parent_id}.zarr"
            tree_sha, shard_files, shard_bytes = _tree_hash(shard_path)
            if (
                tree_sha != str(shard_row["shard_tree_sha256"])
                or shard_files != int(shard_row["shard_file_count"])
                or shard_bytes != int(shard_row["shard_bytes"])
            ):
                raise RuntimeError(f"sealed C01-C08 shard drift: {parent_id}")

            primary = mesh_root / parent_id / "primary"
            mesh_path = primary / "mesh.obj"
            mesh_expected = str(mesh_rows[parent_id]["primary"]["mesh_sha256"])
            mesh_actual = _sha256(mesh_path)
            if mesh_actual != mesh_expected:
                raise RuntimeError(f"native perception mesh drift: {parent_id}")
            graph = load_json(primary / "graph.json")
            geometry = load_json(primary / "geometry_parameters.json")
            fta = float(geometry["fta_distance_m"])
            if not math.isfinite(fta):
                raise RuntimeError(f"FTA drift: {parent_id}")
            nodes = _event_nodes(parent_id, graph, fta)
            node_targets = np.stack([row["target"] for row in nodes])
            node_types = np.asarray([row["event_type"] for row in nodes], dtype=object)
            for node in nodes:
                identity = str(node["identity"])
                if identity in event_stats:
                    raise RuntimeError(f"cross-world event identity collision: {identity}")
                event_stats[identity] = {
                    "identity": identity,
                    "world": parent_id,
                    "partition": partition,
                    "family": family,
                    "event_type": node["event_type"],
                    "degree": int(node["degree"]),
                    "visible_observations": 0,
                    "minimum_distance_m": None,
                    "maximum_distance_m": None,
                    "headings_deg": [],
                }

            group = zarr.open_group(str(shard_path), mode="r")
            if (
                str(group.attrs.get("split")) != "train"
                or str(group.attrs.get("parent_id")) != parent_id
                or not bool(group.attrs.get("pose_teacher_only"))
            ):
                raise RuntimeError(f"dataset shard semantic drift: {parent_id}")
            references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
            global_indices = np.asarray(group["global_sequence_index"][:], dtype=np.int64)
            sensor_all = np.asarray(group["sensor_xyz_m"][:], dtype=np.float64)
            yaw_all = np.asarray(group["yaw_deg"][:], dtype=np.float64)
            if references.ndim != 2 or references.shape[1] != 5 or len(global_indices) != len(references):
                raise RuntimeError(f"causal five-frame reference drift: {parent_id}")
            current_frames = references[:, -1]
            if (
                np.any(current_frames < 0)
                or np.any(current_frames >= len(sensor_all))
                or sensor_all.shape != (len(yaw_all), 3)
            ):
                raise RuntimeError(f"current causal pose drift: {parent_id}")
            sensors = sensor_all[current_frames]
            yaws = yaw_all[current_frames]
            if not np.all(np.isfinite(sensors)) or not np.all(np.isfinite(yaws)):
                raise RuntimeError(f"nonfinite current causal pose: {parent_id}")

            cardinality = np.zeros(len(sensors), dtype=np.int16)
            has_terminal = np.zeros(len(sensors), dtype=bool)
            has_junction = np.zeros(len(sensors), dtype=bool)
            world_candidates = 0
            world_visible = 0
            world_blocked = 0
            scene = _scene(mesh_path)

            observation_batch = 4096
            for obs_start in range(0, len(sensors), observation_batch):
                obs_stop = min(obs_start + observation_batch, len(sensors))
                delta = node_targets[None, :, :] - sensors[obs_start:obs_stop, None, :]
                distances = np.linalg.norm(delta, axis=2)
                candidate_obs_local, candidate_nodes = np.nonzero(distances <= MAX_EVENT_RANGE_M + 1e-9)
                if len(candidate_nodes) == 0:
                    continue
                candidate_obs = candidate_obs_local + obs_start
                pair_distance = distances[candidate_obs_local, candidate_nodes]
                origins = sensors[candidate_obs]
                directions = delta[candidate_obs_local, candidate_nodes].copy()
                nonzero = pair_distance > 1e-8
                directions[nonzero] /= pair_distance[nonzero, None]
                directions[~nonzero] = np.asarray((1.0, 0.0, 0.0))
                hits = _cast_pairs(scene, origins, directions)
                if np.any(np.isnan(hits)) or np.any(hits < 0.0):
                    raise RuntimeError(f"native mesh raycast drift: {parent_id}")
                visible = (~np.isfinite(hits)) | (hits >= pair_distance - LOS_MARGIN_M)
                visible_obs = candidate_obs[visible]
                visible_nodes = candidate_nodes[visible]
                np.add.at(cardinality, visible_obs, 1)
                terminal_mask = node_types[visible_nodes] == "terminal"
                junction_mask = node_types[visible_nodes] == "junction"
                has_terminal[visible_obs[terminal_mask]] = True
                has_junction[visible_obs[junction_mask]] = True

                world_candidates += len(candidate_nodes)
                world_visible += int(np.sum(visible))
                world_blocked += int(np.sum(~visible))
                for obs_row, node_index, distance, hit in zip(
                    visible_obs,
                    visible_nodes,
                    pair_distance[visible],
                    hits[visible],
                    strict=True,
                ):
                    node = nodes[int(node_index)]
                    identity = str(node["identity"])
                    record = event_stats[identity]
                    record["visible_observations"] += 1
                    record["minimum_distance_m"] = (
                        float(distance)
                        if record["minimum_distance_m"] is None
                        else min(float(record["minimum_distance_m"]), float(distance))
                    )
                    record["maximum_distance_m"] = (
                        float(distance)
                        if record["maximum_distance_m"] is None
                        else max(float(record["maximum_distance_m"]), float(distance))
                    )
                    record["headings_deg"].append(float(yaws[int(obs_row)]))

                batch_global = global_indices[candidate_obs]
                rare_mask = np.fromiter((int(value) in rare_by_global for value in batch_global), bool)
                if np.any(rare_mask):
                    for global_index in np.unique(batch_global[rare_mask]):
                        obs_matches = candidate_obs[batch_global == global_index]
                        if len(np.unique(obs_matches)) != 1:
                            raise RuntimeError("rare observation row mapping drift")
                        obs_row = int(obs_matches[0])
                        pair_mask = candidate_obs == obs_row
                        tokens = []
                        for node_index, distance, hit, is_visible in zip(
                            candidate_nodes[pair_mask],
                            pair_distance[pair_mask],
                            hits[pair_mask],
                            visible[pair_mask],
                            strict=True,
                        ):
                            node = nodes[int(node_index)]
                            raw_hit = float(hit)
                            tokens.append(
                                {
                                    "identity": node["identity"],
                                    "event_type": node["event_type"],
                                    "degree": int(node["degree"]),
                                    "distance_m": float(distance),
                                    "relative_xyz_m": list(
                                        robot_relative_xyz(
                                            sensors[obs_row], yaws[obs_row], node["target"]
                                        )
                                    ),
                                    "visible": bool(is_visible),
                                    "first_hit_distance_m": raw_hit if math.isfinite(raw_hit) else None,
                                }
                            )
                        rare_evidence[int(global_index)] = {
                            "source": rare_by_global[int(global_index)],
                            "sensor_xyz_m": sensors[obs_row].tolist(),
                            "yaw_deg": float(yaws[obs_row]),
                            "candidate_tokens": sorted(tokens, key=lambda row: (row["distance_m"], row["identity"])),
                        }

            world_histogram = Counter(int(value) for value in cardinality)
            maximum_cardinality = max(maximum_cardinality, int(np.max(cardinality, initial=0)))
            partition_histograms[partition].update(world_histogram)
            co_visible = has_terminal & has_junction
            multi_event = cardinality >= 2
            family_partition_covisibility[(partition, family)] += int(np.sum(co_visible))
            partition_totals[partition].update(
                {
                    "worlds": 1,
                    "observations": len(sensors),
                    "candidate_pairs": world_candidates,
                    "visible_pairs": world_visible,
                    "blocked_pairs": world_blocked,
                    "zero_event_observations": int(np.sum(cardinality == 0)),
                    "single_event_observations": int(np.sum(cardinality == 1)),
                    "multi_event_observations": int(np.sum(multi_event)),
                    "junction_terminal_covisible_observations": int(np.sum(co_visible)),
                    "junction_visible_observations": int(np.sum(has_junction)),
                    "terminal_visible_observations": int(np.sum(has_terminal)),
                }
            )
            world_count += 1
            observation_count += len(sensors)
            candidate_pair_count += world_candidates
            visible_pair_count += world_visible
            blocked_pair_count += world_blocked
            summary_row = {
                "parent_id": parent_id,
                "partition": partition,
                "family": family,
                "observations": len(sensors),
                "event_identities": len(nodes),
                "candidate_pairs": world_candidates,
                "visible_pairs": world_visible,
                "blocked_pairs": world_blocked,
                "cardinality_histogram": {str(key): value for key, value in sorted(world_histogram.items())},
                "maximum_cardinality": int(np.max(cardinality, initial=0)),
                "multi_event_observations": int(np.sum(multi_event)),
                "junction_terminal_covisible_observations": int(np.sum(co_visible)),
                "dataset_shard_tree_sha256": tree_sha,
                "native_mesh_sha256": mesh_actual,
            }
            source_rows.append(summary_row)
            _json_line(world_stream, summary_row)
            print(
                json.dumps(
                    {
                        "world": parent_id,
                        "index": world_index,
                        "of": len(train_shards),
                        "observations": len(sensors),
                        "visible_pairs": world_visible,
                        "multi_event": int(np.sum(multi_event)),
                        "co_visible": int(np.sum(co_visible)),
                        "max_tokens": int(np.max(cardinality, initial=0)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if world_count != EXPECTED_WORLDS or observation_count != EXPECTED_OBSERVATIONS:
        raise RuntimeError("full C01-C08 population drift")
    if len(rare_evidence) != 4:
        raise RuntimeError("rare endpoint rows not all resolved")

    # Convert the per-identity accumulator into compact scientific evidence.
    identity_rows = []
    for identity in sorted(event_stats):
        record = event_stats[identity]
        headings = record.pop("headings_deg")
        record["opposite_heading_visible"] = has_opposite_headings(headings)
        record["distinct_heading_degrees_rounded"] = len({round(value, 3) for value in headings})
        identity_rows.append(record)
    identity_path = output / "event_identity_coverage.jsonl"
    with identity_path.open("w", encoding="utf-8") as stream:
        for row in identity_rows:
            _json_line(stream, row)

    objective_identity_count = sum(
        1
        for shard_row in train_shards
        for node in load_json(mesh_root / str(shard_row["parent_id"]) / "primary/graph.json")["nodes"]
        if event_type_for_degree(int(node["degree"])) is not None
    )
    if objective_identity_count != len(identity_rows):
        raise RuntimeError("objective event identity accounting drift")
    visible_identity_count = sum(int(row["visible_observations"]) > 0 for row in identity_rows)
    opposite_identity_count = sum(bool(row["opposite_heading_visible"]) for row in identity_rows)
    opposite_fraction = opposite_identity_count / objective_identity_count if objective_identity_count else 0.0

    rare_checks: dict[str, bool] = {}
    for global_index, evidence in sorted(rare_evidence.items()):
        source = evidence["source"]
        visible_tokens = [row for row in evidence["candidate_tokens"] if row["visible"]]
        visible_ids = {str(row["identity"]) for row in visible_tokens}
        rare_checks[f"rare_{global_index}_objective_identity_visible"] = str(source["identity"]) in visible_ids
    rare_checks["S04_44299_terminal_and_junction_covisible"] = {
        row["event_type"] for row in rare_evidence[44299]["candidate_tokens"] if row["visible"]
    } >= {"terminal", "junction"}
    rare_checks["S07_110361_terminal_and_junction_covisible"] = {
        row["event_type"] for row in rare_evidence[110361]["candidate_tokens"] if row["visible"]
    } >= {"terminal", "junction"}

    partition_summary = {
        name: {
            **dict(values),
            "cardinality_histogram": {
                str(key): value for key, value in sorted(partition_histograms[name].items())
            },
            "families_with_junction_terminal_covisibility": sorted(
                family
                for (part, family), count in family_partition_covisibility.items()
                if part == name and count > 0
            ),
        }
        for name, values in partition_totals.items()
    }
    gates = {
        "exact_80_C01_C08_worlds": world_count == EXPECTED_WORLDS,
        "exact_188126_causal_observations": observation_count == EXPECTED_OBSERVATIONS,
        "all_objective_event_identities_visible": visible_identity_count == objective_identity_count,
        "opposite_heading_identity_coverage_at_least_0_90": opposite_fraction >= MINIMUM_OPPOSITE_HEADING_COVERAGE,
        "bounded_maximum_event_set_at_most_16": maximum_cardinality <= MAXIMUM_TOKEN_CAPACITY,
        "fit_has_multi_event_and_junction_terminal_covisibility": (
            partition_totals["fit"]["multi_event_observations"] > 0
            and partition_totals["fit"]["junction_terminal_covisible_observations"] > 0
        ),
        "selection_has_multi_event_and_junction_terminal_covisibility": (
            partition_totals["selection"]["multi_event_observations"] > 0
            and partition_totals["selection"]["junction_terminal_covisible_observations"] > 0
        ),
        "unique_event_set_identity_and_position_mapping": (
            duplicate_set_identity_count == 0 and nonfinite_target_count == 0
        ),
        "all_four_rare_rows_resolved": len(rare_evidence) == 4,
        **rare_checks,
        "zero_training_inference_threshold_selection": True,
        "zero_C09_C10_MTARE_reads": True,
    }
    gates["all_passed"] = all(gates.values())
    status = PASS if gates["all_passed"] else FAIL

    figure_source = {
        "partition_cardinality_histograms": {
            name: {str(key): value for key, value in sorted(hist.items())}
            for name, hist in partition_histograms.items()
        },
        "partition_summary": partition_summary,
        "objective_identity_count": objective_identity_count,
        "visible_identity_count": visible_identity_count,
        "opposite_heading_identity_count": opposite_identity_count,
        "opposite_heading_fraction": opposite_fraction,
        "rare_evidence": rare_evidence,
    }
    write_json(output / "rare_failure_los_evidence.json", rare_evidence)
    write_json(output / "figure_source.json", figure_source)

    max_bin = max(max(hist) for hist in partition_histograms.values())
    bins = np.arange(max_bin + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    width = 0.38
    for offset, (name, color) in zip((-width / 2, width / 2), (("fit", "#2468a2"), ("selection", "#e07a2d")), strict=True):
        values = np.asarray([partition_histograms[name][int(value)] for value in bins], dtype=np.float64)
        values = values / max(1.0, values.sum()) * 100.0
        axes[0].bar(bins + offset, values, width=width, label=name, color=color)
    axes[0].set_xlabel("Visible structural events in one causal observation")
    axes[0].set_ylabel("Observations (%)")
    axes[0].set_xticks(bins)
    axes[0].legend(frameon=False)
    axes[0].set_title("Multi-event Teacher cardinality")

    names = ["Objective\nidentities visible", "Opposite-heading\ncoverage", "Rare endpoints\ncovered"]
    values = [
        100.0 * visible_identity_count / max(1, objective_identity_count),
        100.0 * opposite_fraction,
        100.0 * sum(rare_checks[key] for key in rare_checks if key.endswith("objective_identity_visible")) / 4.0,
    ]
    axes[1].bar(names, values, color=["#4d9f55", "#6750a4", "#b14c48"])
    axes[1].axhline(90.0, color="black", linestyle="--", linewidth=1.0, label="90% direction gate")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_title("Teacher coverage and directional diversity")
    axes[1].legend(frameon=False, loc="lower right")
    for index, value in enumerate(values):
        axes[1].text(index, min(value + 1.5, 102.0), f"{value:.1f}%", ha="center", fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_spatial_multi_event_teacher_feasibility.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(fig)

    summary = {
        "schema_version": "gse_spatial_multi_event_teacher_feasibility_v1",
        "status": status,
        "question": "Can native-mesh LOS define unique bounded-cardinality relative event sets over all C01-C08 causal observations?",
        "population": {
            "worlds": world_count,
            "observations": observation_count,
            "fit_worlds": partition_totals["fit"]["worlds"],
            "selection_worlds": partition_totals["selection"]["worlds"],
            "objective_event_identities": objective_identity_count,
            "visible_event_identities": visible_identity_count,
            "candidate_pairs_within_50m": candidate_pair_count,
            "visible_pairs": visible_pair_count,
            "blocked_pairs": blocked_pair_count,
        },
        "parameters": {
            "maximum_event_range_m": MAX_EVENT_RANGE_M,
            "los_margin_m": LOS_MARGIN_M,
            "maximum_token_capacity": MAXIMUM_TOKEN_CAPACITY,
            "minimum_opposite_heading_coverage": MINIMUM_OPPOSITE_HEADING_COVERAGE,
            "minimum_target_separation_m": MINIMUM_TARGET_SEPARATION_M,
            "target_height_rule": "TNG axis z + fta_distance_m + 1.0 m",
            "relative_frame": "forward-left-up at current (fifth) causal frame",
        },
        "partition_summary": partition_summary,
        "maximum_visible_event_set_cardinality": maximum_cardinality,
        "opposite_heading_identity_count": opposite_identity_count,
        "opposite_heading_identity_fraction": opposite_fraction,
        "rare_failure_rows": len(rare_evidence),
        "gates": gates,
        "recommendation": (
            "freeze_spatial_multi_event_teacher_data_card_before_model_design"
            if status == PASS
            else "stop_and_revise_teacher_or_data_before_any_training"
        ),
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "model_updates": 0,
        "threshold_selection_steps": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "student_inputs_read": 0,
        "full_teacher_export_rows": 0,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if status == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
