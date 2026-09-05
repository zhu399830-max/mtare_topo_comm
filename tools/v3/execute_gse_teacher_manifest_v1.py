#!/usr/bin/env python3
"""Build the exact development GSE traversal/teacher/association manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_teacher_manifest import EVENT_NAMES, world_teacher_manifest
from mtare_topo.governance import load_json, write_json


EXPECTED = {
    "train": {
        "world_count": 80,
        "edge_count": 8039,
        "directed_traversal_count": 16078,
        "sequence_count": 188126,
        "unique_frame_count": 252430,
        "referenced_frame_count": 940630,
    },
    "validation": {
        "world_count": 10,
        "edge_count": 1027,
        "directed_traversal_count": 2054,
        "sequence_count": 24462,
        "unique_frame_count": 32678,
        "referenced_frame_count": 122310,
    },
}
EXPECTED_TRAIN_EVENTS = {
    "corridor": 135123,
    "junction": 26608,
    "terminal": 7525,
    "turn": 2003,
    "geometry_transition": 16867,
}
RINGS = 16
AZIMUTH = 720
RANGE_BYTES = 4
MASK_BYTES = 1
REFERENCE_BYTES = 8


def _scene(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty mesh: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _batch_cast(scene: o3d.t.geometry.RaycastingScene):
    def cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        expanded = np.broadcast_to(origins[:, None, :], directions.shape)
        rays = np.concatenate((expanded, directions), axis=2).astype(np.float32, copy=False)
        return (
            scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"]
            .numpy()
            .reshape(directions.shape[:2])
        )

    return cast


def _write_row(stream, row: dict) -> None:
    stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    registry = load_json(args.registry.resolve())
    mesh_root = args.mesh_root.resolve()
    started = time.monotonic()
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    schema = registry["sampling_contract"]["row_schema"]
    decoded = [dict(zip(schema, row, strict=True)) if isinstance(row, list) else dict(row) for row in registry["rows"]]
    rows = sorted(
        (row for row in decoded if row["split"] in {"train", "validation"}),
        key=lambda row: str(row["parent_id"]),
    )
    if Counter(str(row["split"]) for row in rows) != Counter({"train": 80, "validation": 10}):
        raise RuntimeError("development split count mismatch")
    if any(str(row["parent_id"]).endswith("_C10") for row in rows):
        raise RuntimeError("strict-test parent leaked into teacher manifest")

    traversal_stream = (run_dir / "artifacts/traversal_manifest.jsonl").open("w", encoding="utf-8")
    observation_stream = (run_dir / "artifacts/teacher_observations.jsonl").open("w", encoding="utf-8")
    pair_stream = (run_dir / "artifacts/association_pairs.jsonl").open("w", encoding="utf-8")
    world_stream = (run_dir / "artifacts/world_manifest_summary.jsonl").open("w", encoding="utf-8")
    identity_stream = (run_dir / "artifacts/edge_event_identities.jsonl").open("w", encoding="utf-8")

    split_totals: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
    split_events: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
    split_pairs: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
    global_frame_offset = 0
    global_sequence_offset = 0
    all_observation_identity: dict[str, tuple[str, str, str | None]] = {}
    pair_key_count = 0
    pair_keys: set[tuple[str, str, str]] = set()

    try:
        for world_index, row in enumerate(rows, start=1):
            parent_id = str(row["parent_id"])
            split = str(row["split"])
            primary = mesh_root / parent_id / "primary"
            result = world_teacher_manifest(
                parent_id=parent_id,
                split=split,
                graph=load_json(primary / "graph.json"),
                spline_document=load_json(primary / "splines.json"),
                geometry_parameters=load_json(primary / "geometry_parameters.json"),
                cast_distances=_batch_cast(_scene(primary / "mesh.obj")),
            )
            summary = result["summary"]
            split_totals[split].update(
                {
                    "world_count": 1,
                    "edge_count": summary["edge_count"],
                    "directed_traversal_count": summary["directed_traversal_count"],
                    "zero_sequence_traversal_count": summary["zero_sequence_traversal_count"],
                    "sequence_count": summary["sequence_count"],
                    "unique_frame_count": summary["unique_frame_count"],
                    "referenced_frame_count": summary["referenced_frame_count"],
                    "structural_observation_count": summary["structural_observation_count"],
                    "association_identity_count": summary["association_identity_count"],
                    "positive_anchor_count": summary["positive_anchor_count"],
                    "hard_negative_anchor_count": summary["hard_negative_anchor_count"],
                }
            )
            split_events[split].update(summary["event_counts"])
            split_pairs[split].update(summary["association_pair_counts"])

            offsets: dict[str, tuple[int, int]] = {}
            for traversal in result["traversals"]:
                frame_count = (
                    traversal["sequence_count"] + traversal["history_frames"] - 1
                    if traversal["sequence_count"] > 0
                    else 0
                )
                traversal_row = {
                    **traversal,
                    "global_frame_offset": global_frame_offset,
                    "global_sequence_offset": global_sequence_offset,
                    "unique_frame_count": frame_count,
                    "referenced_frame_count": traversal["sequence_count"] * traversal["history_frames"],
                }
                offsets[traversal["traversal_id"]] = (global_frame_offset, global_sequence_offset)
                _write_row(traversal_stream, traversal_row)
                global_frame_offset += frame_count
                global_sequence_offset += traversal["sequence_count"]
            for observation in result["observations"]:
                frame_offset, sequence_offset = offsets[observation["traversal_id"]]
                row_out = {
                    **observation,
                    "global_sequence_index": sequence_offset + observation["sequence_index"],
                    "global_frame_references": [
                        frame_offset + value for value in observation["local_frame_references"]
                    ],
                }
                del row_out["local_frame_references"]
                _write_row(observation_stream, row_out)
                all_observation_identity[observation["observation_id"]] = (
                    parent_id,
                    observation["event"],
                    observation["identity"],
                )
            for pair in result["association_pairs"]:
                anchor = all_observation_identity[pair["anchor_observation_id"]]
                paired = all_observation_identity[pair["paired_observation_id"]]
                if anchor[0] != paired[0] or anchor[1] != paired[1]:
                    raise RuntimeError("association pair crosses parent or event")
                if bool(anchor[2] == paired[2]) != bool(pair["same_identity"]):
                    raise RuntimeError("association pair identity label is inconsistent")
                key = (
                    pair["anchor_observation_id"],
                    pair["paired_observation_id"],
                    pair["pair_kind"],
                )
                pair_key_count += 1
                pair_keys.add(key)
                _write_row(pair_stream, {"parent_id": parent_id, "split": split, **pair})
            for identity in result["edge_event_identities"]:
                _write_row(identity_stream, {"parent_id": parent_id, "split": split, **identity})
            _write_row(world_stream, {"parent_id": parent_id, "split": split, **summary})
            print(
                json.dumps(
                    {
                        "world": parent_id,
                        "index": world_index,
                        "of": len(rows),
                        "sequences": summary["sequence_count"],
                        "structural": summary["structural_observation_count"],
                        "pairs": sum(summary["association_pair_counts"].values()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        for stream in (
            traversal_stream,
            observation_stream,
            pair_stream,
            world_stream,
            identity_stream,
        ):
            stream.close()

    checks: dict[str, bool] = {}
    for split in ("train", "validation"):
        for key, expected in EXPECTED[split].items():
            checks[f"exact_{split}_{key}"] = split_totals[split][key] == expected
    checks.update(
        {
            "exact_total_directed_traversals": sum(x["directed_traversal_count"] for x in split_totals.values()) == 18132,
            "exact_total_sequences": sum(x["sequence_count"] for x in split_totals.values()) == 212588,
            "exact_total_unique_frames": global_frame_offset == 285108,
            "exact_total_referenced_frames": sum(x["referenced_frame_count"] for x in split_totals.values()) == 1062940,
            "exact_two_zero_sequence_traversals": sum(x["zero_sequence_traversal_count"] for x in split_totals.values()) == 2,
            "train_event_distribution_matches_sealed_proof": dict(split_events["train"]) == EXPECTED_TRAIN_EVENTS,
            "all_observation_ids_unique": len(all_observation_identity) == 212588,
            "all_association_pair_keys_unique": pair_key_count == len(pair_keys),
            "train_cross_traversal_positives_nonzero": split_pairs["train"]["cross_traversal_positive"] > 0,
            "train_hard_negatives_nonzero": split_pairs["train"]["same_event_geometry_hard_negative"] > 0,
            "validation_cross_traversal_positives_nonzero": split_pairs["validation"]["cross_traversal_positive"] > 0,
            "validation_hard_negatives_nonzero": split_pairs["validation"]["same_event_geometry_hard_negative"] > 0,
            "zero_strict_test_mtare_reads": True,
        }
    )
    passed = all(checks.values())
    total_sequences = sum(x["sequence_count"] for x in split_totals.values())
    unique_frames = global_frame_offset
    resource = {
        "range_shape_per_frame": [RINGS, AZIMUTH],
        "unique_range_float32_plus_valid_uint8_bytes": unique_frames * RINGS * AZIMUTH * (RANGE_BYTES + MASK_BYTES),
        "sequence_reference_int64_bytes": total_sequences * 5 * REFERENCE_BYTES,
        "forbidden_duplicated_five_frame_payload_bytes": total_sequences * 5 * RINGS * AZIMUTH * (RANGE_BYTES + MASK_BYTES),
        "unique_payload_gib": unique_frames * RINGS * AZIMUTH * (RANGE_BYTES + MASK_BYTES) / 1024**3,
        "forbidden_duplicated_payload_gib": total_sequences * 5 * RINGS * AZIMUTH * (RANGE_BYTES + MASK_BYTES) / 1024**3,
        "compression_estimate": "not claimed; teacher export must enforce an observed disk limit",
    }
    write_json(run_dir / "artifacts/resource_estimate.json", resource)
    summary = {
        "schema_version": "gse_teacher_manifest_v1",
        "overall_status": "PASS_GSE_TEACHER_MANIFEST_V1" if passed else "FAIL_GSE_TEACHER_MANIFEST_V1",
        "split_totals": {split: dict(values) for split, values in split_totals.items()},
        "event_counts": {split: {name: int(split_events[split][name]) for name in EVENT_NAMES} for split in ("train", "validation")},
        "association_pair_counts": {split: dict(split_pairs[split]) for split in ("train", "validation")},
        "total_association_pair_count": pair_key_count,
        "total_observation_count": len(all_observation_identity),
        "resource_estimate": resource,
        "checks": checks,
        "validation_worlds_read": 10,
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
        raise RuntimeError("GSE teacher manifest checks failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
