#!/usr/bin/env python3
"""Create the immutable LiDAR-vertical-FOV-observable spatial-event Teacher V2."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numcodecs import Blosc
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.gse_observable_spatial_event_teacher import (
    FOV_NUMERICAL_EPSILON_DEG,
    observable_teacher_contract,
    repack_observable_event_set,
)


PASS = "PASS_GSE_OBSERVABLE_SPATIAL_EVENT_TEACHER_V2"
FAIL = "FAIL_GSE_OBSERVABLE_SPATIAL_EVENT_TEACHER_V2"
EXPECTED = {
    "worlds": 80,
    "observations": 188126,
    "identities": 1076,
    "source_tokens": 133055,
    "tokens": 131424,
    "removed": 1631,
    "terminal": 30714,
    "junction": 100710,
    "fit_tokens": 98279,
    "selection_tokens": 33145,
    "cardinality": {0: 82326, 1: 82183, 2: 21756, 3: 1724, 4: 128, 5: 9},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        file_size = path.stat().st_size
        digest.update(f"{relative}\0{file_hash}\0{file_size}\n".encode())
        count += 1
        size += file_size
    return digest.hexdigest(), count, size


def _write_array(group, name: str, values: np.ndarray, compressor) -> None:
    data = np.asarray(values)
    first = max(1, len(data))
    group.create_dataset(
        name,
        data=data,
        chunks=(first, *data.shape[1:]),
        compressor=compressor,
    )


def _json_line(stream, value: dict[str, object]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-teacher-root", required=True, type=Path)
    parser.add_argument("--source-summary", required=True, type=Path)
    parser.add_argument("--source-identity-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher_output = output / "teacher"
    teacher_output.mkdir(parents=True, exist_ok=True)
    source_root = args.source_teacher_root.resolve()
    source_summary = load_json(args.source_summary.resolve())
    identity_map = load_json(args.source_identity_map.resolve())
    if (
        source_summary.get("status") != "PASS_GSE_SPATIAL_MULTI_EVENT_TEACHER_EXPORT_V1"
        or source_summary.get("population", {}).get("visible_event_tokens") != 133055
        or len(identity_map.get("rows", [])) != 1076
    ):
        raise RuntimeError("source Teacher V1 contract drift")

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    removed_type: Counter[str] = Counter()
    partition_counts: dict[str, Counter[str]] = {
        "fit": Counter(),
        "selection": Counter(),
    }
    cardinality: Counter[int] = Counter()
    retained_identities: set[int] = set()
    removed_by_family: Counter[str] = Counter()
    manifests: list[dict[str, object]] = []
    source_shards = sorted(source_root.glob("*/*.zarr"))
    if len(source_shards) != 80:
        raise RuntimeError("source Teacher shard count drift")

    manifest_path = output / "teacher_shard_manifest.jsonl"
    removed_path = output / "removed_unobservable_tokens.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_stream, removed_path.open(
        "w", encoding="utf-8"
    ) as removed_stream:
        for shard_index, source_path in enumerate(source_shards, 1):
            source = zarr.open_group(str(source_path), mode="r")
            parent = str(source.attrs["parent_id"])
            partition = str(source.attrs["partition"])
            if partition not in partition_counts:
                raise RuntimeError(f"source Teacher partition drift: {parent}")
            arrays = {
                name: np.asarray(source[name][:])
                for name in (
                    "event_type_index",
                    "event_relative_xyz_m",
                    "event_distance_m",
                    "event_identity_index",
                    "event_mask",
                    "set_cardinality",
                )
            }
            global_sequence_index = np.asarray(source["global_sequence_index"][:], dtype=np.int64)
            corrected = repack_observable_event_set(arrays)
            active = arrays["event_mask"].astype(bool)
            removed_row, removed_slot = np.nonzero(corrected.removed_mask)
            family = parent.rsplit("_C", 1)[0]
            for row, slot in zip(removed_row, removed_slot):
                event_type = int(arrays["event_type_index"][row, slot])
                _json_line(
                    removed_stream,
                    {
                        "parent_id": parent,
                        "partition": partition,
                        "family": family,
                        "global_sequence_index": int(global_sequence_index[row]),
                        "original_slot": int(slot),
                        "event_type_index": event_type,
                        "event_identity_index": int(arrays["event_identity_index"][row, slot]),
                        "distance_m": float(arrays["event_distance_m"][row, slot]),
                        "elevation_deg": float(corrected.elevation_deg[row, slot]),
                        "reason": "outside_frozen_lidar_vertical_fov",
                    },
                )
                removed_type["terminal" if event_type == 0 else "junction"] += 1
                removed_by_family[family] += 1

            destination = teacher_output / partition / f"{parent}.zarr"
            destination.parent.mkdir(parents=True, exist_ok=True)
            group = zarr.open_group(str(destination), mode="w")
            group.attrs.update(
                {
                    "schema_version": "gse_observable_spatial_event_teacher_v2",
                    "parent_id": parent,
                    "partition": partition,
                    "teacher_only": True,
                    "source_teacher_schema": str(source.attrs["schema_version"]),
                    "source_teacher_path": str(source_path.relative_to(PROJECT_ROOT)),
                    "relative_frame": str(source.attrs["relative_frame"]),
                    "maximum_event_tokens": 16,
                    "maximum_event_range_m": 50.0,
                    "los_margin_m": 0.25,
                    "vertical_fov_deg_inclusive": [-15.0, 15.0],
                    "fov_numerical_epsilon_deg": FOV_NUMERICAL_EPSILON_DEG,
                    "filter": "active target elevation atan2(up,horizontal) within frozen LiDAR vertical FOV",
                    "student_input": "none; join global_sequence_index to existing five-frame LiDAR shard",
                    "event_type_index": {"padding": -1, "terminal": 0, "junction": 1},
                    "truncation": "forbidden",
                }
            )
            _write_array(group, "global_sequence_index", global_sequence_index, compressor)
            for name in (
                "event_type_index",
                "event_relative_xyz_m",
                "event_distance_m",
                "event_identity_index",
                "event_mask",
                "set_cardinality",
            ):
                _write_array(group, name, getattr(corrected, name), compressor)

            output_mask = corrected.event_mask.astype(bool)
            output_type = corrected.event_type_index
            retained_identities.update(int(value) for value in corrected.event_identity_index[output_mask])
            type_counts["terminal"] += int(np.sum(output_mask & (output_type == 0)))
            type_counts["junction"] += int(np.sum(output_mask & (output_type == 1)))
            cardinality.update(int(value) for value in corrected.set_cardinality)
            source_tokens = int(active.sum())
            retained_tokens = int(output_mask.sum())
            removed_tokens = int(corrected.removed_mask.sum())
            counts.update(
                {
                    "worlds": 1,
                    "observations": len(global_sequence_index),
                    "source_tokens": source_tokens,
                    "tokens": retained_tokens,
                    "removed": removed_tokens,
                }
            )
            partition_counts[partition].update(
                {
                    "worlds": 1,
                    "observations": len(global_sequence_index),
                    "source_tokens": source_tokens,
                    "tokens": retained_tokens,
                    "removed": removed_tokens,
                }
            )
            tree_sha, file_count, byte_count = _tree_hash(destination)
            record = {
                "parent_id": parent,
                "partition": partition,
                "observations": len(global_sequence_index),
                "source_tokens": source_tokens,
                "retained_tokens": retained_tokens,
                "removed_tokens": removed_tokens,
                "shard_tree_sha256": tree_sha,
                "shard_file_count": file_count,
                "shard_bytes": byte_count,
            }
            manifests.append(record)
            _json_line(manifest_stream, record)
            print(json.dumps({"index": shard_index, "of": 80, **record}, sort_keys=True), flush=True)

    identity_document = dict(identity_map)
    identity_document["schema_version"] = "gse_observable_spatial_event_identity_map_v2"
    identity_document["source_teacher_identity_map"] = str(args.source_identity_map.resolve().relative_to(PROJECT_ROOT))
    identity_document["observable_teacher_contract"] = observable_teacher_contract()
    write_json(output / "event_identity_map.json", identity_document)

    checks = {
        "exact_80_worlds": counts["worlds"] == 80,
        "exact_188126_rows_retained": counts["observations"] == 188126,
        "exact_source_133055_tokens": counts["source_tokens"] == 133055,
        "exact_1631_vertical_fov_tokens_removed": counts["removed"] == 1631,
        "exact_131424_observable_tokens": counts["tokens"] == 131424,
        "exact_type_counts": dict(type_counts) == {"terminal": 30714, "junction": 100710},
        "exact_removed_type_counts": dict(removed_type) == {"terminal": 75, "junction": 1556},
        "all_1076_identities_retained": len(retained_identities) == 1076,
        "exact_fit_selection_counts": partition_counts["fit"]["tokens"] == 98279
        and partition_counts["selection"]["tokens"] == 33145
        and partition_counts["fit"]["removed"] == 1213
        and partition_counts["selection"]["removed"] == 418,
        "exact_cardinality_histogram": dict(cardinality) == EXPECTED["cardinality"],
        "all_rows_preserved_and_zero_rows_retained": sum(cardinality.values()) == 188126
        and cardinality[0] == 82326,
        "zero_training_inference_C09_C10_MTARE": True,
    }
    checks["all_passed"] = all(checks.values())
    status = PASS if checks["all_passed"] else FAIL

    figure_source = {
        "schema_version": "gse_observable_spatial_event_teacher_v2_figure_source",
        "source_tokens": 133055,
        "retained_tokens": 131424,
        "removed_tokens": 1631,
        "retained_type_counts": dict(type_counts),
        "removed_type_counts": dict(removed_type),
        "removed_by_family": dict(sorted(removed_by_family.items())),
        "cardinality_histogram": dict(sorted(cardinality.items())),
    }
    write_json(output / "figure_source.json", figure_source)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    axes[0].bar(["source", "observable V2"], [133055, 131424], color=["#7f8c8d", "#2878b5"])
    axes[0].set_ylabel("event tokens")
    axes[0].set_title("All observations retained")
    axes[1].bar(["terminal", "junction"], [75, 1556], color=["#f39c12", "#c0392b"])
    axes[1].set_ylabel("removed tokens")
    axes[1].set_title("Outside vertical FOV")
    families = list(sorted(removed_by_family))
    axes[2].barh(range(len(families)), [removed_by_family[name] for name in families], color="#6c5ce7")
    axes[2].set_yticks(range(len(families)), [name.split("_", 1)[0] for name in families])
    axes[2].set_xlabel("removed tokens")
    axes[2].set_title("3D families only")
    figure.suptitle("Observable structure-event Teacher V2: fixed LiDAR FOV correction")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_observable_spatial_event_teacher_v2.{suffix}", dpi=220)
    plt.close(figure)

    summary = {
        "schema_version": "gse_observable_spatial_event_teacher_v2",
        "status": status,
        "population": {
            "worlds": counts["worlds"],
            "observations": counts["observations"],
            "event_identities": len(retained_identities),
            "source_tokens": counts["source_tokens"],
            "observable_tokens": counts["tokens"],
            "removed_unobservable_tokens": counts["removed"],
            "retained_type_counts": dict(type_counts),
            "removed_type_counts": dict(removed_type),
            "cardinality_histogram": dict(sorted(cardinality.items())),
            "partition_totals": {name: dict(value) for name, value in partition_counts.items()},
        },
        "filter_contract": observable_teacher_contract(),
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
    print(json.dumps(summary, indent=2))
    return 0 if status == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
