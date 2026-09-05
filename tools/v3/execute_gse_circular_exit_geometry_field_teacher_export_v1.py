#!/usr/bin/env python3
"""Export the qualified C01-C08 circular exit-geometry peak-field Teacher."""

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

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.representation.gse_circular_exit_geometry_field import (
    BEARING_BINS,
    BIN_WIDTH_DEG,
    encode_exit_geometry_peaks,
    heading_unit_to_bearing_bins,
)


PASS = "PASS_GSE_CIRCULAR_EXIT_GEOMETRY_FIELD_TEACHER_EXPORT_V1"
FAIL = "FAIL_GSE_CIRCULAR_EXIT_GEOMETRY_FIELD_TEACHER_EXPORT_V1"
COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE)
TARGET_NAMES = (
    "global_sequence_index", "global_frame_references", "local_frame_references",
    "presence", "heading_residual_deg", "opening_width_m", "width_valid_mask",
    "vertical_profile_m", "exit_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    total = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(item)))
        total += item.stat().st_size
    return digest.hexdigest(), len(files), total


def _partition(parent: str) -> str:
    if parent.endswith(tuple(f"_C{i:02d}" for i in range(1, 7))):
        return "fit"
    if parent.endswith(("_C07", "_C08")):
        return "selection"
    raise RuntimeError(f"forbidden world in circular Teacher export: {parent}")


def _write_array(group: zarr.hierarchy.Group, name: str, value: np.ndarray) -> None:
    chunks = (min(1024, len(value)),) + value.shape[1:]
    group.create_dataset(name, data=value, chunks=chunks, compressor=COMPRESSOR, overwrite=False)


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    cardinality = summary["cardinality"]
    x = np.asarray([1, 2, 3, 4])
    axes[0].bar(x, [cardinality[str(item)] for item in x], color="#4e79a7")
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Exit peaks per observation")
    axes[0].set_ylabel("Observations")
    axes[0].set_title("A  Exported peak population")
    split = summary["partition"]
    axes[1].bar([0, 1], [split[name]["visible_exit_tokens"] for name in ("fit", "selection")], color=["#59a14f", "#f28e2b"])
    axes[1].set_xticks([0, 1], ["C01-C06", "C07-C08"])
    axes[1].set_ylabel("Exported peaks")
    axes[1].set_title("B  Split-preserved targets")
    axes[2].bar([0, 1], [split[name]["compressed_bytes"] / 2**20 for name in ("fit", "selection")], color=["#76b7b2", "#e15759"])
    axes[2].set_xticks([0, 1], ["C01-C06", "C07-C08"])
    axes[2].set_ylabel("Compressed shard size (MiB)")
    axes[2].set_title("C  No LiDAR duplication")
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph circular exit-geometry Teacher export")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_circular_exit_geometry_field_teacher_export_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--shard-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher_root = output / "teacher"
    source_manifest = json.loads(args.shard_manifest.read_text(encoding="utf-8"))
    source_records = {row["parent_id"]: row for row in source_manifest["shards"] if row["split"] == "train"}
    source_shards = sorted(args.dataset.glob("*.zarr"))
    if len(source_shards) != 80 or len(source_records) != 80:
        raise RuntimeError("circular Teacher source population drift")
    manifest_rows = []
    totals = Counter()
    cardinality = Counter()
    partition_totals = {name: Counter() for name in ("fit", "selection")}
    max_heading_error = 0.0
    for source_path in source_shards:
        parent = source_path.stem
        split = _partition(parent)
        record = source_records.get(parent)
        if record is None:
            raise RuntimeError(f"missing source manifest row: {parent}")
        source_tree, source_files, source_bytes = _tree_hash(source_path)
        if source_tree != record["shard_tree_sha256"] or source_files != int(record["shard_file_count"]) or source_bytes != int(record["shard_bytes"]):
            raise RuntimeError(f"source shard drift: {parent}")
        source = zarr.open_group(str(source_path), mode="r")
        mask = np.asarray(source["exit_mask"], dtype=np.bool_)
        heading = np.asarray(source["exit_heading_unit"], dtype=np.float32)
        width = np.asarray(source["exit_opening_width_m"], dtype=np.float32)
        width_mask = np.asarray(source["exit_width_valid_mask"], dtype=np.bool_)
        profile = np.asarray(source["exit_vertical_profile_m"], dtype=np.float32)
        field = encode_exit_geometry_peaks(
            exit_mask=mask, heading_unit=heading, opening_width_m=width,
            width_valid_mask=width_mask, vertical_profile_m=profile,
        )
        arrays = {
            "global_sequence_index": np.asarray(source["global_sequence_index"], dtype=np.int64),
            "global_frame_references": np.asarray(source["global_frame_references"], dtype=np.int64),
            "local_frame_references": np.asarray(source["local_frame_references"], dtype=np.int32),
            "presence": field.presence.astype(np.uint8),
            "heading_residual_deg": field.heading_residual_deg.astype(np.float32),
            "opening_width_m": field.opening_width_m.astype(np.float32),
            "width_valid_mask": field.width_valid_mask.astype(np.uint8),
            "vertical_profile_m": field.vertical_profile_m.astype(np.float32),
            "exit_count": field.presence.sum(axis=1).astype(np.uint8),
        }
        destination = teacher_root / split / f"{parent}.zarr"
        destination.parent.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(str(destination), mode="w")
        group.attrs.update({
            "schema_version": "gse_circular_exit_geometry_field_teacher_v1",
            "parent_id": parent,
            "partition": split,
            "bearing_bins": BEARING_BINS,
            "bin_width_deg": BIN_WIDTH_DEG,
            "source_shard_tree_sha256": source_tree,
            "contains_lidar": False,
            "contains_teacher_identity": False,
        })
        for name in TARGET_NAMES:
            _write_array(group, name, arrays[name])
        reopened = zarr.open_group(str(destination), mode="r")
        if set(reopened.array_keys()) != set(TARGET_NAMES):
            raise RuntimeError(f"exported array set drift: {parent}")
        for name, expected in arrays.items():
            actual = np.asarray(reopened[name])
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"exported Teacher round-trip mismatch: {parent}/{name}")
        if any(forbidden in reopened for forbidden in ("range_m", "valid_mask", "exit_identity", "association_identity", "parent_id")):
            raise RuntimeError("exported Teacher contains forbidden LiDAR or identity")
        angle, bins, _ = heading_unit_to_bearing_bins(heading)
        for row in range(len(mask)):
            slots = np.flatnonzero(mask[row])
            active = bins[row, slots]
            recovered = (active.astype(np.float64) * BIN_WIDTH_DEG + arrays["heading_residual_deg"][row, active]) % 360.0
            error = np.abs((recovered - angle[row, slots] + 180.0) % 360.0 - 180.0)
            max_heading_error = max(max_heading_error, float(np.max(error, initial=0.0)))
        tree, files, size = _tree_hash(destination)
        count = arrays["exit_count"]
        for value, frequency in zip(*np.unique(count, return_counts=True), strict=True):
            cardinality[int(value)] += int(frequency)
        visible = int(arrays["presence"].sum())
        valid_width = int(arrays["width_valid_mask"].sum())
        totals["worlds"] += 1
        totals["observations"] += len(mask)
        totals["visible_exit_tokens"] += visible
        totals["width_valid_tokens"] += valid_width
        totals["compressed_bytes"] += size
        partition_totals[split]["worlds"] += 1
        partition_totals[split]["observations"] += len(mask)
        partition_totals[split]["visible_exit_tokens"] += visible
        partition_totals[split]["width_valid_tokens"] += valid_width
        partition_totals[split]["compressed_bytes"] += size
        manifest_rows.append({
            "parent_id": parent, "partition": split, "observations": len(mask),
            "visible_exit_tokens": visible, "width_valid_tokens": valid_width,
            "source_shard_tree_sha256": source_tree, "teacher_shard_tree_sha256": tree,
            "teacher_shard_file_count": files, "teacher_shard_bytes": size,
        })
    manifest_rows.sort(key=lambda row: row["parent_id"])
    with (output / "teacher_shard_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    partition = {name: {key: int(value) for key, value in values.items()} for name, values in partition_totals.items()}
    checks = {
        "exact_population": totals["worlds"] == 80 and totals["observations"] == 188_126 and totals["visible_exit_tokens"] == 396_913 and totals["width_valid_tokens"] == 389_026,
        "exact_partition": partition["fit"]["worlds"] == 60 and partition["selection"]["worlds"] == 20 and partition["fit"]["observations"] == 142_184 and partition["selection"]["observations"] == 45_942 and partition["fit"]["visible_exit_tokens"] == 299_872 and partition["selection"]["visible_exit_tokens"] == 97_041,
        "exact_cardinality": dict(cardinality) == {1: 7_525, 2: 154_279, 3: 24_458, 4: 1_864},
        "all_shards_roundtrip_exact": len(manifest_rows) == 80,
        "heading_roundtrip_at_most_1e4_deg": max_heading_error <= 1e-4,
        "no_lidar_or_identity_export": True,
        "compressed_size_below_1_gib": totals["compressed_bytes"] < 2**30,
        "zero_forbidden_operations": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_circular_exit_geometry_field_teacher_export_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_CIRCULAR_EXIT_GEOMETRY_FIELD_MODEL_READINESS" if scientific_pass else "STOP_CIRCULAR_EXIT_GEOMETRY_FIELD_BEFORE_TRAINING",
        "population": {"worlds": int(totals["worlds"]), "fit_worlds": int(partition["fit"]["worlds"]), "selection_worlds": int(partition["selection"]["worlds"]), "observations": int(totals["observations"]), "fit_observations": int(partition["fit"]["observations"]), "selection_observations": int(partition["selection"]["observations"]), "visible_exit_tokens": int(totals["visible_exit_tokens"]), "width_valid_tokens": int(totals["width_valid_tokens"]), "width_invalid_tokens": int(totals["visible_exit_tokens"] - totals["width_valid_tokens"]), "bearing_bins_per_observation": BEARING_BINS},
        "partition": partition,
        "cardinality": {str(key): int(value) for key, value in sorted(cardinality.items())},
        "maximum_heading_roundtrip_error_deg": max_heading_error,
        "compressed_bytes": int(totals["compressed_bytes"]),
        "compressed_mib": float(totals["compressed_bytes"] / 2**20),
        "arrays": list(TARGET_NAMES),
        "contains_lidar": False,
        "contains_teacher_identity": False,
        "checks": checks,
        "optimizer_steps": 0, "model_inference_frames": 0, "model_updates": 0,
        "normalization_steps": 0, "threshold_selection_steps": 0, "graph_replays": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_circular_exit_geometry_field_teacher_export_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
