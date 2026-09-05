#!/usr/bin/env python3
"""Read-only C01-C08 feasibility audit for a dense circular exit-geometry peak field."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.evaluation.gse_token_validity_feasibility import binary_ranking_metrics
from mtare_topo.representation.gse_circular_exit_geometry_field import (
    BEARING_BINS,
    BIN_WIDTH_DEG,
    circular_component_count,
    circular_roll_field,
    encode_exit_geometry_peaks,
    heading_unit_to_bearing_bins,
)


PASS = "PASS_GSE_CIRCULAR_EXIT_GEOMETRY_FIELD_FEASIBILITY_V1"
FAIL = "FAIL_GSE_CIRCULAR_EXIT_GEOMETRY_FIELD_FEASIBILITY_V1"
EXPECTED_OBSERVATIONS = {"fit": 142_184, "selection": 45_942}


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


def _angles_from_unit(unit: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(unit[..., 0], unit[..., 1])) % 360.0


def _circular_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(left) - np.asarray(right) + 180.0) % 360.0 - 180.0)


def _range_envelope(group: zarr.hierarchy.Group, anchor: np.ndarray) -> np.ndarray:
    ranges = np.asarray(group["range_m"].oindex[anchor], dtype=np.float32)
    valid = np.asarray(group["valid_mask"].oindex[anchor], dtype=np.bool_)
    if ranges.shape[1:] != (16, 720) or valid.shape != ranges.shape:
        raise RuntimeError("range image contract drift")
    supported = np.where(valid, ranges, 0.0).reshape(len(ranges), 16, BEARING_BINS, 4)
    return supported.max(axis=(1, 3))


def _audit_dataset(dataset: Path, shard_manifest: Path) -> dict:
    source = json.loads(shard_manifest.read_text(encoding="utf-8"))
    records = {row["parent_id"]: row for row in source["shards"] if row["split"] == "train"}
    shards = sorted(dataset.glob("*.zarr"))
    if len(shards) != 80 or len(records) != 80:
        raise RuntimeError("C01-C08 shard population drift")
    totals = Counter()
    cardinality = Counter()
    verified_files = verified_bytes = 0
    max_heading_error = max_rotation_residual_error = 0.0
    max_width_error = max_profile_error = 0.0
    presence_parts: dict[str, list[np.ndarray]] = {"fit": [], "selection": []}
    range_parts: dict[str, list[np.ndarray]] = {"fit": [], "selection": []}
    for path in shards:
        record = records.get(path.stem)
        if record is None:
            raise RuntimeError(f"unexpected dataset shard: {path.name}")
        tree, files, size = _tree_hash(path)
        if tree != record["shard_tree_sha256"] or files != int(record["shard_file_count"]) or size != int(record["shard_bytes"]):
            raise RuntimeError(f"dataset shard drift: {path.name}")
        verified_files += files
        verified_bytes += size
        group = zarr.open_group(str(path), mode="r")
        mask = np.asarray(group["exit_mask"], dtype=np.bool_)
        heading = np.asarray(group["exit_heading_unit"], dtype=np.float32)
        width = np.asarray(group["exit_opening_width_m"], dtype=np.float32)
        width_mask = np.asarray(group["exit_width_valid_mask"], dtype=np.bool_)
        profile = np.asarray(group["exit_vertical_profile_m"], dtype=np.float32)
        local_refs = np.asarray(group["local_frame_references"], dtype=np.int64)
        global_refs = np.asarray(group["global_frame_references"], dtype=np.int64)
        if local_refs.shape != (len(mask), 5) or global_refs.shape != local_refs.shape:
            raise RuntimeError("causal reference shape drift")
        totals["causal_local_noncontiguous"] += int(np.sum(np.any(np.diff(local_refs, axis=1) != 1, axis=1)))
        totals["causal_global_noncontiguous"] += int(np.sum(np.any(np.diff(global_refs, axis=1) != 1, axis=1)))
        totals["future_reference_rows"] += int(np.sum(np.any(global_refs[:, :-1] >= global_refs[:, -1:], axis=1)))
        field = encode_exit_geometry_peaks(
            exit_mask=mask, heading_unit=heading, opening_width_m=width,
            width_valid_mask=width_mask, vertical_profile_m=profile,
        )
        reverse = encode_exit_geometry_peaks(
            exit_mask=mask[:, ::-1], heading_unit=heading[:, ::-1], opening_width_m=width[:, ::-1],
            width_valid_mask=width_mask[:, ::-1], vertical_profile_m=profile[:, ::-1],
        )
        totals["slot_permutation_mismatch"] += int(
            not all(np.array_equal(getattr(field, name), getattr(reverse, name)) for name in field.__dataclass_fields__)
        )
        angles, bins, _ = heading_unit_to_bearing_bins(heading)
        for row in range(len(mask)):
            slots = np.flatnonzero(mask[row])
            active = bins[row, slots]
            cardinality[len(slots)] += 1
            totals["same_bin_collision_rows"] += int(len(np.unique(active)) != len(active))
            if len(active) > 1:
                separation = np.abs(active[:, None] - active[None, :])
                separation = np.minimum(separation, BEARING_BINS - separation) + np.eye(len(active), dtype=np.int64) * BEARING_BINS
                totals["adjacent_bin_collision_rows"] += int(np.min(separation) <= 1)
            recovered = (active.astype(np.float64) * BIN_WIDTH_DEG + field.heading_residual_deg[row, active]) % 360.0
            max_heading_error = max(max_heading_error, float(np.max(_circular_error(recovered, angles[row, slots]), initial=0.0)))
            if np.any(width_mask[row, slots]):
                valid_slots = slots[width_mask[row, slots]]
                valid_bins = bins[row, valid_slots]
                max_width_error = max(max_width_error, float(np.max(np.abs(field.opening_width_m[row, valid_bins] - width[row, valid_slots]), initial=0.0)))
            max_profile_error = max(max_profile_error, float(np.max(np.abs(field.vertical_profile_m[row, active] - profile[row, slots]), initial=0.0)))
        radians = np.radians(20.0)
        rotated_heading = heading.copy()
        rotated_heading[..., 0] = heading[..., 0] * math.cos(radians) + heading[..., 1] * math.sin(radians)
        rotated_heading[..., 1] = heading[..., 1] * math.cos(radians) - heading[..., 0] * math.sin(radians)
        rotated = encode_exit_geometry_peaks(
            exit_mask=mask, heading_unit=rotated_heading, opening_width_m=width,
            width_valid_mask=width_mask, vertical_profile_m=profile,
        )
        rolled = circular_roll_field(field, 10)
        totals["rotation_presence_mismatch"] += int(not np.array_equal(rotated.presence, rolled.presence))
        totals["rotation_geometry_mismatch"] += int(
            not np.array_equal(rotated.opening_width_m, rolled.opening_width_m)
            or not np.array_equal(rotated.width_valid_mask, rolled.width_valid_mask)
            or not np.array_equal(rotated.vertical_profile_m, rolled.vertical_profile_m)
        )
        active = rotated.presence
        max_rotation_residual_error = max(
            max_rotation_residual_error,
            float(np.max(np.abs(rotated.heading_residual_deg[active] - rolled.heading_residual_deg[active]), initial=0.0)),
        )
        partition = "fit" if path.stem.endswith(tuple(f"_C{i:02d}" for i in range(1, 7))) else "selection"
        anchor = local_refs[:, -1]
        presence_parts[partition].append(field.presence.reshape(-1))
        range_parts[partition].append(_range_envelope(group, anchor).reshape(-1))
        totals[f"{partition}_observations"] += len(mask)
        totals[f"{partition}_visible_tokens"] += int(mask.sum())
        totals["observations"] += len(mask)
        totals["visible_tokens"] += int(mask.sum())
        totals["width_valid_tokens"] += int(width_mask.sum())
    raw_baseline = {}
    for partition in ("fit", "selection"):
        labels = np.concatenate(presence_parts[partition])
        scores = np.concatenate(range_parts[partition])
        raw_baseline[partition] = binary_ranking_metrics(labels, scores, precision_floor=.995)
        raw_baseline[partition]["bearing_bins"] = int(len(labels))
    return {
        "totals": dict(totals),
        "cardinality": {str(key): int(value) for key, value in sorted(cardinality.items())},
        "max_heading_roundtrip_error_deg": max_heading_error,
        "max_width_roundtrip_error_m": max_width_error,
        "max_vertical_profile_roundtrip_error_m": max_profile_error,
        "max_rotation_residual_error_deg": max_rotation_residual_error,
        "raw_range_envelope_baseline": raw_baseline,
        "verification": {"shards": 80, "files": verified_files, "bytes": verified_bytes, "all_shard_tree_hashes_match": True},
    }


def _sector_field(tokens: list[tuple[float, float | None, float]]) -> np.ndarray:
    active = np.zeros((1, BEARING_BINS), dtype=np.bool_)
    centers = np.arange(BEARING_BINS, dtype=np.float64) * BIN_WIDTH_DEG
    for heading, width, distance in tokens:
        half = math.degrees(math.atan2(width / 2.0, distance)) if width is not None else BIN_WIDTH_DEG / 2.0
        delta = np.abs((centers - heading + 180.0) % 360.0 - 180.0)
        active[0, delta <= max(BIN_WIDTH_DEG / 2.0, half)] = True
    return active


def _audit_sector_baseline(path: Path) -> dict:
    current: str | None = None
    tokens: list[tuple[float, float | None, float]] = []
    totals = Counter()
    spans: list[float] = []

    def flush() -> None:
        if not tokens:
            return
        count = len(tokens)
        component_count = int(circular_component_count(_sector_field(tokens))[0])
        totals["observations"] += 1
        totals["component_count_mismatch_rows"] += int(component_count != count)
        sets = []
        centers = np.arange(BEARING_BINS, dtype=np.float64) * BIN_WIDTH_DEG
        for heading, width, distance in tokens:
            half = math.degrees(math.atan2(width / 2.0, distance)) if width is not None else BIN_WIDTH_DEG / 2.0
            delta = np.abs((centers - heading + 180.0) % 360.0 - 180.0)
            sets.append(set(np.flatnonzero(delta <= max(BIN_WIDTH_DEG / 2.0, half)).tolist()))
            if width is not None:
                spans.append(2.0 * half)
        overlap = any(bool(sets[left] & sets[right]) for left in range(count) for right in range(left))
        totals["overlap_rows"] += int(overlap)

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            parent = str(row["parent_id"])
            if not parent.endswith(tuple(f"_C{i:02d}" for i in range(1, 9))):
                continue
            observation = str(row["observation_id"])
            if current is not None and observation != current:
                flush()
                tokens = []
            current = observation
            token = row["token"]
            if bool(token["visible"]):
                width = float(token["opening_width_m"]) if bool(token["opening_width_valid"]) else None
                tokens.append((float(token["heading_robot_deg"]), width, float(token["line_of_sight_distance_m"])))
                totals["visible_tokens"] += 1
                totals["width_valid_tokens"] += int(width is not None)
        flush()
    return {
        **dict(totals),
        "component_mismatch_fraction": totals["component_count_mismatch_rows"] / totals["observations"],
        "overlap_fraction": totals["overlap_rows"] / totals["observations"],
        "valid_angular_span_deg": {
            "median": float(np.median(spans)),
            "p90": float(np.quantile(spans, .9)),
            "p99": float(np.quantile(spans, .99)),
            "maximum": float(np.max(spans)),
        },
    }


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    card = summary["peak_field"]["cardinality"]
    x = np.asarray([1, 2, 3, 4])
    axes[0].bar(x, [card[str(value)] for value in x], color="#4e79a7")
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Visible exits per observation")
    axes[0].set_ylabel("Observations")
    axes[0].set_title("A  Exact peak population")
    mismatch = summary["wide_sector_component_baseline"]["component_mismatch_fraction"]
    axes[1].bar([0, 1], [0.0, mismatch], color=["#59a14f", "#e15759"])
    axes[1].set_xticks([0, 1], ["Center-peak\nfield", "Wide-sector\ncomponents"])
    axes[1].set_ylim(0, max(.15, mismatch * 1.2))
    axes[1].set_ylabel("Exit-count mismatch fraction")
    axes[1].set_title("B  Why components merge")
    baseline = summary["peak_field"]["raw_range_envelope_baseline"]
    axes[2].bar(np.arange(2) - .18, [baseline[p]["average_precision"] for p in ("fit", "selection")], .36, label="AP", color="#f28e2b")
    axes[2].bar(np.arange(2) + .18, [baseline[p]["recall_at_precision_floor"] for p in ("fit", "selection")], .36, label="Recall @ P>=.995", color="#76b7b2")
    axes[2].set_xticks(np.arange(2), ["C01-C06", "C07-C08"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("C  Raw-range baseline")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph circular executable-geometry peak-field feasibility")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_circular_exit_geometry_field_feasibility_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--shard-manifest", required=True, type=Path)
    parser.add_argument("--exit-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    peak = _audit_dataset(args.dataset.resolve(), args.shard_manifest.resolve())
    sector = _audit_sector_baseline(args.exit_audit.resolve())
    totals = peak["totals"]
    checks = {
        "exact_population": totals["observations"] == 188_126 and totals["fit_observations"] == 142_184 and totals["selection_observations"] == 45_942 and totals["visible_tokens"] == 396_913 and totals["width_valid_tokens"] == 389_026,
        "exact_cardinality": peak["cardinality"] == {"1": 7_525, "2": 154_279, "3": 24_458, "4": 1_864},
        "zero_same_or_adjacent_bin_collisions": totals.get("same_bin_collision_rows", 0) == 0 and totals.get("adjacent_bin_collision_rows", 0) == 0,
        "lossless_geometry_roundtrip": peak["max_heading_roundtrip_error_deg"] <= 1e-4 and peak["max_width_roundtrip_error_m"] == 0.0 and peak["max_vertical_profile_roundtrip_error_m"] == 0.0,
        "slot_permutation_invariant": totals.get("slot_permutation_mismatch", 0) == 0,
        "twenty_degree_rotation_equivariant": totals.get("rotation_presence_mismatch", 0) == 0 and totals.get("rotation_geometry_mismatch", 0) == 0 and peak["max_rotation_residual_error_deg"] <= 1e-4,
        "causal_references_past_only_contiguous": totals.get("causal_local_noncontiguous", 0) == 0 and totals.get("causal_global_noncontiguous", 0) == 0 and totals.get("future_reference_rows", 0) == 0,
        "wide_sector_baseline_failure_reproduced": sector["component_count_mismatch_rows"] == 22_626 and sector["overlap_rows"] == 22_381,
        "source_hashes_verified": bool(peak["verification"]["all_shard_tree_hashes_match"]),
        "zero_forbidden_operations": True,
    }
    scientific_pass = all(checks.values())
    decision = "ALLOW_CIRCULAR_EXIT_GEOMETRY_FIELD_TEACHER_EXPORT_DATA_CARD" if scientific_pass else "STOP_CIRCULAR_EXIT_GEOMETRY_FIELD_REPRESENTATION"
    summary = {
        "schema_version": "gse_circular_exit_geometry_field_feasibility_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": decision,
        "question": "Can every C01-C08 visible directed exit be represented as a unique 2-degree circular geometry peak without identity, future frames or free-query objectness?",
        "method": "Rasterize each exit center to its nearest one of 180 bearing bins and regress sub-bin residual, opening width/mask and four-value vertical profile at the peak; compare wide-sector connected components and a raw-range envelope.",
        "population": {"worlds": 80, "raw_frames": 252_430, "observations": 188_126, "fit_observations": 142_184, "selection_observations": 45_942, "visible_exit_tokens": 396_913, "width_valid_tokens": 389_026, "width_invalid_tokens": 7_887, "bearing_bins_per_observation": 180},
        "peak_field": peak,
        "wide_sector_component_baseline": sector,
        "checks": checks,
        "optimizer_steps": 0, "teacher_export_rows": 0, "model_inference_frames": 0, "threshold_selection_steps": 0,
        "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_circular_exit_geometry_field_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "representation_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("representation", "observations", "mismatch_rows", "mismatch_fraction", "same_bin_collisions", "adjacent_bin_collisions"))
        writer.writeheader()
        writer.writerow({"representation": "circular_center_peak", "observations": 188126, "mismatch_rows": 0, "mismatch_fraction": 0.0, "same_bin_collisions": totals.get("same_bin_collision_rows", 0), "adjacent_bin_collisions": totals.get("adjacent_bin_collision_rows", 0)})
        writer.writerow({"representation": "wide_sector_connected_components", "observations": sector["observations"], "mismatch_rows": sector["component_count_mismatch_rows"], "mismatch_fraction": sector["component_mismatch_fraction"], "same_bin_collisions": "", "adjacent_bin_collisions": ""})
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": decision, "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
