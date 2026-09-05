#!/usr/bin/env python3
"""Audit whether Observable Teacher V2 admits deterministic dense polar slots."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.gse_observable_spatial_event_dataset import (
    load_observable_spatial_event_teacher,
)


PASS = "PASS_GSE_STRUCTURED_POLAR_TEACHER_FEASIBILITY_V1"
FAIL = "FAIL_GSE_STRUCTURED_POLAR_TEACHER_FEASIBILITY_V1"
POLAR_BINS = 180
BIN_WIDTH_DEG = 2.0
BIN_CENTER_OFFSET_DEG = 0.75
ROTATION_SHIFT_BINS = 17
ELEVATION_CENTERS_DEG = np.asarray((-12.0, -4.0, 4.0, 12.0), dtype=np.float64)


def assign_azimuth_bin(xyz: np.ndarray) -> np.ndarray:
    bearing = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0])) % 360.0
    return np.floor((bearing + 0.25) / BIN_WIDTH_DEG).astype(np.int64) % POLAR_BINS


def assign_elevation_band(xyz: np.ndarray) -> np.ndarray:
    horizontal = np.linalg.norm(xyz[..., :2], axis=-1)
    elevation = np.degrees(np.arctan2(xyz[..., 2], horizontal))
    return np.abs(elevation[..., None] - ELEVATION_CENTERS_DEG).argmin(axis=-1).astype(np.int64)


def circular_bin_distance(left: int, right: int) -> int:
    direct = abs(int(left) - int(right))
    return min(direct, POLAR_BINS - direct)


def wrapped_angle_error_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--readiness-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = load_observable_spatial_event_teacher(args.teacher_root.resolve())
    readiness = json.loads(args.readiness_summary.resolve().read_text(encoding="utf-8"))
    support = readiness.get("scan_support", {})
    support_ok = (
        readiness.get("status") == "PASS_GSE_GEOMETRY_ANCHORED_SPATIAL_EVENT_READINESS_V2"
        and readiness.get("population", {}).get("tokens") == 131424
        and support.get("unsupported_tokens") == 0
        and support.get("margin_m", {}).get("minimum", -1.0) >= 0.0
    )

    same_azimuth_pairs = 0
    adjacent_azimuth_pairs = 0
    same_azimuth_elevation_pairs = 0
    adjacent_same_elevation_pairs = 0
    multi_rows = 0
    minimum_separations: list[float] = []
    residuals: list[float] = []
    elevation_residuals: list[float] = []
    bin_histogram = np.zeros(POLAR_BINS, dtype=np.int64)
    band_histogram = np.zeros(4, dtype=np.int64)
    collision_examples: list[dict[str, object]] = []
    max_same_azimuth_occupancy = 0
    max_same_cell_occupancy = 0
    rotation_mismatches = 0
    token_count = 0
    partition_tokens: Counter[str] = Counter()

    for row in range(len(teacher.global_sequence_index)):
        active = np.flatnonzero(teacher.event_mask[row].astype(bool))
        if not len(active):
            continue
        xyz = teacher.event_relative_xyz_m[row, active].astype(np.float64)
        bins = assign_azimuth_bin(xyz)
        bands = assign_elevation_band(xyz)
        bearing = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
        horizontal = np.linalg.norm(xyz[:, :2], axis=1)
        elevation = np.degrees(np.arctan2(xyz[:, 2], horizontal))
        centers = (bins.astype(np.float64) * BIN_WIDTH_DEG + BIN_CENTER_OFFSET_DEG) % 360.0
        residuals.extend(wrapped_angle_error_deg(value, center) for value, center in zip(bearing, centers, strict=True))
        elevation_residuals.extend(abs(float(value) - float(ELEVATION_CENTERS_DEG[band])) for value, band in zip(elevation, bands, strict=True))
        bin_histogram += np.bincount(bins, minlength=POLAR_BINS)
        band_histogram += np.bincount(bands, minlength=4)
        token_count += len(active)
        partition_tokens["fit" if teacher.partition_code[row] == 0 else "selection"] += len(active)
        az_counts = Counter(int(value) for value in bins)
        cell_counts = Counter((int(bin_value), int(band)) for bin_value, band in zip(bins, bands, strict=True))
        max_same_azimuth_occupancy = max(max_same_azimuth_occupancy, max(az_counts.values()))
        max_same_cell_occupancy = max(max_same_cell_occupancy, max(cell_counts.values()))
        if len(active) >= 2:
            multi_rows += 1
            pair_separations = []
            for left in range(len(active)):
                for right in range(left + 1, len(active)):
                    separation = wrapped_angle_error_deg(bearing[left], bearing[right])
                    pair_separations.append(separation)
                    bin_distance = circular_bin_distance(int(bins[left]), int(bins[right]))
                    same_azimuth_pairs += int(bin_distance == 0)
                    adjacent_azimuth_pairs += int(bin_distance <= 1)
                    same_azimuth_elevation_pairs += int(bin_distance == 0 and bands[left] == bands[right])
                    adjacent_same_elevation_pairs += int(bin_distance <= 1 and bands[left] == bands[right])
                    if bin_distance <= 1 and len(collision_examples) < 64:
                        collision_examples.append({
                            "parent_id": str(teacher.parent_id[row]),
                            "global_sequence_index": int(teacher.global_sequence_index[row]),
                            "slots": [int(active[left]), int(active[right])],
                            "identities": [int(teacher.event_identity_index[row, active[left]]), int(teacher.event_identity_index[row, active[right]])],
                            "types": [int(teacher.event_type_index[row, active[left]]), int(teacher.event_type_index[row, active[right]])],
                            "azimuth_bins": [int(bins[left]), int(bins[right])],
                            "elevation_bands": [int(bands[left]), int(bands[right])],
                            "angular_separation_deg": separation,
                            "radial_distances_m": [float(np.linalg.norm(xyz[left])), float(np.linalg.norm(xyz[right]))],
                        })
            minimum_separations.append(min(pair_separations))
        rotation = math.radians(ROTATION_SHIFT_BINS * BIN_WIDTH_DEG)
        rotated = xyz.copy()
        rotated[:, 0] = math.cos(rotation) * xyz[:, 0] - math.sin(rotation) * xyz[:, 1]
        rotated[:, 1] = math.sin(rotation) * xyz[:, 0] + math.cos(rotation) * xyz[:, 1]
        expected = (bins + ROTATION_SHIFT_BINS) % POLAR_BINS
        rotation_mismatches += int(np.count_nonzero(assign_azimuth_bin(rotated) != expected))

    if not support_ok or rotation_mismatches:
        decision = "TEACHER_SUPPORT_OR_INDEX_CONTRACT_FAIL"
        status = FAIL
    elif adjacent_azimuth_pairs == 0:
        decision = "DENSE_180_BIN_LOCAL_MAX_SINGLE_SLOT_ALLOWED"
        status = PASS
    elif same_azimuth_pairs == 0:
        decision = "DENSE_180_BIN_TOPK_WITHOUT_NEIGHBOR_SUPPRESSION_ALLOWED"
        status = PASS
    elif same_azimuth_elevation_pairs == 0:
        decision = "DENSE_180x4_AZIMUTH_ELEVATION_SLOT_REQUIRED"
        status = PASS
    else:
        decision = "MULTI_DEPTH_POLAR_SLOT_REQUIRED"
        status = PASS

    residual_array = np.asarray(residuals, dtype=np.float64)
    elevation_array = np.asarray(elevation_residuals, dtype=np.float64)
    separation_array = np.asarray(minimum_separations, dtype=np.float64)
    summary = {
        "schema_version": "gse_structured_polar_teacher_feasibility_v1",
        "status": status,
        "decision": decision,
        "population": {
            "worlds": int(len(np.unique(teacher.parent_id))),
            "observations": int(len(teacher.global_sequence_index)),
            "tokens": token_count,
            "event_identities": int(len(np.unique(teacher.event_identity_index[teacher.event_mask.astype(bool)]))),
            "fit_tokens": int(partition_tokens["fit"]),
            "selection_tokens": int(partition_tokens["selection"]),
            "multi_event_rows": multi_rows,
        },
        "slot_contract": {
            "azimuth_bins": POLAR_BINS,
            "bin_width_deg": BIN_WIDTH_DEG,
            "bin_center_offset_deg": BIN_CENTER_OFFSET_DEG,
            "elevation_centers_deg": ELEVATION_CENTERS_DEG.tolist(),
            "rotation_shift_bins": ROTATION_SHIFT_BINS,
        },
        "conflicts": {
            "same_azimuth_bin_pairs": same_azimuth_pairs,
            "within_one_azimuth_bin_pairs_including_same": adjacent_azimuth_pairs,
            "same_azimuth_and_elevation_pairs": same_azimuth_elevation_pairs,
            "within_one_azimuth_bin_same_elevation_pairs": adjacent_same_elevation_pairs,
            "max_same_azimuth_occupancy_per_row": max_same_azimuth_occupancy,
            "max_same_azimuth_elevation_occupancy_per_row": max_same_cell_occupancy,
            "examples_saved": len(collision_examples),
        },
        "geometry": {
            "azimuth_bin_residual_deg": {
                "maximum": float(residual_array.max()),
                "mean": float(residual_array.mean()),
                "p99": float(np.quantile(residual_array, 0.99)),
            },
            "elevation_band_residual_deg": {
                "maximum": float(elevation_array.max()),
                "mean": float(elevation_array.mean()),
                "p99": float(np.quantile(elevation_array, 0.99)),
            },
            "multi_event_minimum_angular_separation_deg": {
                "minimum": float(separation_array.min()),
                "median": float(np.median(separation_array)),
                "p01": float(np.quantile(separation_array, 0.01)),
            },
            "rotation_index_mismatches": rotation_mismatches,
            "azimuth_bin_token_histogram": bin_histogram.astype(int).tolist(),
            "elevation_band_token_histogram": band_histogram.astype(int).tolist(),
        },
        "sealed_support": {
            "readiness_status": readiness.get("status"),
            "tokens": support.get("tokens"),
            "unsupported_tokens": support.get("unsupported_tokens"),
            "minimum_margin_m": support.get("margin_m", {}).get("minimum"),
            "definition": support.get("definition"),
            "support_ok": support_ok,
        },
        "checks": {
            "exact_population": token_count == 131424 and partition_tokens == Counter({"fit": 98279, "selection": 33145}),
            "all_tokens_have_sealed_local_support": support_ok,
            "azimuth_residual_within_one_degree": float(residual_array.max()) <= 1.0 + 1e-9,
            "integer_two_degree_rotation_exact": rotation_mismatches == 0,
            "one_unambiguous_slot_scheme_selected": decision != "TEACHER_SUPPORT_OR_INDEX_CONTRACT_FAIL",
        },
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "threshold_selection_steps": 0,
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    summary["checks"]["all_passed"] = all(summary["checks"].values())
    if not summary["checks"]["all_passed"]:
        summary["status"] = FAIL

    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "conflict_examples.json").write_text(json.dumps(collision_examples, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), constrained_layout=True)
    axes[0].bar(np.arange(POLAR_BINS), bin_histogram, width=1.0, color="#1971c2")
    axes[0].set_xlabel("2° azimuth bin")
    axes[0].set_ylabel("Teacher tokens")
    axes[0].set_xlim(0, POLAR_BINS - 1)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].hist(residual_array, bins=40, color="#2f9e44")
    axes[1].set_xlabel("absolute azimuth residual to assigned bin center (deg)")
    axes[1].set_ylabel("tokens")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].hist(separation_array, bins=40, color="#f08c00")
    axes[2].axvline(2.0, color="#c92a2a", linestyle="--", label="one-bin neighborhood")
    axes[2].set_xlabel("minimum pair separation in multi-event row (deg)")
    axes[2].set_ylabel("rows")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.suptitle(f"Structured polar Teacher feasibility: {decision.lower().replace('_', ' ')}")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_structured_polar_teacher_feasibility_v1.{suffix}", dpi=220)
    plt.close(figure)
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_structured_polar_teacher_feasibility_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
