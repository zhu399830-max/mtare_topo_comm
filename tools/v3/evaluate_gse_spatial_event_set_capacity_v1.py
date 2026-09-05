#!/usr/bin/env python3
"""Compare three spatial-set seeds with exclusive and analytic baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG
from mtare_topo.data.gse_spatial_event_set_cache import (
    load_spatial_event_teacher,
)
from mtare_topo.evaluation.gse_spatial_event_set_metrics import select_fixed_grid_threshold
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES
from mtare_topo.semantics.nonlearning_geometry_observation import (
    GEOMETRY_TRANSITION_THRESHOLD_M,
    TURN_CURVATURE_THRESHOLD_PER_M,
)
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
from mtare_topo.semantics.range_geometry_baseline import RangeGeometryBaseline


THRESHOLD_GRID = tuple(round(value * 0.05, 2) for value in range(1, 20))
PASS = "PASS_GSE_SPATIAL_EVENT_SET_CAPACITY_V1"
FAIL = "FAIL_GSE_SPATIAL_EVENT_SET_CAPACITY_V1"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - values.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _exclusive_baseline(legacy_logits, projections, rows, global_index):
    probabilities = np.mean(
        [_softmax(np.asarray(values[rows], dtype=np.float32)) for values in legacy_logits], axis=0
    )
    event_values = probabilities[:, [EVENT_NAMES.index("terminal"), EVENT_NAMES.index("junction")]]
    event_type = np.argmax(event_values, axis=1).astype(np.int8)
    confidence = np.max(event_values, axis=1).astype(np.float32)
    vectors = []
    for path in projections:
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["global_sequence_index"], global_index):
                raise RuntimeError("exclusive center projection global index drift")
            vectors.append(archive["predicted_local_vector_m"][rows].astype(np.float32))
    relative = np.mean(vectors, axis=0).astype(np.float32)
    return {
        "confidence": confidence[:, None],
        "event_type": event_type[:, None],
        "relative_xyz_m": relative[:, None],
    }


def _nonlearning_baseline(dataset_run: Path, selection_global: np.ndarray):
    import zarr

    confidence = np.zeros((len(selection_global), 1), dtype=np.float32)
    event_type = np.zeros((len(selection_global), 1), dtype=np.int8)
    relative = np.zeros((len(selection_global), 1, 3), dtype=np.float32)
    output_offset = 0
    exit_method = RangeExitBaseline()
    geometry_method = RangeGeometryBaseline()
    paths = [
        path
        for path in sorted((dataset_run / "artifacts/dataset/train").glob("*.zarr"))
        if path.stem.endswith(("C07", "C08"))
    ]
    if len(paths) != 20:
        raise RuntimeError("nonlearning baseline must read exactly C07--C08")
    for world_index, path in enumerate(paths, start=1):
        group = zarr.open_group(str(path), mode="r")
        references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
        world_global = np.asarray(group["global_sequence_index"][:], dtype=np.int64)
        stop_offset = output_offset + len(references)
        if not np.array_equal(selection_global[output_offset:stop_offset], world_global):
            raise RuntimeError("nonlearning selection global index drift")
        range_all = group["range_m"]
        valid_all = group["valid_mask"]
        unique_frames = len(range_all)
        width = np.empty(unique_frames, dtype=np.float32)
        height = np.empty(unique_frames, dtype=np.float32)
        curvature = np.empty(unique_frames, dtype=np.float32)
        branch_count = np.empty(unique_frames, dtype=np.int8)
        peak_heading = np.zeros(unique_frames, dtype=np.float32)
        peak_distance = np.zeros(unique_frames, dtype=np.float32)
        for frame in range(unique_frames):
            current_range = np.asarray(range_all[frame], dtype=np.float32)
            current_valid = np.asarray(valid_all[frame], dtype=np.uint8)
            geometry = geometry_method.predict(current_range, current_valid)
            exits = exit_method.predict(current_range, current_valid, ELEVATION_DEG)
            width[frame] = geometry["width_m"]
            height[frame] = geometry["height_m"]
            curvature[frame] = geometry["curvature_per_m"]
            branch_count[frame] = exits["branch_count"]
            if exits["sectors"]:
                sector = max(
                    exits["sectors"],
                    key=lambda value: (
                        float(value["peak_range_m"]), -float(value["heading_robot_deg"])
                    ),
                )
                peak_heading[frame] = sector["heading_robot_deg"]
                peak_distance[frame] = sector["peak_range_m"]
        for row, frame_references in enumerate(references):
            current = int(frame_references[-1])
            count = int(branch_count[current])
            width_before = float(np.median(width[frame_references[:2]]))
            width_after = float(np.median(width[frame_references[-2:]]))
            height_before = float(np.median(height[frame_references[:2]]))
            height_after = float(np.median(height[frame_references[-2:]]))
            if count >= 3:
                predicted_name = "junction"
            elif count <= 1:
                predicted_name = "terminal"
            elif (
                abs(width_after - width_before) >= GEOMETRY_TRANSITION_THRESHOLD_M
                or abs(height_after - height_before) >= GEOMETRY_TRANSITION_THRESHOLD_M
            ):
                predicted_name = "geometry_transition"
            elif float(curvature[current]) >= TURN_CURVATURE_THRESHOLD_PER_M:
                predicted_name = "turn"
            else:
                predicted_name = "corridor"
            destination = output_offset + row
            if predicted_name not in {"terminal", "junction"} or peak_distance[current] <= 0.0:
                continue
            heading = math.radians(float(peak_heading[current]))
            distance = float(peak_distance[current])
            confidence[destination, 0] = 1.0
            event_type[destination, 0] = 0 if predicted_name == "terminal" else 1
            relative[destination, 0] = (
                distance * math.cos(heading), distance * math.sin(heading), 0.0
            )
        output_offset = stop_offset
        print(json.dumps({"baseline_world": world_index, "parent_id": path.stem}), flush=True)
    if output_offset != len(selection_global):
        raise RuntimeError("nonlearning baseline population drift")
    return {"confidence": confidence, "event_type": event_type, "relative_xyz_m": relative}


def _plot(output: Path, main, exclusive, nonlearning):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    colors = ("#1864ab", "#2b8a3e", "#e67700")
    for seed, (summary, color) in enumerate(zip(main, colors, strict=True)):
        rows = summary["threshold_records"]
        axes[0].plot(
            [row["recall"] for row in rows], [row["precision"] for row in rows],
            marker="o", markersize=2.5, color=color, label=f"GSE seed {seed}",
        )
    for label, result, style in (
        ("exclusive single-center", exclusive, "--"),
        ("non-learning geometry", nonlearning, ":"),
    ):
        rows = result["threshold_records"]
        axes[0].plot(
            [row["recall"] for row in rows], [row["precision"] for row in rows],
            linestyle=style, linewidth=2.0, label=label,
        )
    axes[0].set_xlabel("event recall (same type, ≤4 m)")
    axes[0].set_ylabel("event precision")
    axes[0].set_xlim(left=0.0); axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)

    labels = ["exclusive", "non-learning", "GSE mean"]
    f1 = [
        exclusive["selected_threshold_metrics"]["f1"],
        nonlearning["selected_threshold_metrics"]["f1"],
        float(np.mean([row["selected_threshold_metrics"]["f1"] for row in main])),
    ]
    multi = [
        exclusive["selected_threshold_metrics"]["multi_event_recall"],
        nonlearning["selected_threshold_metrics"]["multi_event_recall"],
        float(np.mean([row["selected_threshold_metrics"]["multi_event_recall"] for row in main])),
    ]
    x = np.arange(3); width = 0.34
    axes[1].bar(x - width / 2, f1, width, label="set F1", color="#4c6ef5")
    axes[1].bar(x + width / 2, multi, width, label="multi-event recall", color="#20c997")
    axes[1].set_xticks(x, labels, rotation=12)
    axes[1].set_ylim(0.0, 1.0); axes[1].set_ylabel("score")
    axes[1].grid(axis="y", alpha=0.25); axes[1].legend(fontsize=8)
    fig.suptitle("Spatial structure event-set capacity on C07–C08")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_spatial_event_set_capacity.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--legacy-logits", action="append", required=True, type=Path)
    parser.add_argument("--model-dir", action="append", required=True, type=Path)
    parser.add_argument("--projection", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.legacy_logits) != 3 or len(args.model_dir) != 3 or len(args.projection) != 3:
        raise ValueError("capacity evaluation requires exactly three seed inputs")
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = load_spatial_event_teacher(args.teacher_root.resolve())
    rows = teacher.selection_rows
    targets = teacher.targets(rows)
    legacy_logits = [np.load(path, mmap_mode="r") for path in args.legacy_logits]
    if any(values.shape != (188_126, 5) or values.dtype != np.float16 for values in legacy_logits):
        raise RuntimeError("exclusive legacy event-logit archive drift")

    seed_summaries = []
    for seed, model_dir in enumerate(args.model_dir):
        summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("seed") != seed or summary.get("trainable_parameters") != 93_638:
            raise RuntimeError("spatial event seed summary drift")
        seed_summaries.append(summary)
    exclusive_predictions = _exclusive_baseline(
        legacy_logits, args.projection, rows, teacher.global_sequence_index
    )
    exclusive_best, exclusive_grid = select_fixed_grid_threshold(
        exclusive_predictions, targets, THRESHOLD_GRID
    )
    nonlearning_predictions = _nonlearning_baseline(
        args.dataset_run.resolve(), teacher.global_sequence_index[rows]
    )
    nonlearning_best, nonlearning_grid = select_fixed_grid_threshold(
        nonlearning_predictions, targets, THRESHOLD_GRID
    )
    exclusive = {"selected_threshold_metrics": exclusive_best, "threshold_records": exclusive_grid}
    nonlearning = {"selected_threshold_metrics": nonlearning_best, "threshold_records": nonlearning_grid}
    best_baseline_f1 = max(exclusive_best["f1"], nonlearning_best["f1"])
    best_baseline_multi = max(
        exclusive_best["multi_event_recall"], nonlearning_best["multi_event_recall"]
    )
    per_seed_pass = []
    for summary in seed_summaries:
        metrics = summary["selected_threshold_metrics"]
        per_seed_pass.append(
            metrics["precision"] >= 0.90
            and metrics["recall"] >= 0.25
            and metrics["per_type"]["terminal"]["recall"] >= 0.20
            and metrics["per_type"]["junction"]["recall"] >= 0.20
            and metrics["f1"] >= best_baseline_f1 + 0.05
            and metrics["multi_event_recall"] >= best_baseline_multi + 0.10
            and metrics["matched_localization_mae_m"] is not None
            and metrics["matched_localization_mae_m"] <= 4.0
        )
    scientific_pass = all(per_seed_pass)
    summary = {
        "schema_version": "gse_spatial_event_set_capacity_evaluation_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "seed_pass": per_seed_pass,
        "seed_metrics": [row["selected_threshold_metrics"] for row in seed_summaries],
        "exclusive_single_center_baseline": exclusive,
        "nonlearning_geometry_baseline": nonlearning,
        "acceptance": {
            "all_seeds_precision_at_least": 0.90,
            "all_seeds_recall_at_least": 0.25,
            "all_seeds_per_type_recall_at_least": 0.20,
            "all_seeds_f1_gain_over_best_baseline_at_least": 0.05,
            "all_seeds_multi_event_recall_gain_at_least": 0.10,
            "all_seeds_matched_localization_mae_at_most_m": 4.0,
        },
        "population": {
            "worlds": 20,
            "observations": len(rows),
            "target_tokens": int(targets["event_mask"].sum()),
            "multi_event_rows": int(np.sum(targets["event_mask"].sum(axis=1) >= 2)),
        },
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "baseline_selection_outputs.npz",
        global_sequence_index=teacher.global_sequence_index[rows],
        exclusive_confidence=exclusive_predictions["confidence"],
        exclusive_event_type=exclusive_predictions["event_type"],
        exclusive_relative_xyz_m=exclusive_predictions["relative_xyz_m"],
        nonlearning_confidence=nonlearning_predictions["confidence"],
        nonlearning_event_type=nonlearning_predictions["event_type"],
        nonlearning_relative_xyz_m=nonlearning_predictions["relative_xyz_m"],
    )
    figure_source = {
        "schema_version": "gse_spatial_event_set_capacity_figure_source_v1",
        "seed_summaries": seed_summaries,
        "exclusive": exclusive,
        "nonlearning": nonlearning,
    }
    (output / "figure_source.json").write_text(
        json.dumps(figure_source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(output, seed_summaries, exclusive, nonlearning)
    print(json.dumps(summary, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
