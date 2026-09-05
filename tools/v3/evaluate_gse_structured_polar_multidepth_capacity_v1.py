#!/usr/bin/env python3
"""Evaluate three structured-polar seeds on the frozen C07-C08 population."""
from __future__ import annotations

import argparse
import csv
import json
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
from mtare_topo.evaluation.gse_spatial_event_set_metrics import (
    _maximum_valid_matching,
    select_fixed_grid_threshold,
)


PASS = "PASS_GSE_STRUCTURED_POLAR_MULTIDEPTH_CAPACITY_V1"
FAIL = "FAIL_GSE_STRUCTURED_POLAR_MULTIDEPTH_CAPACITY_V1"
THRESHOLD_GRID = tuple(round(value * 0.05, 2) for value in range(1, 20))


def _prediction_archive(path: Path, expected_global: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], expected_global):
            raise RuntimeError(f"selection output global identity drift: {path}")
        return {
            "confidence": np.asarray(archive["confidence"], dtype=np.float32),
            "event_type": np.asarray(archive["event_type"], dtype=np.int8),
            "relative_xyz_m": np.asarray(archive["relative_xyz_m"], dtype=np.float32),
        }


def _macro_f1(metrics: dict[str, object]) -> float:
    per_type = metrics["per_type"]
    return float((per_type["terminal"]["f1"] + per_type["junction"]["f1"]) / 2.0)


def _second_depth_target_mask(targets: dict[str, np.ndarray]) -> np.ndarray:
    """Mark the farther target when two targets occupy one frozen azimuth bin."""

    mask = np.asarray(targets["event_mask"], dtype=bool)
    xyz = np.asarray(targets["event_relative_xyz_m"], dtype=np.float64)
    identity = np.asarray(targets["event_identity_index"], dtype=np.int64)
    if mask.ndim != 2 or mask.shape[1] != 16 or xyz.shape != (*mask.shape, 3):
        raise ValueError("second-depth target shape drift")
    radial = np.linalg.norm(xyz, axis=-1)
    bearing_deg = np.degrees(np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]), 2.0 * np.pi))
    bins = np.mod(np.floor((bearing_deg + 0.25) / 2.0).astype(np.int64), 180)
    same = mask[:, :, None] & mask[:, None, :] & (bins[:, :, None] == bins[:, None, :])
    earlier = (radial[:, None, :] < radial[:, :, None]) | (
        (radial[:, None, :] == radial[:, :, None])
        & (identity[:, None, :] < identity[:, :, None])
    )
    slots = np.sum(same & earlier, axis=2)
    if np.any(slots[mask] >= 2):
        raise ValueError("Teacher exceeds the frozen two-depth layout")
    return mask & (slots == 1)


def _second_depth_recall(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    *,
    threshold: float,
    maximum_error_m: float = 4.0,
) -> dict[str, object]:
    slot1 = _second_depth_target_mask(targets)
    target_count = int(slot1.sum())
    true_positive = 0
    confidence = predictions["confidence"]
    predicted_type = predictions["event_type"]
    predicted_xyz = predictions["relative_xyz_m"]
    target_mask = np.asarray(targets["event_mask"], dtype=bool)
    target_type = np.asarray(targets["event_type_index"], dtype=np.int64)
    target_xyz = np.asarray(targets["event_relative_xyz_m"], dtype=np.float64)
    for row in range(len(confidence)):
        selected = np.flatnonzero(confidence[row] >= float(threshold))
        selected = np.asarray(
            sorted(
                selected.tolist(),
                key=lambda index: (
                    int(predicted_type[row, index]),
                    *tuple(float(value) for value in predicted_xyz[row, index]),
                    -float(confidence[row, index]),
                ),
            ),
            dtype=np.int64,
        )
        active = np.flatnonzero(target_mask[row])
        matches = _maximum_valid_matching(
            predicted_type[row, selected],
            predicted_xyz[row, selected],
            target_type[row, active],
            target_xyz[row, active],
            maximum_error_m=maximum_error_m,
        )
        for _, target_local, _ in matches:
            true_positive += int(slot1[row, active[target_local]])
    return {
        "true_positive": true_positive,
        "target": target_count,
        "recall": float(true_positive / target_count) if target_count else 0.0,
    }


def _evaluate(predictions, targets):
    best, records = select_fixed_grid_threshold(
        predictions, targets, THRESHOLD_GRID, maximum_error_m=4.0
    )
    for record in records:
        record["macro_f1"] = _macro_f1(record)
    best["second_depth"] = _second_depth_recall(
        predictions, targets, threshold=float(best["threshold"])
    )
    best["second_depth_recall"] = best["second_depth"]["recall"]
    return {"selected_threshold_metrics": best, "threshold_records": records}


def _core_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in ("second_depth", "second_depth_recall")
    }


def _plot(output: Path, structured, baselines) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 4.3), constrained_layout=True)
    colors = ("#1864ab", "#2b8a3e", "#e67700")
    for seed, (result, color) in enumerate(zip(structured, colors, strict=True)):
        records = result["threshold_records"]
        axes[0].plot(
            [row["recall"] for row in records],
            [row["precision"] for row in records],
            marker="o",
            markersize=2.2,
            color=color,
            label=f"structured seed {seed}",
        )
    for label, result, style in (
        ("exclusive center", baselines["exclusive"], "--"),
        ("free-query best", max(baselines["free_query"], key=lambda item: item["selected_threshold_metrics"]["f1"]), "-."),
        ("frozen-set best", max(baselines["frozen_set"], key=lambda item: item["selected_threshold_metrics"]["f1"]), ":"),
        ("nonlearning", baselines["nonlearning"], (0, (3, 1, 1, 1))),
    ):
        records = result["threshold_records"]
        axes[0].plot(
            [row["recall"] for row in records],
            [row["precision"] for row in records],
            linestyle=style,
            linewidth=1.8,
            label=label,
        )
    axes[0].set_xlabel("event recall (same type, ≤4 m)")
    axes[0].set_ylabel("event precision")
    axes[0].set_xlim(left=0.0)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=6.5)

    baseline_selected = [
        baselines["exclusive"]["selected_threshold_metrics"],
        max(baselines["free_query"], key=lambda item: item["selected_threshold_metrics"]["f1"])["selected_threshold_metrics"],
        max(baselines["frozen_set"], key=lambda item: item["selected_threshold_metrics"]["f1"])["selected_threshold_metrics"],
        baselines["nonlearning"]["selected_threshold_metrics"],
    ]
    structured_selected = [item["selected_threshold_metrics"] for item in structured]
    labels = ["exclusive", "free query", "frozen set", "nonlearn", "structured"]
    metric_sources = baseline_selected + [
        {
            key: float(np.mean([item[key] for item in structured_selected]))
            for key in ("f1", "macro_f1", "multi_event_recall", "second_depth_recall")
        }
    ]
    x = np.arange(len(labels))
    width = 0.19
    for offset, (name, color) in enumerate(
        (
            ("F1", "#4c6ef5"),
            ("macro F1", "#ae3ec9"),
            ("multi recall", "#20c997"),
            ("second-depth recall", "#f08c00"),
        )
    ):
        key = ("f1", "macro_f1", "multi_event_recall", "second_depth_recall")[offset]
        axes[1].bar(
            x + (offset - 1.5) * width,
            [item[key] for item in metric_sources],
            width,
            label=name,
            color=color,
        )
    axes[1].set_xticks(x, labels, rotation=13)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("score")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=6.5)

    axes[2].bar(
        [f"seed {index}" for index in range(3)],
        [item["matched_localization_mae_m"] or 0.0 for item in structured_selected],
        color=colors,
    )
    axes[2].axhline(4.0, color="#c92a2a", linestyle="--", linewidth=1.5, label="4 m gate")
    axes[2].set_ylabel("matched position MAE (m)")
    axes[2].set_title("Structured localization")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)

    best_second = max(item["second_depth_recall"] for item in baseline_selected)
    axes[3].bar(
        [f"seed {index}" for index in range(3)],
        [item["second_depth_recall"] for item in structured_selected],
        color=colors,
    )
    axes[3].axhline(
        best_second + 0.10,
        color="#c92a2a",
        linestyle="--",
        linewidth=1.5,
        label="best baseline + 0.10",
    )
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_ylabel("farther same-bin target recall")
    axes[3].set_title("Two-depth capability (79 targets)")
    axes[3].grid(axis="y", alpha=0.25)
    axes[3].legend(fontsize=7)
    figure.suptitle("Structured polar multi-depth event capacity on C07–C08")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"gse_structured_polar_multidepth_capacity_v1.{suffix}", dpi=220
        )
    plt.close(figure)


def _write_tables(output: Path, records: list[dict[str, object]]) -> None:
    fields = (
        "method",
        "seed",
        "threshold",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "multi_event_recall",
        "second_depth_recall",
        "matched_localization_mae_m",
    )
    with (output / "capacity_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: record.get(name) for name in fields} for record in records)
    lines = [
        "| Method | Seed | Precision | Recall | F1 | Macro-F1 | Multi recall | Second-depth recall | MAE (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['macro_f1']:.4f} | {row['multi_event_recall']:.4f} | "
            f"{row['second_depth_recall']:.4f} | {row['matched_localization_mae_m'] if row['matched_localization_mae_m'] is not None else 'NA'} |"
        )
    (output / "capacity_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--structured-model-dir", action="append", required=True, type=Path)
    parser.add_argument("--free-query-model-dir", action="append", required=True, type=Path)
    parser.add_argument("--frozen-set-output", action="append", required=True, type=Path)
    parser.add_argument("--baseline-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if any(
        len(values) != 3
        for values in (
            args.structured_model_dir,
            args.free_query_model_dir,
            args.frozen_set_output,
        )
    ):
        raise ValueError("capacity evaluation requires exactly three seeds per learned method")
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = load_observable_spatial_event_teacher(args.teacher_root.resolve())
    rows = teacher.selection_rows
    expected_global = teacher.global_sequence_index[rows]
    targets = teacher.targets(rows)
    second_depth_targets = int(_second_depth_target_mask(targets).sum())
    if second_depth_targets != 79:
        raise RuntimeError(f"selection second-depth population drift: {second_depth_targets}")

    structured = []
    for seed, model_dir in enumerate(args.structured_model_dir):
        summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
        if (
            summary.get("seed") != seed
            or summary.get("trainable_parameters") != 172430
            or summary.get("smoke_limited")
        ):
            raise RuntimeError("structured-polar seed summary drift")
        predictions = _prediction_archive(model_dir / "selection_outputs.npz", expected_global)
        result = _evaluate(predictions, targets)
        if _core_metrics(result["selected_threshold_metrics"]) != summary["selected_threshold_metrics"]:
            raise RuntimeError(f"structured-polar seed metric replay drift: {seed}")
        structured.append(result)

    free_query = [
        _evaluate(_prediction_archive(path / "selection_outputs.npz", expected_global), targets)
        for path in args.free_query_model_dir
    ]
    frozen_set = [
        _evaluate(_prediction_archive(path, expected_global), targets)
        for path in args.frozen_set_output
    ]
    with np.load(args.baseline_output.resolve(), allow_pickle=False) as baseline:
        if not np.array_equal(baseline["global_sequence_index"], expected_global):
            raise RuntimeError("baseline global identity drift")
        exclusive = _evaluate(
            {
                "confidence": baseline["exclusive_confidence"],
                "event_type": baseline["exclusive_event_type"],
                "relative_xyz_m": baseline["exclusive_relative_xyz_m"],
            },
            targets,
        )
        nonlearning = _evaluate(
            {
                "confidence": baseline["nonlearning_confidence"],
                "event_type": baseline["nonlearning_event_type"],
                "relative_xyz_m": baseline["nonlearning_relative_xyz_m"],
            },
            targets,
        )
    baselines = {
        "exclusive": exclusive,
        "free_query": free_query,
        "frozen_set": frozen_set,
        "nonlearning": nonlearning,
    }
    named_baselines = [
        ("exclusive_single_center", None, exclusive),
        ("nonlearning_geometry", None, nonlearning),
        *[("free_query_joint", seed, result) for seed, result in enumerate(free_query)],
        *[("frozen_encoder_set", seed, result) for seed, result in enumerate(frozen_set)],
    ]
    baseline_metrics = [item[2]["selected_threshold_metrics"] for item in named_baselines]
    best_baseline = {
        key: max(float(item[key]) for item in baseline_metrics)
        for key in ("f1", "macro_f1", "multi_event_recall", "second_depth_recall")
    }
    seed_pass = []
    checks = []
    for result in structured:
        metrics = result["selected_threshold_metrics"]
        record = {
            "precision": metrics["precision"] >= 0.90,
            "recall": metrics["recall"] >= 0.25,
            "terminal_recall": metrics["per_type"]["terminal"]["recall"] >= 0.20,
            "junction_recall": metrics["per_type"]["junction"]["recall"] >= 0.20,
            "f1_gain": metrics["f1"] >= best_baseline["f1"] + 0.05,
            "macro_f1_gain": metrics["macro_f1"] >= best_baseline["macro_f1"] + 0.05,
            "multi_event_recall_gain": metrics["multi_event_recall"] >= best_baseline["multi_event_recall"] + 0.10,
            "second_depth_recall_gain": metrics["second_depth_recall"] >= best_baseline["second_depth_recall"] + 0.10,
            "localization_mae": metrics["matched_localization_mae_m"] is not None and metrics["matched_localization_mae_m"] <= 4.0,
        }
        checks.append(record)
        seed_pass.append(all(record.values()))
    scientific_pass = all(seed_pass)
    summary = {
        "schema_version": "gse_structured_polar_multidepth_capacity_evaluation_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "seed_pass": seed_pass,
        "seed_checks": checks,
        "seed_metrics": [item["selected_threshold_metrics"] for item in structured],
        "baselines": {
            "exclusive_single_center": exclusive,
            "free_query_joint": free_query,
            "frozen_encoder_set": frozen_set,
            "nonlearning_geometry": nonlearning,
        },
        "best_recomputed_baseline": {
            **best_baseline,
            "includes": [
                "exclusive_single_center",
                "nonlearning_geometry",
                "free_query_joint_seed0",
                "free_query_joint_seed1",
                "free_query_joint_seed2",
                "frozen_encoder_set_seed0",
                "frozen_encoder_set_seed1",
                "frozen_encoder_set_seed2",
            ],
        },
        "acceptance": {
            "precision_min": 0.90,
            "recall_min": 0.25,
            "per_type_recall_min": 0.20,
            "f1_gain_min": 0.05,
            "macro_f1_gain_min": 0.05,
            "multi_event_recall_gain_min": 0.10,
            "second_depth_recall_gain_min": 0.10,
            "matched_localization_mae_max_m": 4.0,
        },
        "population": {
            "worlds": 20,
            "observations": len(rows),
            "target_tokens": int(targets["event_mask"].sum()),
            "multi_event_rows": int(np.sum(targets["event_mask"].sum(axis=1) >= 2)),
            "second_depth_targets": second_depth_targets,
        },
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    table_records = []
    for method, seed, result in named_baselines + [
        ("structured_polar_multidepth", seed, result)
        for seed, result in enumerate(structured)
    ]:
        table_records.append(
            {
                "method": method,
                "seed": "" if seed is None else seed,
                **result["selected_threshold_metrics"],
            }
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "figure_source.json").write_text(
        json.dumps(
            {
                "schema_version": "gse_structured_polar_multidepth_capacity_figure_source_v1",
                "summary": summary,
                "table": table_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tables(output, table_records)
    _plot(output, structured, baselines)
    print(json.dumps(summary, indent=2))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
