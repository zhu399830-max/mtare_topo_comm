#!/usr/bin/env python3
"""Evaluate joint geometry-anchored seeds against all frozen V2 baselines."""
from __future__ import annotations

import argparse
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
from mtare_topo.evaluation.gse_spatial_event_set_metrics import select_fixed_grid_threshold


PASS = "PASS_GSE_GEOMETRY_ANCHORED_JOINT_CAPACITY_V1"
FAIL = "FAIL_GSE_GEOMETRY_ANCHORED_JOINT_CAPACITY_V1"
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


def _evaluate(predictions, targets):
    best, records = select_fixed_grid_threshold(predictions, targets, THRESHOLD_GRID, maximum_error_m=4.0)
    best = dict(best)
    best["macro_f1"] = _macro_f1(best)
    return {"selected_threshold_metrics": best, "threshold_records": records}


def _plot(output: Path, joint, exclusive, frozen, nonlearning) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), constrained_layout=True)
    colors = ("#1864ab", "#2b8a3e", "#e67700")
    for seed, (result, color) in enumerate(zip(joint, colors, strict=True)):
        records = result["threshold_records"]
        axes[0].plot([row["recall"] for row in records], [row["precision"] for row in records], marker="o", markersize=2.2, color=color, label=f"joint seed {seed}")
    for label, result, style in (("exclusive center", exclusive, "--"), ("frozen encoder set (best)", max(frozen, key=lambda item: item["selected_threshold_metrics"]["f1"]), "-."), ("nonlearning geometry", nonlearning, ":")):
        records = result["threshold_records"]
        axes[0].plot([row["recall"] for row in records], [row["precision"] for row in records], linestyle=style, linewidth=2.0, label=label)
    axes[0].set_xlabel("event recall (same type, ≤4 m)")
    axes[0].set_ylabel("event precision")
    axes[0].set_xlim(left=0.0)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    labels = ["exclusive", "frozen set", "nonlearning", "joint mean"]
    frozen_best = max(frozen, key=lambda item: item["selected_threshold_metrics"]["f1"])["selected_threshold_metrics"]
    sources = [exclusive["selected_threshold_metrics"], frozen_best, nonlearning["selected_threshold_metrics"]]
    f1 = [item["f1"] for item in sources] + [float(np.mean([item["selected_threshold_metrics"]["f1"] for item in joint]))]
    macro = [item["macro_f1"] for item in sources] + [float(np.mean([item["selected_threshold_metrics"]["macro_f1"] for item in joint]))]
    multi = [item["multi_event_recall"] for item in sources] + [float(np.mean([item["selected_threshold_metrics"]["multi_event_recall"] for item in joint]))]
    x = np.arange(len(labels)); width = 0.25
    axes[1].bar(x - width, f1, width, label="set F1", color="#4c6ef5")
    axes[1].bar(x, macro, width, label="macro F1", color="#ae3ec9")
    axes[1].bar(x + width, multi, width, label="multi recall", color="#20c997")
    axes[1].set_xticks(x, labels, rotation=12)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("score")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7)

    seed_metrics = [item["selected_threshold_metrics"] for item in joint]
    axes[2].bar([f"seed {index}" for index in range(3)], [item["matched_localization_mae_m"] or 0.0 for item in seed_metrics], color=colors)
    axes[2].axhline(4.0, color="#c92a2a", linestyle="--", linewidth=1.5, label="4 m gate")
    axes[2].set_ylabel("matched position MAE (m)")
    axes[2].set_title("Joint geometry localization")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.suptitle("Geometry-anchored spatial event capacity on C07–C08")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_geometry_anchored_joint_capacity_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--joint-model-dir", action="append", required=True, type=Path)
    parser.add_argument("--frozen-set-output", action="append", required=True, type=Path)
    parser.add_argument("--baseline-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.joint_model_dir) != 3 or len(args.frozen_set_output) != 3:
        raise ValueError("joint evaluation requires exactly three joint and frozen-set seeds")
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = load_observable_spatial_event_teacher(args.teacher_root.resolve())
    rows = teacher.selection_rows
    expected_global = teacher.global_sequence_index[rows]
    targets = teacher.targets(rows)

    joint = []
    for seed, model_dir in enumerate(args.joint_model_dir):
        summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("seed") != seed or summary.get("trainable_parameters") != 264134 or summary.get("smoke_limited"):
            raise RuntimeError("joint seed summary drift")
        predictions = _prediction_archive(model_dir / "selection_outputs.npz", expected_global)
        result = _evaluate(predictions, targets)
        if result["selected_threshold_metrics"] != summary["selected_threshold_metrics"]:
            raise RuntimeError(f"joint seed metric replay drift: {seed}")
        joint.append(result)

    frozen = [_evaluate(_prediction_archive(path, expected_global), targets) for path in args.frozen_set_output]
    with np.load(args.baseline_output.resolve(), allow_pickle=False) as baseline:
        if not np.array_equal(baseline["global_sequence_index"], expected_global):
            raise RuntimeError("baseline global identity drift")
        exclusive = _evaluate({"confidence": baseline["exclusive_confidence"], "event_type": baseline["exclusive_event_type"], "relative_xyz_m": baseline["exclusive_relative_xyz_m"]}, targets)
        nonlearning = _evaluate({"confidence": baseline["nonlearning_confidence"], "event_type": baseline["nonlearning_event_type"], "relative_xyz_m": baseline["nonlearning_relative_xyz_m"]}, targets)

    baseline_metrics = [
        exclusive["selected_threshold_metrics"],
        nonlearning["selected_threshold_metrics"],
        *(result["selected_threshold_metrics"] for result in frozen),
    ]
    best_baseline_f1 = max(item["f1"] for item in baseline_metrics)
    best_baseline_macro = max(item["macro_f1"] for item in baseline_metrics)
    best_baseline_multi = max(item["multi_event_recall"] for item in baseline_metrics)
    seed_pass = []
    reasons = []
    for result in joint:
        metrics = result["selected_threshold_metrics"]
        checks = {
            "precision": metrics["precision"] >= 0.90,
            "recall": metrics["recall"] >= 0.25,
            "terminal_recall": metrics["per_type"]["terminal"]["recall"] >= 0.20,
            "junction_recall": metrics["per_type"]["junction"]["recall"] >= 0.20,
            "f1_gain": metrics["f1"] >= best_baseline_f1 + 0.05,
            "macro_f1_gain": metrics["macro_f1"] >= best_baseline_macro + 0.05,
            "multi_event_recall_gain": metrics["multi_event_recall"] >= best_baseline_multi + 0.10,
            "localization_mae": metrics["matched_localization_mae_m"] is not None and metrics["matched_localization_mae_m"] <= 4.0,
        }
        seed_pass.append(all(checks.values()))
        reasons.append(checks)
    scientific_pass = all(seed_pass)
    summary = {
        "schema_version": "gse_geometry_anchored_joint_capacity_evaluation_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "seed_pass": seed_pass,
        "seed_checks": reasons,
        "seed_metrics": [result["selected_threshold_metrics"] for result in joint],
        "exclusive_single_center_baseline": exclusive,
        "frozen_encoder_set_baselines": frozen,
        "nonlearning_geometry_baseline": nonlearning,
        "best_recomputed_baseline": {
            "f1": best_baseline_f1,
            "macro_f1": best_baseline_macro,
            "multi_event_recall": best_baseline_multi,
            "includes": [
                "exclusive_single_center",
                "nonlearning_geometry",
                "frozen_encoder_set_seed0",
                "frozen_encoder_set_seed1",
                "frozen_encoder_set_seed2",
            ],
        },
        "acceptance": {"precision_min": 0.90, "recall_min": 0.25, "per_type_recall_min": 0.20, "f1_gain_min": 0.05, "macro_f1_gain_min": 0.05, "multi_event_recall_gain_min": 0.10, "matched_localization_mae_max_m": 4.0},
        "population": {"worlds": 20, "observations": len(rows), "target_tokens": int(targets["event_mask"].sum()), "multi_event_rows": int(np.sum(targets["event_mask"].sum(axis=1) >= 2))},
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    figure_source = {"schema_version": "gse_geometry_anchored_joint_capacity_figure_source_v1", "summary": summary}
    write = lambda path, value: path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write(output / "summary.json", summary)
    write(output / "figure_source.json", figure_source)
    _plot(output, joint, exclusive, frozen, nonlearning)
    print(json.dumps(summary, indent=2))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
