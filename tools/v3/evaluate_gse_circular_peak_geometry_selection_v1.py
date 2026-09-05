#!/usr/bin/env python3
"""Select one C07 ensemble peak threshold and transfer it once to C08."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from mtare_topo.evaluation.gse_circular_peak_metrics import (
    action_macro_f1,
    apply_peak_threshold,
    ranked_peak_metrics,
    select_safe_threshold,
)
from mtare_topo.governance import write_json


PASS = "PASS_GSE_CIRCULAR_PEAK_GEOMETRY_SELECTION_V1"
FAIL = "FAIL_GSE_CIRCULAR_PEAK_GEOMETRY_SELECTION_V1"
PRECISION_FLOOR = 0.995
RAW_RANGE_AP = 0.05149281407549527


def _load_split(suffix: str, teacher_root: Path, source_root: Path, prediction_roots: list[Path]) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {}
    for teacher_path in sorted(teacher_root.glob(f"selection/*_{suffix}.zarr")):
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = str(teacher.attrs["parent_id"])
        source = zarr.open_group(str(source_root / f"{parent}.zarr"), mode="r")
        predictions = [np.load(root / f"{parent}.npz") for root in prediction_roots]
        global_sequence = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
        if not np.array_equal(global_sequence, np.asarray(source["global_sequence_index"][:], dtype=np.int64)) or any(not np.array_equal(global_sequence, item["global_sequence_index"]) for item in predictions):
            raise RuntimeError(f"circular selection join drift: {parent}")
        values = {
            "global_sequence_index": global_sequence,
            "parent_id": np.full(len(global_sequence), parent, dtype=f"U{len(parent)}"),
            "presence": np.asarray(teacher["presence"][:], dtype=np.uint8),
            "heading_residual_target": np.asarray(teacher["heading_residual_deg"][:], dtype=np.float32),
            "width_target": np.asarray(teacher["opening_width_m"][:], dtype=np.float32),
            "width_valid": np.asarray(teacher["width_valid_mask"][:], dtype=np.uint8),
            "profile_target": np.asarray(teacher["vertical_profile_m"][:], dtype=np.float32),
            "axis_target": np.asarray(source["local_axis_robot"][:], dtype=np.float32),
            "geometry_target": np.asarray(source["geometry"][:], dtype=np.float32),
            "geometry_valid": np.asarray(source["geometry_valid_mask"][:], dtype=np.uint8),
            "confidence": np.mean([np.asarray(item["confidence"], dtype=np.float32) for item in predictions], axis=0),
            "heading_residual": np.mean([np.asarray(item["heading_residual_deg"], dtype=np.float32) for item in predictions], axis=0),
            "width": np.mean([np.asarray(item["opening_width_m"], dtype=np.float32) for item in predictions], axis=0),
            "profile": np.mean([np.asarray(item["vertical_profile_m"], dtype=np.float32) for item in predictions], axis=0),
            "axis": np.mean([np.asarray(item["local_axis"], dtype=np.float32) for item in predictions], axis=0),
            "geometry": np.mean([np.asarray(item["geometry"], dtype=np.float32) for item in predictions], axis=0),
        }
        for seed, item in enumerate(predictions):
            values[f"seed{seed}_confidence"] = np.asarray(item["confidence"], dtype=np.float32)
        for name, value in values.items():
            output.setdefault(name, []).append(value)
    if len(output.get("global_sequence_index", [])) != 10:
        raise RuntimeError(f"circular selection {suffix} world count drift")
    return {name: np.concatenate(parts) for name, parts in output.items()}


def _evaluate(data: dict[str, np.ndarray], threshold: float) -> dict:
    selected, detection = apply_peak_threshold(data["confidence"], data["presence"], threshold)
    truth = data["presence"].astype(bool)
    matched = selected & truth
    width_mask = matched & data["width_valid"].astype(bool)
    heading_mae = float(np.mean(np.abs(data["heading_residual"][matched] - data["heading_residual_target"][matched]))) if matched.any() else math.inf
    width_mae = float(np.mean(np.abs(data["width"][width_mask] - data["width_target"][width_mask]))) if width_mask.any() else math.inf
    profile_mae = float(np.mean(np.abs(data["profile"][matched] - data["profile_target"][matched]))) if matched.any() else math.inf
    target_count = truth.sum(axis=1)
    predicted_count = selected.sum(axis=1)
    action = action_macro_f1(predicted_count, target_count)
    exact_set = float(np.mean(np.all(selected == truth, axis=1)))
    predicted_axis = data["axis"] / np.clip(np.linalg.norm(data["axis"], axis=1, keepdims=True), 1e-12, None)
    target_axis = data["axis_target"] / np.clip(np.linalg.norm(data["axis_target"], axis=1, keepdims=True), 1e-12, None)
    axis_error = float(np.degrees(np.arccos(np.clip(np.sum(predicted_axis * target_axis, axis=1), -1.0, 1.0))).mean())
    global_mae = []
    for index in range(4):
        valid = data["geometry_valid"][:, index].astype(bool)
        global_mae.append(float(np.mean(np.abs(data["geometry"][valid, index] - data["geometry_target"][valid, index]))))
    per_world = []
    for parent in np.unique(data["parent_id"]):
        rows = data["parent_id"] == parent
        _, metrics = apply_peak_threshold(data["confidence"][rows], data["presence"][rows], threshold)
        per_world.append({"parent_id": str(parent), **metrics})
    return {
        "detection": detection,
        "average_precision": ranked_peak_metrics(data["confidence"], data["presence"])["average_precision"],
        "action": action,
        "exact_peak_set_fraction": exact_set,
        "peak_geometry": {"heading_mae_deg": heading_mae, "width_mae_m": width_mae, "vertical_profile_mae_m": profile_mae, "matched_peaks": int(matched.sum()), "matched_width_peaks": int(width_mask.sum())},
        "global_geometry": {"axis_mean_error_deg": axis_error, "width_mae_m": global_mae[0], "height_mae_m": global_mae[1], "slope_mae_deg": global_mae[2], "curvature_mae_per_m": global_mae[3]},
        "per_world": per_world,
    }


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    splits = ("c07", "c08")
    x = np.arange(2)
    axes[0].bar(x - .18, [RAW_RANGE_AP, RAW_RANGE_AP], .36, label="raw range", color="#bab0ac")
    axes[0].bar(x + .18, [summary[name]["average_precision"] for name in splits], .36, label="GSE peak field", color="#4e79a7")
    axes[0].set_xticks(x, ("C07", "C08"))
    axes[0].set_ylabel("Peak average precision")
    axes[0].set_title("A  Learned executable peaks")
    axes[0].legend(frameon=False)
    axes[1].bar(x, [summary[name]["detection"]["recall"] for name in splits], color=["#59a14f", "#f28e2b"])
    axes[1].axhline(.5, color="#e15759", linestyle="--", label="required")
    axes[1].set_xticks(x, ("C07", "C08"))
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Recall at fixed high-precision threshold")
    axes[1].set_title("B  Safe peak recovery")
    axes[1].legend(frameon=False)
    width = .22
    names = ("heading_mae_deg", "width_mae_m", "vertical_profile_mae_m")
    labels = ("heading (deg)", "opening width (m)", "vertical profile (m)")
    for index, split in enumerate(splits):
        axes[2].bar(np.arange(3) + (index - .5) * width, [summary[split]["peak_geometry"][name] for name in names], width, label=split.upper())
    axes[2].set_xticks(np.arange(3), labels, rotation=15)
    axes[2].set_ylabel("Matched-peak MAE")
    axes[2].set_title("C  Executable geometry")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph circular peak geometry selection and transfer")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_circular_peak_geometry_selection_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--prediction-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if len(args.prediction_root) != 3:
        raise RuntimeError("circular selection requires three seed prediction roots")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    c07 = _load_split("C07", args.teacher_root.resolve(), args.source_root.resolve(), args.prediction_root)
    c08 = _load_split("C08", args.teacher_root.resolve(), args.source_root.resolve(), args.prediction_root)
    chosen = select_safe_threshold(c07["confidence"], c07["presence"], precision_floor=PRECISION_FLOOR)
    threshold = 2.0 if chosen is None else float(chosen["threshold"])
    c07_metrics = _evaluate(c07, threshold)
    c08_metrics = _evaluate(c08, threshold)
    seed_diagnostics = []
    for seed in range(3):
        seed_choice = select_safe_threshold(c07[f"seed{seed}_confidence"], c07["presence"], precision_floor=PRECISION_FLOOR)
        seed_threshold = 2.0 if seed_choice is None else float(seed_choice["threshold"])
        _, c07_detection = apply_peak_threshold(c07[f"seed{seed}_confidence"], c07["presence"], seed_threshold)
        _, c08_detection = apply_peak_threshold(c08[f"seed{seed}_confidence"], c08["presence"], seed_threshold)
        seed_diagnostics.append({"seed": seed, "c07_selected": seed_choice, "c07_detection": c07_detection, "c08_transfer_detection": c08_detection, "c07_ap": ranked_peak_metrics(c07[f"seed{seed}_confidence"], c07["presence"])["average_precision"], "c08_ap": ranked_peak_metrics(c08[f"seed{seed}_confidence"], c08["presence"])["average_precision"]})
    checks = {
        "c07_safe_precision_and_recall": c07_metrics["detection"]["precision"] >= .995 and c07_metrics["detection"]["recall"] >= .50,
        "c08_safe_precision_and_recall": c08_metrics["detection"]["precision"] >= .995 and c08_metrics["detection"]["recall"] >= .50,
        "ap_improves_raw_range_by_0p10": c07_metrics["average_precision"] >= RAW_RANGE_AP + .10 and c08_metrics["average_precision"] >= RAW_RANGE_AP + .10,
        "action_macro_f1_at_least_0p80": c07_metrics["action"]["macro_f1"] >= .80 and c08_metrics["action"]["macro_f1"] >= .80,
        "peak_geometry_contract": all(metrics["peak_geometry"]["heading_mae_deg"] <= 1.0 and metrics["peak_geometry"]["width_mae_m"] <= 3.0 and metrics["peak_geometry"]["vertical_profile_mae_m"] <= 1.0 for metrics in (c07_metrics, c08_metrics)),
        "global_geometry_contract": all(metrics["global_geometry"]["axis_mean_error_deg"] <= 10.0 and metrics["global_geometry"]["width_mae_m"] <= 2.0 and metrics["global_geometry"]["height_mae_m"] <= 2.0 and metrics["global_geometry"]["slope_mae_deg"] <= 2.0 and metrics["global_geometry"]["curvature_mae_per_m"] <= .02 for metrics in (c07_metrics, c08_metrics)),
        "c08_not_used_for_selection": True,
        "zero_test_graph_planner": True,
    }
    scientific_pass = chosen is not None and all(checks.values())
    summary = {
        "schema_version": "gse_circular_peak_geometry_selection_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_CIRCULAR_PEAK_DESCRIPTOR_ASSOCIATION_STAGE" if scientific_pass else "STOP_CIRCULAR_PEAK_GEOMETRY_BEFORE_GRAPH",
        "selected_on": "C07 only", "transferred_once_to": "C08 with zero adaptation",
        "precision_floor": PRECISION_FLOOR, "ensemble_threshold": None if chosen is None else threshold,
        "c07_threshold_selection": chosen,
        "population": {"c07_observations": len(c07["presence"]), "c07_peaks": int(c07["presence"].sum()), "c08_observations": len(c08["presence"]), "c08_peaks": int(c08["presence"].sum())},
        "c07": c07_metrics, "c08": c08_metrics, "seed_diagnostics": seed_diagnostics,
        "checks": checks,
        "optimizer_steps": 0, "c08_checkpoint_observations": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "graph_replays": 0, "planner_calls": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", {"schema_version": "gse_circular_peak_geometry_selection_figure_source_v1", "summary": summary})
    with (output / "per_world_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "parent_id", "precision", "recall", "true_positive", "false_positive", "false_negative", "selected_peaks"))
        writer.writeheader()
        for split, metrics in (("C07", c07_metrics), ("C08", c08_metrics)):
            for row in metrics["per_world"]:
                writer.writerow({"split": split, **row})
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "threshold": summary["ensemble_threshold"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
