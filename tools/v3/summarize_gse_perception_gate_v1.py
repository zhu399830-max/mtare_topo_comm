#!/usr/bin/env python3
"""Combine sealed validation calibration and baseline results into one gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_perception_gate import summarize_gse_perception_gate
from mtare_topo.governance import load_json, write_json


def summarize(training_run: Path, calibration_dir: Path, exit_only_dir: Path, geometry_dir: Path, output: Path) -> dict:
    root = PROJECT_ROOT.resolve()
    paths = [path.resolve() for path in (training_run, calibration_dir, exit_only_dir, geometry_dir)]
    for path in paths:
        path.relative_to(root)
    training_run, calibration_dir, exit_only_dir, geometry_dir = paths
    training = load_json(training_run / "metrics/summary.json")
    calibration = load_json(calibration_dir / "summary.json")
    exit_only = load_json(exit_only_dir / "summary.json")
    geometry = load_json(geometry_dir / "summary.json")
    if (
        training.get("overall_status") != "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"
        or calibration.get("overall_status") != "PASS_GSE_VALIDATION_CALIBRATION_V1"
        or exit_only.get("status") != "PASS_GSE_EXIT_ONLY_BASELINE_VALIDATION_V1"
        or geometry.get("overall_status") != "PASS_GSE_NONLEARNING_GEOMETRY_VALIDATION_V1"
    ):
        raise RuntimeError("GSE perception gate sources are not completed PASS evidence")
    learned = [
        {
            "seed": int(seed["seed"]),
            "geometry_mae": seed["best_validation"]["geometry_mae"],
            "axis": seed["best_validation"]["axis"],
        }
        for seed in training["seeds"]
    ]
    calibrated = [
        {
            "seed": int(seed["seed"]),
            "event_macro_f1": float(seed["event_macro_f1"]),
            "place_association_precision": float(seed["place_association_precision"]),
            "place_false_accept_rate": float(seed["place_false_loop_merge_rate"]),
            "place_accepted": int(seed["place_association_accepted"]),
            "exit_association_precision": float(seed["exit_descriptor_association_precision"]),
            "exit_false_accept_rate": float(seed["exit_descriptor_false_accept_rate"]),
            "exit_accepted": int(seed["exit_descriptor_association_accepted"]),
        }
        for seed in calibration["seeds"]
    ]
    decision = summarize_gse_perception_gate(
        learned_seed_metrics=learned,
        calibrated_seed_metrics=calibrated,
        exit_only_seed_metrics=exit_only["per_seed"],
        nonlearning_geometry_mae=geometry["geometry_mae"],
    )
    result = {
        "schema_version": "gse_perception_gate_summary_v1",
        "overall_status": decision["scientific_status"],
        **decision,
        "learned_axis_diagnostic": {str(item["seed"]): item["axis"] for item in learned},
        "sources": {
            "training_run": str(training_run.relative_to(root)),
            "calibration_dir": str(calibration_dir.relative_to(root)),
            "exit_only_dir": str(exit_only_dir.relative_to(root)),
            "nonlearning_geometry_dir": str(geometry_dir.relative_to(root)),
        },
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "optimizer_steps": 0,
        "model_updates": 0,
    }
    write_json(output.resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--exit-only-dir", required=True, type=Path)
    parser.add_argument("--geometry-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.training_run, args.calibration_dir, args.exit_only_dir, args.geometry_dir, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
