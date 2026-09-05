#!/usr/bin/env python3
"""C07 selection and one C08 transfer for COUST."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

import evaluate_gse_cardinality_conditioned_circular_slot_transport_selection_v1 as base
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import BRANCH_SLICE


PASS = "PASS_GSE_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_SELECTION_V1"
FAIL = "FAIL_GSE_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_SELECTION_V1"
BASELINE = {
    "c07": {"overall_exact2": 0.3412845739743828, "count3_exact2": 0.013004791238877482, "count4_exact2": 0.0, "safe_recall": 0.00021976090014064697},
    "c08": {"overall_exact2": 0.27777322292366974, "count3_exact2": 0.007140731925022315, "count4_exact2": 0.0, "safe_recall": 0.00015522828259308845},
}


def _align_cyclic_order(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if len(reference) != len(candidate):
        raise ValueError("COUST slot alignment count drift")
    orders = [np.roll(np.arange(len(candidate)), -shift) for shift in range(len(candidate))]
    return min(
        orders,
        key=lambda order: sum(base._angular_error(float(reference[index]), float(candidate[order[index]])) for index in range(len(reference))),
    )


def _ensemble(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    observations = len(data["presence"])
    count = np.argmax(data["count_probability"], axis=1).astype(np.int64) + 1
    bearing = np.zeros((observations, 4))
    concentration = np.zeros((observations, 4))
    width = np.zeros((observations, 4))
    profile = np.zeros((observations, 4, 4))
    mass = np.zeros((observations, 4, 180), dtype=np.float32)
    azimuth = np.arange(180) * 2 * np.pi / 180
    for row in range(observations):
        cardinality = int(count[row])
        branch = np.arange(BRANCH_SLICE[cardinality].start, BRANCH_SLICE[cardinality].stop)
        reference = data["seed0_slot_bearing_deg"][row, branch]
        aligned = []
        for seed in range(3):
            candidate = data[f"seed{seed}_slot_bearing_deg"][row, branch]
            order = np.arange(cardinality) if seed == 0 else _align_cyclic_order(reference, candidate)
            aligned.append(branch[order])
        for slot in range(cardinality):
            mass[row, slot] = np.mean([
                data[f"seed{seed}_slot_mass"][row, aligned[seed][slot]] for seed in range(3)
            ], axis=0)
            mass[row, slot] /= mass[row, slot].sum()
            cosine = float((mass[row, slot] * np.cos(azimuth)).sum())
            sine = float((mass[row, slot] * np.sin(azimuth)).sum())
            bearing[row, slot] = np.degrees(np.arctan2(sine, cosine)) % 360
            concentration[row, slot] = np.hypot(cosine, sine)
            width[row, slot] = np.mean([
                data[f"seed{seed}_slot_opening_width_m"][row, aligned[seed][slot]] for seed in range(3)
            ])
            profile[row, slot] = np.mean([
                data[f"seed{seed}_slot_vertical_profile_m"][row, aligned[seed][slot]] for seed in range(3)
            ], axis=0)
    score = data["count_probability"].max(axis=1) * np.asarray([
        concentration[row, :count[row]].min() for row in range(observations)
    ])
    return {"count": count, "bearing": bearing, "concentration": concentration, "width": width, "profile": profile, "score": score}


def _stratum(metrics: dict, cardinality: int) -> dict:
    return next(row for row in metrics["cardinality"]["strata"] if row["target_count"] == cardinality)


def _baseline_improvement(split: str, metrics: dict) -> dict[str, float | bool]:
    baseline = BASELINE[split]
    values = {
        "overall_exact2_gain": metrics["refusal"]["raw_exact_set_fraction_2deg"] - baseline["overall_exact2"],
        "count3_exact2_gain": _stratum(metrics, 3)["exact_set_fraction_2deg"] - baseline["count3_exact2"],
        "count4_exact2_gain": _stratum(metrics, 4)["exact_set_fraction_2deg"] - baseline["count4_exact2"],
        "safe_recall_gain": metrics["detection"]["recall"] - baseline["safe_recall"],
    }
    values["passes_registered_gain"] = bool(
        values["overall_exact2_gain"] >= 0.05
        and values["count3_exact2_gain"] >= 0.05
        and values["count4_exact2_gain"] >= 0.05
        and values["safe_recall_gain"] >= 0.05
    )
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if len(args.prediction_root) != 3:
        raise RuntimeError("COUST selection requires three seeds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = [path.resolve() for path in args.prediction_root]
    c07_data = base._load_split("C07", args.teacher_root.resolve(), roots)
    c08_data = base._load_split("C08", args.teacher_root.resolve(), roots)
    c07_decoded, c08_decoded = _ensemble(c07_data), _ensemble(c08_data)
    c07_match = base._match(c07_data, c07_decoded, 2.0)
    choice = base._select(c07_decoded["score"], c07_match, c07_decoded["count"], int(c07_data["presence"].sum()))
    threshold = 2.0 if choice is None else float(choice["threshold"])
    c07 = base._evaluate(c07_data, c07_decoded, threshold, args.source_root.resolve())
    c08 = base._evaluate(c08_data, c08_decoded, threshold, args.source_root.resolve())
    improvements = {"c07": _baseline_improvement("c07", c07), "c08": _baseline_improvement("c08", c08)}
    checks = {
        "c07_safe_precision_recall": c07["detection"]["precision"] >= .995 and c07["detection"]["recall"] >= .50,
        "c08_safe_precision_recall": c08["detection"]["precision"] >= .995 and c08["detection"]["recall"] >= .50,
        "raw_cardinality_accuracy": c07["cardinality"]["raw_accuracy"] >= .80 and c08["cardinality"]["raw_accuracy"] >= .80,
        "deployed_action_macro_f1": c07["cardinality"]["deployed_action"]["macro_f1"] >= .80 and c08["cardinality"]["deployed_action"]["macro_f1"] >= .80,
        "overall_exact_set": all(metrics["refusal"]["raw_exact_set_fraction_2deg"] >= .50 and metrics["refusal"]["safe_exact_set_fraction_all"] >= .50 for metrics in (c07, c08)),
        "three_four_exit_recovery": all(
            _stratum(metrics, 3)["exact_set_fraction_2deg"] >= .40
            and _stratum(metrics, 3)["safe_exact_accepted_fraction_2deg"] >= .30
            and _stratum(metrics, 4)["exact_set_fraction_2deg"] >= .20
            and _stratum(metrics, 4)["safe_exact_accepted_fraction_2deg"] >= .10
            for metrics in (c07, c08)
        ),
        "registered_baseline_gain": improvements["c07"]["passes_registered_gain"] and improvements["c08"]["passes_registered_gain"],
        "slot_geometry_contract": all(
            metrics["slot_geometry"]["bearing_mae_deg"] <= 1
            and metrics["slot_geometry"]["opening_width_mae_m"] <= 3
            and metrics["slot_geometry"]["vertical_profile_mae_m"] <= 1
            for metrics in (c07, c08)
        ),
        "global_geometry_contract": all(
            metrics["global_geometry"]["axis_mean_error_deg"] <= 10
            and metrics["global_geometry"]["width_mae_m"] <= 2
            and metrics["global_geometry"]["height_mae_m"] <= 2
            and metrics["global_geometry"]["slope_mae_deg"] <= 2
            and metrics["global_geometry"]["curvature_mae_per_m"] <= .02
            for metrics in (c07, c08)
        ),
        "c08_not_used_for_selection": True,
        "zero_test_graph_planner": True,
    }
    scientific_pass = choice is not None and all(checks.values())
    summary = {
        "schema_version": "gse_cyclic_ordered_unimodal_slot_transport_selection_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_GSE_STRUCTURE_NODE_GENERATION_READINESS" if scientific_pass else "STOP_COUST_BEFORE_GRAPH",
        "selected_on": "C07 only", "transferred_once_to": "C08 zero adaptation",
        "precision_floor": base.PRECISION_FLOOR, "bearing_tolerance_deg": base.BEARING_TOLERANCE_DEG,
        "threshold_choice": choice, "frozen_baseline": BASELINE, "baseline_improvement": improvements,
        "c07": c07, "c08": c08, "checks": checks, "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "c08_checkpoint_observations": 0, "c09_worlds_read": 0,
        "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", summary)
    with (output / "per_world_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "parent_id", "precision", "recall", "accepted_observations", "observations"))
        writer.writeheader()
        for split, metrics in (("C07", c07), ("C08", c08)):
            for row in metrics["per_world"]:
                writer.writerow({"split": split, **row})
    base._plot(output, summary)
    for suffix in ("png", "pdf", "svg"):
        source = output / f"gse_cardinality_conditioned_circular_slot_transport_selection_v1.{suffix}"
        source.rename(output / f"gse_cyclic_ordered_unimodal_slot_transport_selection_v1.{suffix}")
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "threshold": None if choice is None else choice["threshold"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
