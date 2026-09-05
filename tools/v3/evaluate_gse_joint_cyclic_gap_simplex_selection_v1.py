#!/usr/bin/env python3
"""C07 selection and one C08 transfer for Joint Cyclic Gap Simplex."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

import evaluate_gse_cardinality_conditioned_circular_slot_transport_selection_v1 as base
from evaluate_gse_cyclic_ordered_unimodal_slot_transport_selection_v1 import _align_cyclic_order
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import BRANCH_SLICE


PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_SELECTION_V1"
FAIL = "FAIL_GSE_JOINT_CYCLIC_GAP_SIMPLEX_SELECTION_V1"
BASELINE = {"c07": {"overall": .4155838124, "k3": .0023956194, "k4": 0.0}, "c08": {"overall": .3869804050, "k3": .0017851830, "k4": .0043290043}}


def _load_split(suffix: str, teacher_root: Path, prediction_roots: list[Path]) -> dict[str, np.ndarray]:
    output = defaultdict(list)
    for teacher_path in sorted(teacher_root.glob(f"selection/*_{suffix}.zarr")):
        teacher = zarr.open_group(str(teacher_path), mode="r"); parent = str(teacher.attrs["parent_id"]); sequence = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64); predictions = [np.load(root / f"{parent}.npz") for root in prediction_roots]
        if any(not np.array_equal(sequence, item["global_sequence_index"]) for item in predictions): raise RuntimeError(f"gap-simplex join drift:{parent}")
        values = {"parent_id": np.full(len(sequence), parent, dtype=f"U{len(parent)}"), "global_sequence_index": sequence, "presence": np.asarray(teacher["presence"][:], dtype=np.uint8), "heading_target": np.asarray(teacher["heading_residual_deg"][:], dtype=np.float32), "width_target": np.asarray(teacher["opening_width_m"][:], dtype=np.float32), "width_valid": np.asarray(teacher["width_valid_mask"][:], dtype=np.uint8), "profile_target": np.asarray(teacher["vertical_profile_m"][:], dtype=np.float32), "count_probability": np.mean([np.asarray(item["exit_count_probability"], dtype=np.float32) for item in predictions], axis=0), "axis": np.mean([np.asarray(item["local_axis"], dtype=np.float32) for item in predictions], axis=0), "geometry": np.mean([np.asarray(item["geometry"], dtype=np.float32) for item in predictions], axis=0)}
        for seed, item in enumerate(predictions):
            for name in ("joint_bearing_deg", "joint_bearing_scale_deg", "phase_concentration", "exit_opening_width_m", "exit_vertical_profile_m"): values[f"seed{seed}_{name}"] = np.asarray(item[name], dtype=np.float32)
        for name, value in values.items(): output[name].append(value)
    if len(output["presence"]) != 10: raise RuntimeError(f"gap-simplex {suffix} world count drift")
    return {name: np.concatenate(parts) for name, parts in output.items()}


def _ensemble(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count_probability = data["count_probability"]; count = np.argmax(count_probability, axis=1).astype(np.int64) + 1; observations = len(count)
    bearing = np.zeros((observations, 4)); concentration = np.zeros((observations, 4)); width = np.zeros((observations, 4)); profile = np.zeros((observations, 4, 4)); scale = np.zeros((observations, 4)); phase_concentration = np.zeros(observations)
    for row, cardinality in enumerate(count):
        branch = np.arange(BRANCH_SLICE[int(cardinality)].start, BRANCH_SLICE[int(cardinality)].stop); reference = data["seed0_joint_bearing_deg"][row, branch]; aligned = []
        for seed in range(3):
            candidate = data[f"seed{seed}_joint_bearing_deg"][row, branch]; order = np.arange(cardinality) if seed == 0 else _align_cyclic_order(reference, candidate); aligned.append(branch[order])
        phase_concentration[row] = np.mean([data[f"seed{seed}_phase_concentration"][row, int(cardinality)-1] for seed in range(3)])
        for slot in range(cardinality):
            angles = np.deg2rad([data[f"seed{seed}_joint_bearing_deg"][row, aligned[seed][slot]] for seed in range(3)]); cosine, sine = np.mean(np.cos(angles)), np.mean(np.sin(angles)); bearing[row, slot] = np.degrees(np.arctan2(sine, cosine)) % 360.0; concentration[row, slot] = np.hypot(cosine, sine)
            width[row, slot] = np.mean([data[f"seed{seed}_exit_opening_width_m"][row, aligned[seed][slot]] for seed in range(3)]); profile[row, slot] = np.mean([data[f"seed{seed}_exit_vertical_profile_m"][row, aligned[seed][slot]] for seed in range(3)], axis=0); scale[row, slot] = np.mean([data[f"seed{seed}_joint_bearing_scale_deg"][row, aligned[seed][slot]] for seed in range(3)])
    score = count_probability.max(axis=1) * phase_concentration * np.exp(-np.asarray([scale[row, :count[row]].max() for row in range(observations)]) / 180.0)
    return {"count": count, "bearing": bearing, "concentration": concentration, "width": width, "profile": profile, "score": score, "bearing_scale": scale, "phase_concentration": phase_concentration}


def _stratum(metrics: dict, cardinality: int) -> dict: return next(row for row in metrics["cardinality"]["strata"] if row["target_count"] == cardinality)


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    splits = ("c07", "c08")
    x = np.arange(2)
    axes[0].bar(x - .18, [summary[s]["detection"]["precision"] for s in splits], .36, label="precision")
    axes[0].bar(x + .18, [summary[s]["detection"]["recall"] for s in splits], .36, label="recall")
    axes[0].axhline(.995, color="#e15759", linestyle="--")
    axes[0].set_xticks(x, ("C07", "C08")); axes[0].set_ylim(0, 1.03); axes[0].set_title("A  Safe exit sets"); axes[0].legend(frameon=False)
    for split, color in (("c07", "#4e79a7"), ("c08", "#f28e2b")):
        axes[1].plot(range(1, 5), [_stratum(summary[split], k)["exact_set_fraction_2deg"] for k in range(1, 5)], marker="o", label=split.upper(), color=color)
    axes[1].set_xticks(range(1, 5)); axes[1].set_ylim(0, 1); axes[1].set(xlabel="true exits", ylabel="raw exact-set fraction", title="B  Joint-set recovery"); axes[1].legend(frameon=False)
    names = ("bearing_mae_deg", "opening_width_mae_m", "vertical_profile_mae_m")
    axes[2].bar(np.arange(3) - .15, [summary["c07"]["slot_geometry"][name] for name in names], .3, label="C07")
    axes[2].bar(np.arange(3) + .15, [summary["c08"]["slot_geometry"][name] for name in names], .3, label="C08")
    axes[2].set_xticks(np.arange(3), ("bearing deg", "width m", "profile m"), rotation=15); axes[2].set_title("C  Matched exit geometry"); axes[2].legend(frameon=False)
    for axis in axes: axis.grid(axis="y", alpha=.25); axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph joint cyclic gap-simplex selection and transfer")
    for suffix in ("png", "pdf", "svg"): figure.savefig(output / f"gse_joint_cyclic_gap_simplex_selection_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--teacher-root", required=True, type=Path); parser.add_argument("--source-root", required=True, type=Path); parser.add_argument("--prediction-root", required=True, action="append", type=Path); parser.add_argument("--output-dir", required=True, type=Path); args = parser.parse_args(); started = time.monotonic()
    if len(args.prediction_root) != 3: raise RuntimeError("gap-simplex selection requires three seeds")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False); roots = [path.resolve() for path in args.prediction_root]
    data07, data08 = _load_split("C07", args.teacher_root.resolve(), roots), _load_split("C08", args.teacher_root.resolve(), roots); decoded07, decoded08 = _ensemble(data07), _ensemble(data08)
    match07 = base._match(data07, decoded07, 2.0); choice = base._select(decoded07["score"], match07, decoded07["count"], int(data07["presence"].sum())); threshold = 2.0 if choice is None else float(choice["threshold"])
    c07 = base._evaluate(data07, decoded07, threshold, args.source_root.resolve()); c08 = base._evaluate(data08, decoded08, threshold, args.source_root.resolve())
    gains = {split: {"overall": metrics["refusal"]["raw_exact_set_fraction_2deg"] - BASELINE[split]["overall"], "k3": _stratum(metrics, 3)["exact_set_fraction_2deg"] - BASELINE[split]["k3"], "k4": _stratum(metrics, 4)["exact_set_fraction_2deg"] - BASELINE[split]["k4"]} for split, metrics in (("c07", c07), ("c08", c08))}
    checks = {"c07_safe_precision_recall": c07["detection"]["precision"] >= .995 and c07["detection"]["recall"] >= .50, "c08_safe_precision_recall": c08["detection"]["precision"] >= .995 and c08["detection"]["recall"] >= .50, "raw_cardinality_accuracy": c07["cardinality"]["raw_accuracy"] >= .80 and c08["cardinality"]["raw_accuracy"] >= .80, "deployed_action_macro_f1": c07["cardinality"]["deployed_action"]["macro_f1"] >= .80 and c08["cardinality"]["deployed_action"]["macro_f1"] >= .80, "overall_exact_set": all(metrics["refusal"]["raw_exact_set_fraction_2deg"] >= .50 and metrics["refusal"]["safe_exact_set_fraction_all"] >= .50 for metrics in (c07, c08)), "three_four_exit_recovery": all(_stratum(metrics, 3)["exact_set_fraction_2deg"] >= .40 and _stratum(metrics, 3)["safe_exact_accepted_fraction_2deg"] >= .30 and _stratum(metrics, 4)["exact_set_fraction_2deg"] >= .20 and _stratum(metrics, 4)["safe_exact_accepted_fraction_2deg"] >= .10 for metrics in (c07, c08)), "registered_coust_gain": all(gains[split][name] >= .05 for split in ("c07", "c08") for name in ("overall", "k3", "k4")), "exit_geometry_contract": all(metrics["slot_geometry"]["bearing_mae_deg"] <= 1 and metrics["slot_geometry"]["opening_width_mae_m"] <= 3 and metrics["slot_geometry"]["vertical_profile_mae_m"] <= 1 for metrics in (c07, c08)), "global_geometry_contract": all(metrics["global_geometry"]["axis_mean_error_deg"] <= 10 and metrics["global_geometry"]["width_mae_m"] <= 2 and metrics["global_geometry"]["height_mae_m"] <= 2 and metrics["global_geometry"]["slope_mae_deg"] <= 2 and metrics["global_geometry"]["curvature_mae_per_m"] <= .02 for metrics in (c07, c08)), "c08_not_used_for_selection": True, "zero_test_graph_planner": True}
    scientific_pass = choice is not None and all(checks.values()); summary = {"schema_version": "gse_joint_cyclic_gap_simplex_selection_v1", "status": PASS if scientific_pass else FAIL, "scientific_pass": scientific_pass, "decision": "ALLOW_GSE_STRUCTURE_NODE_GENERATION_READINESS" if scientific_pass else "STOP_JOINT_CYCLIC_GAP_SIMPLEX_BEFORE_GRAPH", "selected_on": "C07 only", "transferred_once_to": "C08 zero adaptation", "precision_floor": base.PRECISION_FLOOR, "bearing_tolerance_deg": base.BEARING_TOLERANCE_DEG, "threshold_choice": choice, "baseline": BASELINE, "baseline_gain": gains, "c07": c07, "c08": c08, "checks": checks, "duration_seconds": time.monotonic()-started, "optimizer_steps": 0, "c08_checkpoint_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0}
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    with (output / "per_world_metrics.csv").open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=("split", "parent_id", "precision", "recall", "accepted_observations", "observations")); writer.writeheader(); [writer.writerow({"split": split, **row}) for split, metrics in (("C07", c07), ("C08", c08)) for row in metrics["per_world"]]
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "threshold": None if choice is None else choice["threshold"], "checks": checks}, indent=2, sort_keys=True)); return 0 if scientific_pass else 2


if __name__ == "__main__": raise SystemExit(main())
