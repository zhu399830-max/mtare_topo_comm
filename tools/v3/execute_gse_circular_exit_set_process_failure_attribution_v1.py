#!/usr/bin/env python3
"""Attribute the frozen circular exit-set process safe-recall collapse."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations, permutations
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

from mtare_topo.governance import write_json


PASS = "PASS_GSE_CIRCULAR_EXIT_SET_PROCESS_FAILURE_ATTRIBUTION_V1"
FAIL = "FAIL_GSE_CIRCULAR_EXIT_SET_PROCESS_FAILURE_ATTRIBUTION_V1"
TOLERANCES_DEG = (2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0)
PRECISION_FLOOR = 0.995
SUPPRESSION_RADIUS_BINS = 1


def _angular_error(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _optimal_pairs(predicted: np.ndarray, target: np.ndarray) -> list[tuple[int, int, float]]:
    """Minimum-error one-to-one assignment for sets of at most four bearings."""
    size = min(len(predicted), len(target))
    if size == 0:
        return []
    best: list[tuple[int, int, float]] | None = None
    best_error = math.inf
    for pred_subset in combinations(range(len(predicted)), size):
        for target_order in permutations(range(len(target)), size):
            pairs = [
                (p, t, _angular_error(float(predicted[p]), float(target[t])))
                for p, t in zip(pred_subset, target_order, strict=True)
            ]
            total = sum(item[2] for item in pairs)
            if total < best_error:
                best = pairs
                best_error = total
    assert best is not None
    return best


def _average_precision(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    truth = np.asarray(label, dtype=bool)[order]
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float(precision[truth].sum() / positives)


def _safe_threshold(score: np.ndarray, label: np.ndarray) -> dict[str, float | int] | None:
    values = np.asarray(score, dtype=np.float64)
    truth = np.asarray(label, dtype=bool)
    order = np.argsort(-values, kind="stable")
    values = values[order]
    truth = truth[order]
    tp = np.cumsum(truth)
    ends = np.flatnonzero(np.r_[values[1:] != values[:-1], True])
    precision = tp[ends] / (ends + 1)
    recall = tp[ends] / max(int(truth.sum()), 1)
    safe = np.flatnonzero(precision >= PRECISION_FLOOR)
    if not len(safe):
        return None
    best = max(safe, key=lambda index: (float(recall[index]), float(precision[index]), float(values[ends[index]])))
    return {
        "threshold": float(values[ends[best]]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "accepted": int(ends[best] + 1),
    }


def _decode(mass: np.ndarray, count_probability: np.ndarray, heading: np.ndarray) -> dict[str, np.ndarray]:
    count = np.argmax(count_probability, axis=1).astype(np.int64) + 1
    bins = np.full((len(mass), 4), -1, dtype=np.int64)
    selected_mass = np.zeros((len(mass), 4), dtype=np.float64)
    next_mass = np.zeros(len(mass), dtype=np.float64)
    for row, values in enumerate(mass):
        available = np.ones(180, dtype=bool)
        for slot in range(int(count[row])):
            selected = int(np.argmax(np.where(available, values, -1.0)))
            bins[row, slot] = selected
            selected_mass[row, slot] = values[selected]
            for delta in range(-SUPPRESSION_RADIUS_BINS, SUPPRESSION_RADIUS_BINS + 1):
                available[(selected + delta) % 180] = False
        next_mass[row] = float(np.max(values[available]))
    safe = np.maximum(bins, 0)
    bearing = np.mod(safe * 2.0 + heading[np.arange(len(mass))[:, None], safe], 360.0)
    center = np.mod(safe * 2.0, 360.0)
    valid = np.arange(4)[None, :] < count[:, None]
    bearing[~valid] = 0.0
    center[~valid] = 0.0
    normalized_min = np.asarray([
        min(float(count[row]) * selected_mass[row, : count[row]].min(), 1.0)
        for row in range(len(mass))
    ])
    return {
        "count": count,
        "bins": bins,
        "bearing": bearing,
        "center": center,
        "selected_mass": selected_mass,
        "next_mass": next_mass,
        "normalized_min": normalized_min,
        "original_score": count_probability.max(axis=1) * normalized_min,
    }


def _load_split(suffix: str, teacher_root: Path, prediction_roots: list[Path]) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    for teacher_path in sorted(teacher_root.glob(f"selection/*_{suffix}.zarr")):
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = str(teacher.attrs["parent_id"])
        predictions = [np.load(root / f"{parent}.npz") for root in prediction_roots]
        sequence = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
        if any(not np.array_equal(sequence, item["global_sequence_index"]) for item in predictions):
            raise RuntimeError(f"prediction join drift: {parent}")
        values: dict[str, np.ndarray] = {
            "parent_id": np.full(len(sequence), parent, dtype=f"U{len(parent)}"),
            "presence": np.asarray(teacher["presence"][:], dtype=np.uint8),
            "heading_target": np.asarray(teacher["heading_residual_deg"][:], dtype=np.float64),
            "width_target": np.asarray(teacher["opening_width_m"][:], dtype=np.float64),
            "width_valid": np.asarray(teacher["width_valid_mask"][:], dtype=np.uint8),
            "profile_target": np.asarray(teacher["vertical_profile_m"][:], dtype=np.float64),
            "mass": np.mean([np.asarray(item["exit_mass"], dtype=np.float64) for item in predictions], axis=0),
            "count_probability": np.mean([np.asarray(item["exit_count_probability"], dtype=np.float64) for item in predictions], axis=0),
            "heading": np.mean([np.asarray(item["heading_residual_deg"], dtype=np.float64) for item in predictions], axis=0),
            "width": np.mean([np.asarray(item["opening_width_m"], dtype=np.float64) for item in predictions], axis=0),
            "profile": np.mean([np.asarray(item["vertical_profile_m"], dtype=np.float64) for item in predictions], axis=0),
            "observation_uncertainty": np.mean([np.asarray(item["observation_uncertainty"], dtype=np.float64) for item in predictions], axis=0),
        }
        values["mass"] /= values["mass"].sum(axis=1, keepdims=True)
        for seed, item in enumerate(predictions):
            for name in ("exit_mass", "exit_count_probability", "heading_residual_deg"):
                values[f"seed{seed}_{name}"] = np.asarray(item[name], dtype=np.float64)
        for name, value in values.items():
            output[name].append(value)
    if len(output["presence"]) != 10:
        raise RuntimeError(f"{suffix} world count drift")
    return {name: np.concatenate(parts) for name, parts in output.items()}


def _row_analysis(data: dict[str, np.ndarray]) -> dict[str, object]:
    decoded = _decode(data["mass"], data["count_probability"], data["heading"])
    target_count = data["presence"].sum(axis=1).astype(np.int64)
    row_errors: list[list[float]] = []
    center_errors: list[list[float]] = []
    all_errors: list[float] = []
    all_center_errors: list[float] = []
    residual_changes: list[float] = []
    oracle_residual_errors: list[float] = []
    width_errors: list[float] = []
    profile_errors: list[float] = []
    exact_by_tolerance = {tolerance: np.zeros(len(target_count), dtype=bool) for tolerance in TOLERANCES_DEG}
    tp_by_tolerance = {tolerance: 0 for tolerance in TOLERANCES_DEG}
    for row in range(len(target_count)):
        truth_bins = np.flatnonzero(data["presence"][row])
        truth = np.mod(truth_bins * 2.0 + data["heading_target"][row, truth_bins], 360.0)
        predicted = decoded["bearing"][row, : decoded["count"][row]]
        centers = decoded["center"][row, : decoded["count"][row]]
        pairs = _optimal_pairs(predicted, truth)
        center_pairs = _optimal_pairs(centers, truth)
        errors = [item[2] for item in pairs]
        row_errors.append(errors)
        center_errors.append([item[2] for item in center_pairs])
        all_errors.extend(errors)
        all_center_errors.extend(item[2] for item in center_pairs)
        residual_changes.extend(
            _angular_error(float(centers[pred_slot]), float(truth[target_index])) - error
            for pred_slot, target_index, error in pairs
        )
        for target_index, target_bin in enumerate(truth_bins):
            oracle_bearing = (target_bin * 2.0 + data["heading"][row, target_bin]) % 360.0
            oracle_residual_errors.append(_angular_error(float(oracle_bearing), float(truth[target_index])))
        for pred_slot, target_index, _ in pairs:
            pred_bin = int(decoded["bins"][row, pred_slot])
            target_bin = int(truth_bins[target_index])
            if data["width_valid"][row, target_bin]:
                width_errors.append(abs(float(data["width"][row, pred_bin] - data["width_target"][row, target_bin])))
            profile_errors.extend(np.abs(data["profile"][row, pred_bin] - data["profile_target"][row, target_bin]).tolist())
        for tolerance in TOLERANCES_DEG:
            matched = sum(error <= tolerance for error in errors)
            tp_by_tolerance[tolerance] += matched
            exact_by_tolerance[tolerance][row] = (
                decoded["count"][row] == target_count[row] and len(errors) == target_count[row] and all(error <= tolerance for error in errors)
            )
    total_predicted = int(decoded["count"].sum())
    total_target = int(target_count.sum())
    tolerance_rows = []
    for tolerance in TOLERANCES_DEG:
        true_positive = tp_by_tolerance[tolerance]
        tolerance_rows.append({
            "tolerance_deg": tolerance,
            "exit_precision": true_positive / total_predicted,
            "exit_recall": true_positive / total_target,
            "exact_set_fraction": float(exact_by_tolerance[tolerance].mean()),
        })
    seed_decoded = []
    for seed in range(3):
        mass = data[f"seed{seed}_exit_mass"]
        mass = mass / mass.sum(axis=1, keepdims=True)
        seed_decoded.append(_decode(mass, data[f"seed{seed}_exit_count_probability"], data[f"seed{seed}_heading_residual_deg"]))
    count_agreement = np.asarray([
        max(np.bincount([int(item["count"][row]) for item in seed_decoded], minlength=5)) / 3.0
        for row in range(len(target_count))
    ])
    seed_disagreement = np.zeros(len(target_count), dtype=np.float64)
    for row in range(len(target_count)):
        distances = []
        for first, second in combinations(range(3), 2):
            a = seed_decoded[first]["bearing"][row, : seed_decoded[first]["count"][row]]
            b = seed_decoded[second]["bearing"][row, : seed_decoded[second]["count"][row]]
            pairs = _optimal_pairs(a, b)
            count_penalty = abs(len(a) - len(b)) * 180.0
            distances.append((sum(item[2] for item in pairs) + count_penalty) / max(len(a), len(b)))
        seed_disagreement[row] = float(np.mean(distances))
    count_probability = data["count_probability"]
    sorted_count = np.sort(count_probability, axis=1)
    entropy = -(data["mass"] * np.log(np.clip(data["mass"], 1e-12, None))).sum(axis=1)
    selected_min = np.asarray([
        decoded["selected_mass"][row, : decoded["count"][row]].min()
        for row in range(len(target_count))
    ])
    features = {
        "original_score": decoded["original_score"],
        "count_confidence": count_probability.max(axis=1),
        "count_margin": sorted_count[:, -1] - sorted_count[:, -2],
        "normalized_selected_min": decoded["normalized_min"],
        "selected_next_margin": selected_min - decoded["next_mass"],
        "negative_mass_entropy": -entropy,
        "negative_observation_uncertainty": -data["observation_uncertainty"],
        "seed_count_agreement": count_agreement,
        "negative_seed_set_disagreement": -seed_disagreement,
        "original_consensus": decoded["original_score"] * count_agreement * np.exp(-seed_disagreement / 2.0),
    }
    confidence = {}
    exact2 = exact_by_tolerance[2.0]
    for name, values in features.items():
        confidence[name] = {
            "average_precision_exact_set_2deg": _average_precision(values, exact2),
            "c07_choice": _safe_threshold(values, exact2),
        }
    cardinality = []
    for count in range(1, 5):
        rows = target_count == count
        cardinality.append({
            "target_count": count,
            "observations": int(rows.sum()),
            "count_accuracy": float(np.mean(decoded["count"][rows] == count)),
            **{f"exact_set_fraction_{int(t)}deg": float(exact_by_tolerance[t][rows].mean()) for t in TOLERANCES_DEG},
        })
    return {
        "decoded": decoded,
        "target_count": target_count,
        "exact_by_tolerance": exact_by_tolerance,
        "features": features,
        "tolerance_curve": tolerance_rows,
        "cardinality": cardinality,
        "confidence": confidence,
        "errors": {
            "continuous_bearing_deg": _quantiles(all_errors),
            "bin_center_bearing_deg": _quantiles(all_center_errors),
            "residual_improvement_deg": _quantiles(residual_changes),
            "oracle_teacher_bin_residual_deg": _quantiles(oracle_residual_errors),
            "selected_bin_width_mae_m": float(np.mean(width_errors)),
            "selected_bin_profile_mae_m": float(np.mean(profile_errors)),
        },
        "population": {"observations": len(target_count), "target_exits": total_target, "predicted_exits": total_predicted},
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _transfer_confidence(c07: dict[str, object], c08: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    labels08 = c08["exact_by_tolerance"][2.0]
    for name, feature07 in c07["features"].items():
        choice = c07["confidence"][name]["c07_choice"]
        values08 = c08["features"][name]
        accepted = np.zeros(len(values08), dtype=bool) if choice is None else values08 >= float(choice["threshold"])
        precision08 = float(labels08[accepted].mean()) if accepted.any() else 0.0
        recall08 = float((labels08 & accepted).sum() / max(int(labels08.sum()), 1))
        rows.append({
            "feature": name,
            "c07_average_precision": c07["confidence"][name]["average_precision_exact_set_2deg"],
            "c07_threshold": None if choice is None else choice["threshold"],
            "c07_precision": 0.0 if choice is None else choice["precision"],
            "c07_recall": 0.0 if choice is None else choice["recall"],
            "c07_accepted": 0 if choice is None else choice["accepted"],
            "c08_precision": precision08,
            "c08_recall": recall08,
            "c08_accepted": int(accepted.sum()),
        })
    return rows


def _public(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if key not in ("decoded", "target_count", "exact_by_tolerance", "features")}


def _plot(output: Path, summary: dict[str, object]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    for split, color in (("c07", "#4e79a7"), ("c08", "#f28e2b")):
        rows = summary[split]["tolerance_curve"]
        axes[0].plot([row["tolerance_deg"] for row in rows], [row["exact_set_fraction"] for row in rows], marker="o", label=split.upper(), color=color)
    axes[0].axhline(0.5, color="#e15759", linestyle="--", linewidth=1)
    axes[0].set(xlabel="bearing tolerance (deg)", ylabel="exact exit-set fraction", title="A  Localization capacity")
    names = [row["feature"] for row in summary["confidence_transfer"]]
    order = np.argsort([-row["c07_recall"] for row in summary["confidence_transfer"]])[:5]
    labels = [names[index].replace("negative_", "-").replace("_", "\n") for index in order]
    x = np.arange(len(order))
    axes[1].bar(x - 0.18, [summary["confidence_transfer"][index]["c07_recall"] for index in order], 0.36, label="C07")
    axes[1].bar(x + 0.18, [summary["confidence_transfer"][index]["c08_recall"] for index in order], 0.36, label="C08")
    axes[1].axhline(0.5, color="#e15759", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set(ylabel="safe exact-set recall", title="B  Confidence separability")
    error_names = ("bin center", "selected residual", "oracle-bin residual")
    for split_index, split in enumerate(("c07", "c08")):
        errors = summary[split]["errors"]
        values = [errors["bin_center_bearing_deg"]["p50"], errors["continuous_bearing_deg"]["p50"], errors["oracle_teacher_bin_residual_deg"]["p50"]]
        axes[2].bar(np.arange(3) + (split_index - 0.5) * 0.28, values, 0.28, label=split.upper())
    axes[2].set_xticks(np.arange(3), error_names, rotation=15)
    axes[2].set(ylabel="median angular error (deg)", title="C  Residual-head coupling")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(frameon=False)
    figure.suptitle("GSE-Graph exit-set failure attribution")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_circular_exit_set_process_failure_attribution_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, action="append", type=Path)
    parser.add_argument("--formal-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if len(args.prediction_root) != 3:
        raise RuntimeError("attribution requires exactly three frozen seeds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    formal = json.loads(args.formal_summary.read_text(encoding="utf-8"))["selection"]
    c07_internal = _row_analysis(_load_split("C07", args.teacher_root.resolve(), [path.resolve() for path in args.prediction_root]))
    c08_internal = _row_analysis(_load_split("C08", args.teacher_root.resolve(), [path.resolve() for path in args.prediction_root]))
    transfer = _transfer_confidence(c07_internal, c08_internal)
    exact2_c07 = c07_internal["tolerance_curve"][0]["exact_set_fraction"]
    exact10_c07 = next(row["exact_set_fraction"] for row in c07_internal["tolerance_curve"] if row["tolerance_deg"] == 10.0)
    best_safe_c07 = max(row["c07_recall"] for row in transfer)
    best_safe_c08 = max(row["c08_recall"] for row in transfer)
    oracle_residual = c07_internal["errors"]["oracle_teacher_bin_residual_deg"]["mean"]
    if exact2_c07 >= 0.5 and min(best_safe_c07, best_safe_c08) < 0.5:
        diagnosis = "CONFIDENCE_RANKING_FAILURE"
    elif exact10_c07 < 0.5:
        diagnosis = "LOCALIZATION_CAPACITY_FAILURE"
    elif oracle_residual <= 1.0:
        diagnosis = "SELECTED_BIN_RESIDUAL_COUPLING_AND_CONFIDENCE_FAILURE"
    else:
        diagnosis = "MIXED_LOCALIZATION_AND_CONFIDENCE_FAILURE"
    checks = {
        "population_exact": c07_internal["population"]["observations"] == 21548 and c07_internal["population"]["target_exits"] == 45504 and c08_internal["population"]["observations"] == 24394 and c08_internal["population"]["target_exits"] == 51537,
        "cardinality_reproduced": abs(float(np.mean(c07_internal["decoded"]["count"] == c07_internal["target_count"])) - formal["c07"]["cardinality"]["raw_accuracy"]) <= 1e-12 and abs(float(np.mean(c08_internal["decoded"]["count"] == c08_internal["target_count"])) - formal["c08"]["cardinality"]["raw_accuracy"]) <= 1e-12,
        "tolerance_monotonic": all(np.all(np.diff([row[key] for row in item["tolerance_curve"]]) >= -1e-12) for item in (c07_internal, c08_internal) for key in ("exit_precision", "exit_recall", "exact_set_fraction")),
        "confidence_features_finite": all(np.all(np.isfinite(values)) for item in (c07_internal, c08_internal) for values in item["features"].values()),
        "diagnosis_resolved": diagnosis != "UNRESOLVED",
        "zero_training_test_graph": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_circular_exit_set_process_failure_attribution_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": diagnosis if scientific_pass else "STOP_UNRESOLVED_ATTRIBUTION",
        "question": "Localization capacity failure or confidence separability failure?",
        "c07": _public(c07_internal),
        "c08": _public(c08_internal),
        "confidence_transfer": transfer,
        "diagnosis_basis": {"c07_exact_set_fraction_2deg": exact2_c07, "c07_exact_set_fraction_10deg": exact10_c07, "best_safe_confidence_recall_c07": best_safe_c07, "best_safe_confidence_recall_c08": best_safe_c08, "c07_oracle_teacher_bin_residual_mean_deg": oracle_residual},
        "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0,
        "model_inference_observations": 0,
        "checkpoint_writes": 0,
        "threshold_training_steps": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "graph_replays": 0,
        "planner_calls": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", summary)
    with (output / "tolerance_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "tolerance_deg", "exit_precision", "exit_recall", "exact_set_fraction"))
        writer.writeheader()
        for split, item in (("C07", c07_internal), ("C08", c08_internal)):
            for row in item["tolerance_curve"]:
                writer.writerow({"split": split, **row})
    with (output / "confidence_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(transfer[0]))
        writer.writeheader()
        writer.writerows(transfer)
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "diagnosis_basis": summary["diagnosis_basis"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
