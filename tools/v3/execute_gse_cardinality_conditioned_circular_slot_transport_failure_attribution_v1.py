#!/usr/bin/env python3
"""Attribute strict-localization failure in frozen circular slot transport outputs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations, permutations
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
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import BRANCH_SLICE


PASS = "PASS_GSE_CARDINALITY_CONDITIONED_CIRCULAR_SLOT_TRANSPORT_FAILURE_ATTRIBUTION_V1"
PRECISION_FLOOR = 0.995
TOLERANCES_DEG = (2.0, 4.0, 10.0)
MODES = ("circular_mean", "argmax", "local_mode")


def angular_error(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circular_bearing(mass: np.ndarray) -> float:
    angle = np.arange(180, dtype=np.float64) * 2.0 * np.pi / 180.0
    return float(np.degrees(np.arctan2(np.sum(mass * np.sin(angle)), np.sum(mass * np.cos(angle)))) % 360.0)


def local_mode_bearing(mass: np.ndarray) -> float:
    """Circular mean over the argmax bin and its two immediate neighbours."""
    peak = int(np.argmax(mass))
    indices = np.remainder(peak + np.arange(-1, 2), 180)
    local = mass[indices].astype(np.float64)
    angle = indices * 2.0 * np.pi / 180.0
    return float(np.degrees(np.arctan2(np.sum(local * np.sin(angle)), np.sum(local * np.cos(angle)))) % 360.0)


def binary_auc(score: np.ndarray, positive: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        return math.nan
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def _align_order(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, ...]:
    return min(
        permutations(range(len(candidate))),
        key=lambda order: sum(angular_error(float(reference[index]), float(candidate[order[index]])) for index in range(len(reference))),
    )


def _optimal_pairs(predicted: np.ndarray, target: np.ndarray, tolerance: float = math.inf) -> list[tuple[int, int, float]]:
    best: list[tuple[int, int, float]] = []
    best_error = math.inf
    for size in range(1, min(len(predicted), len(target)) + 1):
        for subset in combinations(range(len(predicted)), size):
            for order in permutations(range(len(target)), size):
                pairs = [(p, t, angular_error(float(predicted[p]), float(target[t]))) for p, t in zip(subset, order, strict=True)]
                error = sum(item[2] for item in pairs)
                if all(item[2] <= tolerance for item in pairs) and (len(pairs) > len(best) or (len(pairs) == len(best) and error < best_error)):
                    best, best_error = pairs, error
    return best


def _load_split(suffix: str, teacher_root: Path, prediction_roots: list[Path]) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    for teacher_path in sorted(teacher_root.glob(f"selection/*_{suffix}.zarr")):
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = str(teacher.attrs["parent_id"])
        sequence = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
        predictions = [np.load(root / f"{parent}.npz") for root in prediction_roots]
        if any(not np.array_equal(sequence, item["global_sequence_index"]) for item in predictions):
            raise RuntimeError(f"attribution join drift: {parent}")
        output["parent_id"].append(np.full(len(sequence), parent, dtype=f"U{len(parent)}"))
        output["global_sequence_index"].append(sequence)
        output["presence"].append(np.asarray(teacher["presence"][:], dtype=np.uint8))
        output["heading_target"].append(np.asarray(teacher["heading_residual_deg"][:], dtype=np.float32))
        for seed, prediction in enumerate(predictions):
            for name in ("slot_mass", "slot_bearing_deg", "slot_concentration", "exit_count_probability"):
                output[f"seed{seed}_{name}"].append(np.asarray(prediction[name], dtype=np.float32))
    if len(output["presence"]) != 10:
        raise RuntimeError(f"attribution {suffix} world count drift")
    return {name: np.concatenate(parts) for name, parts in output.items()}


def _target_bearings(data: dict[str, np.ndarray], row: int) -> np.ndarray:
    bins = np.flatnonzero(data["presence"][row])
    return np.remainder(bins * 2.0 + data["heading_target"][row, bins], 360.0)


def _bearing_from_mass(mass: np.ndarray, mode: str) -> float:
    if mode == "circular_mean":
        return circular_bearing(mass)
    if mode == "argmax":
        return float(np.argmax(mass) * 2.0)
    if mode == "local_mode":
        return local_mode_bearing(mass)
    raise ValueError(mode)


def _decode_seed(data: dict[str, np.ndarray], seed: int, mode: str) -> dict[str, np.ndarray]:
    count = np.argmax(data[f"seed{seed}_exit_count_probability"], axis=1).astype(np.int64) + 1
    bearing = np.zeros((len(count), 4), dtype=np.float64)
    masses = np.zeros((len(count), 4, 180), dtype=np.float32)
    for row, cardinality in enumerate(count):
        branch = np.arange(BRANCH_SLICE[int(cardinality)].start, BRANCH_SLICE[int(cardinality)].stop)
        for slot, index in enumerate(branch):
            mass = data[f"seed{seed}_slot_mass"][row, index].astype(np.float64)
            mass /= mass.sum()
            masses[row, slot] = mass
            bearing[row, slot] = _bearing_from_mass(mass, mode)
    return {"count": count, "bearing": bearing, "mass": masses, "count_probability": data[f"seed{seed}_exit_count_probability"]}


def _decode_ensemble(data: dict[str, np.ndarray], mode: str, *, oracle_count: bool = False) -> dict[str, np.ndarray]:
    count_probability = np.mean([data[f"seed{seed}_exit_count_probability"] for seed in range(3)], axis=0)
    count = data["presence"].sum(axis=1).astype(np.int64) if oracle_count else np.argmax(count_probability, axis=1).astype(np.int64) + 1
    bearing = np.zeros((len(count), 4), dtype=np.float64)
    masses = np.zeros((len(count), 4, 180), dtype=np.float32)
    disagreement = np.zeros(len(count), dtype=np.float64)
    for row, cardinality in enumerate(count):
        branch = np.arange(BRANCH_SLICE[int(cardinality)].start, BRANCH_SLICE[int(cardinality)].stop)
        reference = data["seed0_slot_bearing_deg"][row, branch]
        aligned: list[np.ndarray] = []
        aligned_bearings: list[np.ndarray] = []
        for seed in range(3):
            candidate = data[f"seed{seed}_slot_bearing_deg"][row, branch]
            order = np.arange(cardinality) if seed == 0 else np.asarray(_align_order(reference, candidate))
            aligned.append(branch[order])
            aligned_bearings.append(candidate[order])
        for slot in range(cardinality):
            mass = np.mean([data[f"seed{seed}_slot_mass"][row, aligned[seed][slot]] for seed in range(3)], axis=0).astype(np.float64)
            mass /= mass.sum()
            masses[row, slot] = mass
            bearing[row, slot] = _bearing_from_mass(mass, mode)
            disagreement[row] = max(disagreement[row], max(angular_error(float(aligned_bearings[a][slot]), float(aligned_bearings[b][slot])) for a, b in combinations(range(3), 2)))
    return {"count": count, "bearing": bearing, "mass": masses, "count_probability": count_probability, "seed_disagreement_deg": disagreement}


def _score_decode(data: dict[str, np.ndarray], decoded: dict[str, np.ndarray]) -> dict:
    target_count = data["presence"].sum(axis=1).astype(np.int64)
    exact_by_tolerance = {}
    row_tp_by_tolerance = {}
    for tolerance in TOLERANCES_DEG:
        exact = np.zeros(len(target_count), dtype=bool)
        row_tp = np.zeros(len(target_count), dtype=np.int64)
        for row in range(len(target_count)):
            predicted = decoded["bearing"][row, : decoded["count"][row]]
            target = _target_bearings(data, row)
            pairs = _optimal_pairs(predicted, target, tolerance)
            row_tp[row] = len(pairs)
            exact[row] = len(pairs) == len(predicted) == len(target)
        exact_by_tolerance[tolerance] = exact
        row_tp_by_tolerance[tolerance] = row_tp
    strata = []
    for cardinality in range(1, 5):
        rows = target_count == cardinality
        strata.append({
            "target_count": cardinality, "observations": int(rows.sum()),
            "count_accuracy": float(np.mean(decoded["count"][rows] == cardinality)),
            **{f"exact_set_fraction_{int(tolerance)}deg": float(exact_by_tolerance[tolerance][rows].mean()) for tolerance in TOLERANCES_DEG},
        })
    return {
        "count_accuracy": float(np.mean(decoded["count"] == target_count)),
        "exact": {f"{int(tolerance)}deg": float(exact_by_tolerance[tolerance].mean()) for tolerance in TOLERANCES_DEG},
        "strata": strata, "exact_mask_2deg": exact_by_tolerance[2.0], "row_tp_2deg": row_tp_by_tolerance[2.0],
    }


def _confidence_features(decoded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rows = len(decoded["count"])
    concentration = np.zeros((rows, 4), dtype=np.float64)
    peak = np.zeros((rows, 4), dtype=np.float64)
    margin = np.zeros((rows, 4), dtype=np.float64)
    entropy = np.zeros((rows, 4), dtype=np.float64)
    angle = np.arange(180, dtype=np.float64) * 2.0 * np.pi / 180.0
    for row, cardinality in enumerate(decoded["count"]):
        for slot in range(cardinality):
            mass = decoded["mass"][row, slot].astype(np.float64)
            concentration[row, slot] = math.hypot(float(np.sum(mass * np.cos(angle))), float(np.sum(mass * np.sin(angle))))
            top = np.partition(mass, -2)[-2:]
            peak[row, slot] = top[-1]
            margin[row, slot] = top[-1] - top[-2]
            entropy[row, slot] = -float(np.sum(mass * np.log(np.clip(mass, 1e-12, None)))) / math.log(180.0)
    count_confidence = decoded["count_probability"].max(axis=1)
    minimum = lambda values: np.asarray([values[row, : decoded["count"][row]].min() for row in range(rows)])
    maximum = lambda values: np.asarray([values[row, : decoded["count"][row]].max() for row in range(rows)])
    features = {
        "count_confidence": count_confidence,
        "minimum_concentration": minimum(concentration),
        "count_times_minimum_concentration": count_confidence * minimum(concentration),
        "minimum_peak_mass": minimum(peak),
        "count_times_minimum_peak_mass": count_confidence * minimum(peak),
        "minimum_peak_margin": minimum(margin),
        "negative_maximum_entropy": -maximum(entropy),
    }
    if "seed_disagreement_deg" in decoded:
        features["negative_seed_disagreement"] = -decoded["seed_disagreement_deg"]
    return features


def _select_safe_threshold(score: np.ndarray, row_tp: np.ndarray, predicted_count: np.ndarray, total_targets: int) -> dict | None:
    order = np.argsort(-score, kind="stable")
    values = score[order]
    cumulative_tp = np.cumsum(row_tp[order])
    cumulative_predictions = np.cumsum(predicted_count[order])
    ends = np.flatnonzero(np.r_[values[1:] != values[:-1], True])
    precision = cumulative_tp[ends] / np.maximum(cumulative_predictions[ends], 1)
    recall = cumulative_tp[ends] / total_targets
    eligible = np.flatnonzero(precision >= PRECISION_FLOOR)
    if not len(eligible):
        return None
    choice = max(eligible, key=lambda index: (float(recall[index]), float(precision[index]), float(values[ends[index]])))
    return {"threshold": float(values[ends[choice]]), "precision": float(precision[choice]), "recall": float(recall[choice]), "accepted_observations": int(ends[choice] + 1)}


def _apply_threshold(score: np.ndarray, threshold: float, scored: dict, decoded: dict[str, np.ndarray], total_targets: int) -> dict:
    accepted = score >= threshold
    tp = int(scored["row_tp_2deg"][accepted].sum())
    predictions = int(decoded["count"][accepted].sum())
    return {"precision": tp / max(predictions, 1), "recall": tp / total_targets, "accepted_observations": int(accepted.sum()), "exact_set_coverage": float((accepted & scored["exact_mask_2deg"]).mean())}


def _matched_error_diagnostics(data: dict[str, np.ndarray], decoded: dict[str, np.ndarray]) -> dict:
    errors = defaultdict(list)
    neighbourhood_mass = defaultdict(list)
    target_count = data["presence"].sum(axis=1).astype(np.int64)
    for row, cardinality in enumerate(decoded["count"]):
        if cardinality != target_count[row]:
            continue
        target = _target_bearings(data, row)
        pairs = _optimal_pairs(decoded["bearing"][row, :cardinality], target)
        for slot, target_index, error in pairs:
            errors[int(cardinality)].append(error)
            continuous_bin = target[target_index] / 2.0
            lower = int(math.floor(continuous_bin)) % 180
            upper = (lower + 1) % 180
            neighbourhood_mass[int(cardinality)].append(float(decoded["mass"][row, slot, lower] + decoded["mass"][row, slot, upper]))
    result = {}
    for cardinality in range(1, 5):
        values = np.asarray(errors[cardinality], dtype=np.float64)
        mass = np.asarray(neighbourhood_mass[cardinality], dtype=np.float64)
        result[str(cardinality)] = {
            "matched_exits": len(values),
            "angular_error_deg_p50_p90_p99": [float(np.percentile(values, q)) for q in (50, 90, 99)] if len(values) else [math.nan] * 3,
            "target_two_bin_mass_p50_p10": [float(np.percentile(mass, q)) for q in (50, 10)] if len(mass) else [math.nan] * 2,
        }
    return result


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    colors = {"c07": "#4e79a7", "c08": "#f28e2b"}
    for split in ("c07", "c08"):
        x = np.arange(3)
        values = [summary[split]["decoders"]["ensemble"][mode]["exact"]["2deg"] for mode in MODES]
        axes[0].plot(x, values, marker="o", label=split.upper(), color=colors[split])
    axes[0].set_xticks(np.arange(3), ("mean", "argmax", "local mode"), rotation=15)
    axes[0].set_ylim(0, 0.5); axes[0].set_ylabel("exact-set fraction at 2°"); axes[0].set_title("A  Decoder attribution"); axes[0].legend(frameon=False)
    for split in ("c07", "c08"):
        strata = summary[split]["decoders"]["ensemble"]["circular_mean"]["strata"]
        axes[1].plot(range(1, 5), [row["exact_set_fraction_10deg"] for row in strata], marker="o", label=split.upper(), color=colors[split])
    axes[1].set_xticks(range(1, 5)); axes[1].set_ylim(0, 1); axes[1].set_xlabel("true exits"); axes[1].set_ylabel("exact-set fraction at 10°"); axes[1].set_title("B  Residual cardinality failure")
    features = summary["confidence_transfer"]
    names = list(features)
    axes[2].bar(np.arange(len(names)) - 0.18, [features[name]["c07"]["recall"] for name in names], 0.36, label="C07")
    axes[2].bar(np.arange(len(names)) + 0.18, [features[name]["c08"]["recall"] for name in names], 0.36, label="C08")
    axes[2].set_xticks(np.arange(len(names)), [name.replace("minimum_", "min ").replace("count_times_", "count×") for name in names], rotation=55, ha="right")
    axes[2].set_ylabel("exit recall at precision ≥0.995"); axes[2].set_title("C  Frozen confidence transfer"); axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25); axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph slot-transport frozen failure attribution")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.prediction_root) != 3:
        raise RuntimeError("failure attribution requires exactly three frozen seeds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data_by_split = {suffix.lower(): _load_split(suffix, args.teacher_root.resolve(), [path.resolve() for path in args.prediction_root]) for suffix in ("C07", "C08")}
    summary: dict = {
        "schema_version": "gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1",
        "status": PASS, "scientific_pass": True, "precision_floor": PRECISION_FLOOR,
        "optimizer_steps": 0, "new_model_inference_observations": 0, "c09_worlds_read": 0,
        "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    decoded_by_split = {}
    scored_by_split = {}
    for split, data in data_by_split.items():
        decoders = {f"seed{seed}": {mode: _decode_seed(data, seed, mode) for mode in MODES} for seed in range(3)}
        decoders["ensemble"] = {mode: _decode_ensemble(data, mode) for mode in MODES}
        oracle = _decode_ensemble(data, "circular_mean", oracle_count=True)
        decoded_by_split[split] = decoders
        metrics = {name: {mode: _score_decode(data, decoded) for mode, decoded in modes.items()} for name, modes in decoders.items()}
        oracle_metrics = _score_decode(data, oracle)
        scored_by_split[split] = metrics
        clean_metrics = {}
        for name, modes in metrics.items():
            clean_metrics[name] = {}
            for mode, values in modes.items():
                clean_metrics[name][mode] = {key: value for key, value in values.items() if key not in ("exact_mask_2deg", "row_tp_2deg")}
        summary[split] = {
            "observations": len(data["presence"]), "visible_exits": int(data["presence"].sum()),
            "decoders": clean_metrics,
            "oracle_count_ensemble": {key: value for key, value in oracle_metrics.items() if key not in ("exact_mask_2deg", "row_tp_2deg")},
            "matched_error": _matched_error_diagnostics(data, decoders["ensemble"]["circular_mean"]),
        }

    c07_decoded = decoded_by_split["c07"]["ensemble"]["circular_mean"]
    c08_decoded = decoded_by_split["c08"]["ensemble"]["circular_mean"]
    c07_scored = scored_by_split["c07"]["ensemble"]["circular_mean"]
    c08_scored = scored_by_split["c08"]["ensemble"]["circular_mean"]
    c07_features = _confidence_features(c07_decoded)
    c08_features = _confidence_features(c08_decoded)
    confidence_transfer = {}
    for name in c07_features:
        choice = _select_safe_threshold(c07_features[name], c07_scored["row_tp_2deg"], c07_decoded["count"], int(data_by_split["c07"]["presence"].sum()))
        if choice is None:
            confidence_transfer[name] = {"threshold_choice": None, "auc_exact_set": binary_auc(c07_features[name], c07_scored["exact_mask_2deg"]), "c07": {"precision": 0.0, "recall": 0.0, "accepted_observations": 0, "exact_set_coverage": 0.0}, "c08": {"precision": 0.0, "recall": 0.0, "accepted_observations": 0, "exact_set_coverage": 0.0}}
            continue
        confidence_transfer[name] = {
            "threshold_choice": choice, "auc_exact_set": binary_auc(c07_features[name], c07_scored["exact_mask_2deg"]),
            "c07": _apply_threshold(c07_features[name], choice["threshold"], c07_scored, c07_decoded, int(data_by_split["c07"]["presence"].sum())),
            "c08": _apply_threshold(c08_features[name], choice["threshold"], c08_scored, c08_decoded, int(data_by_split["c08"]["presence"].sum())),
        }
    summary["confidence_transfer"] = confidence_transfer

    ensemble_2 = {split: summary[split]["decoders"]["ensemble"]["circular_mean"]["exact"]["2deg"] for split in ("c07", "c08")}
    best_single_2 = {split: max(summary[split]["decoders"][f"seed{seed}"]["circular_mean"]["exact"]["2deg"] for seed in range(3)) for split in ("c07", "c08")}
    mode_best_2 = {split: max(summary[split]["decoders"]["ensemble"][mode]["exact"]["2deg"] for mode in ("argmax", "local_mode")) for split in ("c07", "c08")}
    oracle_2 = {split: summary[split]["oracle_count_ensemble"]["exact"]["2deg"] for split in ("c07", "c08")}
    best_safe = max(confidence_transfer, key=lambda name: confidence_transfer[name]["c07"]["recall"])
    checks = {
        "population_exact": summary["c07"]["observations"] == 21548 and summary["c08"]["observations"] == 24394 and summary["c07"]["visible_exits"] == 45504 and summary["c08"]["visible_exits"] == 51537,
        "reproduces_formal_ensemble": abs(ensemble_2["c07"] - 0.3412845739743828) <= 1e-12 and abs(ensemble_2["c08"] - 0.27777322292366974) <= 1e-12,
        "all_tolerance_curves_monotonic": all(all(values["exact"][f"{a}deg"] <= values["exact"][f"{b}deg"] for a, b in ((2, 4), (4, 10))) for split in ("c07", "c08") for modes in summary[split]["decoders"].values() for values in modes.values()),
        "zero_forbidden_operations": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"failure-attribution invariant failed: {checks}")
    causes = {
        "circular_mean_decoder_primary": all(mode_best_2[split] >= ensemble_2[split] + 0.03 for split in ("c07", "c08")),
        "ensemble_alignment_primary": all(best_single_2[split] >= ensemble_2[split] + 0.03 for split in ("c07", "c08")),
        "cardinality_coupling_primary": all(oracle_2[split] >= ensemble_2[split] + 0.03 for split in ("c07", "c08")),
        "confidence_posthoc_sufficient": confidence_transfer[best_safe]["c07"]["recall"] >= 0.50 and confidence_transfer[best_safe]["c08"]["recall"] >= 0.50,
    }
    decision = "STRICT_SLOT_DISTRIBUTION_LOCALIZATION_AND_CONFIDENCE_FAILURE"
    if causes["circular_mean_decoder_primary"]:
        decision = "LOCAL_MODE_DECODER_CORRECTIVE_SUPPORTED"
    elif causes["ensemble_alignment_primary"]:
        decision = "ENSEMBLE_ALIGNMENT_CORRECTIVE_SUPPORTED"
    elif causes["cardinality_coupling_primary"]:
        decision = "CARDINALITY_COUPLING_CORRECTIVE_SUPPORTED"
    summary.update({
        "decision": decision, "causes": causes, "best_safe_confidence": best_safe,
        "comparison": {"ensemble_exact_2deg": ensemble_2, "best_single_exact_2deg": best_single_2, "best_mode_exact_2deg": mode_best_2, "oracle_count_exact_2deg": oracle_2},
        "checks": checks, "duration_seconds": time.monotonic() - started,
    })
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", summary)
    with (output / "decoder_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "model", "mode", "tolerance_deg", "exact_set_fraction"))
        writer.writeheader()
        for split in ("c07", "c08"):
            for model, modes in summary[split]["decoders"].items():
                for mode, values in modes.items():
                    for tolerance in TOLERANCES_DEG:
                        writer.writerow({"split": split.upper(), "model": model, "mode": mode, "tolerance_deg": tolerance, "exact_set_fraction": values["exact"][f"{int(tolerance)}deg"]})
    _plot(output, summary)
    print(json.dumps({"status": PASS, "decision": decision, "comparison": summary["comparison"], "best_safe_confidence": best_safe, "best_safe": confidence_transfer[best_safe]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
