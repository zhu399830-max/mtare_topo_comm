#!/usr/bin/env python3
"""Attribute COUST K>=3 collapse and confidence-transfer failure from frozen outputs."""
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

import execute_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1 as base
from evaluate_gse_cyclic_ordered_unimodal_slot_transport_selection_v1 import _align_cyclic_order
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import BRANCH_SLICE
from mtare_topo.representation.gse_cyclic_ordered_unimodal_slot_transport import cyclic_orders


PASS = "PASS_GSE_COUST_COMPLEX_CARDINALITY_FAILURE_ATTRIBUTION_V1"


def circular_pair_gaps(bearings: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.remainder(np.asarray(bearings, dtype=np.float64), 360.0))
    if len(ordered) < 2:
        return np.asarray([360.0])
    return np.diff(np.r_[ordered, ordered[0] + 360.0])


def cyclic_assignment_diagnostic(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | bool | list[int]]:
    cardinality = len(predicted)
    costs = np.asarray([[base.angular_error(float(p), float(t)) for t in target] for p in predicted])
    all_orders = list(permutations(range(cardinality)))
    cyclic = set(cyclic_orders(cardinality))
    totals = np.asarray([sum(costs[slot, order[slot]] for slot in range(cardinality)) for order in all_orders])
    best_index = int(np.argmin(totals))
    best_order = all_orders[best_index]
    cyclic_totals = [sum(costs[slot, order[slot]] for slot in range(cardinality)) for order in cyclic]
    return {
        "unrestricted_mean_error_deg": float(totals[best_index] / cardinality),
        "cyclic_mean_error_deg": float(min(cyclic_totals) / cardinality),
        "unrestricted_is_cyclic": best_order in cyclic,
        "unrestricted_order": list(best_order),
    }


def _load_split(suffix: str, teacher_root: Path, prediction_roots: list[Path]) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    for teacher_path in sorted(teacher_root.glob(f"selection/*_{suffix}.zarr")):
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = str(teacher.attrs["parent_id"])
        sequence = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
        predictions = [np.load(root / f"{parent}.npz") for root in prediction_roots]
        if any(not np.array_equal(sequence, item["global_sequence_index"]) for item in predictions):
            raise RuntimeError(f"COUST attribution join drift: {parent}")
        output["parent_id"].append(np.full(len(sequence), parent, dtype=f"U{len(parent)}"))
        output["global_sequence_index"].append(sequence)
        output["presence"].append(np.asarray(teacher["presence"][:], dtype=np.uint8))
        output["heading_target"].append(np.asarray(teacher["heading_residual_deg"][:], dtype=np.float32))
        for seed, prediction in enumerate(predictions):
            for name in ("slot_mass", "slot_bearing_deg", "slot_concentration", "slot_kappa", "exit_count_probability"):
                output[f"seed{seed}_{name}"].append(np.asarray(prediction[name], dtype=np.float32))
    if len(output["presence"]) != 10:
        raise RuntimeError(f"COUST attribution {suffix} world count drift")
    return {name: np.concatenate(parts) for name, parts in output.items()}


def _decode_ensemble(data: dict[str, np.ndarray], *, unrestricted: bool) -> dict[str, np.ndarray]:
    count_probability = np.mean([data[f"seed{seed}_exit_count_probability"] for seed in range(3)], axis=0)
    count = np.argmax(count_probability, axis=1).astype(np.int64) + 1
    observations = len(count)
    bearing = np.zeros((observations, 4), dtype=np.float64)
    masses = np.zeros((observations, 4, 180), dtype=np.float32)
    kappa = np.zeros((observations, 4), dtype=np.float64)
    concentration = np.zeros((observations, 4), dtype=np.float64)
    alignment_disagreement = np.zeros(observations, dtype=np.float64)
    for row, cardinality in enumerate(count):
        branch = np.arange(BRANCH_SLICE[int(cardinality)].start, BRANCH_SLICE[int(cardinality)].stop)
        reference = data["seed0_slot_bearing_deg"][row, branch]
        aligned = []
        aligned_bearings = []
        for seed in range(3):
            candidate = data[f"seed{seed}_slot_bearing_deg"][row, branch]
            if seed == 0:
                order = np.arange(cardinality)
            elif unrestricted:
                order = np.asarray(base._align_order(reference, candidate))
            else:
                order = _align_cyclic_order(reference, candidate)
            aligned.append(branch[order])
            aligned_bearings.append(candidate[order])
        for slot in range(cardinality):
            mass = np.mean([data[f"seed{seed}_slot_mass"][row, aligned[seed][slot]] for seed in range(3)], axis=0).astype(np.float64)
            mass /= mass.sum()
            masses[row, slot] = mass
            bearing[row, slot] = base.circular_bearing(mass)
            kappa[row, slot] = float(np.mean([data[f"seed{seed}_slot_kappa"][row, aligned[seed][slot]] for seed in range(3)]))
            concentration[row, slot] = float(np.mean([data[f"seed{seed}_slot_concentration"][row, aligned[seed][slot]] for seed in range(3)]))
            alignment_disagreement[row] = max(
                alignment_disagreement[row],
                max(base.angular_error(float(aligned_bearings[a][slot]), float(aligned_bearings[b][slot])) for a, b in combinations(range(3), 2)),
            )
    return {
        "count": count, "bearing": bearing, "mass": masses, "count_probability": count_probability,
        "slot_kappa": kappa, "slot_concentration": concentration,
        "seed_disagreement_deg": alignment_disagreement,
    }


def _decode_seed(data: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    decoded = base._decode_seed(data, seed, "circular_mean")
    observations = len(decoded["count"])
    kappa = np.zeros((observations, 4), dtype=np.float64)
    concentration = np.zeros((observations, 4), dtype=np.float64)
    for row, cardinality in enumerate(decoded["count"]):
        branch = np.arange(BRANCH_SLICE[int(cardinality)].start, BRANCH_SLICE[int(cardinality)].stop)
        kappa[row, :cardinality] = data[f"seed{seed}_slot_kappa"][row, branch]
        concentration[row, :cardinality] = data[f"seed{seed}_slot_concentration"][row, branch]
    decoded.update({"slot_kappa": kappa, "slot_concentration": concentration})
    return decoded


def _clean_score(scored: dict) -> dict:
    return {key: value for key, value in scored.items() if key not in ("exact_mask_2deg", "row_tp_2deg")}


def _complex_diagnostics(data: dict[str, np.ndarray], decoded: dict[str, np.ndarray]) -> dict:
    target_count = data["presence"].sum(axis=1).astype(np.int64)
    records = {}
    for cardinality in (3, 4):
        eligible = np.flatnonzero((target_count == cardinality) & (decoded["count"] == cardinality))
        ratios, predicted_gap, target_gap, cyclic_excess, cyclic_ok = [], [], [], [], []
        for row in eligible:
            predicted = decoded["bearing"][row, :cardinality]
            target = base._target_bearings(data, int(row))
            pred_min = float(circular_pair_gaps(predicted).min())
            true_min = float(circular_pair_gaps(target).min())
            assignment = cyclic_assignment_diagnostic(predicted, target)
            predicted_gap.append(pred_min); target_gap.append(true_min); ratios.append(pred_min / max(true_min, 1e-9))
            cyclic_excess.append(float(assignment["cyclic_mean_error_deg"] - assignment["unrestricted_mean_error_deg"]))
            cyclic_ok.append(bool(assignment["unrestricted_is_cyclic"]))
        records[str(cardinality)] = {
            "correct_count_rows": len(eligible),
            "predicted_min_gap_deg_p10_p50_p90": [float(np.percentile(predicted_gap, q)) for q in (10, 50, 90)] if predicted_gap else [math.nan] * 3,
            "target_min_gap_deg_p10_p50_p90": [float(np.percentile(target_gap, q)) for q in (10, 50, 90)] if target_gap else [math.nan] * 3,
            "predicted_to_target_min_gap_ratio_p10_p50_p90": [float(np.percentile(ratios, q)) for q in (10, 50, 90)] if ratios else [math.nan] * 3,
            "collapse_fraction_predicted_gap_below_half_target": float(np.mean(np.asarray(ratios) < 0.5)) if ratios else math.nan,
            "unrestricted_best_assignment_is_cyclic_fraction": float(np.mean(cyclic_ok)) if cyclic_ok else math.nan,
            "cyclic_assignment_excess_error_deg_p50_p90": [float(np.percentile(cyclic_excess, q)) for q in (50, 90)] if cyclic_excess else [math.nan] * 2,
        }
    return records


def _confidence_diagnostics(data: dict[str, np.ndarray], decoded: dict[str, np.ndarray], scored: dict) -> dict:
    target_count = data["presence"].sum(axis=1).astype(np.int64)
    minimum_kappa = np.asarray([decoded["slot_kappa"][row, :decoded["count"][row]].min() for row in range(len(target_count))])
    minimum_concentration = np.asarray([decoded["slot_concentration"][row, :decoded["count"][row]].min() for row in range(len(target_count))])
    exact = scored["exact_mask_2deg"]
    output = {}
    for name, values in (("minimum_kappa", minimum_kappa), ("minimum_concentration", minimum_concentration)):
        output[name] = {
            "auc_exact_set": base.binary_auc(values, exact),
            "median_exact": float(np.median(values[exact])) if exact.any() else math.nan,
            "median_incorrect": float(np.median(values[~exact])) if (~exact).any() else math.nan,
            "by_cardinality": {
                str(cardinality): {
                    "median": float(np.median(values[target_count == cardinality])),
                    "auc_exact_set": base.binary_auc(values[target_count == cardinality], exact[target_count == cardinality]),
                } for cardinality in range(1, 5)
            },
        }
    return output


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    methods = ("seed0", "seed1", "seed2", "cyclic_ensemble", "unrestricted_ensemble")
    x = np.arange(len(methods))
    for offset, cardinality in ((-0.18, 3), (0.18, 4)):
        values = [next(row for row in summary["c07"][name]["score"]["strata"] if row["target_count"] == cardinality)["exact_set_fraction_2deg"] for name in methods]
        axes[0].bar(x + offset, values, .36, label=f"K={cardinality}")
    axes[0].set_xticks(x, ("s0", "s1", "s2", "cyclic ens", "free ens"), rotation=20); axes[0].set_ylabel("exact-set at 2°"); axes[0].set_title("A  Complex-set recovery"); axes[0].legend(frameon=False)
    for split, color in (("c07", "#4e79a7"), ("c08", "#f28e2b")):
        values = [summary[split]["cyclic_ensemble"]["complex"][str(cardinality)]["predicted_to_target_min_gap_ratio_p10_p50_p90"][1] for cardinality in (3, 4)]
        axes[1].plot((3, 4), values, marker="o", label=split.upper(), color=color)
    axes[1].axhline(1, color="#59a14f", linestyle="--"); axes[1].axhline(.5, color="#e15759", linestyle=":")
    axes[1].set_xticks((3, 4)); axes[1].set(xlabel="true exits", ylabel="median predicted/target min gap", title="B  Slot-spacing collapse"); axes[1].legend(frameon=False)
    confidence_names = ("minimum_kappa", "minimum_concentration")
    axes[2].bar(np.arange(2)-.18, [summary["c07"]["cyclic_ensemble"]["confidence"][name]["auc_exact_set"] for name in confidence_names], .36, label="C07")
    axes[2].bar(np.arange(2)+.18, [summary["c08"]["cyclic_ensemble"]["confidence"][name]["auc_exact_set"] for name in confidence_names], .36, label="C08")
    axes[2].axhline(.5, color="#e15759", linestyle="--"); axes[2].set_xticks(np.arange(2), ("kappa", "concentration")); axes[2].set_ylim(0,1); axes[2].set(ylabel="AUC of exact set", title="C  Confidence separability"); axes[2].legend(frameon=False)
    for axis in axes: axis.grid(axis="y", alpha=.25); axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph COUST frozen complex-cardinality attribution")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_coust_complex_cardinality_failure_attribution_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.prediction_root) != 3:
        raise RuntimeError("COUST attribution requires three frozen seeds")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {
        "schema_version": "gse_coust_complex_cardinality_failure_attribution_v1",
        "status": PASS, "scientific_pass": True,
        "optimizer_steps": 0, "new_model_inference_observations": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "graph_replays": 0, "planner_calls": 0,
    }
    rows_csv = []
    for suffix in ("C07", "C08"):
        split = suffix.lower()
        data = _load_split(suffix, args.teacher_root.resolve(), [path.resolve() for path in args.prediction_root])
        decoders = {f"seed{seed}": _decode_seed(data, seed) for seed in range(3)}
        decoders["cyclic_ensemble"] = _decode_ensemble(data, unrestricted=False)
        decoders["unrestricted_ensemble"] = _decode_ensemble(data, unrestricted=True)
        result = {"observations": len(data["presence"]), "visible_exits": int(data["presence"].sum())}
        for name, decoded in decoders.items():
            scored = base._score_decode(data, decoded)
            result[name] = {
                "score": _clean_score(scored),
                "complex": _complex_diagnostics(data, decoded),
                "confidence": _confidence_diagnostics(data, decoded, scored),
            }
            for cardinality in (3, 4):
                stratum = next(row for row in result[name]["score"]["strata"] if row["target_count"] == cardinality)
                complex_record = result[name]["complex"][str(cardinality)]
                rows_csv.append({
                    "split": suffix, "decoder": name, "cardinality": cardinality,
                    "exact_set_fraction_2deg": stratum["exact_set_fraction_2deg"],
                    "exact_set_fraction_10deg": stratum["exact_set_fraction_10deg"],
                    "min_gap_ratio_median": complex_record["predicted_to_target_min_gap_ratio_p10_p50_p90"][1],
                    "collapse_fraction": complex_record["collapse_fraction_predicted_gap_below_half_target"],
                    "cyclic_optimal_fraction": complex_record["unrestricted_best_assignment_is_cyclic_fraction"],
                })
        summary[split] = result
    def stratum(split, name, cardinality):
        return next(row for row in summary[split][name]["score"]["strata"] if row["target_count"] == cardinality)
    ensemble_gain = {
        split: {
            str(cardinality): stratum(split, "unrestricted_ensemble", cardinality)["exact_set_fraction_2deg"] - stratum(split, "cyclic_ensemble", cardinality)["exact_set_fraction_2deg"]
            for cardinality in (3, 4)
        } for split in ("c07", "c08")
    }
    best_single = {
        split: {
            str(cardinality): max(stratum(split, f"seed{seed}", cardinality)["exact_set_fraction_2deg"] for seed in range(3))
            for cardinality in (3, 4)
        } for split in ("c07", "c08")
    }
    collapse = {
        split: {str(cardinality): summary[split]["cyclic_ensemble"]["complex"][str(cardinality)]["collapse_fraction_predicted_gap_below_half_target"] for cardinality in (3,4)}
        for split in ("c07", "c08")
    }
    summary["cross_decoder_attribution"] = {
        "unrestricted_minus_cyclic_exact2_gain": ensemble_gain,
        "best_single_exact2": best_single,
        "cyclic_ensemble_collapse_fraction": collapse,
    }
    alignment_primary = all(ensemble_gain[split]["3"] >= .03 for split in ("c07", "c08"))
    single_seed_primary = all(best_single[split]["3"] - stratum(split, "cyclic_ensemble", 3)["exact_set_fraction_2deg"] >= .03 for split in ("c07", "c08"))
    representation_collapse = all(collapse[split]["3"] >= .25 for split in ("c07", "c08"))
    summary["decision_tests"] = {
        "ensemble_cyclic_alignment_primary": alignment_primary,
        "single_seed_ensemble_interference_primary": single_seed_primary,
        "k3_representation_spacing_collapse": representation_collapse,
    }
    if alignment_primary:
        decision = "COUST_ENSEMBLE_CYCLIC_ALIGNMENT_FAILURE"
    elif single_seed_primary:
        decision = "COUST_CROSS_SEED_INTERFERENCE_FAILURE"
    elif representation_collapse:
        decision = "COUST_PER_SEED_K_GE_3_SLOT_SPACING_COLLAPSE"
    else:
        decision = "COUST_K_GE_3_LOCALIZATION_AND_CONFIDENCE_OBJECTIVE_FAILURE"
    summary["decision"] = decision
    summary["duration_seconds"] = time.monotonic() - started
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    with (output / "complex_cardinality_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows_csv[0])); writer.writeheader(); writer.writerows(rows_csv)
    _plot(output, summary)
    print(json.dumps({"status": PASS, "decision": decision, "decision_tests": summary["decision_tests"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
