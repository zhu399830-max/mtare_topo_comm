#!/usr/bin/env python3
"""Read-only attribution of the sealed JCGS C07/C08 predictions.

This audit separates four possible causes that are inseparable in the formal
complete-set score: cardinality, cyclic seed alignment, phase, and gap shape.
No model is loaded and no new inference is performed.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

import evaluate_gse_cardinality_conditioned_circular_slot_transport_selection_v1 as base
import evaluate_gse_joint_cyclic_gap_simplex_selection_v1 as formal
from evaluate_gse_cyclic_ordered_unimodal_slot_transport_selection_v1 import _align_cyclic_order
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import BRANCH_SLICE


PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_FAILURE_ATTRIBUTION_V1"
TOLERANCES = (2.0, 4.0, 10.0)
PRECISION_FLOOR = 0.995


def angular_error(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def binary_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney AUC without an sklearn dependency."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
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


def _load_split(suffix: str, teacher_root: Path, prediction_roots: list[Path]) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    for teacher_path in sorted(teacher_root.glob(f"selection/*_{suffix}.zarr")):
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = str(teacher.attrs["parent_id"])
        sequence = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
        predictions = [np.load(root / f"{parent}.npz") for root in prediction_roots]
        if any(not np.array_equal(sequence, item["global_sequence_index"]) for item in predictions):
            raise RuntimeError(f"JCGS attribution join drift: {parent}")
        values = {
            "parent_id": np.full(len(sequence), parent, dtype=f"U{len(parent)}"),
            "global_sequence_index": sequence,
            "presence": np.asarray(teacher["presence"][:], dtype=np.uint8),
            "heading_target": np.asarray(teacher["heading_residual_deg"][:], dtype=np.float32),
            "count_probability": np.mean([
                np.asarray(item["exit_count_probability"], dtype=np.float32) for item in predictions
            ], axis=0),
        }
        for seed, item in enumerate(predictions):
            for name in (
                "joint_bearing_deg", "joint_bearing_scale_deg", "phase_concentration",
                "gap_fraction", "exit_count_probability", "exit_opening_width_m",
                "exit_vertical_profile_m",
            ):
                values[f"seed{seed}_{name}"] = np.asarray(item[name], dtype=np.float32)
        for name, value in values.items():
            output[name].append(value)
    if len(output["presence"]) != 10:
        raise RuntimeError(f"JCGS attribution {suffix} world-count drift")
    return {name: np.concatenate(parts) for name, parts in output.items()}


def _target_bearings(data: dict[str, np.ndarray], row: int) -> np.ndarray:
    bins = np.flatnonzero(data["presence"][row])
    values = np.remainder(bins * 2.0 + data["heading_target"][row, bins], 360.0)
    return np.sort(values.astype(np.float64))


def _cyclic_gaps(bearings: np.ndarray) -> np.ndarray:
    bearings = np.asarray(bearings, dtype=np.float64)
    if len(bearings) == 1:
        return np.asarray([360.0])
    return np.remainder(np.roll(bearings, -1) - bearings, 360.0)


def _decode_seed(data: dict[str, np.ndarray], seed: int, *, oracle_count: bool) -> dict[str, np.ndarray]:
    count_probability = data[f"seed{seed}_exit_count_probability"]
    count = data["presence"].sum(axis=1).astype(np.int64) if oracle_count else np.argmax(count_probability, axis=1).astype(np.int64) + 1
    rows = len(count)
    bearing = np.zeros((rows, 4), dtype=np.float64)
    gap = np.zeros((rows, 4), dtype=np.float64)
    scale = np.zeros((rows, 4), dtype=np.float64)
    phase_concentration = np.zeros(rows, dtype=np.float64)
    for row, cardinality in enumerate(count):
        branch = np.arange(BRANCH_SLICE[int(cardinality)].start, BRANCH_SLICE[int(cardinality)].stop)
        bearing[row, :cardinality] = data[f"seed{seed}_joint_bearing_deg"][row, branch]
        gap[row, :cardinality] = data[f"seed{seed}_gap_fraction"][row, branch] * 360.0
        scale[row, :cardinality] = data[f"seed{seed}_joint_bearing_scale_deg"][row, branch]
        phase_concentration[row] = data[f"seed{seed}_phase_concentration"][row, cardinality - 1]
    score = count_probability.max(axis=1) * phase_concentration * np.exp(-np.asarray([
        scale[row, :count[row]].max() for row in range(rows)
    ]) / 180.0)
    return {"count": count, "bearing": bearing, "gap": gap, "bearing_scale": scale,
            "phase_concentration": phase_concentration, "count_probability": count_probability, "score": score}


def _ensemble_for_count(data: dict[str, np.ndarray], count: np.ndarray, *, target_aligned: bool) -> dict[str, np.ndarray]:
    rows = len(count)
    bearing = np.zeros((rows, 4), dtype=np.float64)
    gap = np.zeros((rows, 4), dtype=np.float64)
    scale = np.zeros((rows, 4), dtype=np.float64)
    phase_concentration = np.zeros(rows, dtype=np.float64)
    count_probability = data["count_probability"]
    for row, cardinality_value in enumerate(count):
        cardinality = int(cardinality_value)
        branch = np.arange(BRANCH_SLICE[cardinality].start, BRANCH_SLICE[cardinality].stop)
        reference = _target_bearings(data, row) if target_aligned else data["seed0_joint_bearing_deg"][row, branch]
        aligned: list[np.ndarray] = []
        for seed in range(3):
            candidate = data[f"seed{seed}_joint_bearing_deg"][row, branch]
            if seed == 0 and not target_aligned:
                order = np.arange(cardinality)
            else:
                order = np.asarray(_align_cyclic_order(reference, candidate), dtype=np.int64)
            aligned.append(branch[order])
        for slot in range(cardinality):
            angles = np.deg2rad([data[f"seed{seed}_joint_bearing_deg"][row, aligned[seed][slot]] for seed in range(3)])
            bearing[row, slot] = np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))) % 360.0
            gap[row, slot] = np.mean([data[f"seed{seed}_gap_fraction"][row, aligned[seed][slot]] for seed in range(3)]) * 360.0
            scale[row, slot] = np.mean([data[f"seed{seed}_joint_bearing_scale_deg"][row, aligned[seed][slot]] for seed in range(3)])
        phase_concentration[row] = np.mean([
            data[f"seed{seed}_phase_concentration"][row, cardinality - 1] for seed in range(3)
        ])
    score = count_probability.max(axis=1) * phase_concentration * np.exp(-np.asarray([
        scale[row, :count[row]].max() for row in range(rows)
    ]) / 180.0)
    return {"count": count.copy(), "bearing": bearing, "gap": gap, "bearing_scale": scale,
            "phase_concentration": phase_concentration, "count_probability": count_probability, "score": score}


def _score(data: dict[str, np.ndarray], decoded: dict[str, np.ndarray]) -> tuple[dict, dict[float, np.ndarray], dict[float, np.ndarray]]:
    target_count = data["presence"].sum(axis=1).astype(np.int64)
    exact_masks: dict[float, np.ndarray] = {}
    tp_rows: dict[float, np.ndarray] = {}
    for tolerance in TOLERANCES:
        matched = base._match(data, decoded, tolerance)
        exact_masks[tolerance] = matched["exact"]
        tp_rows[tolerance] = matched["row_tp"]
    strata = []
    for cardinality in range(1, 5):
        mask = target_count == cardinality
        strata.append({
            "target_count": cardinality,
            "observations": int(mask.sum()),
            "count_accuracy": float(np.mean(decoded["count"][mask] == cardinality)),
            **{f"exact_set_fraction_{int(t)}deg": float(exact_masks[t][mask].mean()) for t in TOLERANCES},
        })
    complex_mask = target_count >= 3
    metrics = {
        "count_accuracy": float(np.mean(decoded["count"] == target_count)),
        "exact": {f"{int(t)}deg": float(exact_masks[t].mean()) for t in TOLERANCES},
        "complex_exact": {f"{int(t)}deg": float(exact_masks[t][complex_mask].mean()) for t in TOLERANCES},
        "strata": strata,
    }
    return metrics, exact_masks, tp_rows


def _confusion(target: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for truth, estimate in zip(target, predicted, strict=True):
        matrix[int(truth) - 1, int(estimate) - 1] += 1
    return matrix.tolist()


def _is_exact(predicted: np.ndarray, target: np.ndarray, tolerance: float) -> bool:
    pairs = base._optimal_pairs(predicted, target, tolerance)
    return len(pairs) == len(predicted) == len(target)


def _component_diagnostics(data: dict[str, np.ndarray], decoded: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    """Use true count and separate phase error from cyclic gap-shape error."""
    target_count = data["presence"].sum(axis=1).astype(np.int64)
    if not np.array_equal(decoded["count"], target_count):
        raise ValueError("component attribution requires oracle count")
    phase_error = np.zeros(len(target_count), dtype=np.float64)
    gap_mae = np.zeros(len(target_count), dtype=np.float64)
    min_gap_ratio = np.zeros(len(target_count), dtype=np.float64)
    gap_closure_error = np.zeros(len(target_count), dtype=np.float64)
    oracle_phase_exact = {t: np.zeros(len(target_count), dtype=bool) for t in TOLERANCES}
    oracle_gaps_exact = {t: np.zeros(len(target_count), dtype=bool) for t in TOLERANCES}
    for row, cardinality_value in enumerate(target_count):
        cardinality = int(cardinality_value)
        predicted = decoded["bearing"][row, :cardinality]
        target = _target_bearings(data, row)
        predicted_gaps = _cyclic_gaps(predicted)
        candidates = []
        for shift in range(cardinality):
            ordered_target = np.roll(target, -shift)
            target_gaps = _cyclic_gaps(ordered_target)
            candidates.append((float(np.mean(np.abs(predicted_gaps - target_gaps))), angular_error(float(predicted[0]), float(ordered_target[0])), ordered_target, target_gaps))
        _, phase_value, ordered_target, target_gaps = min(candidates, key=lambda item: (item[0], item[1]))
        phase_error[row] = phase_value
        gap_mae[row] = np.mean(np.abs(predicted_gaps - target_gaps))
        min_gap_ratio[row] = float(predicted_gaps.min() / max(float(target_gaps.min()), 1e-9))
        gap_closure_error[row] = abs(float(decoded["gap"][row, :cardinality].sum()) - 360.0)
        predicted_prefix = np.r_[0.0, np.cumsum(predicted_gaps[:-1])]
        target_prefix = np.r_[0.0, np.cumsum(target_gaps[:-1])]
        oracle_phase = np.remainder(ordered_target[0] + predicted_prefix, 360.0)
        oracle_gaps = np.remainder(predicted[0] + target_prefix, 360.0)
        for tolerance in TOLERANCES:
            oracle_phase_exact[tolerance][row] = _is_exact(oracle_phase, target, tolerance)
            oracle_gaps_exact[tolerance][row] = _is_exact(oracle_gaps, target, tolerance)
    strata = []
    for cardinality in range(1, 5):
        mask = target_count == cardinality
        percentile = lambda values, quantiles: [float(np.percentile(values, q)) for q in quantiles] if len(values) else [math.nan for _ in quantiles]
        strata.append({
            "target_count": cardinality, "observations": int(mask.sum()),
            "phase_error_deg_p50_p90": percentile(phase_error[mask], (50, 90)),
            "gap_mae_deg_p50_p90": percentile(gap_mae[mask], (50, 90)),
            "minimum_gap_ratio_p10_p50": percentile(min_gap_ratio[mask], (10, 50)),
            "exported_gap_closure_error_deg_max": float(gap_closure_error[mask].max()) if mask.any() else math.nan,
            **{f"oracle_phase_exact_{int(t)}deg": float(oracle_phase_exact[t][mask].mean()) if mask.any() else math.nan for t in TOLERANCES},
            **{f"oracle_gaps_exact_{int(t)}deg": float(oracle_gaps_exact[t][mask].mean()) if mask.any() else math.nan for t in TOLERANCES},
        })
    complex_mask = target_count >= 3
    summary = {
        "meaning": {
            "oracle_phase": "replace predicted phase with target phase; remaining failure is gap shape",
            "oracle_gaps": "replace predicted gaps with target gaps; remaining failure is phase",
        },
        "phase_error_deg_p50_p90": [float(np.percentile(phase_error, q)) for q in (50, 90)],
        "gap_mae_deg_p50_p90": [float(np.percentile(gap_mae, q)) for q in (50, 90)],
        "oracle_phase_exact": {f"{int(t)}deg": float(oracle_phase_exact[t].mean()) for t in TOLERANCES},
        "oracle_gaps_exact": {f"{int(t)}deg": float(oracle_gaps_exact[t].mean()) for t in TOLERANCES},
        "complex_oracle_phase_exact": {f"{int(t)}deg": float(oracle_phase_exact[t][complex_mask].mean()) for t in TOLERANCES},
        "complex_oracle_gaps_exact": {f"{int(t)}deg": float(oracle_gaps_exact[t][complex_mask].mean()) for t in TOLERANCES},
        "strata": strata,
    }
    arrays = {"phase_error": phase_error, "gap_mae": gap_mae, "min_gap_ratio": min_gap_ratio,
              **{f"oracle_phase_{int(t)}": v for t, v in oracle_phase_exact.items()},
              **{f"oracle_gaps_{int(t)}": v for t, v in oracle_gaps_exact.items()}}
    return summary, arrays


def _apply_threshold(score: np.ndarray, threshold: float, decoded: dict[str, np.ndarray], tp_rows: np.ndarray, total_targets: int) -> dict:
    accepted = score >= threshold
    predictions = int(decoded["count"][accepted].sum())
    true_positives = int(tp_rows[accepted].sum())
    return {
        "precision": true_positives / max(predictions, 1),
        "recall": true_positives / total_targets,
        "accepted_observations": int(accepted.sum()),
    }


def _confidence(split_data: dict[str, np.ndarray], decoded: dict[str, np.ndarray], exact2: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    target_count = split_data["presence"].sum(axis=1).astype(np.int64)
    max_scale = np.asarray([decoded["bearing_scale"][row, :decoded["count"][row]].max() for row in range(len(target_count))])
    count_confidence = decoded["count_probability"].max(axis=1)
    features = {
        "count_confidence": count_confidence,
        "phase_concentration": decoded["phase_concentration"],
        "negative_max_scale": -max_scale,
        "count_times_phase": count_confidence * decoded["phase_concentration"],
        "formal_set_score": decoded["score"],
    }
    diagnostics = {name: {
        "auc_exact2": binary_auc(value, exact2),
        "auc_count_correct": binary_auc(value, decoded["count"] == target_count),
    } for name, value in features.items()}
    return features, diagnostics


def _diagnosis(summary: dict) -> dict:
    split_codes = {}
    for split in ("c07", "c08"):
        formal2 = summary[split]["decoders"]["formal_ensemble"]["exact"]["2deg"]
        oracle2 = summary[split]["decoders"]["oracle_count_ensemble"]["exact"]["2deg"]
        aligned2 = summary[split]["decoders"]["target_aligned_ensemble"]["exact"]["2deg"]
        gap_quality = summary[split]["components"]["complex_oracle_phase_exact"]["10deg"]
        phase_quality = summary[split]["components"]["complex_oracle_gaps_exact"]["10deg"]
        if gap_quality >= 0.50 and phase_quality < 0.20:
            component = "PHASE_PRIMARY"
        elif phase_quality >= 0.50 and gap_quality < 0.20:
            component = "GAP_SHAPE_PRIMARY"
        elif phase_quality < 0.20 and gap_quality < 0.20:
            component = "JOINT_PHASE_AND_GAP_FAILURE"
        else:
            component = "MIXED_PHASE_GAP_FAILURE"
        split_codes[split] = {
            "component": component,
            "cardinality_exact2_gain": oracle2 - formal2,
            "target_alignment_exact2_gain": aligned2 - oracle2,
            "complex_gap_quality_with_oracle_phase_at_10deg": gap_quality,
            "complex_phase_quality_with_oracle_gaps_at_10deg": phase_quality,
        }
    stable = split_codes["c07"]["component"] == split_codes["c08"]["component"]
    return {
        "predeclared_rule": "K>=3 exact10 under oracle phase/gaps isolates gap/phase; gains >=0.05 identify count/alignment contribution",
        "by_split": split_codes,
        "cross_split_component_stable": stable,
        "decision": "REPLACE_COMPLETE_SET_REGRESSION_WITH_EVENT_PLUS_RELATION_FACTORIZATION" if stable and split_codes["c08"]["component"] in ("JOINT_PHASE_AND_GAP_FAILURE", "MIXED_PHASE_GAP_FAILURE") else "RETAIN_ONLY_EMPIRICALLY_SUPPORTED_COMPONENTS_AND_REDESIGN_FAILED_FACTOR",
    }


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.3), constrained_layout=True)
    splits = ("c07", "c08")
    colors = ("#4e79a7", "#f28e2b")
    names = ("formal_ensemble", "oracle_count_ensemble", "target_aligned_ensemble")
    labels = ("formal", "true count", "target-aligned seeds")
    x = np.arange(len(names))
    for offset, (split, color) in enumerate(zip(splits, colors, strict=True)):
        axes[0, 0].bar(x + (offset - .5) * .34, [summary[split]["decoders"][name]["exact"]["2deg"] for name in names], .34, label=split.upper(), color=color)
    axes[0, 0].set_xticks(x, labels, rotation=12); axes[0, 0].set_ylim(0, 1); axes[0, 0].set_title("A  Count and seed-alignment ceilings"); axes[0, 0].legend(frameon=False)
    component_names = ("complex_oracle_phase_exact", "complex_oracle_gaps_exact")
    component_labels = ("fix phase → test gaps", "fix gaps → test phase")
    for offset, (split, color) in enumerate(zip(splits, colors, strict=True)):
        axes[0, 1].bar(np.arange(2) + (offset - .5) * .34, [summary[split]["components"][name]["10deg"] for name in component_names], .34, label=split.upper(), color=color)
    axes[0, 1].set_xticks(np.arange(2), component_labels, rotation=10); axes[0, 1].set_ylim(0, 1); axes[0, 1].set_title("B  K≥3 phase/gap isolation at 10°")
    matrix = np.asarray(summary["c08"]["count_confusion"]["formal_ensemble"])
    image = axes[1, 0].imshow(matrix, cmap="Blues")
    for row in range(4):
        for column in range(4):
            axes[1, 0].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=9)
    axes[1, 0].set_xticks(range(4), range(1, 5)); axes[1, 0].set_yticks(range(4), range(1, 5)); axes[1, 0].set(xlabel="predicted exits", ylabel="true exits", title="C  C08 cardinality confusion")
    figure.colorbar(image, ax=axes[1, 0], shrink=.75)
    features = list(summary["confidence_transfer"])
    axes[1, 1].bar(np.arange(len(features)) - .18, [summary["confidence_transfer"][name]["c07_auc_exact2"] for name in features], .36, label="C07")
    axes[1, 1].bar(np.arange(len(features)) + .18, [summary["confidence_transfer"][name]["c08_auc_exact2"] for name in features], .36, label="C08")
    axes[1, 1].set_xticks(np.arange(len(features)), [name.replace("_", " ") for name in features], rotation=35, ha="right"); axes[1, 1].set_ylim(0, 1); axes[1, 1].set_title("D  Confidence separates exact sets?"); axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    figure.suptitle("JCGS frozen failure attribution: what must the next method replace?")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_joint_cyclic_gap_simplex_failure_attribution_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, action="append", type=Path)
    parser.add_argument("--formal-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.prediction_root) != 3:
        raise RuntimeError("JCGS attribution requires exactly three frozen seeds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data_by_split = {name.lower(): _load_split(name, args.teacher_root.resolve(), [path.resolve() for path in args.prediction_root]) for name in ("C07", "C08")}
    source_summary = load_json(args.formal_summary.resolve())
    summary: dict = {
        "schema_version": "gse_joint_cyclic_gap_simplex_failure_attribution_v1",
        "status": PASS, "scientific_pass": True, "precision_floor": PRECISION_FLOOR,
        "optimizer_steps": 0, "new_model_inference_observations": 0, "checkpoint_selection_observations": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    runtime: dict[str, dict] = {}
    for split, data in data_by_split.items():
        target_count = data["presence"].sum(axis=1).astype(np.int64)
        formal_decoded = formal._ensemble(data)
        oracle_count = _ensemble_for_count(data, target_count, target_aligned=False)
        target_aligned = _ensemble_for_count(data, target_count, target_aligned=True)
        predicted_metrics, predicted_masks, predicted_tp = _score(data, formal_decoded)
        oracle_metrics, oracle_masks, _ = _score(data, oracle_count)
        target_metrics, target_masks, _ = _score(data, target_aligned)
        seed_decoded = {f"seed{seed}": _decode_seed(data, seed, oracle_count=False) for seed in range(3)}
        seed_oracle = {f"seed{seed}": _decode_seed(data, seed, oracle_count=True) for seed in range(3)}
        seed_metrics = {name: _score(data, decoded)[0] for name, decoded in seed_decoded.items()}
        seed_oracle_scored = {name: _score(data, decoded) for name, decoded in seed_oracle.items()}
        component_summary, component_arrays = _component_diagnostics(data, oracle_count)
        best_seed = {}
        for tolerance in TOLERANCES:
            mask = np.logical_or.reduce([seed_oracle_scored[f"seed{seed}"][1][tolerance] for seed in range(3)])
            best_seed[f"{int(tolerance)}deg"] = float(mask.mean())
        confidence_features, confidence_diagnostics = _confidence(data, oracle_count if False else {
            **formal_decoded,
            "count_probability": data["count_probability"],
        }, predicted_masks[2.0])
        expected = float(source_summary[split]["refusal"]["raw_exact_set_fraction_2deg"])
        reproduced = float(predicted_metrics["exact"]["2deg"])
        if abs(reproduced - expected) > 1e-12:
            raise RuntimeError(f"formal {split} exact2 reproduction drift: {reproduced} versus {expected}")
        summary[split] = {
            "worlds": len(np.unique(data["parent_id"])), "observations": len(target_count), "visible_exits": int(data["presence"].sum()),
            "formal_exact2_expected": expected, "formal_exact2_reproduced": reproduced,
            "decoders": {"formal_ensemble": predicted_metrics, "oracle_count_ensemble": oracle_metrics,
                         "target_aligned_ensemble": target_metrics, **seed_metrics},
            "oracle_count_single_seeds": {name: values[0] for name, values in seed_oracle_scored.items()},
            "best_single_seed_oracle_count_upper_bound_exact": best_seed,
            "components": component_summary,
            "count_confusion": {
                "formal_ensemble": _confusion(target_count, formal_decoded["count"]),
                **{name: _confusion(target_count, decoded["count"]) for name, decoded in seed_decoded.items()},
            },
            "confidence_auc": confidence_diagnostics,
        }
        runtime[split] = {"data": data, "decoded": formal_decoded, "features": confidence_features,
                          "metrics": predicted_metrics, "masks": predicted_masks, "tp": predicted_tp,
                          "components": component_arrays, "oracle_masks": oracle_masks, "target_masks": target_masks}
    transfer = {}
    for name in runtime["c07"]["features"]:
        c07_score = runtime["c07"]["features"][name]
        choice = base._select(c07_score, {"row_tp": runtime["c07"]["tp"][2.0]}, runtime["c07"]["decoded"]["count"], int(runtime["c07"]["data"]["presence"].sum()))
        record = {
            "c07_auc_exact2": summary["c07"]["confidence_auc"][name]["auc_exact2"],
            "c08_auc_exact2": summary["c08"]["confidence_auc"][name]["auc_exact2"],
            "c07_selection": choice,
        }
        if choice is not None:
            threshold = float(choice["threshold"])
            record["c08_transfer"] = _apply_threshold(runtime["c08"]["features"][name], threshold, runtime["c08"]["decoded"], runtime["c08"]["tp"][2.0], int(runtime["c08"]["data"]["presence"].sum()))
        else:
            record["c08_transfer"] = None
        transfer[name] = record
    summary["confidence_transfer"] = transfer
    summary["diagnosis"] = _diagnosis(summary)
    summary["decision"] = summary["diagnosis"]["decision"]
    summary["duration_seconds"] = time.monotonic() - started
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", summary)
    with (output / "decoder_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "decoder", "count_accuracy", "exact2", "exact4", "exact10", "complex_exact10"))
        writer.writeheader()
        for split in ("c07", "c08"):
            for name, metrics in summary[split]["decoders"].items():
                writer.writerow({"split": split, "decoder": name, "count_accuracy": metrics["count_accuracy"],
                                 "exact2": metrics["exact"]["2deg"], "exact4": metrics["exact"]["4deg"],
                                 "exact10": metrics["exact"]["10deg"], "complex_exact10": metrics["complex_exact"]["10deg"]})
    with (output / "component_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("split", "target_count", "observations", "phase_p50", "phase_p90", "gap_p50", "gap_p90", "oracle_phase_exact10", "oracle_gaps_exact10")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for split in ("c07", "c08"):
            for row in summary[split]["components"]["strata"]:
                writer.writerow({"split": split, "target_count": row["target_count"], "observations": row["observations"],
                                 "phase_p50": row["phase_error_deg_p50_p90"][0], "phase_p90": row["phase_error_deg_p50_p90"][1],
                                 "gap_p50": row["gap_mae_deg_p50_p90"][0], "gap_p90": row["gap_mae_deg_p50_p90"][1],
                                 "oracle_phase_exact10": row["oracle_phase_exact_10deg"], "oracle_gaps_exact10": row["oracle_gaps_exact_10deg"]})
    _plot(output, summary)
    print(json.dumps({"status": PASS, "decision": summary["decision"], "diagnosis": summary["diagnosis"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
