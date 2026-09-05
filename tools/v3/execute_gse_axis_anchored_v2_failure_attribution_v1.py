#!/usr/bin/env python3
"""Attribute the sealed Axis-Anchored V1 failure before a minimal V2."""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import evaluate_gse_axis_anchored_event_relation_selection_v1 as evaluation
import train_gse_axis_anchored_event_relation_v1 as training
from mtare_topo.governance import load_json, write_json


PASS = "PASS_GSE_AXIS_ANCHORED_V2_FAILURE_ATTRIBUTION_V1"
RELATION_NAMES = ("persistent", "reveal", "withdraw")
RADII_BINS = (0, 1, 2, 4)
EXPECTED = {
    7: {"observations": 21548, "valid_pairs": 66752, "positives": (139346, 1636, 1728)},
    8: {"observations": 24394, "valid_pairs": 77490, "positives": (161597, 2107, 2181)},
}


def binary_average_precision(score: np.ndarray, truth: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    positives = int(truth.sum())
    if score.shape != truth.shape or positives == 0 or positives == len(truth):
        raise ValueError("average-precision population must contain positives and negatives")
    order = np.argsort(-score, kind="stable")
    ranked = truth[order]
    return float((np.cumsum(ranked)[ranked] / (np.flatnonzero(ranked) + 1)).sum() / positives)


def binary_auc(score: np.ndarray, truth: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    positive = int(truth.sum()); negative = len(truth) - positive
    if score.shape != truth.shape or positive == 0 or negative == 0:
        raise ValueError("AUC population must contain positives and negatives")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[truth].sum() - positive * (positive + 1) / 2) / (positive * negative))


def precision_recall_envelope(score: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    order = np.argsort(-score, kind="stable")
    ranked = truth[order]
    tp = np.cumsum(ranked); accepted = np.arange(1, len(ranked) + 1); positives = int(ranked.sum())
    precision = tp / accepted; recall = tp / positives
    at_recall = precision[recall >= 0.25]
    at_precision = recall[precision >= 0.98]
    threshold_metrics = evaluation._binary_metrics(score, truth, 0.5)
    return {
        "average_precision": binary_average_precision(score, truth),
        "precision_at_0p5": threshold_metrics["precision"],
        "recall_at_0p5": threshold_metrics["recall"],
        "maximum_precision_at_recall_0p25": float(at_recall.max()) if len(at_recall) else 0.0,
        "maximum_recall_at_precision_0p98": float(at_precision.max()) if len(at_precision) else 0.0,
    }


def circular_assignment_distances(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if predicted.ndim != 1 or target.ndim != 1 or len(predicted) != len(target) or not len(target):
        raise ValueError("circular assignment requires equal non-empty one-dimensional sets")
    difference = np.abs(predicted[:, None] - target[None, :])
    distance = np.minimum(difference, 180 - difference)
    rows, columns = linear_sum_assignment(distance)
    return distance[rows, columns]


def oracle_count_localization(score: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    score = np.asarray(score, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    if score.shape != truth.shape or score.ndim != 2 or score.shape[1] != 180:
        raise ValueError("localization arrays must be aligned [N,180]")
    positive_rows = np.flatnonzero(truth.any(axis=1))
    empty_rows = np.flatnonzero(~truth.any(axis=1))
    matched = []
    for row in positive_rows:
        target = np.flatnonzero(truth[row]); count = len(target)
        predicted = np.argpartition(score[row], -count)[-count:]
        matched.extend(circular_assignment_distances(predicted, target).tolist())
    distances = np.asarray(matched, dtype=np.int64)
    slice_score = score.max(axis=1)
    return {
        "positive_slices": int(len(positive_rows)),
        "empty_slices": int(len(empty_rows)),
        "relation_instances": int(len(distances)),
        "positive_vs_empty_slice_auc": binary_auc(slice_score, truth.any(axis=1)),
        "median_assignment_error_bins": float(np.median(distances)),
        "p90_assignment_error_bins": float(np.percentile(distances, 90)),
        **{f"matched_fraction_within_{radius}_bins": float(np.mean(distances <= radius)) for radius in RADII_BINS},
    }


def split_attribution(split: dict, condition: int) -> dict:
    expected = EXPECTED[condition]
    arrays = split["arrays"]
    valid_steps = arrays["relation_truth"][..., 0, 0] >= 0
    if len(arrays["event_truth"]) != expected["observations"] or int(valid_steps.sum()) != expected["valid_pairs"]:
        raise RuntimeError(f"C{condition:02d} attribution population drift")
    result = {
        "observations": int(len(arrays["event_truth"])),
        "valid_relation_pairs": int(valid_steps.sum()),
        "axis_mean_error_deg": float(split["axis_mean_error_deg"]),
        "event_macro_f1": float(split["event"]["macro_f1"]),
        "relation": {},
    }
    for channel, name in enumerate(RELATION_NAMES):
        score = arrays["relation"][..., channel][valid_steps]
        truth = arrays["relation_truth"][..., channel][valid_steps] == 1
        if int(truth.sum()) != expected["positives"][channel]:
            raise RuntimeError(f"C{condition:02d} {name} positive count drift")
        metrics = precision_recall_envelope(score, truth)
        metrics.update(oracle_count_localization(score, truth))
        metrics["positive_bins"] = int(truth.sum())
        metrics["positive_rate"] = float(truth.mean())
        result["relation"][name] = metrics
    return result


def _plot(output: Path, summary: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    axes[0, 0].bar(("V1 C07", "V1 C08", "Slot C07", "Slot C08"), (
        summary["c07"]["axis_mean_error_deg"], summary["c08"]["axis_mean_error_deg"],
        summary["axis_predecessor"]["c07_axis_deg"], summary["axis_predecessor"]["c08_axis_deg"],
    ), color=("#e15759", "#e15759", "#59a14f", "#59a14f"))
    axes[0, 0].axhline(10.0, color="black", linestyle="--"); axes[0, 0].tick_params(axis="x", rotation=18); axes[0, 0].set_title("A  Axis error (deg)")
    weights = summary["relation_objective"]["effective_positive_weights"]
    axes[0, 1].bar(RELATION_NAMES, [weights[name] for name in RELATION_NAMES], color="#f28e2b"); axes[0, 1].set_yscale("log"); axes[0, 1].set_title("B  Dense BCE positive weight")
    x = np.arange(3); width = .35
    axes[1, 0].bar(x-width/2, [summary["c07"]["relation"][name]["average_precision"] for name in RELATION_NAMES], width, label="C07")
    axes[1, 0].bar(x+width/2, [summary["c08"]["relation"][name]["average_precision"] for name in RELATION_NAMES], width, label="C08")
    axes[1, 0].set(xticks=x, xticklabels=RELATION_NAMES, ylim=(0, 1), title="C  Exact-bin average precision"); axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(x-width/2, [summary["c07"]["relation"][name]["matched_fraction_within_4_bins"] for name in RELATION_NAMES], width, label="C07")
    axes[1, 1].bar(x+width/2, [summary["c08"]["relation"][name]["matched_fraction_within_4_bins"] for name in RELATION_NAMES], width, label="C08")
    axes[1, 1].set(xticks=x, xticklabels=RELATION_NAMES, ylim=(0, 1), title="D  Oracle-count localization within 8 deg"); axes[1, 1].legend(frameon=False)
    for axis in axes.flat: axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    fig.suptitle("Why Axis-Anchored V1 is not the final GSE-Graph method")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_axis_anchored_v2_failure_attribution_v1.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--association-pairs", required=True, type=Path)
    parser.add_argument("--prediction-root", action="append", required=True, type=Path)
    parser.add_argument("--formal-summary", required=True, type=Path)
    parser.add_argument("--slot-summary", required=True, type=Path)
    parser.add_argument("--current-source", required=True, type=Path)
    parser.add_argument("--predecessor-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic()
    if len(args.prediction_root) != 3:
        raise ValueError("attribution requires exactly three frozen seed outputs")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    traversals = training.manifest_traversals(args.sequence_manifest.resolve())
    c07_split = evaluation._split(7, args.dataset_root.resolve(), traversals, [path.resolve() for path in args.prediction_root], args.association_pairs.resolve())
    c07 = split_attribution(c07_split, 7); del c07_split
    c08_split = evaluation._split(8, args.dataset_root.resolve(), traversals, [path.resolve() for path in args.prediction_root], args.association_pairs.resolve())
    c08 = split_attribution(c08_split, 8); del c08_split
    formal = load_json(args.formal_summary.resolve()); slot = load_json(args.slot_summary.resolve())
    formal_selection = formal["selection"]
    reproduction = {
        "c07_axis_absolute_error": abs(c07["axis_mean_error_deg"] - formal_selection["c07"]["axis_mean_error_deg"]),
        "c08_axis_absolute_error": abs(c08["axis_mean_error_deg"] - formal_selection["c08"]["axis_mean_error_deg"]),
        "c07_event_absolute_error": abs(c07["event_macro_f1"] - formal_selection["c07"]["event"]["macro_f1"]),
        "c08_event_absolute_error": abs(c08["event_macro_f1"] - formal_selection["c08"]["event"]["macro_f1"]),
    }
    slot_selection = slot["selection"]
    predecessor = {
        "c07_axis_deg": float(slot_selection["c07"]["global_geometry"]["axis_mean_error_deg"]),
        "c08_axis_deg": float(slot_selection["c08"]["global_geometry"]["axis_mean_error_deg"]),
    }
    current_source = args.current_source.read_text(encoding="utf-8")
    predecessor_source = args.predecessor_source.read_text(encoding="utf-8")
    architecture = {
        "current_axis_uses_last_frame_directional": "current_directional = directional[:, -1]" in current_source and "self.axis_azimuth_head(current_directional)" in current_source,
        "predecessor_has_five_frame_directional_temporal": "self.directional_temporal" in predecessor_source and "self.axis_azimuth_head(directional)" in predecessor_source,
    }
    rates = training.RELATION_POSITIVE_RATE.astype(np.float64)
    objective = {
        "fit_positive_rates": {name: float(rates[index]) for index, name in enumerate(RELATION_NAMES)},
        "effective_positive_weights": {name: float(0.5 / rates[index]) for index, name in enumerate(RELATION_NAMES)},
        "effective_negative_weights": {name: float(0.5 / (1.0 - rates[index])) for index, name in enumerate(RELATION_NAMES)},
    }
    local_signal = all(
        split["relation"][name]["matched_fraction_within_4_bins"] >= 0.25
        for split, name in product((c07, c08), RELATION_NAMES)
    )
    decision = "ALLOW_SPARSE_CIRCULAR_RELATION_TRANSPORT_V2_READINESS" if local_signal else "REQUIRE_OBJECT_CENTRIC_RELATION_TRANSPORT_V2_READINESS"
    checks = {
        "formal_metrics_reproduced": max(reproduction.values()) <= 1e-12,
        "axis_failure_explained_by_missing_directional_temporal": all(architecture.values()) and c07["axis_mean_error_deg"] - predecessor["c07_axis_deg"] >= 70.0 and c08["axis_mean_error_deg"] - predecessor["c08_axis_deg"] >= 70.0 and max(predecessor.values()) <= 10.0,
        "rare_relation_weighting_is_extreme": objective["effective_positive_weights"]["reveal"] >= 3000.0 and objective["effective_positive_weights"]["withdraw"] >= 3000.0,
        "dense_relation_safety_failure_reproduced": all(split["relation"][name]["maximum_recall_at_precision_0p98"] < 0.25 for split, name in product((c07, c08), RELATION_NAMES)),
        "diagnosis_selects_one_minimal_transport_route": decision in ("ALLOW_SPARSE_CIRCULAR_RELATION_TRANSPORT_V2_READINESS", "REQUIRE_OBJECT_CENTRIC_RELATION_TRANSPORT_V2_READINESS"),
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_axis_anchored_v2_failure_attribution_v1",
        "status": PASS if scientific_pass else "FAIL_GSE_AXIS_ANCHORED_V2_FAILURE_ATTRIBUTION_V1",
        "scientific_pass": scientific_pass, "decision": decision,
        "c07": c07, "c08": c08, "formal_reproduction": reproduction,
        "axis_predecessor": predecessor, "architecture": architecture,
        "relation_objective": objective, "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "new_model_inference_observations": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    with (output / "relation_attribution.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("split", "relation", "positive_rate", "average_precision", "precision_at_0p5", "recall_at_0p5", "maximum_recall_at_precision_0p98", "positive_vs_empty_slice_auc", "matched_fraction_within_0_bins", "matched_fraction_within_1_bins", "matched_fraction_within_2_bins", "matched_fraction_within_4_bins", "median_assignment_error_bins", "p90_assignment_error_bins")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for split_name, split in (("C07", c07), ("C08", c08)):
            for name in RELATION_NAMES:
                writer.writerow({"split": split_name, "relation": name, **{field: split["relation"][name][field] for field in fields[2:]}})
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": decision, "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
