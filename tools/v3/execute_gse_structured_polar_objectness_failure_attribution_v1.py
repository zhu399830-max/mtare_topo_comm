#!/usr/bin/env python3
"""Read-only proposal/objectness attribution for structured-polar top16 outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from evaluate_gse_structured_polar_multidepth_capacity_v1 import (
    _second_depth_target_mask,
)
from mtare_topo.data.gse_observable_spatial_event_dataset import (
    load_observable_spatial_event_teacher,
)
from mtare_topo.evaluation.gse_spatial_event_set_metrics import (
    _maximum_valid_matching,
    spatial_event_set_metrics,
)


PASS = "PASS_GSE_STRUCTURED_POLAR_OBJECTNESS_FAILURE_ATTRIBUTION_V1"
MATCH_CAP_M = 4.0
F1_GAIN = 0.05
RECALL_GAIN = 0.10
DISTANCE_EDGES_M = (0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0, 50.0001)


def _safe(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _load_predictions(path: Path, rows: np.ndarray, global_index: np.ndarray):
    with np.load(path, allow_pickle=False) as archive:
        if (
            not np.array_equal(archive["observation_row"], rows)
            or not np.array_equal(archive["global_sequence_index"], global_index)
        ):
            raise RuntimeError(f"prediction/Teacher identity drift: {path}")
        return {
            "confidence": np.asarray(archive["confidence"], dtype=np.float32),
            "event_type": np.asarray(archive["event_type"], dtype=np.int8),
            "relative_xyz_m": np.asarray(archive["relative_xyz_m"], dtype=np.float32),
            "azimuth_bin": np.asarray(archive["azimuth_bin"], dtype=np.int16),
            "depth_slot": np.asarray(archive["depth_slot"], dtype=np.int8),
        }


def _metrics(predictions, targets, threshold: float, *, ignore_type: bool = False):
    predicted_type = predictions["event_type"]
    target_type = targets["event_type_index"]
    if ignore_type:
        predicted_type = np.zeros_like(predicted_type)
        target_type = np.zeros_like(target_type)
    metrics = spatial_event_set_metrics(
        confidence=predictions["confidence"],
        predicted_type=predicted_type,
        predicted_xyz_m=predictions["relative_xyz_m"],
        target_type=target_type,
        target_xyz_m=targets["event_relative_xyz_m"],
        target_mask=targets["event_mask"],
        threshold=threshold,
        maximum_error_m=MATCH_CAP_M,
    )
    if not ignore_type:
        metrics["macro_f1"] = float(
            (metrics["per_type"]["terminal"]["f1"] + metrics["per_type"]["junction"]["f1"])
            / 2.0
        )
    return metrics


def _matching_labels(predictions, targets, selected_mask, *, ignore_type: bool = False):
    prediction_match = np.zeros_like(selected_mask, dtype=bool)
    target_match = np.zeros_like(targets["event_mask"], dtype=bool)
    errors = []
    for row in range(len(selected_mask)):
        selected = np.flatnonzero(selected_mask[row])
        selected = np.asarray(
            sorted(
                selected.tolist(),
                key=lambda index: (
                    int(predictions["event_type"][row, index]),
                    *tuple(float(value) for value in predictions["relative_xyz_m"][row, index]),
                    -float(predictions["confidence"][row, index]),
                ),
            ),
            dtype=np.int64,
        )
        active = np.flatnonzero(targets["event_mask"][row])
        predicted_type = predictions["event_type"][row, selected]
        target_type = targets["event_type_index"][row, active]
        if ignore_type:
            predicted_type = np.zeros_like(predicted_type)
            target_type = np.zeros_like(target_type)
        matches = _maximum_valid_matching(
            predicted_type,
            predictions["relative_xyz_m"][row, selected],
            target_type,
            targets["event_relative_xyz_m"][row, active],
            maximum_error_m=MATCH_CAP_M,
        )
        for prediction_local, target_local, error in matches:
            prediction_match[row, selected[prediction_local]] = True
            target_match[row, active[target_local]] = True
            errors.append(error)
    return prediction_match, target_match, np.asarray(errors, dtype=np.float64)


def _ranking_diagnostic(confidence: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    scores = np.asarray(confidence, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    ordered_truth = truth[order]
    tp = np.cumsum(ordered_truth, dtype=np.int64)
    fp = np.cumsum(~ordered_truth, dtype=np.int64)
    positives = int(truth.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(positives, 1)
    average_precision = float(np.sum(precision[ordered_truth]) / positives) if positives else 0.0
    eligible = np.flatnonzero(recall >= 0.25)
    if len(eligible):
        local = int(eligible[np.argmax(precision[eligible])])
        precision_at_recall25 = float(precision[local])
        threshold_at_recall25 = float(scores[order[local]])
        attained_recall = float(recall[local])
    else:
        precision_at_recall25 = threshold_at_recall25 = attained_recall = 0.0
    quantiles = lambda values: {
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
    }
    return {
        "candidates": len(scores),
        "oracle_positive_candidates": positives,
        "prevalence": _safe(positives, len(scores)),
        "average_precision": average_precision,
        "precision_at_candidate_recall_at_least_0p25": precision_at_recall25,
        "candidate_recall_at_reported_point": attained_recall,
        "diagnostic_threshold_at_reported_point": threshold_at_recall25,
        "positive_confidence": quantiles(scores[truth]),
        "negative_confidence": quantiles(scores[~truth]),
    }


def _slot_attribution(predictions, targets, threshold: float) -> dict[str, object]:
    selected = predictions["confidence"] >= threshold
    matched, _, _ = _matching_labels(predictions, targets, selected)
    rows_empty = targets["event_mask"].sum(axis=1) == 0
    records = {}
    for slot in (0, 1):
        mask = selected & (predictions["depth_slot"] == slot)
        true = int(np.sum(mask & matched))
        total = int(mask.sum())
        records[f"slot{slot}"] = {
            "selected": total,
            "matched": true,
            "false_positive": total - true,
            "precision": _safe(true, total),
            "empty_row_selected": int(mask[rows_empty].sum()),
        }
    both = np.zeros(len(selected), dtype=bool)
    both_selected_events = 0
    both_false_events = 0
    for row in range(len(selected)):
        chosen = np.flatnonzero(selected[row])
        unique, counts = np.unique(predictions["azimuth_bin"][row, chosen], return_counts=True)
        duplicate_bins = set(int(value) for value in unique[counts >= 2])
        if duplicate_bins:
            both[row] = True
            duplicate = np.asarray(
                [index for index in chosen if int(predictions["azimuth_bin"][row, index]) in duplicate_bins],
                dtype=np.int64,
            )
            both_selected_events += len(duplicate)
            both_false_events += int(np.sum(~matched[row, duplicate]))
    return {
        **records,
        "same_bin_double_selected_rows": int(both.sum()),
        "same_bin_double_selected_events": both_selected_events,
        "same_bin_double_false_events": both_false_events,
    }


def _one_per_bin(predictions):
    confidence = predictions["confidence"].copy()
    suppressed = 0
    for row in range(len(confidence)):
        for bin_index in np.unique(predictions["azimuth_bin"][row]):
            indices = np.flatnonzero(predictions["azimuth_bin"][row] == bin_index)
            if len(indices) <= 1:
                continue
            keep = int(indices[np.argmax(confidence[row, indices])])
            drop = indices[indices != keep]
            confidence[row, drop] = 0.0
            suppressed += len(drop)
    return {**predictions, "confidence": confidence}, suppressed


def _oracle_cardinality(predictions, targets):
    confidence = np.zeros_like(predictions["confidence"], dtype=np.float32)
    cardinality = targets["event_mask"].sum(axis=1).astype(np.int64)
    for row, count in enumerate(cardinality):
        if count:
            order = np.argsort(-predictions["confidence"][row], kind="stable")
            confidence[row, order[:count]] = 1.0
    return {**predictions, "confidence": confidence}


def _distance_strata(targets, target_match: np.ndarray) -> dict[str, object]:
    radial = np.linalg.norm(targets["event_relative_xyz_m"], axis=-1)
    records = {}
    for left, right in zip(DISTANCE_EDGES_M[:-1], DISTANCE_EDGES_M[1:], strict=True):
        selected = targets["event_mask"] & (radial >= left) & (radial < right)
        count = int(selected.sum())
        records[f"{left:g}-{right:g}m"] = {
            "targets": count,
            "typed_oracle_recall": _safe(int(np.sum(target_match & selected)), count),
        }
    return records


def _seed_record(predictions, targets, threshold: float) -> dict[str, object]:
    selected_metrics = _metrics(predictions, targets, threshold)
    all_selected = np.ones_like(predictions["confidence"], dtype=bool)
    typed_prediction_match, typed_target_match, typed_errors = _matching_labels(
        predictions, targets, all_selected
    )
    _, position_target_match, _ = _matching_labels(
        predictions, targets, all_selected, ignore_type=True
    )
    target_count = int(targets["event_mask"].sum())
    second_depth = _second_depth_target_mask(targets)
    one_per_bin, suppressed = _one_per_bin(predictions)
    slot0 = {**predictions, "confidence": np.where(predictions["depth_slot"] == 0, predictions["confidence"], 0.0)}
    slot1 = {**predictions, "confidence": np.where(predictions["depth_slot"] == 1, predictions["confidence"], 0.0)}
    oracle_cardinality = _oracle_cardinality(predictions, targets)
    return {
        "sealed_threshold": threshold,
        "sealed_metrics": selected_metrics,
        "proposal_oracle": {
            "typed_recall": _safe(int(typed_target_match.sum()), target_count),
            "position_recall": _safe(int(position_target_match.sum()), target_count),
            "typed_true_positive": int(typed_target_match.sum()),
            "position_true_positive": int(position_target_match.sum()),
            "target_count": target_count,
            "typed_localization_mae_m": float(typed_errors.mean()),
            "second_depth_typed_recall": _safe(int(np.sum(typed_target_match & second_depth)), int(second_depth.sum())),
        },
        "ranking": _ranking_diagnostic(predictions["confidence"], typed_prediction_match),
        "slot_attribution": _slot_attribution(predictions, targets, threshold),
        "one_per_bin": {
            "suppressed_candidates": suppressed,
            "metrics": _metrics(one_per_bin, targets, threshold),
        },
        "slot0_only_metrics": _metrics(slot0, targets, threshold),
        "slot1_only_metrics": _metrics(slot1, targets, threshold),
        "teacher_cardinality_upper_bound": _metrics(oracle_cardinality, targets, 0.5),
        "distance_strata": _distance_strata(targets, typed_target_match),
    }


def _plot(output: Path, records: list[dict], baseline: dict, decision: str) -> None:
    colors = ("#1864ab", "#2b8a3e", "#e67700")
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 4.3), constrained_layout=True)
    x = np.arange(3)
    axes[0].bar(x - 0.18, [r["sealed_metrics"]["recall"] for r in records], 0.18, label="sealed recall")
    axes[0].bar(x, [r["proposal_oracle"]["typed_recall"] for r in records], 0.18, label="typed oracle")
    axes[0].bar(x + 0.18, [r["proposal_oracle"]["position_recall"] for r in records], 0.18, label="position oracle")
    axes[0].axhline(baseline["recall"] + RECALL_GAIN, color="#c92a2a", linestyle="--", label="baseline +0.10")
    axes[0].set_xticks(x, ["seed 0", "seed 1", "seed 2"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("target recall")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.25)

    slot0 = [r["slot_attribution"]["slot0"]["false_positive"] for r in records]
    slot1 = [r["slot_attribution"]["slot1"]["false_positive"] for r in records]
    axes[1].bar(x, slot0, color="#f08c00", label="slot0 false")
    axes[1].bar(x, slot1, bottom=slot0, color="#c92a2a", label="slot1 false")
    axes[1].set_xticks(x, ["seed 0", "seed 1", "seed 2"])
    axes[1].set_ylabel("false events at 0.95")
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x - 0.16, [r["ranking"]["positive_confidence"]["median"] for r in records], 0.32, color="#2b8a3e", label="oracle-positive")
    axes[2].bar(x + 0.16, [r["ranking"]["negative_confidence"]["median"] for r in records], 0.32, color="#c92a2a", label="oracle-negative")
    axes[2].set_xticks(x, ["seed 0", "seed 1", "seed 2"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("median confidence")
    axes[2].legend(fontsize=7)
    axes[2].grid(axis="y", alpha=0.25)

    keys = [f"{left:g}-{right:g}m" for left, right in zip(DISTANCE_EDGES_M[:-1], DISTANCE_EDGES_M[1:], strict=True)]
    distance_x = np.arange(len(keys))
    for seed, (record, color) in enumerate(zip(records, colors, strict=True)):
        axes[3].plot(distance_x, [record["distance_strata"][key]["typed_oracle_recall"] for key in keys], marker="o", color=color, label=f"seed {seed}")
    axes[3].set_xticks(distance_x, keys, rotation=27)
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("typed oracle recall")
    axes[3].set_xlabel("Teacher event distance")
    axes[3].legend(fontsize=7)
    axes[3].grid(alpha=0.25)
    figure.suptitle(f"Structured-polar objectness attribution: {decision}")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_structured_polar_objectness_failure_attribution_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--capacity-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = load_observable_spatial_event_teacher(args.teacher_root.resolve())
    rows = teacher.selection_rows
    targets = teacher.targets(rows)
    global_index = teacher.global_sequence_index[rows]
    capacity = args.capacity_run.resolve()
    evaluation = json.loads((capacity / "metrics/capacity/summary.json").read_text(encoding="utf-8"))
    if evaluation.get("status") != "FAIL_GSE_STRUCTURED_POLAR_MULTIDEPTH_CAPACITY_V1":
        raise RuntimeError("attribution requires sealed structured-polar capacity FAIL")
    baseline = evaluation["baselines"]["exclusive_single_center"]["selected_threshold_metrics"]
    records = []
    for seed in (0, 1, 2):
        model_dir = capacity / f"artifacts/models/seed{seed}"
        model_summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
        if model_summary.get("seed") != seed or model_summary.get("optimizer_steps") != 9096:
            raise RuntimeError(f"seed summary drift: {seed}")
        threshold = float(model_summary["selected_threshold_metrics"]["threshold"])
        if threshold != 0.95:
            raise RuntimeError("sealed structured-polar threshold drift")
        predictions = _load_predictions(model_dir / "selection_outputs.npz", rows, global_index)
        record = _seed_record(predictions, targets, threshold)
        if record["sealed_metrics"] != model_summary["selected_threshold_metrics"]:
            raise RuntimeError(f"seed sealed metric replay drift: {seed}")
        records.append({"seed": seed, **record})
    proposal_gate = all(r["proposal_oracle"]["typed_recall"] >= baseline["recall"] + RECALL_GAIN for r in records)
    one_per_bin_gate = all(r["one_per_bin"]["metrics"]["f1"] >= baseline["f1"] + F1_GAIN for r in records)
    cardinality_gate = all(r["teacher_cardinality_upper_bound"]["f1"] >= baseline["f1"] + F1_GAIN for r in records)
    ranking_safe = all(r["ranking"]["precision_at_candidate_recall_at_least_0p25"] >= 0.90 for r in records)
    if not proposal_gate:
        decision = "STOP_DENSE_PROPOSAL_ROUTE"
    elif one_per_bin_gate:
        decision = "ONE_PER_BIN_CORRECTIVE_JUSTIFIED"
    elif cardinality_gate:
        decision = "CARDINALITY_RANKING_CORRECTIVE_JUSTIFIED"
    else:
        decision = "FROZEN_GEOMETRY_OBJECTNESS_REFIT_REQUIRED"
    mask = targets["event_mask"].astype(bool)
    population = {
        "worlds": 20,
        "observations": len(rows),
        "target_tokens": int(mask.sum()),
        "target_types": np.bincount(targets["event_type_index"][mask].astype(np.int64), minlength=2).astype(int).tolist(),
        "event_identities": int(len(np.unique(targets["event_identity_index"][mask]))),
        "multi_event_rows": int(np.sum(mask.sum(axis=1) >= 2)),
        "second_depth_targets": int(_second_depth_target_mask(targets).sum()),
        "cardinality": np.bincount(mask.sum(axis=1).astype(np.int64), minlength=6).astype(int).tolist(),
    }
    if population != {
        "worlds": 20,
        "observations": 45942,
        "target_tokens": 33145,
        "target_types": [7578, 25567],
        "event_identities": 275,
        "multi_event_rows": 6073,
        "second_depth_targets": 79,
        "cardinality": [19159, 20710, 5786, 285, 2, 0],
    }:
        raise RuntimeError(f"attribution population drift: {population}")
    summary = {
        "schema_version": "gse_structured_polar_objectness_failure_attribution_v1",
        "status": PASS,
        "decision": decision,
        "decision_gates": {
            "all_seed_typed_oracle_recall_at_least_exclusive_plus_0p10": proposal_gate,
            "all_seed_one_per_bin_f1_at_least_exclusive_plus_0p05": one_per_bin_gate,
            "all_seed_teacher_cardinality_f1_at_least_exclusive_plus_0p05": cardinality_gate,
            "all_seed_candidate_ranking_precision_at_recall0p25_at_least_0p90": ranking_safe,
            "exclusive_reference": {"recall": baseline["recall"], "f1": baseline["f1"]},
        },
        "population": population,
        "seed_records": records,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "threshold_selection_steps": 0,
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_structured_polar_objectness_failure_attribution_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ("seed", "sealed_precision", "sealed_recall", "sealed_f1", "typed_oracle_recall", "position_oracle_recall", "second_depth_oracle_recall", "candidate_average_precision", "candidate_precision_at_recall0p25", "slot0_false_positive", "slot1_false_positive", "one_per_bin_f1", "teacher_cardinality_f1")
    with (output / "seed_attribution.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "seed": record["seed"],
                "sealed_precision": record["sealed_metrics"]["precision"],
                "sealed_recall": record["sealed_metrics"]["recall"],
                "sealed_f1": record["sealed_metrics"]["f1"],
                "typed_oracle_recall": record["proposal_oracle"]["typed_recall"],
                "position_oracle_recall": record["proposal_oracle"]["position_recall"],
                "second_depth_oracle_recall": record["proposal_oracle"]["second_depth_typed_recall"],
                "candidate_average_precision": record["ranking"]["average_precision"],
                "candidate_precision_at_recall0p25": record["ranking"]["precision_at_candidate_recall_at_least_0p25"],
                "slot0_false_positive": record["slot_attribution"]["slot0"]["false_positive"],
                "slot1_false_positive": record["slot_attribution"]["slot1"]["false_positive"],
                "one_per_bin_f1": record["one_per_bin"]["metrics"]["f1"],
                "teacher_cardinality_f1": record["teacher_cardinality_upper_bound"]["f1"],
            })
    _plot(output, records, baseline, decision)
    print(json.dumps({"status": PASS, "decision": decision, "decision_gates": summary["decision_gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
