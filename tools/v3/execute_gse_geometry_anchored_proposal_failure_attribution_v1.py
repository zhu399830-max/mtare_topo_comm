#!/usr/bin/env python3
"""Read-only attribution of the geometry-anchored joint proposal failure."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.gse_observable_spatial_event_dataset import (
    load_observable_spatial_event_teacher,
)
from mtare_topo.evaluation.gse_spatial_event_set_metrics import (
    _maximum_valid_matching,
    spatial_event_set_metrics,
)


PASS = "PASS_GSE_GEOMETRY_ANCHORED_PROPOSAL_FAILURE_ATTRIBUTION_V1"
MATCH_CAP_M = 4.0
GAIN_F1 = 0.05
GAIN_RECALL = 0.10
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


def _nms(predictions, threshold: float):
    confidence = predictions["confidence"].copy()
    suppressed = 0
    before = int(np.sum(confidence >= threshold))
    for row in range(len(confidence)):
        for event_type in (0, 1):
            indices = np.flatnonzero(
                (confidence[row] >= threshold)
                & (predictions["event_type"][row] == event_type)
            )
            ordered = sorted(
                indices.tolist(),
                key=lambda index: (
                    -float(confidence[row, index]),
                    *tuple(float(value) for value in predictions["relative_xyz_m"][row, index]),
                    index,
                ),
            )
            kept: list[int] = []
            for index in ordered:
                if any(
                    np.linalg.norm(
                        predictions["relative_xyz_m"][row, index]
                        - predictions["relative_xyz_m"][row, previous]
                    ) <= MATCH_CAP_M
                    for previous in kept
                ):
                    confidence[row, index] = 0.0
                    suppressed += 1
                else:
                    kept.append(index)
    return {**predictions, "confidence": confidence}, {
        "selected_before": before,
        "suppressed": suppressed,
        "suppressed_fraction": _safe(suppressed, before),
        "selected_after": before - suppressed,
    }


def _angle_deg(xyz: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))


def _wrapped_angle_error_deg(a: float, b: np.ndarray) -> np.ndarray:
    return np.abs((b - a + 180.0) % 360.0 - 180.0)


def _proposal_support(predictions, targets):
    mask = targets["event_mask"].astype(bool)
    target_count = int(mask.sum())
    typed_independent = 0
    position_independent = 0
    missing_same_type = 0
    nearest_typed_error: list[float] = []
    nearest_position_error: list[float] = []
    nearest_typed_azimuth_error: list[float] = []
    nearest_typed_radius_error: list[float] = []
    distance_records: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in range(len(mask)):
        active = np.flatnonzero(mask[row])
        predicted_xyz = predictions["relative_xyz_m"][row]
        predicted_type = predictions["event_type"][row]
        predicted_azimuth = _angle_deg(predicted_xyz)
        predicted_radius = np.linalg.norm(predicted_xyz, axis=1)
        for target_index in active:
            target_xyz = targets["event_relative_xyz_m"][row, target_index]
            target_type = int(targets["event_type_index"][row, target_index])
            errors = np.linalg.norm(predicted_xyz - target_xyz, axis=1)
            nearest_position = float(np.min(errors))
            nearest_position_error.append(nearest_position)
            position_independent += int(nearest_position <= MATCH_CAP_M)
            same = np.flatnonzero(predicted_type == target_type)
            nearest_typed = float("inf")
            if len(same):
                local = int(same[int(np.argmin(errors[same]))])
                nearest_typed = float(errors[local])
                nearest_typed_error.append(nearest_typed)
                target_azimuth = float(_angle_deg(target_xyz))
                nearest_typed_azimuth_error.append(
                    float(_wrapped_angle_error_deg(target_azimuth, predicted_azimuth[local]))
                )
                nearest_typed_radius_error.append(
                    abs(float(predicted_radius[local]) - float(np.linalg.norm(target_xyz)))
                )
            else:
                missing_same_type += 1
            typed_independent += int(nearest_typed <= MATCH_CAP_M)
            distance = float(np.linalg.norm(target_xyz))
            bin_index = min(
                np.searchsorted(DISTANCE_EDGES_M, distance, side="right") - 1,
                len(DISTANCE_EDGES_M) - 2,
            )
            key = f"{DISTANCE_EDGES_M[bin_index]:g}-{DISTANCE_EDGES_M[bin_index + 1]:g}m"
            distance_records[key][0] += 1
            distance_records[key][1] += int(nearest_typed <= MATCH_CAP_M)
            distance_records[key][2] += int(nearest_position <= MATCH_CAP_M)
    all_confidence = np.ones_like(predictions["confidence"], dtype=np.float32)
    all_predictions = {**predictions, "confidence": all_confidence}
    typed_matching = _metrics(all_predictions, targets, 0.5)
    position_matching = _metrics(all_predictions, targets, 0.5, ignore_type=True)
    typed_errors = np.asarray(nearest_typed_error, dtype=np.float64)
    position_errors = np.asarray(nearest_position_error, dtype=np.float64)
    azimuth_errors = np.asarray(nearest_typed_azimuth_error, dtype=np.float64)
    radius_errors = np.asarray(nearest_typed_radius_error, dtype=np.float64)
    quantiles = lambda values: {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }
    return {
        "target_count": target_count,
        "typed_independent_support": typed_independent,
        "typed_independent_recall": _safe(typed_independent, target_count),
        "position_independent_support": position_independent,
        "position_independent_recall": _safe(position_independent, target_count),
        "typed_one_to_one_oracle": typed_matching,
        "position_one_to_one_oracle": position_matching,
        "assignment_conflict_targets": typed_independent - int(typed_matching["true_positive"]),
        "type_confusion_upper_bound_targets": position_independent - typed_independent,
        "no_nearby_position_targets": target_count - position_independent,
        "missing_same_type_query_count": missing_same_type,
        "nearest_typed_3d_error_m": quantiles(typed_errors),
        "nearest_position_3d_error_m": quantiles(position_errors),
        "nearest_typed_azimuth_error_deg": quantiles(azimuth_errors),
        "nearest_typed_radius_error_m": quantiles(radius_errors),
        "distance_strata": {
            key: {
                "targets": values[0],
                "typed_independent_recall": _safe(values[1], values[0]),
                "position_independent_recall": _safe(values[2], values[0]),
            }
            for key, values in distance_records.items()
        },
    }


def _threshold_diagnostics(predictions, targets, threshold: float):
    selected = predictions["confidence"] >= threshold
    predicted_count = selected.sum(axis=1).astype(np.int64)
    target_count = targets["event_mask"].sum(axis=1).astype(np.int64)
    empty = target_count == 0
    nonempty = ~empty
    metrics = _metrics(predictions, targets, threshold)
    empty_predictions = int(predicted_count[empty].sum())
    nonempty_predictions = int(predicted_count[nonempty].sum())
    duplicate_pairs = 0
    for row in range(len(selected)):
        indices = np.flatnonzero(selected[row])
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                i, j = int(indices[left]), int(indices[right])
                if (
                    predictions["event_type"][row, i] == predictions["event_type"][row, j]
                    and np.linalg.norm(
                        predictions["relative_xyz_m"][row, i]
                        - predictions["relative_xyz_m"][row, j]
                    ) <= MATCH_CAP_M
                ):
                    duplicate_pairs += 1
    return {
        "metrics": metrics,
        "empty_target_rows": int(empty.sum()),
        "empty_rows_with_prediction": int(np.sum(predicted_count[empty] > 0)),
        "empty_row_false_positive_events": empty_predictions,
        "nonempty_target_rows": int(nonempty.sum()),
        "nonempty_rows_with_prediction": int(np.sum(predicted_count[nonempty] > 0)),
        "nonempty_row_predicted_events": nonempty_predictions,
        "predicted_cardinality_mean": float(np.mean(predicted_count)),
        "target_cardinality_mean": float(np.mean(target_count)),
        "cardinality_mae": float(np.mean(np.abs(predicted_count - target_count))),
        "cardinality_exact_fraction": float(np.mean(predicted_count == target_count)),
        "predicted_cardinality_histogram": {
            str(key): int(value) for key, value in sorted(Counter(predicted_count.tolist()).items())
        },
        "target_cardinality_histogram": {
            str(key): int(value) for key, value in sorted(Counter(target_count.tolist()).items())
        },
        "same_type_within_4m_selected_pairs": duplicate_pairs,
    }


def _plot(output: Path, records: list[dict], baseline: dict, decision: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.4), constrained_layout=True)
    keys = [
        f"{DISTANCE_EDGES_M[index]:g}-{DISTANCE_EDGES_M[index + 1]:g}m"
        for index in range(len(DISTANCE_EDGES_M) - 1)
    ]
    x = np.arange(len(keys))
    colors = ("#1864ab", "#2b8a3e", "#e67700")
    for seed, (record, color) in enumerate(zip(records, colors, strict=True)):
        strata = record["support"]["distance_strata"]
        axes[0].plot(x, [strata[key]["typed_independent_recall"] for key in keys], marker="o", color=color, label=f"seed {seed} typed")
        axes[0].plot(x, [strata[key]["position_independent_recall"] for key in keys], linestyle="--", color=color, alpha=0.65, label=f"seed {seed} position")
    axes[0].set_xticks(x, keys, rotation=28)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("all-query target coverage")
    axes[0].set_xlabel("Teacher event distance")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    labels = ["selected match", "latent typed", "wrong type nearby", "no nearby proposal"]
    decomposition = []
    for record in records:
        target = record["support"]["target_count"]
        selected = record["threshold"]["metrics"]["true_positive"]
        typed = record["support"]["typed_one_to_one_oracle"]["true_positive"]
        position = record["support"]["position_one_to_one_oracle"]["true_positive"]
        decomposition.append(np.asarray([selected, typed - selected, position - typed, target - position]) / target)
    bottom = np.zeros(3)
    for index, label in enumerate(labels):
        values = [row[index] for row in decomposition]
        axes[1].bar(np.arange(3), values, bottom=bottom, label=label)
        bottom += values
    axes[1].set_xticks(np.arange(3), ["seed 0", "seed 1", "seed 2"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction of Teacher targets")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7)

    false_empty = [record["threshold"]["empty_row_false_positive_events"] for record in records]
    total_fp = [record["threshold"]["metrics"]["false_positive"] for record in records]
    false_nonempty = [total - empty for total, empty in zip(total_fp, false_empty, strict=True)]
    axes[2].bar(np.arange(3), false_empty, label="false events on empty rows", color="#c92a2a")
    axes[2].bar(np.arange(3), false_nonempty, bottom=false_empty, label="false events on nonempty rows", color="#f08c00")
    axes[2].set_xticks(np.arange(3), ["seed 0", "seed 1", "seed 2"])
    axes[2].set_ylabel("false predicted events at sealed threshold")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=7)
    short = {
        "DUPLICATE_PROPOSAL_CORRECTIVE_REQUIRED": "duplicate proposal failure",
        "EXPLICIT_OBJECTNESS_CARDINALITY_CORRECTIVE_REQUIRED": "objectness/cardinality failure",
        "TYPE_CONDITIONING_CORRECTIVE_REQUIRED": "event type conditioning failure",
        "STRUCTURED_POLAR_PROPOSAL_REQUIRED": "free-query spatial proposal failure",
    }[decision]
    figure.suptitle(f"Geometry-anchored proposal attribution: {short}; exclusive F1={baseline['f1']:.3f}")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_geometry_anchored_proposal_failure_attribution_v1.{suffix}", dpi=220)
    plt.close(figure)


def _population_summary(targets) -> dict[str, object]:
    mask = targets["event_mask"].astype(bool)
    cardinality = targets["event_mask"].sum(axis=1).astype(np.int64)
    return {
        "worlds": 20,
        "observations": len(mask),
        "target_tokens": int(mask.sum()),
        "target_types": np.bincount(
            targets["event_type_index"][mask].astype(np.int64), minlength=2
        ).astype(int).tolist(),
        "cardinality": np.bincount(cardinality, minlength=6).astype(int).tolist(),
        "event_identities": int(len(np.unique(targets["event_identity_index"][mask]))),
    }


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
    if evaluation.get("status") != "FAIL_GSE_GEOMETRY_ANCHORED_JOINT_CAPACITY_V1":
        raise RuntimeError("attribution requires sealed geometry-anchored capacity FAIL")
    baseline = evaluation["exclusive_single_center_baseline"]["selected_threshold_metrics"]
    records = []
    for seed in (0, 1, 2):
        model_dir = capacity / f"artifacts/models/seed{seed}"
        model_summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
        if model_summary.get("seed") != seed or model_summary.get("optimizer_steps") != 9096:
            raise RuntimeError(f"seed summary drift: {seed}")
        threshold = float(model_summary["selected_threshold_metrics"]["threshold"])
        predictions = _load_predictions(model_dir / "selection_outputs.npz", rows, global_index)
        threshold_record = _threshold_diagnostics(predictions, targets, threshold)
        if threshold_record["metrics"] != model_summary["selected_threshold_metrics"]:
            raise RuntimeError(f"seed sealed metric replay drift: {seed}")
        nms_predictions, nms_counts = _nms(predictions, threshold)
        records.append({
            "seed": seed,
            "threshold_value": threshold,
            "threshold": threshold_record,
            "nms": nms_counts,
            "nms_metrics": _metrics(nms_predictions, targets, threshold),
            "support": _proposal_support(predictions, targets),
        })
    duplicate_gate = all(record["nms_metrics"]["f1"] >= baseline["f1"] + GAIN_F1 for record in records)
    typed_gate = all(record["support"]["typed_one_to_one_oracle"]["recall"] >= baseline["recall"] + GAIN_RECALL for record in records)
    position_gate = all(record["support"]["position_one_to_one_oracle"]["recall"] >= baseline["recall"] + GAIN_RECALL for record in records)
    if duplicate_gate:
        decision = "DUPLICATE_PROPOSAL_CORRECTIVE_REQUIRED"
    elif typed_gate:
        decision = "EXPLICIT_OBJECTNESS_CARDINALITY_CORRECTIVE_REQUIRED"
    elif position_gate:
        decision = "TYPE_CONDITIONING_CORRECTIVE_REQUIRED"
    else:
        decision = "STRUCTURED_POLAR_PROPOSAL_REQUIRED"
    summary = {
        "schema_version": "gse_geometry_anchored_proposal_failure_attribution_v1",
        "status": PASS,
        "decision": decision,
        "decision_gates": {
            "all_seed_nms_f1_at_least_exclusive_plus_0p05": duplicate_gate,
            "all_seed_typed_oracle_recall_at_least_exclusive_plus_0p10": typed_gate,
            "all_seed_position_oracle_recall_at_least_exclusive_plus_0p10": position_gate,
            "exclusive_reference": {"f1": baseline["f1"], "recall": baseline["recall"]},
        },
        "population": _population_summary(targets),
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
    (output / "figure_source.json").write_text(json.dumps({"schema_version": "gse_geometry_anchored_proposal_failure_attribution_figure_source_v1", "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(output, records, baseline, decision)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
