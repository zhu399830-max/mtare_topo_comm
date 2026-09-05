#!/usr/bin/env python3
"""Attribute the sealed spatial-set failure without training or threshold changes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.gse_spatial_event_set_cache import load_spatial_event_teacher
from mtare_topo.evaluation.gse_spatial_event_set_metrics import (
    _maximum_valid_matching,
    spatial_event_set_metrics,
)


PASS = "PASS_GSE_SPATIAL_EVENT_SET_FAILURE_ATTRIBUTION_V1"
DISTANCE_EDGES_M = (0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0, 50.0001)
MATCH_CAP_M = 4.0
EXCLUSIVE_REFERENCE_F1 = 0.39130627633621645
EXCLUSIVE_REFERENCE_RECALL = 0.26284897059261686


def _family(parent_id: str) -> str:
    value = re.sub(r"^S\d+_", "", parent_id)
    return re.sub(r"_C\d+$", "", value)


def _nms_predictions(predictions: dict[str, np.ndarray], threshold: float):
    confidence = predictions["confidence"].copy()
    suppressed = 0
    selected_before = int(np.sum(confidence >= threshold))
    for row in range(len(confidence)):
        for event_type in (0, 1):
            candidates = np.flatnonzero(
                (confidence[row] >= threshold)
                & (predictions["event_type"][row] == event_type)
            )
            ordered = sorted(
                candidates.tolist(),
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
        "selected_before": selected_before,
        "suppressed": suppressed,
        "suppressed_fraction": suppressed / selected_before if selected_before else 0.0,
        "selected_after": selected_before - suppressed,
    }


def _target_coverage(
    predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray],
    parents: np.ndarray, threshold: float,
):
    strata: dict[str, dict[str, list[int]]] = {
        "distance": defaultdict(lambda: [0, 0, 0]),
        "cardinality": defaultdict(lambda: [0, 0, 0]),
        "type": defaultdict(lambda: [0, 0, 0]),
        "family": defaultdict(lambda: [0, 0, 0]),
    }
    total = [0, 0, 0]
    nearest_typed = []
    nearest_position = []
    target_distances = []
    for row in range(len(parents)):
        active = np.flatnonzero(targets["event_mask"][row].astype(bool))
        if len(active) == 0:
            continue
        selected = np.flatnonzero(predictions["confidence"][row] >= threshold)
        actual_matches = _maximum_valid_matching(
            predictions["event_type"][row, selected],
            predictions["relative_xyz_m"][row, selected],
            targets["event_type_index"][row, active],
            targets["event_relative_xyz_m"][row, active],
            maximum_error_m=MATCH_CAP_M,
        )
        actual_targets = {target_local for _, target_local, _ in actual_matches}
        for target_local, target_index in enumerate(active):
            target_type = int(targets["event_type_index"][row, target_index])
            target_xyz = targets["event_relative_xyz_m"][row, target_index]
            distance = float(np.linalg.norm(target_xyz))
            target_distances.append(distance)
            all_distances = np.linalg.norm(
                predictions["relative_xyz_m"][row] - target_xyz, axis=1
            )
            typed = all_distances[predictions["event_type"][row] == target_type]
            typed_nearest = float(np.min(typed)) if len(typed) else float("inf")
            position_nearest = float(np.min(all_distances))
            nearest_typed.append(typed_nearest)
            nearest_position.append(position_nearest)
            flags = (
                int(target_local in actual_targets),
                int(typed_nearest <= MATCH_CAP_M),
                int(position_nearest <= MATCH_CAP_M),
            )
            for index, value in enumerate(flags):
                total[index] += value
            bin_index = min(
                np.searchsorted(DISTANCE_EDGES_M, distance, side="right") - 1,
                len(DISTANCE_EDGES_M) - 2,
            )
            keys = {
                "distance": f"{DISTANCE_EDGES_M[bin_index]:g}-{DISTANCE_EDGES_M[bin_index + 1]:g}m",
                "cardinality": str(len(active)),
                "type": "terminal" if target_type == 0 else "junction",
                "family": _family(str(parents[row])),
            }
            for category, key in keys.items():
                values = strata[category][key]
                values[0] += 1
                values[1] += flags[0]
                values[2] += flags[1]
    target_count = len(target_distances)
    records = {}
    for category, values in strata.items():
        if category == "distance":
            ordered_keys = [
                f"{DISTANCE_EDGES_M[index]:g}-{DISTANCE_EDGES_M[index + 1]:g}m"
                for index in range(len(DISTANCE_EDGES_M) - 1)
                if f"{DISTANCE_EDGES_M[index]:g}-{DISTANCE_EDGES_M[index + 1]:g}m" in values
            ]
        else:
            ordered_keys = sorted(values)
        records[category] = {
            key: {
                "targets": values[key][0],
                "actual_recall": values[key][1] / values[key][0],
                "typed_query_oracle_recall": values[key][2] / values[key][0],
            }
            for key in ordered_keys
        }
    typed_array = np.asarray(nearest_typed, dtype=np.float64)
    typed_finite = typed_array[np.isfinite(typed_array)]
    position_array = np.asarray(nearest_position, dtype=np.float64)
    return {
        "targets": target_count,
        "actual_recall": total[0] / target_count,
        "typed_query_oracle_recall": total[1] / target_count,
        "position_only_query_oracle_recall": total[2] / target_count,
        "nearest_typed_query_error_m": {
            "finite_mean": float(np.mean(typed_finite)) if len(typed_finite) else None,
            "finite_median": float(np.median(typed_finite)) if len(typed_finite) else None,
            "finite_p90": float(np.quantile(typed_finite, 0.9)) if len(typed_finite) else None,
            "missing_same_type_query_count": int(np.sum(~np.isfinite(typed_array))),
            "missing_same_type_query_fraction": float(np.mean(~np.isfinite(typed_array))),
        },
        "nearest_position_query_error_m": {
            "mean": float(np.mean(position_array)),
            "median": float(np.median(position_array)),
            "p90": float(np.quantile(position_array, 0.9)),
        },
        "target_distance_m": {
            "mean": float(np.mean(target_distances)),
            "median": float(np.median(target_distances)),
            "p90": float(np.quantile(target_distances, 0.9)),
        },
        "strata": records,
    }


def _load_seed(path: Path, rows: np.ndarray, global_index: np.ndarray):
    with np.load(path, allow_pickle=False) as archive:
        if (
            not np.array_equal(archive["observation_row"], rows)
            or not np.array_equal(archive["global_sequence_index"], global_index)
        ):
            raise RuntimeError(f"seed output/Teacher join drift: {path}")
        return {
            "confidence": archive["confidence"].astype(np.float32),
            "event_type": archive["event_type"].astype(np.int8),
            "relative_xyz_m": archive["relative_xyz_m"].astype(np.float32),
        }


def _plot(output: Path, seed_records: list[dict], decision: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    distance_keys = list(seed_records[0]["coverage"]["strata"]["distance"])
    x = np.arange(len(distance_keys))
    for seed, record in enumerate(seed_records):
        values = record["coverage"]["strata"]["distance"]
        axes[0].plot(
            x,
            [values[key]["actual_recall"] for key in distance_keys],
            marker="o", label=f"seed {seed} actual",
        )
        axes[0].plot(
            x,
            [values[key]["typed_query_oracle_recall"] for key in distance_keys],
            linestyle="--", alpha=0.7, label=f"seed {seed} query oracle",
        )
    axes[0].set_xticks(x, distance_keys, rotation=30)
    axes[0].set_ylabel("target recall")
    axes[0].set_xlabel("Teacher event distance")
    axes[0].set_ylim(0.0, 1.0); axes[0].grid(alpha=0.25); axes[0].legend(fontsize=7)

    labels = ["sealed", "4 m NMS", "typed oracle"]
    width = 0.22
    for seed, record in enumerate(seed_records):
        values = [
            record["sealed_metrics"]["f1"],
            record["nms_metrics"]["f1"],
            record["coverage"]["typed_query_oracle_recall"],
        ]
        axes[1].bar(np.arange(3) + (seed - 1) * width, values, width, label=f"seed {seed}")
    axes[1].axhline(EXCLUSIVE_REFERENCE_F1, color="black", linestyle="--", label="exclusive F1")
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylim(0.0, 1.0); axes[1].set_ylabel("score")
    axes[1].grid(axis="y", alpha=0.25); axes[1].legend(fontsize=8)
    short_decision = (
        "new spatial encoder objective required"
        if decision.startswith("ENCODER_SPATIAL_REPRESENTATION")
        else decision.lower().replace("_", " ")
    )
    fig.suptitle(f"Spatial event-set failure attribution: {short_decision}")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_spatial_event_set_failure_attribution.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--capacity-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = load_spatial_event_teacher(args.teacher_root.resolve())
    rows = teacher.selection_rows
    global_index = teacher.global_sequence_index[rows]
    targets = teacher.targets(rows)
    parents = teacher.parent_id[rows]
    capacity = args.capacity_run.resolve()
    capacity_summary = json.loads((capacity / "metrics/capacity/summary.json").read_text(encoding="utf-8"))
    if capacity_summary.get("status") != "FAIL_GSE_SPATIAL_EVENT_SET_CAPACITY_V1":
        raise RuntimeError("failure attribution requires the sealed capacity FAIL")
    seed_records = []
    for seed in (0, 1, 2):
        seed_summary = json.loads(
            (capacity / f"artifacts/models/seed{seed}/summary.json").read_text(encoding="utf-8")
        )
        threshold = float(seed_summary["selected_threshold_metrics"]["threshold"])
        predictions = _load_seed(
            capacity / f"artifacts/models/seed{seed}/selection_outputs.npz",
            rows, global_index,
        )
        sealed_metrics = spatial_event_set_metrics(
            confidence=predictions["confidence"], predicted_type=predictions["event_type"],
            predicted_xyz_m=predictions["relative_xyz_m"],
            target_type=targets["event_type_index"], target_xyz_m=targets["event_relative_xyz_m"],
            target_mask=targets["event_mask"], threshold=threshold,
        )
        nms_predictions, duplicate = _nms_predictions(predictions, threshold)
        nms_metrics = spatial_event_set_metrics(
            confidence=nms_predictions["confidence"], predicted_type=nms_predictions["event_type"],
            predicted_xyz_m=nms_predictions["relative_xyz_m"],
            target_type=targets["event_type_index"], target_xyz_m=targets["event_relative_xyz_m"],
            target_mask=targets["event_mask"], threshold=threshold,
        )
        coverage = _target_coverage(predictions, targets, parents, threshold)
        seed_records.append({
            "seed": seed, "threshold": threshold, "sealed_metrics": sealed_metrics,
            "nms": duplicate, "nms_metrics": nms_metrics, "coverage": coverage,
            "predicted_radius_m": {
                "mean": float(np.mean(np.linalg.norm(predictions["relative_xyz_m"], axis=2))),
                "median": float(np.median(np.linalg.norm(predictions["relative_xyz_m"], axis=2))),
                "p90": float(np.quantile(np.linalg.norm(predictions["relative_xyz_m"], axis=2), 0.9)),
            },
        })
    nms_recovers = all(
        record["nms_metrics"]["f1"] >= EXCLUSIVE_REFERENCE_F1 + 0.05
        for record in seed_records
    )
    query_support = all(
        record["coverage"]["typed_query_oracle_recall"]
        >= EXCLUSIVE_REFERENCE_RECALL + 0.10
        for record in seed_records
    )
    if nms_recovers:
        decision = "DUPLICATE_QUERY_DOMINANT_MINIMAL_REPULSION_CORRECTIVE_ALLOWED"
    elif query_support:
        decision = "CONFIDENCE_CARDINALITY_DOMINANT_CALIBRATION_CORRECTIVE_ALLOWED"
    else:
        decision = "ENCODER_SPATIAL_REPRESENTATION_INSUFFICIENT_NEW_ENCODER_OBJECTIVE_REQUIRED"
    summary = {
        "schema_version": "gse_spatial_event_set_failure_attribution_v1",
        "status": PASS,
        "decision": decision,
        "decision_gates": {
            "all_seed_nms_f1_at_least_exclusive_plus_0p05": nms_recovers,
            "all_seed_typed_query_oracle_recall_at_least_exclusive_recall_plus_0p10": query_support,
            "exclusive_reference_f1": EXCLUSIVE_REFERENCE_F1,
            "exclusive_reference_recall": EXCLUSIVE_REFERENCE_RECALL,
        },
        "seed_records": seed_records,
        "population": {
            "worlds": 20, "observations": len(rows),
            "target_tokens": int(targets["event_mask"].sum()),
            "multi_event_rows": int(np.sum(targets["event_mask"].sum(axis=1) >= 2)),
        },
        "optimizer_steps": 0, "model_inference_frames": 0, "threshold_selection_steps": 0,
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure_source = {
        "schema_version": "gse_spatial_event_set_failure_attribution_figure_source_v1",
        "decision": decision, "seed_records": seed_records,
    }
    (output / "figure_source.json").write_text(
        json.dumps(figure_source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(output, seed_records, decision)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
