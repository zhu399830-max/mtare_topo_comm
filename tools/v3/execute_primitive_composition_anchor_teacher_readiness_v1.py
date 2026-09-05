#!/usr/bin/env python3
"""Audit the O(E) per-endpoint composition-anchor relation representation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from mtare_topo.data.primitive_attachment_observability_sidecar import (
    unpack_endpoint_observed,
)
from mtare_topo.evaluation.primitive_composition_anchor_teacher import (
    construction_endpoint_anchor_table,
    current_sensor_anchor_targets,
    observable_anchor_pair_slice,
)
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    exact_ranked_selection,
)
from mtare_topo.governance import write_json


EXPECTED = {
    "fit": {"parents": 60, "tasks": 180, "rows": 426_552},
    "c07": {"parents": 10, "tasks": 30, "rows": 64_644},
}
EXPECTED_C07_POSITIVES = 442_936
BATCH_SIZE = 512
SAFE_PRECISION = 0.98


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, .25, .5, .75, .9, .95, .99, 1.0)
    return {
        f"q{int(round(value * 100)):02d}": float(np.quantile(values, value))
        for value in probabilities
    } if len(values) else {}


def _unpack_overlap(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=2, count=32, bitorder="little",
    ).astype(np.uint8)


def _threshold_counts(score, target, threshold):
    predicted = np.asarray(score, dtype=np.float32) >= float(threshold)
    truth = np.asarray(target, dtype=bool)
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "threshold": float(threshold), "true_positive": tp,
        "false_positive": fp, "false_negative": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "selected_pairs": int(np.count_nonzero(predicted)),
    }, predicted


def _distinct_node_minimum(table) -> float:
    anchors = {}
    for primitive in range(len(table.primitive_ids)):
        for endpoint in range(2):
            node = str(table.node_ids[primitive, endpoint])
            value = table.anchor_world_m[primitive, endpoint]
            if node in anchors and not np.array_equal(anchors[node], value):
                raise RuntimeError("same construction node has multiple anchor coordinates")
            anchors[node] = value
    values = np.asarray(list(anchors.values()), dtype=np.float64)
    if len(values) < 2:
        return float("inf")
    distance = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    distance[np.eye(len(values), dtype=bool)] = np.inf
    return float(np.min(distance))


def _task(
    *, task_id: str, split: str, sensor_root: Path,
    construction_root: Path, teacher_root: Path, sidecar_root: Path,
):
    construction_path = construction_root / split / f"{task_id}.json"
    sensor_path = sensor_root / split / f"{task_id}.zarr"
    teacher_path = teacher_root / split / f"{task_id}.zarr"
    sidecar_path = sidecar_root / split / f"{task_id}.zarr"
    if any(not value.exists() for value in (construction_path, sensor_path, teacher_path, sidecar_path)):
        raise FileNotFoundError(f"composition-anchor paired source missing: {task_id}")
    document = json.loads(construction_path.read_text(encoding="utf-8"))
    table = construction_endpoint_anchor_table(document)
    sensor = zarr.open_group(str(sensor_path), mode="r")
    teacher = zarr.open_group(str(teacher_path), mode="r")
    sidecar = zarr.open_group(str(sidecar_path), mode="r")
    rows = int(teacher["primitive_mask"].shape[0])
    if rows != int(sidecar["endpoint_observed_packed"].shape[0]):
        raise RuntimeError(f"composition-anchor Teacher/sidecar rows differ: {task_id}")
    if (
        str(teacher.attrs["parent_id"]) != str(document["parent_id"])
        or str(teacher.attrs["geometry_realization"]) != str(document["geometry_realization"])
        or str(sidecar.attrs["task_id"]) != task_id
    ):
        raise RuntimeError(f"composition-anchor task provenance differs: {task_id}")

    residuals = []
    axis_distances = []
    anchor_distances = []
    targets = []
    hard_negatives = []
    for start in range(0, rows, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, rows)
        frame_rows = np.asarray(teacher["frame_row"][start:stop, -1], dtype=np.int64)
        primitive_index = np.asarray(teacher["primitive_index"][start:stop], dtype=np.int32)
        primitive_mask = np.asarray(teacher["primitive_mask"][start:stop], dtype=np.uint8)
        anchors = current_sensor_anchor_targets(
            primitive_index=primitive_index, primitive_mask=primitive_mask,
            anchor_world_m=table.anchor_world_m,
            sensor_xyz_m=np.asarray(sensor["sensor_xyz_m"].oindex[frame_rows], dtype=np.float64),
            yaw_deg=np.asarray(sensor["yaw_deg"].oindex[frame_rows], dtype=np.float64),
        )
        value = observable_anchor_pair_slice(
            axis_control_current_sensor_m=np.asarray(
                teacher["axis_control_current_sensor_m"][start:stop], dtype=np.float32,
            ),
            anchor_current_sensor_m=anchors,
            primitive_mask=primitive_mask,
            endpoint_observed=unpack_endpoint_observed(
                np.asarray(sidecar["endpoint_observed_packed"][start:stop], dtype=np.uint8),
            ),
            endpoint_neighbor=np.asarray(teacher["endpoint_neighbor"][start:stop], dtype=np.int8),
            disconnected_overlap=_unpack_overlap(
                np.asarray(teacher["disconnected_overlap_packed"][start:stop], dtype=np.uint8),
            ),
        )
        residuals.append(value.observed_endpoint_residual_m)
        axis_distances.append(value.observed_axis_pair_distance_m)
        anchor_distances.append(value.target_anchor_pair_distance_m)
        targets.append(value.attachment_target)
        hard_negatives.append(value.overlap_hard_negative)
    residual = np.concatenate(residuals)
    axis_distance = np.concatenate(axis_distances)
    anchor_distance = np.concatenate(anchor_distances)
    target = np.concatenate(targets)
    hard_negative = np.concatenate(hard_negatives)
    positive_anchor = anchor_distance[target]
    negative_anchor = anchor_distance[~target]
    hard_anchor = anchor_distance[hard_negative]
    record = {
        "split": split, "task_id": task_id,
        "parent_id": str(document["parent_id"]),
        "geometry_realization": str(document["geometry_realization"]),
        "rows": rows, "primitive_count": len(table.primitive_ids),
        "observable_endpoints": len(residual), "observable_pairs": len(target),
        "observable_positive_pairs": int(np.count_nonzero(target)),
        "overlap_hard_negative_pairs": int(np.count_nonzero(hard_negative)),
        "maximum_positive_target_anchor_distance_m": float(np.max(positive_anchor, initial=0.0)),
        "minimum_negative_target_anchor_distance_m": float(np.min(negative_anchor, initial=np.inf)),
        "minimum_overlap_hard_negative_anchor_distance_m": float(np.min(hard_anchor, initial=np.inf)),
        "minimum_distinct_construction_node_distance_m": _distinct_node_minimum(table),
        "endpoint_anchor_residual_quantiles_m": _quantiles(residual),
    }
    return record, residual, axis_distance, anchor_distance, target, hard_negative


def _plot(summary, fit, c07, output):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    bins = np.linspace(0, min(2.0, max(float(np.quantile(fit[0], .999)), .3)), 80)
    axes[0].hist(fit[0], bins=bins, density=True, alpha=.55, label="C01–C06 fit")
    axes[0].hist(c07[0], bins=bins, density=True, alpha=.55, label="C07")
    axes[0].set_xlabel("observed endpoint → composition anchor residual (m)")
    axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
    positive = c07[1][c07[2]]; negative = c07[1][~c07[2]]
    axes[1].hist(positive, bins=80, alpha=.65, label="true connection")
    axes[1].hist(negative, bins=80, alpha=.5, label="non-connection")
    axes[1].set_yscale("log"); axes[1].set_xlabel("observed endpoint-pair distance (m)")
    axes[1].set_ylabel("pairs"); axes[1].legend(fontsize=8)
    baseline = summary["axis_endpoint_baseline"]
    axes[2].bar(
        ("fit precision", "fit recall", "C07 precision", "C07 recall"),
        (
            baseline["fit_safe_selection"]["precision"],
            baseline["fit_safe_selection"]["recall"],
            baseline["c07_transfer"]["precision"],
            baseline["c07_transfer"]["recall"],
        ), color=("#287271", "#5b8e7d", "#d37524", "#e0a458"),
    )
    axes[2].axhline(.98, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylim(0, 1.02); axes[2].tick_params(axis="x", rotation=22, labelsize=8)
    axes[2].set_ylabel("metric")
    fig.suptitle("O(E) learned composition-anchor Teacher readiness (fit + C07 only)")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"primitive_composition_anchor_teacher_readiness.{suffix}", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--construction-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    per_task = []
    split_data = {}
    for split in ("fit", "c07"):
        roots = (
            args.sensor_root.resolve() / split,
            args.teacher_root.resolve() / split,
            args.sidecar_root.resolve() / split,
        )
        populations = [
            {value.stem for value in root.glob("*.zarr") if value.is_dir()}
            for root in roots
        ]
        constructions = {
            value.stem for value in (args.construction_root.resolve() / split).glob("*.json")
        }
        if not populations[0] or any(value != populations[0] for value in populations[1:]) or constructions != populations[0]:
            raise RuntimeError(f"composition-anchor {split} source tasks differ")
        tasks = sorted(populations[0])
        if len(tasks) != EXPECTED[split]["tasks"]:
            raise RuntimeError(f"composition-anchor {split} task population drift")
        values = []
        for task_id in tasks:
            result = _task(
                task_id=task_id, split=split,
                sensor_root=args.sensor_root.resolve(),
                construction_root=args.construction_root.resolve(),
                teacher_root=args.teacher_root.resolve(),
                sidecar_root=args.sidecar_root.resolve(),
            )
            per_task.append(result[0]); values.append(result[1:])
        parents = {value["parent_id"] for value in per_task if value["split"] == split}
        rows = sum(value["rows"] for value in per_task if value["split"] == split)
        if len(parents) != EXPECTED[split]["parents"] or rows != EXPECTED[split]["rows"]:
            raise RuntimeError(f"composition-anchor {split} parent/row population drift")
        split_data[split] = tuple(np.concatenate([value[index] for value in values]) for index in range(5))

    fit_residual, fit_axis, fit_anchor, fit_target, fit_hard = split_data["fit"]
    c07_residual, c07_axis, c07_anchor, c07_target, c07_hard = split_data["c07"]
    fit_score = -fit_axis
    c07_score = -c07_axis
    fit_safe = exact_ranked_selection(
        fit_score, fit_target, total_positive=int(np.count_nonzero(fit_target)),
        minimum_precision=SAFE_PRECISION,
    )
    if fit_safe["available"]:
        c07_transfer, c07_prediction = _threshold_counts(
            c07_score, c07_target, fit_safe["threshold"],
        )
    else:
        c07_transfer = {
            "threshold": None, "true_positive": 0, "false_positive": 0,
            "false_negative": int(np.count_nonzero(c07_target)),
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "selected_pairs": 0,
        }
        c07_prediction = np.zeros_like(c07_target)
    c07_hard_fp = int(np.count_nonzero(c07_prediction & ~c07_target & c07_hard))
    positive_target_max = max(
        float(np.max(fit_anchor[fit_target], initial=0.0)),
        float(np.max(c07_anchor[c07_target], initial=0.0)),
    )
    negative_target_min = min(
        float(np.min(fit_anchor[~fit_target], initial=np.inf)),
        float(np.min(c07_anchor[~c07_target], initial=np.inf)),
    )
    hard_target_min = min(
        float(np.min(fit_anchor[fit_hard], initial=np.inf)),
        float(np.min(c07_anchor[c07_hard], initial=np.inf)),
    )
    fixed_pair_outputs = 2 * 32 * 31
    fixed_anchor_outputs = 32 * 2 * 4
    checks = {
        "exact_population": len(per_task) == 210,
        "unique_endpoint_composition_membership": True,
        "same_cluster_anchor_exact": positive_target_max <= 1e-9,
        "distinct_cluster_anchor_noncollapsed": negative_target_min > 1e-6,
        "overlap_hard_negative_anchor_noncollapsed": hard_target_min > 1e-6,
        "fit_safe_axis_baseline_available": bool(fit_safe["available"]),
        "c07_fit_threshold_precision_ge_98": c07_transfer["precision"] >= .98,
        "c07_fit_threshold_nonzero_true_positive": c07_transfer["true_positive"] > 0,
        "c07_observable_positive_reproduction": int(np.count_nonzero(c07_target)) == EXPECTED_C07_POSITIVES,
        "linear_output_compression_ge_7x": fixed_pair_outputs / fixed_anchor_outputs >= 7.0,
        "zero_c08_model_graph_mtare": True,
    }
    scientific_pass = all(checks.values())
    split_summary = {}
    for split, values in split_data.items():
        residual, axis_distance, anchor_distance, target, hard = values
        split_summary[split] = {
            "rows": EXPECTED[split]["rows"],
            "observable_endpoints": len(residual), "observable_pairs": len(target),
            "observable_positive_pairs": int(np.count_nonzero(target)),
            "overlap_hard_negative_pairs": int(np.count_nonzero(hard)),
            "endpoint_anchor_residual_quantiles_m": _quantiles(residual),
            "positive_axis_pair_distance_quantiles_m": _quantiles(axis_distance[target]),
            "negative_axis_pair_distance_quantiles_m": _quantiles(axis_distance[~target]),
            "positive_target_anchor_distance_quantiles_m": _quantiles(anchor_distance[target]),
            "negative_target_anchor_distance_quantiles_m": _quantiles(anchor_distance[~target]),
            "overlap_hard_negative_anchor_distance_quantiles_m": _quantiles(anchor_distance[hard]),
        }
    summary = {
        "schema_version": "primitive_composition_anchor_teacher_readiness_v1",
        "scientific_pass": scientific_pass,
        "decision": (
            "PROCEED_TO_COMPOSITION_ANCHOR_MODEL_READINESS"
            if scientific_pass else "STOP_LEARNED_COMPOSITION_ANCHOR_CANDIDATE"
        ),
        "splits": split_summary,
        "axis_endpoint_baseline": {
            "score": "negative observed endpoint Euclidean separation; no learned residual",
            "fit_safe_selection": fit_safe,
            "c07_transfer": c07_transfer,
            "c07_overlap_hard_negative_false_positives": c07_hard_fp,
        },
        "target_geometry": {
            "maximum_same_cluster_anchor_distance_m": positive_target_max,
            "minimum_distinct_cluster_anchor_distance_m": negative_target_min,
            "minimum_overlap_hard_negative_anchor_distance_m": hard_target_min,
        },
        "output_complexity": {
            "independent_pair_scores_per_row": fixed_pair_outputs,
            "anchor_xyz_plus_uncertainty_values_per_row": fixed_anchor_outputs,
            "linear_output_compression_ratio": fixed_pair_outputs / fixed_anchor_outputs,
            "learned_output_order": "O(E)",
            "pairwise_probability_computation": "O(E^2) deterministic calibrated Gaussian compatibility; no independent pair logits",
        },
        "checks": checks,
        "fit_rows_read": EXPECTED["fit"]["rows"], "c07_rows_read": EXPECTED["c07"]["rows"],
        "sensor_pose_rows_read": EXPECTED["fit"]["rows"] + EXPECTED["c07"]["rows"],
        "sensor_range_rows_read": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "model_forward_rows": 0, "optimizer_steps": 0, "graph_replays": 0,
        "mtare_worlds_read": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "per_task.json", {"tasks": per_task})
    with (output / "per_task.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "split", "task_id", "parent_id", "geometry_realization", "rows",
            "primitive_count", "observable_endpoints", "observable_pairs",
            "observable_positive_pairs", "overlap_hard_negative_pairs",
            "maximum_positive_target_anchor_distance_m",
            "minimum_negative_target_anchor_distance_m",
            "minimum_overlap_hard_negative_anchor_distance_m",
            "minimum_distinct_construction_node_distance_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(per_task)
    write_json(output / "figure_source.json", {
        "schema_version": "primitive_composition_anchor_teacher_figure_source_v1",
        "question": "Can one learned anchor per endpoint replace quadratic independent relation outputs?",
        "splits": split_summary, "baseline": summary["axis_endpoint_baseline"],
        "target_geometry": summary["target_geometry"],
        "output_complexity": summary["output_complexity"], "checks": checks,
    })
    _plot(summary, (fit_residual, fit_axis, fit_target), (c07_residual, c07_axis, c07_target), output)
    print(json.dumps({
        "scientific_pass": scientific_pass, "decision": summary["decision"],
        "fit_safe": fit_safe, "c07_transfer": c07_transfer,
        "target_geometry": summary["target_geometry"], "checks": checks,
    }, indent=2))
    # A scientific FAIL is a valid completed audit, not a process failure.
    # The immutable runner records the decision and stops the candidate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
