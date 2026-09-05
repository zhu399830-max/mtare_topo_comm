#!/usr/bin/env python3
"""Execute the C01-C08-only causal geometry/risk-conflict audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.evaluation.gse_causal_geometry_risk_conflict import (
    causal_lag_pairs,
    change_confidence_decomposition,
    fixed_risk_summary,
    hard_negative_tail_summary,
    width_height_change_score,
)
from mtare_topo.governance import write_json


PASS_STATUS = "PASS_GSE_CAUSAL_GEOMETRY_RISK_CONFLICT_AUDIT_V1"
GEOMETRY_NAMES = ("width_m", "height_m", "slope_deg", "curvature_per_m")
FEATURE_SCALE = np.asarray((30.0, 30.0, 45.0, 0.1), dtype=np.float64)
EVENT_INDEX = {
    "corridor": 0,
    "junction": 1,
    "terminal": 2,
    "turn": 3,
    "geometry_transition": 4,
}
LAGS = tuple(range(2, 13))


def _read_teacher(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 188126:
        raise RuntimeError(f"corrected Teacher population drift: {len(rows)}")
    return rows


def _selection_output(path: Path, expected_global: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        global_index = np.asarray(archive["global_sequence_index"], dtype=np.int64)
        probability = np.asarray(archive["probability"], dtype=np.float64)
    if not np.array_equal(global_index, expected_global) or probability.shape != (45942, 5):
        raise RuntimeError(f"selection output identity/shape drift: {path}")
    if not np.all(np.isfinite(probability)) or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-5
    ):
        raise RuntimeError(f"selection probabilities invalid: {path}")
    return probability


def _factor_identity(rows: list[dict]) -> np.ndarray:
    identities = sorted(
        {
            str(row["identity"])
            for row in rows
            if row["event"] != "corridor" and row.get("identity") is not None
        }
    )
    code = {value: index for index, value in enumerate(identities)}
    result = np.full(len(rows), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        if row["event"] != "corridor":
            try:
                result[index] = code[str(row["identity"])]
            except KeyError as exc:
                raise RuntimeError("structural Teacher row lacks identity") from exc
    return result


def _write_figure(
    output: Path,
    confidence: dict[str, dict],
    lag_rows: list[dict],
    hard_negative_rows: list[dict],
) -> None:
    sources = tuple(confidence)
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    x = np.arange(len(sources))
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        [confidence[name]["median_structural_probability"] for name in sources],
        width,
        label="structural confidence",
        color="#376996",
    )
    axes[0, 0].bar(
        x + width / 2,
        [confidence[name]["median_conditional_change_probability"] for name in sources],
        width,
        label="conditional change class",
        color="#E49B32",
    )
    axes[0, 0].set_xticks(x, sources, rotation=16)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Median probability on corrected change frames")
    axes[0, 0].set_title("(a) Class evidence and structural acceptance diverge")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.2)

    for source, color in (("Teacher", "#2A9D8F"), ("Frozen prediction", "#376996")):
        subset = [row for row in lag_rows if row["source"] == source]
        axes[0, 1].plot(
            [row["lag_steps"] for row in subset],
            [row["transition_vs_corridor_auc"] for row in subset],
            marker="o",
            label=source,
            color=color,
        )
    axes[0, 1].axvspan(7, 11, alpha=0.12, color="#E49B32", label="Teacher confirmation delay")
    axes[0, 1].set_ylim(0.6, 0.9)
    axes[0, 1].set_xlabel("Causal geometry lag (m / sequence steps)")
    axes[0, 1].set_ylabel("Transition-vs-corridor ROC-AUC")
    axes[0, 1].set_title("(b) Longer history improves average separability")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    for source, color in (("Teacher", "#2A9D8F"), ("Frozen prediction", "#376996")):
        subset = [row for row in lag_rows if row["source"] == source]
        axes[1, 0].plot(
            [row["lag_steps"] for row in subset],
            [row["covered_change_identities"] for row in subset],
            marker="o",
            label=source,
            color=color,
        )
    axes[1, 0].axhline(7, linestyle="--", color="#C8553D", label="7/17 event gate")
    axes[1, 0].set_xlabel("Causal geometry lag (m / sequence steps)")
    axes[1, 0].set_ylabel("Identities at fit-frozen 1% corridor threshold")
    axes[1, 0].set_ylim(-0.5, 17.5)
    axes[1, 0].set_title("(c) Scalar low-risk coverage remains absent")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    pred_tail = [
        row for row in hard_negative_rows if row["source"] == "Frozen prediction"
    ]
    old = np.asarray([row["removed_old_transition_rows"] for row in pred_tail])
    ordinary = np.asarray([row["ordinary_corridor_rows"] for row in pred_tail])
    lag = np.asarray([row["lag_steps"] for row in pred_tail])
    axes[1, 1].bar(lag, ordinary, label="ordinary corrected corridor", color="#9AA5B1")
    axes[1, 1].bar(
        lag,
        old,
        bottom=ordinary,
        label="removed old transition",
        color="#C8553D",
    )
    axes[1, 1].set_xlabel("Causal geometry lag (m / sequence steps)")
    axes[1, 1].set_ylabel("Selection corridor rows above fit threshold")
    axes[1, 1].set_title("(d) Old labels explain only part of the hard tail")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--feature", action="append", required=True, type=Path)
    parser.add_argument("--old-directional-selection", required=True, type=Path)
    parser.add_argument("--frozen-multitask-selection", required=True, type=Path)
    parser.add_argument("--last-block-selection", required=True, type=Path)
    args = parser.parse_args()
    if len(args.feature) != 3:
        raise RuntimeError("risk-conflict audit requires exactly three feature arrays")
    run_dir = args.run_dir.resolve()
    artifacts = run_dir / "artifacts"
    metrics_dir = run_dir / "metrics"
    previews = run_dir / "previews"
    for directory in (artifacts, metrics_dir, previews):
        directory.mkdir(parents=True, exist_ok=True)

    rows = _read_teacher(args.teacher)
    with np.load(args.pair_cache, allow_pickle=False) as archive:
        global_index = np.asarray(archive["compact_to_global_sequence_index"], dtype=np.int64)
        partition = np.asarray(archive["partition_code"], dtype=np.uint8)
        parent = np.asarray(archive["parent_id"], dtype=str)
    teacher_global = np.asarray([int(row["global_sequence_index"]) for row in rows])
    teacher_parent = np.asarray([str(row["parent_id"]) for row in rows])
    if (
        not np.array_equal(global_index, teacher_global)
        or not np.array_equal(parent, teacher_parent)
        or tuple(np.bincount(partition, minlength=2)) != (142184, 45942)
        or any(value.endswith(("_C09", "_C10")) for value in parent)
    ):
        raise RuntimeError("risk-conflict compact population/split drift")
    event = np.asarray([EVENT_INDEX[str(row["event"])] for row in rows], dtype=np.int8)
    identity = _factor_identity(rows)
    traversal = np.asarray([str(row["traversal_id"]) for row in rows])
    sequence = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)
    geometry = np.asarray(
        [
            [float(row[name]) if row[name] is not None else np.nan for name in GEOMETRY_NAMES]
            for row in rows
        ],
        dtype=np.float64,
    )
    geometry_valid = np.asarray(
        [
            bool(row.get("geometry_valid")) and np.all(np.isfinite(geometry[index]))
            for index, row in enumerate(rows)
        ],
        dtype=np.bool_,
    )
    dataset = GSESequenceDataset(args.dataset_run, "train", augment_azimuth=False)
    row_by_global = {
        int(record["global_sequence_index"]): index
        for index, record in enumerate(dataset.records)
    }
    try:
        dataset_rows = np.asarray([row_by_global[int(value)] for value in global_index])
    except KeyError as exc:
        raise RuntimeError("risk-conflict population absent from LiDAR dataset") from exc
    old_event = dataset.event_labels()[dataset_rows].astype(np.int8)

    predicted_geometry = []
    original_probability = []
    for path in args.feature:
        feature = np.load(path, mmap_mode="r")
        if feature.shape != (188126, 146) or feature.dtype != np.float32:
            raise RuntimeError(f"risk-conflict feature drift: {path}")
        predicted_geometry.append(np.asarray(feature[:, 8:12], dtype=np.float64) * FEATURE_SCALE)
        probability = np.asarray(feature[:, :5], dtype=np.float64)
        probability /= probability.sum(axis=1, keepdims=True)
        original_probability.append(probability)
    predicted_mean = np.mean(np.stack(predicted_geometry), axis=0)
    selection = np.where(partition == 1)[0]
    expected_selection_global = global_index[selection]
    confidence = {
        "Original GSE": change_confidence_decomposition(
            np.mean(np.stack(original_probability), axis=0)[selection], event[selection]
        ),
        "Directional": change_confidence_decomposition(
            _selection_output(args.old_directional_selection, expected_selection_global),
            event[selection],
        ),
        "Frozen delta": change_confidence_decomposition(
            _selection_output(args.frozen_multitask_selection, expected_selection_global),
            event[selection],
        ),
        "Last block": change_confidence_decomposition(
            _selection_output(args.last_block_selection, expected_selection_global),
            event[selection],
        ),
    }

    lag_rows: list[dict] = []
    hard_negative_rows: list[dict] = []
    for lag in LAGS:
        current, past = causal_lag_pairs(
            traversal,
            sequence,
            partition,
            geometry_valid,
            lag_steps=lag,
        )
        for source, value in (("Teacher", geometry), ("Frozen prediction", predicted_mean)):
            score = width_height_change_score(value, current, past)
            summary = fixed_risk_summary(
                score,
                current,
                event,
                identity,
                partition,
                lag_steps=lag,
            )
            lag_rows.append({"source": source, **summary.to_dict()})
            hard_negative_rows.append(
                {
                    "source": source,
                    "lag_steps": lag,
                    **hard_negative_tail_summary(
                        score,
                        event[current],
                        old_event[current],
                        partition[current],
                        threshold=summary.fit_corridor_threshold,
                    ),
                }
            )
    lag4_pred = next(
        row
        for row in lag_rows
        if row["source"] == "Frozen prediction" and row["lag_steps"] == 4
    )
    longer_pred = [
        row
        for row in lag_rows
        if row["source"] == "Frozen prediction" and 7 <= row["lag_steps"] <= 11
    ]
    peak_longer_auc = max(float(row["transition_vs_corridor_auc"]) for row in longer_pred)
    scalar_safe = any(
        row["source"] == "Frozen prediction"
        and int(row["covered_change_identities"]) >= 7
        and float(row["selection_false_positive_rate"]) <= 0.01 + 1e-12
        for row in lag_rows
    )
    mechanism = {
        "longer_history_auc_gain_vs_lag4": peak_longer_auc
        - float(lag4_pred["transition_vs_corridor_auc"]),
        "longer_history_signal_material": peak_longer_auc
        - float(lag4_pred["transition_vs_corridor_auc"])
        >= 0.05,
        "scalar_one_percent_risk_sufficient": scalar_safe,
        "recommended_next": (
            "GEOMETRY_CONDITIONED_IDENTITY_LEVEL_RISK_CAPACITY_PROOF"
            if not scalar_safe
            else "FREEZE_SCALAR_CAUSAL_GEOMETRY_RISK_INTERFACE"
        ),
    }
    if (
        next(row for row in lag_rows if row["source"] == "Teacher" and row["lag_steps"] == 4)["fit_pairs"]
        != 92845
        or lag4_pred["selection_pairs"] != 29920
        or confidence["Original GSE"]["change_frames"] != 240
    ):
        raise RuntimeError("risk-conflict exact evidence count drift")

    def write_csv(name: str, rows_to_write: list[dict]) -> None:
        with (artifacts / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows_to_write[0]))
            writer.writeheader()
            writer.writerows(rows_to_write)

    write_csv("lag_risk_audit.csv", lag_rows)
    write_csv("hard_negative_tail.csv", hard_negative_rows)
    write_json(artifacts / "confidence_decomposition.json", confidence)
    _write_figure(
        previews / "gse_causal_geometry_risk_conflict",
        confidence,
        lag_rows,
        hard_negative_rows,
    )
    result = {
        "schema_version": "gse_causal_geometry_risk_conflict_audit_v1",
        "overall_status": PASS_STATUS,
        "scientific_qualification": False,
        "fit_worlds": 60,
        "fit_observations": 142184,
        "selection_worlds": 20,
        "selection_observations": 45942,
        "selection_change_identities": 17,
        "lags": list(LAGS),
        "confidence_decomposition": confidence,
        "lag_risk_audit": lag_rows,
        "hard_negative_tail": hard_negative_rows,
        "mechanism_conclusion": mechanism,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    # Keep the scientific payload separate from the runner's outer execution
    # summary.  The latter records resource bounds, source immutability and the
    # one-shot state transition without overwriting these audit measurements.
    write_json(metrics_dir / "risk_conflict_summary.json", result)
    write_json(previews / "gse_causal_geometry_risk_conflict_source.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
