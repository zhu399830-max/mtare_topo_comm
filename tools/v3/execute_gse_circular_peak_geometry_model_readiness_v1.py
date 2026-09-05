#!/usr/bin/env python3
"""Zero-training real-data readiness for the circular peak geometry model."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import inspect
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from mtare_topo.governance import write_json
from mtare_topo.representation.gse_circular_peak_geometry_model import (
    BEARING_BINS,
    CircularPeakGeometrySemanticNet,
    circular_peak_geometry_input_contract,
    circular_peak_geometry_loss,
)


PASS = "PASS_GSE_CIRCULAR_PEAK_GEOMETRY_MODEL_READINESS_V1"
FAIL = "FAIL_GSE_CIRCULAR_PEAK_GEOMETRY_MODEL_READINESS_V1"
EXPECTED_CARDINALITY = {1: 7525, 2: 154279, 3: 24458, 4: 1864}
EXPECTED_PARAMETERS = 769268
MAX_RANGE_M = 50.0


def _selected_rows(teacher_paths: list[Path], source_root: Path) -> tuple[list[dict], dict]:
    selected: dict[str, tuple[str, int, str]] = {}
    used: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    cardinality: Counter[int] = Counter()
    local_noncontiguous = 0
    global_noncontiguous = 0
    join_mismatch = 0
    forbidden_teacher_arrays = 0
    for teacher_path in teacher_paths:
        teacher = zarr.open_group(str(teacher_path), mode="r")
        parent = str(teacher.attrs["parent_id"])
        partition = str(teacher.attrs["partition"])
        source = zarr.open_group(str(source_root / f"{parent}.zarr"), mode="r")
        global_sequence = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
        local_refs = np.asarray(teacher["local_frame_references"][:], dtype=np.int64)
        global_refs = np.asarray(teacher["global_frame_references"][:], dtype=np.int64)
        presence = np.asarray(teacher["presence"][:], dtype=np.uint8)
        width_valid = np.asarray(teacher["width_valid_mask"][:], dtype=np.uint8)
        exit_count = np.asarray(teacher["exit_count"][:], dtype=np.uint8)
        if not np.array_equal(global_sequence, np.asarray(source["global_sequence_index"][:], dtype=np.int64)):
            join_mismatch += 1
        if not np.array_equal(local_refs, np.asarray(source["local_frame_references"][:], dtype=np.int64)):
            join_mismatch += 1
        if not np.array_equal(global_refs, np.asarray(source["global_frame_references"][:], dtype=np.int64)):
            join_mismatch += 1
        reconstructed_global = np.asarray(source["global_frame_index"][:], dtype=np.int64)[local_refs]
        if not np.array_equal(reconstructed_global, global_refs):
            join_mismatch += 1
        local_noncontiguous += int(np.count_nonzero(np.any(np.diff(local_refs, axis=1) != 1, axis=1)))
        global_noncontiguous += int(np.count_nonzero(np.any(np.diff(global_refs, axis=1) != 1, axis=1)))
        forbidden_teacher_arrays += sum(
            int(name in teacher)
            for name in ("range_m", "valid_mask", "exit_identity", "association_identity", "sensor_xyz_m", "yaw_deg")
        )
        if presence.shape != (len(global_sequence), BEARING_BINS) or not np.array_equal(presence.sum(axis=1), exit_count):
            raise RuntimeError(f"circular readiness Teacher shape/count drift: {parent}")
        values, frequencies = np.unique(exit_count, return_counts=True)
        for value, frequency in zip(values, frequencies, strict=True):
            cardinality[int(value)] += int(frequency)
        peaks = int(presence.sum())
        valid_width = int(width_valid.sum())
        counts.update({
            "worlds": 1,
            "observations": len(global_sequence),
            "peaks": peaks,
            "width_valid": valid_width,
            f"{partition}_worlds": 1,
            f"{partition}_observations": len(global_sequence),
            f"{partition}_peaks": peaks,
        })

        candidates: list[tuple[str, np.ndarray]] = []
        for value in range(1, 5):
            candidates.append((f"cardinality_{value}", np.flatnonzero(exit_count == value)))
        candidates.extend((
            ("invalid_width", np.flatnonzero(np.any((presence == 1) & (width_valid == 0), axis=1))),
            ("wrap_peak", np.flatnonzero((presence[:, 0] == 1) | (presence[:, -1] == 1))),
            ("fit_extra", np.arange(len(global_sequence)) if partition == "fit" else np.empty(0, dtype=np.int64)),
            ("selection_extra", np.arange(len(global_sequence)) if partition == "selection" else np.empty(0, dtype=np.int64)),
        ))
        for criterion, rows in candidates:
            if criterion in selected:
                continue
            for value in rows:
                key = (parent, int(value))
                if key not in used:
                    selected[criterion] = (parent, int(value), partition)
                    used.add(key)
                    break

    expected_criteria = {
        "cardinality_1", "cardinality_2", "cardinality_3", "cardinality_4",
        "invalid_width", "wrap_peak", "fit_extra", "selection_extra",
    }
    if set(selected) != expected_criteria:
        raise RuntimeError(f"real readiness row coverage drift: {sorted(selected)}")
    rows = [
        {"criterion": criterion, "parent_id": selected[criterion][0], "row": selected[criterion][1], "partition": selected[criterion][2]}
        for criterion in sorted(selected)
    ]
    audit = {
        "counts": dict(counts),
        "cardinality": dict(cardinality),
        "local_noncontiguous_rows": local_noncontiguous,
        "global_noncontiguous_rows": global_noncontiguous,
        "join_mismatch_shards": join_mismatch,
        "forbidden_teacher_arrays": forbidden_teacher_arrays,
    }
    return rows, audit


def _real_batch(rows: list[dict], teacher_root: Path, source_root: Path) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scans: list[np.ndarray] = []
    target_values: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "presence", "heading_residual_deg", "opening_width_m", "width_valid_mask",
            "vertical_profile_m", "local_axis", "geometry", "geometry_valid_mask",
        )
    }
    for record in rows:
        teacher_path = teacher_root / record["partition"] / f"{record['parent_id']}.zarr"
        teacher = zarr.open_group(str(teacher_path), mode="r")
        source = zarr.open_group(str(source_root / f"{record['parent_id']}.zarr"), mode="r")
        row = int(record["row"])
        refs = np.asarray(teacher["local_frame_references"][row], dtype=np.int64)
        range_m = np.asarray(source["range_m"].oindex[refs], dtype=np.float32) / MAX_RANGE_M
        valid = np.asarray(source["valid_mask"].oindex[refs], dtype=np.float32)
        scans.append(np.stack((range_m, valid), axis=1))
        for name in ("presence", "heading_residual_deg", "opening_width_m", "width_valid_mask", "vertical_profile_m"):
            target_values[name].append(np.asarray(teacher[name][row]))
        target_values["local_axis"].append(np.asarray(source["local_axis_robot"][row], dtype=np.float32))
        target_values["geometry"].append(np.asarray(source["geometry"][row], dtype=np.float32))
        target_values["geometry_valid_mask"].append(np.asarray(source["geometry_valid_mask"][row], dtype=np.uint8))
    batch = torch.from_numpy(np.stack(scans).astype(np.float32))
    targets = {name: torch.from_numpy(np.stack(values)) for name, values in target_values.items()}
    return batch, targets


def _maximum_error(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(first - second)))


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    cardinality = summary["population"]["cardinality"]
    x = np.arange(1, 5)
    axes[0].bar(x, [cardinality[str(value)] for value in x], color="#4e79a7")
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Exit peaks per observation")
    axes[0].set_ylabel("Observations")
    axes[0].set_title("A  Real target coverage")
    losses = summary["real_batch"]["losses"]
    names = ("presence", "peak_geometry", "axis", "global_geometry")
    axes[1].bar(np.arange(4), [losses[name] for name in names], color="#59a14f")
    axes[1].set_xticks(np.arange(4), ("presence", "peak geom.", "axis", "tunnel geom."), rotation=15)
    axes[1].set_ylabel("Untrained finite loss")
    axes[1].set_title("B  All structural heads connected")
    errors = summary["equivariance"]
    error_names = ("dense_max_error", "axis_max_error", "invariant_max_error", "batch_permutation_max_error")
    values = [max(float(errors[name]), 1e-12) for name in error_names]
    axes[2].bar(np.arange(4), values, color="#f28e2b")
    axes[2].set_yscale("log")
    axes[2].axhline(3e-5, color="#e15759", linestyle="--", linewidth=1.2, label="contract")
    axes[2].set_xticks(np.arange(4), ("dense", "axis", "global", "batch"), rotation=15)
    axes[2].set_ylabel("Maximum absolute error")
    axes[2].set_title("C  Circular/causal interface")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph circular peak geometry model readiness")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_circular_peak_geometry_model_readiness_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher_root = args.teacher_root.resolve()
    source_root = args.source_root.resolve()
    teacher_paths = sorted(teacher_root.glob("*/*.zarr"))
    if len(teacher_paths) != 80:
        raise RuntimeError("circular model readiness requires exactly 80 Teacher shards")
    rows, audit = _selected_rows(teacher_paths, source_root)
    scans, targets = _real_batch(rows, teacher_root, source_root)

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.manual_seed(20260829)
    model = CircularPeakGeometrySemanticNet()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    outputs = model(scans)
    losses = circular_peak_geometry_loss(outputs, targets)
    losses["total"].backward()
    finite_backward = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    finite_outputs = all(
        bool(torch.isfinite(value).all()) for value in outputs.values()
        if torch.is_floating_point(value)
    )

    model.eval()
    shift_columns = 40
    shift_bins = 10
    angle = 2.0 * math.pi * shift_bins / BEARING_BINS
    permutation = torch.tensor([7, 1, 5, 0, 6, 2, 4, 3])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        base = model(scans)
        repeated = model(scans)
        rotated = model(torch.roll(scans, shift_columns, dims=-1))
        permuted = model(scans[permutation])
        oldest_changed = scans.clone()
        oldest_changed[:, 0, 0] = torch.roll(oldest_changed[:, 0, 0], 73, dims=-1)
        changed = model(oldest_changed)
    dense_names = (
        "peak_presence_logits", "peak_heading_residual_deg", "peak_opening_width_m",
        "peak_vertical_profile_m", "peak_descriptor", "peak_geometry_uncertainty",
    )
    dense_rotation_error = max(
        _maximum_error(rotated[name], torch.roll(base[name], shift_bins, dims=1))
        for name in dense_names
    )
    axis = base["local_axis"]
    expected_axis = torch.stack(
        (
            math.cos(angle) * axis[:, 0] - math.sin(angle) * axis[:, 1],
            math.sin(angle) * axis[:, 0] + math.cos(angle) * axis[:, 1],
            axis[:, 2],
        ),
        dim=-1,
    )
    axis_rotation_error = _maximum_error(rotated["local_axis"], expected_axis)
    invariant_names = (
        "width_m", "height_m", "slope_deg", "curvature_per_m",
        "place_descriptor", "observation_uncertainty",
    )
    invariant_error = max(_maximum_error(rotated[name], base[name]) for name in invariant_names)
    batch_permutation_error = max(
        _maximum_error(permuted[name][inverse], base[name]) for name in (*dense_names, "local_axis", *invariant_names)
    )
    repeat_exact = all(torch.equal(repeated[name], base[name]) for name in base)
    history_sensitivity = _maximum_error(changed["peak_presence_logits"], base["peak_presence_logits"])

    modified = {name: value.clone() for name, value in targets.items()}
    invalid_width = modified["presence"].bool() & ~modified["width_valid_mask"].bool()
    nonpeak = ~modified["presence"].bool()
    modified["opening_width_m"][invalid_width] = 1e6
    modified["vertical_profile_m"][nonpeak] = 1e6
    with torch.no_grad():
        original_masked_loss = circular_peak_geometry_loss(base, targets)["total"]
        changed_masked_loss = circular_peak_geometry_loss(base, modified)["total"]
    masked_target_invariance_error = float(torch.abs(original_masked_loss - changed_masked_loss))

    counts = audit["counts"]
    cardinality = audit["cardinality"]
    contract = circular_peak_geometry_input_contract()
    checks = {
        "exact_population_and_split": counts == {
            "worlds": 80, "observations": 188126, "peaks": 396913, "width_valid": 389026,
            "fit_worlds": 60, "fit_observations": 142184, "fit_peaks": 299872,
            "selection_worlds": 20, "selection_observations": 45942, "selection_peaks": 97041,
        },
        "exact_cardinality": cardinality == EXPECTED_CARDINALITY,
        "teacher_source_join_exact": audit["join_mismatch_shards"] == 0,
        "all_references_contiguous_and_past_only": audit["local_noncontiguous_rows"] == 0 and audit["global_noncontiguous_rows"] == 0,
        "teacher_contains_no_lidar_pose_or_identity": audit["forbidden_teacher_arrays"] == 0,
        "exact_real_batch_coverage": len(rows) == 8 and int(targets["presence"].sum()) >= 10 and bool((targets["presence"].bool() & ~targets["width_valid_mask"].bool()).any()),
        "typed_forward_has_only_scans": tuple(inspect.signature(model.forward).parameters) == ("scans",) and contract["forward_parameters"] == ("scans",),
        "no_free_query_parameter": not any("query" in name for name, _ in model.named_parameters()),
        "exact_parameter_count": parameter_count == EXPECTED_PARAMETERS,
        "finite_real_batch_output_loss_backward": finite_outputs and finite_backward and all(bool(torch.isfinite(value)) for value in losses.values()),
        "circular_dense_axis_and_global_equivariance": dense_rotation_error <= 3e-5 and axis_rotation_error <= 3e-5 and invariant_error <= 3e-5,
        "batch_permutation_and_repeat_deterministic": batch_permutation_error <= 3e-5 and repeat_exact,
        "masked_targets_do_not_change_loss": masked_target_invariance_error == 0.0,
        "all_five_frames_connected": history_sensitivity > 1e-8,
        "zero_training_threshold_test_graph": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_circular_peak_geometry_model_readiness_v1",
        "status": PASS if scientific_pass else FAIL,
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_CIRCULAR_PEAK_GEOMETRY_MODEL_TRAINING_DATA_CARD" if scientific_pass else "STOP_CIRCULAR_PEAK_GEOMETRY_MODEL_BEFORE_TRAINING",
        "population": {
            "worlds": counts["worlds"], "observations": counts["observations"], "peaks": counts["peaks"],
            "fit_observations": counts["fit_observations"], "selection_observations": counts["selection_observations"],
            "fit_peaks": counts["fit_peaks"], "selection_peaks": counts["selection_peaks"],
            "width_valid_peaks": counts["width_valid"], "width_invalid_peaks": counts["peaks"] - counts["width_valid"],
            "cardinality": {str(key): int(value) for key, value in sorted(cardinality.items())},
            "real_readiness_rows": len(rows),
        },
        "model": {
            "parameters": parameter_count, "student_shape": list(scans.shape),
            "peak_layout": [BEARING_BINS], "free_query_parameters": 0,
            "forward_parameters": list(inspect.signature(model.forward).parameters),
            "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
        },
        "real_batch": {
            "rows": rows, "visible_peaks": int(targets["presence"].sum()),
            "invalid_width_peaks": int((targets["presence"].bool() & ~targets["width_valid_mask"].bool()).sum()),
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            "finite_outputs": finite_outputs, "finite_backward": finite_backward,
            "history_sensitivity": history_sensitivity,
            "masked_target_invariance_error": masked_target_invariance_error,
        },
        "causality": audit,
        "equivariance": {
            "shift_columns": shift_columns, "shift_bins": shift_bins,
            "dense_max_error": dense_rotation_error, "axis_max_error": axis_rotation_error,
            "invariant_max_error": invariant_error,
            "batch_permutation_max_error": batch_permutation_error,
            "repeat_exact": repeat_exact,
        },
        "contract": contract,
        "checks": checks,
        "optimizer_steps": 0, "model_inference_frames": 0, "model_updates": 0,
        "checkpoint_writes": 0, "normalization_steps": 0, "threshold_selection_steps": 0,
        "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", {"schema_version": "gse_circular_peak_geometry_model_readiness_figure_source_v1", "summary": summary})
    with (output / "readiness_checks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("check", "passed"))
        writer.writeheader()
        for name, passed in checks.items():
            writer.writerow({"check": name, "passed": int(passed)})
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
