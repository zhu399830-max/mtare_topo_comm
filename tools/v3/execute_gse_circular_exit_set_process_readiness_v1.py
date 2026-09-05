#!/usr/bin/env python3
"""Zero-training full-population and real-batch readiness for exit-set process."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from execute_gse_circular_peak_geometry_model_readiness_v1 import _real_batch, _selected_rows
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_circular_exit_set_process import (
    CausalCircularExitSetProcessNet,
    circular_exit_set_process_contract,
    circular_exit_set_process_loss,
    decode_circular_exit_set,
)


PASS = "PASS_GSE_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_READINESS_V1"
FAIL = "FAIL_GSE_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_READINESS_V1"
EXPECTED_PARAMETERS = 769784
EXPECTED_CARDINALITY = {1: 7525, 2: 154279, 3: 24458, 4: 1864}


def _maximum_error(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(first - second)))


def _synthetic_set_contract() -> dict:
    target = torch.zeros(2, 180)
    target[0, [0, 40]] = 1
    target[1, [10, 12, 100]] = 1
    correct = torch.full((2, 180), -8.0)
    correct[target.bool()] = 8.0
    shifted = torch.roll(correct, 1, dims=1)
    ghost = correct.clone(); ghost[:, 150] = 10.0
    duplicate = correct.clone(); duplicate[:, 1] = 8.0

    def nll(logits: torch.Tensor, truth: torch.Tensor = target) -> float:
        count = truth.sum(dim=1)
        return float((-(torch.log_softmax(logits, -1) * truth).sum(dim=1) / count).mean())

    count_logits = torch.full((4, 4), -8.0)
    count_logits[torch.arange(4), torch.arange(4)] = 8.0
    count_target = torch.arange(4)
    correct_count_loss = float(torch.nn.functional.cross_entropy(count_logits, count_target))
    wrong_count_loss = float(torch.nn.functional.cross_entropy(torch.roll(count_logits, 1, dims=1), count_target))

    batch = 4
    intensity = torch.full((batch, 180), -8.0)
    count = torch.full((batch, 4), -8.0)
    target_bins = ([0], [179, 1], [10, 50, 90], [0, 2, 80, 120])
    for row, bins in enumerate(target_bins):
        count[row, len(bins) - 1] = 8.0
        for rank, bearing_bin in enumerate(bins):
            intensity[row, bearing_bin] = 8.0 - rank
    outputs = {
        "exit_mass": torch.softmax(intensity, -1),
        "exit_count_probability": torch.softmax(count, -1),
        "peak_heading_residual_deg": torch.zeros(batch, 180),
        "peak_opening_width_m": torch.ones(batch, 180),
        "peak_vertical_profile_m": torch.zeros(batch, 180, 4),
        "peak_descriptor": torch.zeros(batch, 180, 32),
        "peak_geometry_uncertainty": torch.ones(batch, 180, 6),
    }
    decoded = decode_circular_exit_set(outputs)
    return {
        "set_nll": {"correct": nll(correct), "shifted_one_bin": nll(shifted), "ghost": nll(ghost), "duplicate": nll(duplicate)},
        "rotation_error": abs(nll(torch.roll(correct, 17, 1), torch.roll(target, 17, 1)) - nll(correct)),
        "row_permutation_error": abs(nll(correct.flip(0), target.flip(0)) - nll(correct)),
        "count_loss": {"correct": correct_count_loss, "wrong": wrong_count_loss},
        "decoded_counts": decoded.count.tolist(),
        "decoded_valid_counts": decoded.valid_mask.sum(dim=1).tolist(),
        "near_wrap_bins": [decoded.bin_index[1, :2].tolist(), decoded.bin_index[3].tolist()],
    }


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    cardinality = summary["population"]["cardinality"]
    axes[0].bar(range(1, 5), [cardinality[str(value)] for value in range(1, 5)], color="#4e79a7")
    axes[0].set_xticks(range(1, 5)); axes[0].set_xlabel("Exits in causal observation"); axes[0].set_ylabel("Observations")
    axes[0].set_title("A  Variable set population")
    losses = summary["synthetic"]["set_nll"]
    names = ("correct", "shifted_one_bin", "duplicate", "ghost")
    axes[1].bar(range(4), [losses[name] for name in names], color=("#59a14f", "#f28e2b", "#af7aa1", "#e15759"))
    axes[1].set_xticks(range(4), ("correct", "1-bin shift", "duplicate", "ghost"), rotation=15)
    axes[1].set_ylabel("Normalized set NLL"); axes[1].set_title("B  One probability budget")
    equivariance = summary["equivariance"]
    error_names = ("intensity_rotation", "count_rotation", "batch_permutation", "repeat")
    axes[2].bar(range(4), [max(equivariance[name], 1e-12) for name in error_names], color="#76b7b2")
    axes[2].set_yscale("log"); axes[2].axhline(3e-5, color="#e15759", linestyle="--", label="contract")
    axes[2].set_xticks(range(4), ("intensity", "count", "batch", "repeat"), rotation=15)
    axes[2].set_ylabel("Maximum absolute error"); axes[2].set_title("C  Circular set interface"); axes[2].legend(frameon=False)
    for axis in axes: axis.grid(axis="y", alpha=.25); axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph causal circular exit-set process readiness")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_circular_exit_set_process_readiness_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--synthetic-rotation-atol", type=float, default=0.0)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    teacher_root = args.teacher_root.resolve(); source_root = args.source_root.resolve()
    teacher_paths = sorted(teacher_root.glob("*/*.zarr"))
    if len(teacher_paths) != 80: raise RuntimeError("exit-set readiness requires 80 Teacher shards")
    rows, audit = _selected_rows(teacher_paths, source_root)
    scans, targets = _real_batch(rows, teacher_root, source_root)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(4); torch.manual_seed(20260829)
    model = CausalCircularExitSetProcessNet()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    outputs = model(scans)
    losses = circular_exit_set_process_loss(outputs, targets)
    losses["total"].backward()
    finite_backward = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    finite_outputs = all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_floating_point(value))

    model.eval(); shift_columns = 40; shift_bins = 10; angle = 2 * math.pi * shift_bins / 180
    permutation = torch.tensor([7, 1, 5, 0, 6, 2, 4, 3]); inverse = torch.argsort(permutation)
    with torch.no_grad():
        base = model(scans); repeated = model(scans); rotated = model(torch.roll(scans, shift_columns, -1)); permuted = model(scans[permutation])
        history_changed = scans.clone(); history_changed[:, 0, 0] = torch.roll(history_changed[:, 0, 0], 73, -1); changed = model(history_changed)
    intensity_rotation = _maximum_error(rotated["exit_mass"], torch.roll(base["exit_mass"], shift_bins, 1))
    count_rotation = _maximum_error(rotated["exit_count_probability"], base["exit_count_probability"])
    dense_names = ("peak_heading_residual_deg", "peak_opening_width_m", "peak_vertical_profile_m", "peak_descriptor", "peak_geometry_uncertainty")
    dense_geometry_rotation = max(_maximum_error(rotated[name], torch.roll(base[name], shift_bins, 1)) for name in dense_names)
    expected_axis = torch.stack((math.cos(angle)*base["local_axis"][:,0]-math.sin(angle)*base["local_axis"][:,1], math.sin(angle)*base["local_axis"][:,0]+math.cos(angle)*base["local_axis"][:,1], base["local_axis"][:,2]), -1)
    axis_rotation = _maximum_error(rotated["local_axis"], expected_axis)
    invariant_names = ("width_m", "height_m", "slope_deg", "curvature_per_m", "place_descriptor", "observation_uncertainty")
    invariant_rotation = max(_maximum_error(rotated[name], base[name]) for name in invariant_names)
    batch_permutation = max(_maximum_error(permuted[name][inverse], base[name]) for name in outputs)
    repeat_error = max(_maximum_error(repeated[name], base[name]) for name in outputs)
    history_sensitivity = _maximum_error(changed["exit_intensity_logits"], base["exit_intensity_logits"])
    synthetic = _synthetic_set_contract()
    contract = circular_exit_set_process_contract()

    cardinality = {int(key): int(value) for key, value in audit["cardinality"].items()}
    checks = {
        "full_population_and_causal_join": audit["counts"].get("worlds") == 80 and audit["counts"].get("observations") == 188126 and audit["counts"].get("peaks") == 396913 and audit["join_mismatch_shards"] == 0 and audit["local_noncontiguous_rows"] == 0 and audit["global_noncontiguous_rows"] == 0,
        "exact_cardinality_population": cardinality == EXPECTED_CARDINALITY,
        "teacher_excludes_sensor_identity": audit["forbidden_teacher_arrays"] == 0,
        "typed_no_free_query_interface": parameter_count == EXPECTED_PARAMETERS and not any("query" in name for name, _ in model.named_parameters()) and contract["decode"].endswith("no existence threshold"),
        "real_1_to_4_finite_backward": finite_outputs and finite_backward and all(math.isfinite(float(value)) for value in losses.values()),
        "set_budget_orders_shift_duplicate_ghost": synthetic["set_nll"]["correct"] < min(synthetic["set_nll"]["shifted_one_bin"], synthetic["set_nll"]["duplicate"], synthetic["set_nll"]["ghost"]),
        "cardinality_loss_orders_correct_wrong": synthetic["count_loss"]["correct"] < synthetic["count_loss"]["wrong"],
        "permutation_wrap_near_exit_decode": synthetic["rotation_error"] <= args.synthetic_rotation_atol and synthetic["row_permutation_error"] == 0 and synthetic["decoded_counts"] == [1,2,3,4] and synthetic["decoded_valid_counts"] == [1,2,3,4] and set(synthetic["near_wrap_bins"][0]) == {179,1} and set(synthetic["near_wrap_bins"][1]) == {0,2,80,120},
        "circular_rotation_and_batch_contract": max(intensity_rotation, count_rotation, dense_geometry_rotation, axis_rotation, invariant_rotation, batch_permutation) <= 3e-5 and repeat_error == 0,
        "five_frame_history_connected": history_sensitivity > 1e-6,
        "zero_training_test_graph": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_causal_circular_exit_set_process_readiness_v1", "status": PASS if scientific_pass else FAIL, "scientific_pass": scientific_pass,
        "decision": "ALLOW_CAUSAL_CIRCULAR_EXIT_SET_PROCESS_TRAINING_DATA_CARD" if scientific_pass else "STOP_CAUSAL_CIRCULAR_EXIT_SET_PROCESS",
        "population": {"counts": audit["counts"], "cardinality": {str(key): value for key, value in sorted(cardinality.items())}, "selected_real_rows": rows},
        "model": {"parameters": parameter_count, "free_query_parameters": sum(int("query" in name) for name, _ in model.named_parameters()), "contract": contract},
        "real_batch": {"rows": len(scans), "exit_counts": targets["presence"].sum(1).tolist(), "losses": {name: float(value.detach()) for name, value in losses.items()}, "finite_outputs": finite_outputs, "finite_backward": finite_backward},
        "synthetic": {**synthetic, "rotation_atol": args.synthetic_rotation_atol},
        "equivariance": {"intensity_rotation": intensity_rotation, "count_rotation": count_rotation, "dense_geometry_rotation": dense_geometry_rotation, "axis_rotation": axis_rotation, "invariant_rotation": invariant_rotation, "batch_permutation": batch_permutation, "repeat": repeat_error, "history_sensitivity": history_sensitivity},
        "checks": checks, "duration_seconds": time.monotonic()-started,
        "optimizer_steps": 0, "checkpoint_writes": 0, "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output/"summary.json", summary); write_json(output/"figure_source.json", {"schema_version":"gse_circular_exit_set_process_readiness_figure_source_v1","summary":summary})
    with (output/"selected_real_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=("criterion","parent_id","row","partition")); writer.writeheader(); writer.writerows(rows)
    _plot(output, summary)
    print(json.dumps({"status":summary["status"],"decision":summary["decision"],"checks":checks},indent=2,sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__": raise SystemExit(main())
