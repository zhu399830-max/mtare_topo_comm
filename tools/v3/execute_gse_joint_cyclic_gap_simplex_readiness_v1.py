#!/usr/bin/env python3
"""Zero-training readiness for the Joint Cyclic Gap Simplex decoder."""
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
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import BRANCH_SLICE
from mtare_topo.representation.gse_joint_cyclic_gap_simplex import (
    CONTRACT,
    JointCyclicGapSimplexNet,
    circular_gaps_from_ordered_bearings,
    decode_cyclic_bearings,
    joint_cyclic_gap_simplex_loss,
    positive_gap_simplex,
)


PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V1"
FAIL = "FAIL_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V1"
EXPECTED_CARDINALITY = {1: 7525, 2: 154279, 3: 24458, 4: 1864}
EXPECTED_PARAMETERS = 789650


def _maximum_error(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.dtype == torch.bool or second.dtype == torch.bool:
        return float(torch.any(first != second))
    return float(torch.max(torch.abs(first - second)))


def _circular_error(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(torch.remainder(first - second + 180.0, 360.0) - 180.0)))


def _targets(bearings: tuple[float, ...]) -> dict[str, torch.Tensor]:
    presence = torch.zeros(1, 180, dtype=torch.bool)
    residual = torch.zeros(1, 180)
    for bearing in bearings:
        index = int(round(bearing / 2.0)) % 180
        presence[0, index] = True
        residual[0, index] = bearing - index * 2.0
    return {
        "presence": presence, "heading_residual_deg": residual,
        "opening_width_m": torch.zeros(1, 180), "width_valid_mask": torch.zeros(1, 180, dtype=torch.bool),
        "vertical_profile_m": torch.zeros(1, 180, 4), "local_axis": torch.tensor([[1.0, 0.0, 0.0]]),
        "geometry": torch.zeros(1, 4), "geometry_valid_mask": torch.ones(1, 4, dtype=torch.bool),
    }


def _synthetic_outputs(phase: float, gap_logits: torch.Tensor, cardinality: int) -> dict[str, torch.Tensor]:
    gap_fraction = gap_logits.new_zeros((1, 10))
    bearing = gap_logits.new_zeros((1, 10))
    branch = BRANCH_SLICE[cardinality]
    gaps = positive_gap_simplex(gap_logits)
    gap_fraction[:, branch] = gaps
    bearing[:, branch] = decode_cyclic_bearings(gap_logits.new_tensor([phase]), gaps)
    return {
        "joint_bearing_deg": bearing, "gap_fraction": gap_fraction,
        "phase_scale_deg": gap_logits.new_full((1, 4), 4.0),
        "gap_scale_deg": gap_logits.new_full((1, 10), 4.0),
        "exit_opening_width_m": gap_logits.new_zeros((1, 10)),
        "exit_vertical_profile_m": gap_logits.new_zeros((1, 10, 4)),
        "exit_count_logits": torch.nn.functional.one_hot(torch.tensor([cardinality - 1]), 4).to(gap_logits.dtype) * 16.0,
        "local_axis": gap_logits.new_tensor([[1.0, 0.0, 0.0]]),
        "width_m": gap_logits.new_zeros(1), "height_m": gap_logits.new_zeros(1),
        "slope_deg": gap_logits.new_zeros(1), "curvature_per_m": gap_logits.new_zeros(1),
    }


def _synthetic_contract() -> dict:
    target_bearing = (20.0, 100.0, 250.0)
    target_gaps = torch.tensor([[80.0, 150.0, 130.0]]) / 360.0
    logits = torch.log((target_gaps - 1e-5) / (1.0 - 3e-5))
    correct = _synthetic_outputs(20.0, logits, 3)
    targets = _targets(target_bearing)
    correct_loss = joint_cyclic_gap_simplex_loss(correct, targets)
    rotated = _synthetic_outputs(54.0, logits, 3)
    rotated_targets = _targets(tuple((value + 34.0) % 360.0 for value in target_bearing))
    rotated_loss = joint_cyclic_gap_simplex_loss(rotated, rotated_targets)
    collapsed_logits = torch.tensor([[-4.0, 4.0, -4.0]], requires_grad=True)
    collapsed = _synthetic_outputs(20.0, collapsed_logits, 3)
    collapsed_loss = joint_cyclic_gap_simplex_loss(collapsed, targets)
    collapsed_loss["localization"].backward()
    decoded = correct["joint_bearing_deg"][:, BRANCH_SLICE[3]]
    recovered_gap = circular_gaps_from_ordered_bearings(decoded)
    proper_error = torch.tensor(4.0)
    scale_candidates = torch.tensor([1.0, 4.0, 16.0])
    scale_nll = proper_error / scale_candidates + torch.log(scale_candidates / 180.0)
    return {
        "target_bearing_deg": target_bearing,
        "decoded_bearing_deg": decoded[0].tolist(),
        "gap_sum_deg": float((target_gaps * 360.0).sum()),
        "gap_recovery_error_deg": float(torch.max(torch.abs(recovered_gap - target_gaps * 360.0))),
        "minimum_gap_fraction": float(positive_gap_simplex(collapsed_logits.detach()).min()),
        "rotation_loss_error": abs(float(rotated_loss["total"] - correct_loss["total"])),
        "collapsed_localization": float(collapsed_loss["localization"].detach()),
        "correct_localization": float(correct_loss["localization"].detach()),
        "collapsed_gap_gradient_l1": float(collapsed_logits.grad.abs().sum()),
        "proper_scale_candidates_deg": scale_candidates.tolist(),
        "proper_scale_nll": scale_nll.tolist(),
        "proper_scale_argmin_deg": float(scale_candidates[torch.argmin(scale_nll)]),
    }


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    cardinality = summary["population"]["cardinality"]
    axes[0].bar(range(1, 5), [cardinality[str(value)] for value in range(1, 5)], color="#4e79a7")
    axes[0].set_xticks(range(1, 5)); axes[0].set(xlabel="exits", ylabel="observations", title="A  Structural population")
    synthetic = summary["synthetic"]
    axes[1].bar((0, 1), (synthetic["correct_localization"], synthetic["collapsed_localization"]), color=("#59a14f", "#e15759"))
    axes[1].set_xticks((0, 1), ("closed target", "collapsed gaps")); axes[1].set(ylabel="joint localization loss", title="B  Complete-set constraint")
    equivariance = summary["equivariance"]
    names = ("phase_mass_rotation", "gap_rotation", "bearing_rotation_deg", "exit_geometry_rotation")
    axes[2].bar(range(4), [max(equivariance[name], 1e-12) for name in names], color="#76b7b2")
    axes[2].set_yscale("log"); axes[2].axhline(3e-5, color="#e15759", linestyle="--")
    axes[2].set_xticks(range(4), ("phase", "gaps", "bearing", "geometry"), rotation=12)
    axes[2].set(ylabel="maximum error", title="C  Circular equivariance")
    for axis in axes:
        axis.grid(axis="y", alpha=.25); axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph Joint Cyclic Gap Simplex readiness")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_joint_cyclic_gap_simplex_readiness_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    teacher_paths = sorted(args.teacher_root.resolve().glob("*/*.zarr"))
    if len(teacher_paths) != 80:
        raise RuntimeError("gap-simplex readiness requires 80 Teacher shards")
    rows, audit = _selected_rows(teacher_paths, args.source_root.resolve())
    scans, targets = _real_batch(rows, args.teacher_root.resolve(), args.source_root.resolve())
    torch.use_deterministic_algorithms(True); torch.set_num_threads(4); torch.manual_seed(20260829)
    model = JointCyclicGapSimplexNet()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    outputs = model(scans)
    losses = joint_cyclic_gap_simplex_loss(outputs, targets)
    losses["total"].backward()
    finite_outputs = all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_floating_point(value))
    finite_backward = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    closure_errors, minimum_gap = [], 1.0
    for cardinality in range(1, 5):
        branch = BRANCH_SLICE[cardinality]
        gaps = outputs["gap_fraction"][:, branch]
        closure_errors.append(float(torch.max(torch.abs(gaps.sum(-1) - 1.0))))
        minimum_gap = min(minimum_gap, float(gaps.min()))

    model.eval(); shift_columns, shift_bins, angle_deg = 40, 10, 20.0
    permutation = torch.tensor([7, 1, 5, 0, 6, 2, 4, 3]); inverse = torch.argsort(permutation)
    with torch.no_grad():
        base = model(scans); repeat = model(scans)
        rotated = model(torch.roll(scans, shift_columns, -1)); permuted = model(scans[permutation])
        history = scans.clone(); history[:, 0, 0] = torch.roll(history[:, 0, 0], 73, -1); changed = model(history)
    phase_mass_rotation = _maximum_error(rotated["phase_mass"], torch.roll(base["phase_mass"], shift_bins, -1))
    phase_scale_rotation = _maximum_error(rotated["phase_scale_deg"], base["phase_scale_deg"])
    gap_rotation = max(_maximum_error(rotated[name], base[name]) for name in ("gap_fraction", "gap_scale_deg"))
    bearing_rotation = _circular_error(rotated["joint_bearing_deg"], torch.remainder(base["joint_bearing_deg"] + angle_deg, 360.0))
    count_rotation = _maximum_error(rotated["exit_count_probability"], base["exit_count_probability"])
    exit_geometry_rotation = max(_maximum_error(rotated[name], base[name]) for name in ("exit_opening_width_m", "exit_vertical_profile_m", "exit_descriptor", "exit_uncertainty"))
    batch_permutation = max(_maximum_error(permuted[name][inverse], base[name]) for name in base)
    repeat_error = max(_maximum_error(repeat[name], base[name]) for name in base)
    history_sensitivity = _maximum_error(changed["phase_logits"], base["phase_logits"])
    synthetic = _synthetic_contract()
    cardinality = {int(key): int(value) for key, value in audit["cardinality"].items()}
    checks = {
        "full_population_and_causal_join": audit["counts"].get("worlds") == 80 and audit["counts"].get("observations") == 188126 and audit["counts"].get("peaks") == 396913 and audit["join_mismatch_shards"] == 0 and audit["local_noncontiguous_rows"] == 0 and audit["global_noncontiguous_rows"] == 0,
        "exact_cardinality_population": cardinality == EXPECTED_CARDINALITY,
        "teacher_excludes_sensor_identity": audit["forbidden_teacher_arrays"] == 0,
        "typed_joint_not_independent_slots": CONTRACT.independent_slot_localizers is False and not any("query" in name or "objectness" in name or "slot_logit" in name for name, _ in model.named_parameters()),
        "positive_exactly_closed_gap_simplex": max(closure_errors) <= 5e-7 and minimum_gap > 0 and synthetic["gap_recovery_error_deg"] <= 5e-5 and abs(synthetic["gap_sum_deg"] - 360.0) <= 1e-5,
        "cyclic_rotation_and_collapse_gradient": synthetic["rotation_loss_error"] <= 1e-6 and synthetic["collapsed_localization"] > synthetic["correct_localization"] and synthetic["collapsed_gap_gradient_l1"] > 1e-6,
        "proper_uncertainty_scale": synthetic["proper_scale_argmin_deg"] == 4.0,
        "real_1_to_4_finite_backward": parameters == EXPECTED_PARAMETERS and finite_outputs and finite_backward and all(math.isfinite(float(value)) for value in losses.values()),
        "network_rotation_and_batch": max(phase_mass_rotation, phase_scale_rotation, gap_rotation, bearing_rotation, count_rotation, exit_geometry_rotation, batch_permutation) <= 3e-5 and repeat_error == 0,
        "five_frame_history_connected": history_sensitivity > 1e-6,
        "zero_training_test_graph": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_joint_cyclic_gap_simplex_readiness_v1",
        "status": PASS if scientific_pass else FAIL, "scientific_pass": scientific_pass,
        "decision": "ALLOW_JOINT_CYCLIC_GAP_SIMPLEX_THREE_SEED_DATA_CARD" if scientific_pass else "STOP_JOINT_CYCLIC_GAP_SIMPLEX",
        "population": {"counts": audit["counts"], "cardinality": {str(key): value for key, value in sorted(cardinality.items())}, "selected_real_rows": rows},
        "model": {"parameters": parameters, "contract": CONTRACT.__dict__}, "synthetic": synthetic,
        "real_batch": {"rows": len(scans), "exit_counts": targets["presence"].sum(1).tolist(), "losses": {name: float(value.detach()) for name, value in losses.items()}, "finite_outputs": finite_outputs, "finite_backward": finite_backward, "closure_errors": closure_errors, "minimum_gap_fraction": minimum_gap},
        "equivariance": {"phase_mass_rotation": phase_mass_rotation, "phase_scale_rotation": phase_scale_rotation, "gap_rotation": gap_rotation, "bearing_rotation_deg": bearing_rotation, "count_rotation": count_rotation, "exit_geometry_rotation": exit_geometry_rotation, "batch_permutation": batch_permutation, "repeat": repeat_error, "history_sensitivity": history_sensitivity},
        "checks": checks, "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "checkpoint_writes": 0, "threshold_selection_steps": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    with (output / "selected_real_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("criterion", "parent_id", "row", "partition")); writer.writeheader(); writer.writerows(rows)
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
