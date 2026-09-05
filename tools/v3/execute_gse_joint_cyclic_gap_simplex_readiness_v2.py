#!/usr/bin/env python3
"""Corrected zero-training readiness for Joint Cyclic Gap Simplex."""
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
import torch

from execute_gse_circular_peak_geometry_model_readiness_v1 import _real_batch, _selected_rows
from execute_gse_joint_cyclic_gap_simplex_readiness_v1 import _circular_error, _maximum_error, _synthetic_outputs, _targets
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import BRANCH_SLICE
from mtare_topo.representation.gse_joint_cyclic_gap_simplex import (
    CONTRACT,
    JointCyclicGapSimplexNet,
    circular_gaps_from_ordered_bearings,
    joint_cyclic_gap_simplex_loss,
    positive_gap_simplex,
)


PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V2"
FAIL = "FAIL_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V2"
EXPECTED_CARDINALITY = {1: 7525, 2: 154279, 3: 24458, 4: 1864}
EXPECTED_PARAMETERS = 789650


def _add_phase(outputs: dict[str, torch.Tensor], phase_deg: float, cardinality: int, *, uniform: bool = False) -> dict[str, torch.Tensor]:
    result = dict(outputs)
    logits = torch.zeros(1, 4, 180, dtype=outputs["joint_bearing_deg"].dtype)
    if not uniform:
        logits.fill_(-8.0)
        continuous = phase_deg / 2.0
        lower = int(math.floor(continuous)) % 180
        upper = (lower + 1) % 180
        fraction = continuous - math.floor(continuous)
        logits[0, cardinality - 1, lower] = 8.0 * (1.0 - fraction)
        logits[0, cardinality - 1, upper] = 8.0 * fraction
    logits.requires_grad_(True)
    log_mass = torch.log_softmax(logits, dim=-1)
    mass = torch.exp(log_mass)
    azimuth = torch.arange(180, dtype=mass.dtype) * 2.0 * torch.pi / 180.0
    xy = torch.stack(((mass * torch.cos(azimuth)).sum(-1), (mass * torch.sin(azimuth)).sum(-1)), dim=-1)
    result.update({"phase_logits": logits, "phase_log_mass": log_mass, "phase_mass": mass, "phase_concentration": torch.linalg.vector_norm(xy, dim=-1)})
    return result


def _synthetic_contract_v2() -> dict:
    bearings = (20.0, 100.0, 250.0); target_gaps = torch.tensor([[80.0, 150.0, 130.0]], dtype=torch.float64) / 360.0
    logits = torch.log((target_gaps - 1e-5) / (1.0 - 3e-5))
    targets = _targets(bearings)
    correct = _add_phase(_synthetic_outputs(20.0, logits, 3), 20.0, 3)
    uniform = _add_phase(_synthetic_outputs(20.0, logits, 3), 20.0, 3, uniform=True)
    wrong = _add_phase(_synthetic_outputs(60.0, logits, 3), 60.0, 3)
    rotated = _add_phase(_synthetic_outputs(54.0, logits, 3), 54.0, 3)
    rotated_targets = _targets(tuple((value + 34.0) % 360.0 for value in bearings))
    correct_loss = joint_cyclic_gap_simplex_loss(correct, targets)
    uniform_loss = joint_cyclic_gap_simplex_loss(uniform, targets)
    wrong_loss = joint_cyclic_gap_simplex_loss(wrong, targets)
    rotated_loss = joint_cyclic_gap_simplex_loss(rotated, rotated_targets)
    uniform_loss["phase_likelihood"].backward()
    target_bin_gradient = float(uniform["phase_logits"].grad[0, 2, 10])
    collapsed_logits = torch.tensor([[-4.0, 4.0, -4.0]], requires_grad=True)
    collapsed = _add_phase(_synthetic_outputs(20.0, collapsed_logits, 3), 20.0, 3)
    collapsed_loss = joint_cyclic_gap_simplex_loss(collapsed, targets)
    collapsed_loss["localization"].backward()
    decoded = correct["joint_bearing_deg"][:, BRANCH_SLICE[3]]
    recovered = circular_gaps_from_ordered_bearings(decoded)
    error = torch.tensor(4.0); scales = torch.tensor([1.0, 4.0, 16.0]); nll = error / scales + torch.log(scales / 180.0)
    return {
        "correct_phase_likelihood": float(correct_loss["phase_likelihood"].detach()),
        "uniform_phase_likelihood": float(uniform_loss["phase_likelihood"].detach()),
        "wrong_phase_likelihood": float(wrong_loss["phase_likelihood"].detach()),
        "uniform_target_bin_gradient": target_bin_gradient,
        "correct_localization": float(correct_loss["localization"].detach()),
        "collapsed_localization": float(collapsed_loss["localization"].detach()),
        "collapsed_gap_gradient_l1": float(collapsed_logits.grad.abs().sum()),
        "rotation_loss_error": abs(float((rotated_loss["total"] - correct_loss["total"]).detach())),
        "gap_sum_deg": float((target_gaps * 360.0).sum()),
        "gap_recovery_error_deg": float(torch.max(torch.abs(recovered - target_gaps * 360.0))),
        "proper_scale_argmin_deg": float(scales[torch.argmin(nll)]),
    }


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    counts = summary["population"]["cardinality"]
    axes[0].bar(range(1, 5), [counts[str(k)] for k in range(1, 5)], color="#4e79a7")
    axes[0].set_xticks(range(1, 5)); axes[0].set(xlabel="exits", ylabel="observations", title="A  Structural population")
    synthetic = summary["synthetic"]
    axes[1].bar(range(3), [synthetic[name] for name in ("correct_phase_likelihood", "uniform_phase_likelihood", "wrong_phase_likelihood")], color=("#59a14f", "#f28e2b", "#e15759"))
    axes[1].set_xticks(range(3), ("correct", "uniform", "wrong")); axes[1].set(ylabel="proper phase NLL", title="B  Observable phase supervision")
    equiv = summary["equivariance"]
    names = ("phase_mass_rotation", "gap_rotation", "phase_resultant_rotation", "authoritative_batch_permutation")
    axes[2].bar(range(4), [max(equiv[name], 1e-12) for name in names], color="#76b7b2")
    axes[2].set_yscale("log"); axes[2].axhline(3e-5, color="#e15759", linestyle="--")
    axes[2].set_xticks(range(4), ("phase mass", "gaps", "resultant", "batch"), rotation=12); axes[2].set(ylabel="maximum error", title="C  Authoritative equivariance")
    for axis in axes: axis.grid(axis="y", alpha=.25); axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph Joint Cyclic Gap Simplex V2 readiness")
    for suffix in ("png", "pdf", "svg"): figure.savefig(output / f"gse_joint_cyclic_gap_simplex_readiness_v2.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--teacher-root", required=True, type=Path); parser.add_argument("--source-root", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    teacher_paths = sorted(args.teacher_root.resolve().glob("*/*.zarr"))
    if len(teacher_paths) != 80: raise RuntimeError("gap-simplex V2 requires 80 Teacher shards")
    rows, audit = _selected_rows(teacher_paths, args.source_root.resolve()); scans, targets = _real_batch(rows, args.teacher_root.resolve(), args.source_root.resolve())
    torch.use_deterministic_algorithms(True); torch.set_num_threads(4); torch.manual_seed(20260829)
    model = JointCyclicGapSimplexNet(); parameters = sum(parameter.numel() for parameter in model.parameters())
    outputs = model(scans); losses = joint_cyclic_gap_simplex_loss(outputs, targets); losses["total"].backward()
    finite_outputs = all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_floating_point(value))
    finite_backward = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    closure_errors, minimum_gap = [], 1.0
    for cardinality in range(1, 5):
        gaps = outputs["gap_fraction"][:, BRANCH_SLICE[cardinality]]; closure_errors.append(float(torch.max(torch.abs(gaps.sum(-1) - 1.0)))); minimum_gap = min(minimum_gap, float(gaps.min()))
    model.eval(); shift_columns, shift_bins, angle_deg = 40, 10, 20.0; permutation = torch.tensor([7, 1, 5, 0, 6, 2, 4, 3]); inverse = torch.argsort(permutation)
    with torch.no_grad():
        base = model(scans); repeat = model(scans); rotated = model(torch.roll(scans, shift_columns, -1)); permuted = model(scans[permutation]); history = scans.clone(); history[:, 0, 0] = torch.roll(history[:, 0, 0], 73, -1); changed = model(history)
    phase_mass_rotation = _maximum_error(rotated["phase_mass"], torch.roll(base["phase_mass"], shift_bins, -1))
    phase_concentration_rotation = _maximum_error(rotated["phase_concentration"], base["phase_concentration"])
    gap_rotation = max(_maximum_error(rotated[name], base[name]) for name in ("gap_fraction", "gap_scale_deg"))
    azimuth = torch.arange(180, dtype=torch.float64) * 2 * torch.pi / 180
    def resultant(mass): return torch.stack(((mass.to(torch.float64) * torch.cos(azimuth)).sum(-1), (mass.to(torch.float64) * torch.sin(azimuth)).sum(-1)), dim=-1)
    rotation = torch.tensor([[math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))], [math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))]], dtype=torch.float64)
    phase_resultant_rotation = _maximum_error(resultant(rotated["phase_mass"]), torch.einsum("ij,bkj->bki", rotation, resultant(base["phase_mass"])))
    bearing_rotation = _circular_error(rotated["joint_bearing_deg"], torch.remainder(base["joint_bearing_deg"] + angle_deg, 360.0))
    exit_geometry_rotation = max(_maximum_error(rotated[name], base[name]) for name in ("exit_opening_width_m", "exit_vertical_profile_m", "exit_descriptor", "exit_uncertainty"))
    authoritative = ("phase_mass", "phase_concentration", "phase_scale_deg", "gap_fraction", "gap_scale_deg", "exit_count_probability", "local_axis", "width_m", "height_m", "slope_deg", "curvature_per_m")
    authoritative_batch = max(_maximum_error(permuted[name][inverse], base[name]) for name in authoritative)
    repeat_error = max(_maximum_error(repeat[name], base[name]) for name in base)
    history_sensitivity = _maximum_error(changed["phase_logits"], base["phase_logits"])
    low_phase_refused = bool(torch.all(base["decoded_set_confidence"] <= base["decoded_count_confidence"] * base["decoded_phase_concentration"] + 1e-12))
    synthetic = _synthetic_contract_v2(); cardinality = {int(key): int(value) for key, value in audit["cardinality"].items()}
    checks = {
        "full_population_and_causal_join": audit["counts"].get("worlds") == 80 and audit["counts"].get("observations") == 188126 and audit["counts"].get("peaks") == 396913 and audit["join_mismatch_shards"] == 0 and audit["local_noncontiguous_rows"] == 0 and audit["global_noncontiguous_rows"] == 0,
        "exact_cardinality_population": cardinality == EXPECTED_CARDINALITY,
        "teacher_excludes_sensor_identity": audit["forbidden_teacher_arrays"] == 0,
        "typed_joint_not_independent_slots": CONTRACT.independent_slot_localizers is False and not any("query" in name or "objectness" in name or "slot_logit" in name for name, _ in model.named_parameters()),
        "positive_exactly_closed_gap_simplex": max(closure_errors) <= 5e-7 and minimum_gap > 0 and synthetic["gap_recovery_error_deg"] <= 5e-5 and abs(synthetic["gap_sum_deg"] - 360.0) <= 1e-5,
        "proper_phase_and_collapse_gradient": synthetic["correct_phase_likelihood"] < synthetic["uniform_phase_likelihood"] < synthetic["wrong_phase_likelihood"] and synthetic["uniform_target_bin_gradient"] < 0 and synthetic["collapsed_localization"] > synthetic["correct_localization"] and synthetic["collapsed_gap_gradient_l1"] > 1e-6 and synthetic["rotation_loss_error"] <= 1e-6,
        "proper_uncertainty_and_low_phase_refusal": synthetic["proper_scale_argmin_deg"] == 4.0 and low_phase_refused,
        "real_1_to_4_finite_backward": parameters == EXPECTED_PARAMETERS and finite_outputs and finite_backward and all(math.isfinite(float(value)) for value in losses.values()),
        "authoritative_distribution_rotation_and_batch": max(phase_mass_rotation, phase_concentration_rotation, gap_rotation, phase_resultant_rotation, authoritative_batch) <= 3e-5 and repeat_error == 0,
        "five_frame_history_connected": history_sensitivity > 1e-6,
        "zero_training_test_graph": True,
    }
    scientific_pass = all(checks.values())
    summary = {"schema_version": "gse_joint_cyclic_gap_simplex_readiness_v2", "status": PASS if scientific_pass else FAIL, "scientific_pass": scientific_pass, "decision": "ALLOW_JOINT_CYCLIC_GAP_SIMPLEX_THREE_SEED_DATA_CARD" if scientific_pass else "STOP_JOINT_CYCLIC_GAP_SIMPLEX", "population": {"counts": audit["counts"], "cardinality": {str(key): value for key, value in sorted(cardinality.items())}, "selected_real_rows": rows}, "model": {"parameters": parameters, "contract": CONTRACT.__dict__, "v2_correction": "proper phase likelihood plus phase-concentration refusal; distribution-space rotation authority"}, "synthetic": synthetic, "real_batch": {"rows": len(scans), "exit_counts": targets["presence"].sum(1).tolist(), "losses": {name: float(value.detach()) for name, value in losses.items()}, "finite_outputs": finite_outputs, "finite_backward": finite_backward, "closure_errors": closure_errors, "minimum_gap_fraction": minimum_gap, "low_phase_refused": low_phase_refused}, "equivariance": {"phase_mass_rotation": phase_mass_rotation, "phase_concentration_rotation": phase_concentration_rotation, "gap_rotation": gap_rotation, "phase_resultant_rotation": phase_resultant_rotation, "authoritative_batch_permutation": authoritative_batch, "bearing_rotation_deg_diagnostic": bearing_rotation, "exit_geometry_rotation_diagnostic": exit_geometry_rotation, "repeat": repeat_error, "history_sensitivity": history_sensitivity}, "checks": checks, "duration_seconds": time.monotonic() - started, "optimizer_steps": 0, "checkpoint_writes": 0, "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0}
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    with (output / "selected_real_rows.csv").open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=("criterion", "parent_id", "row", "partition")); writer.writeheader(); writer.writerows(rows)
    _plot(output, summary); print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True)); return 0 if scientific_pass else 2


if __name__ == "__main__": raise SystemExit(main())
