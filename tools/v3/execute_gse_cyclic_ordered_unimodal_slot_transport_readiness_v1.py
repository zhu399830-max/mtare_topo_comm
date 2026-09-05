#!/usr/bin/env python3
"""Zero-training readiness for Cyclic-Ordered Unimodal Slot Transport."""
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
from mtare_topo.representation.gse_cyclic_ordered_unimodal_slot_transport import (
    CONTRACT,
    CyclicOrderedUnimodalSlotTransportNet,
    cyclic_ordered_unimodal_slot_loss,
    cyclic_orders,
    discrete_von_mises,
)


PASS = "PASS_GSE_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_READINESS_V1"
FAIL = "FAIL_GSE_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_READINESS_V1"
EXPECTED_CARDINALITY = {1: 7525, 2: 154279, 3: 24458, 4: 1864}
EXPECTED_PARAMETERS = 788618


def _maximum_error(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.dtype == torch.bool or second.dtype == torch.bool:
        return float(torch.any(first != second))
    return float(torch.max(torch.abs(first - second)))


def _circular_error(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(torch.remainder(first - second + 180.0, 360.0) - 180.0)))


def _synthetic_batch(slot_bins: tuple[int, int, int] = (10, 50, 100), *, shift: int = 0, require_grad: bool = False):
    target_bins = ((10 + shift) % 180, (50 + shift) % 180, (100 + shift) % 180)
    presence = torch.zeros(1, 180, dtype=torch.bool)
    presence[0, list(target_bins)] = True
    residual = torch.zeros(1, 180)
    logits = torch.full((1, 10, 180), -8.0)
    for slot, bearing_bin in zip(range(BRANCH_SLICE[3].start, BRANCH_SLICE[3].stop), slot_bins, strict=True):
        logits[0, slot, (bearing_bin + shift) % 180] = 8.0
    logits.requires_grad_(require_grad)
    mass = torch.softmax(logits, dim=-1)
    azimuth = torch.arange(180) * 2 * torch.pi / 180
    cosine = (mass * torch.cos(azimuth)).sum(-1)
    sine = (mass * torch.sin(azimuth)).sum(-1)
    outputs = {
        "slot_log_mass": torch.log_softmax(logits, dim=-1),
        "slot_mass": mass,
        "slot_bearing_deg": torch.remainder(torch.rad2deg(torch.atan2(sine, cosine)), 360.0),
        "slot_opening_width_m": torch.zeros(1, 10),
        "slot_vertical_profile_m": torch.zeros(1, 10, 4),
        "exit_count_logits": torch.tensor([[-8.0, -8.0, 8.0, -8.0]]),
        "local_axis": torch.tensor([[1.0, 0.0, 0.0]]),
        "width_m": torch.zeros(1), "height_m": torch.zeros(1),
        "slope_deg": torch.zeros(1), "curvature_per_m": torch.zeros(1),
    }
    targets = {
        "presence": presence, "heading_residual_deg": residual,
        "opening_width_m": torch.zeros(1, 180), "width_valid_mask": torch.zeros(1, 180, dtype=torch.bool),
        "vertical_profile_m": torch.zeros(1, 180, 4), "local_axis": outputs["local_axis"].clone(),
        "geometry": torch.zeros(1, 4), "geometry_valid_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    return outputs, targets, logits


def _permute_branch(outputs: dict[str, torch.Tensor], *, reverse: bool = False) -> dict[str, torch.Tensor]:
    result = dict(outputs)
    permutation = list(range(10))
    for cardinality in range(1, 5):
        branch = list(range(BRANCH_SLICE[cardinality].start, BRANCH_SLICE[cardinality].stop))
        order = list(reversed(branch)) if reverse else branch[1:] + branch[:1]
        for destination, source in zip(branch, order, strict=True):
            permutation[destination] = source
    index = torch.tensor(permutation)
    for name in ("slot_log_mass", "slot_mass", "slot_bearing_deg", "slot_concentration", "slot_kappa", "slot_opening_width_m", "slot_vertical_profile_m", "slot_descriptor", "slot_uncertainty"):
        if name in outputs:
            result[name] = outputs[name][:, index]
    return result


def _synthetic_contract() -> dict:
    correct, targets, _ = _synthetic_batch()
    reversed_output, _, _ = _synthetic_batch((10, 100, 50))
    duplicate, _, duplicate_logits = _synthetic_batch((10, 10, 50), require_grad=True)
    rotated, rotated_targets, _ = _synthetic_batch(shift=17)
    correct_loss = cyclic_ordered_unimodal_slot_loss(correct, targets)
    reverse_loss = cyclic_ordered_unimodal_slot_loss(reversed_output, targets)
    rotated_loss = cyclic_ordered_unimodal_slot_loss(rotated, rotated_targets)
    cyclic_loss = cyclic_ordered_unimodal_slot_loss(_permute_branch(correct), targets)
    duplicate_loss = cyclic_ordered_unimodal_slot_loss(duplicate, targets)["assignment"]
    duplicate_loss.backward()
    missing_gradient = min(float(duplicate_logits.grad[0, slot, 100]) for slot in range(BRANCH_SLICE[3].start, BRANCH_SLICE[3].stop))
    azimuth = torch.arange(180, dtype=torch.float64) * 2 * torch.pi / 180
    unit = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    log_mass, mass, bearing, concentration = discrete_von_mises(unit, torch.tensor([[0.0, 1000.0]], dtype=torch.float64), azimuth)
    sharp = mass[0, 1]
    local_maxima = int(((sharp > torch.roll(sharp, 1)) & (sharp > torch.roll(sharp, -1))).sum())
    return {
        "cyclic_orders_k3": cyclic_orders(3),
        "correct_total": float(correct_loss["total"]),
        "reversed_total": float(reverse_loss["total"]),
        "rotation_total_error": abs(float(rotated_loss["total"] - correct_loss["total"])),
        "cyclic_slot_permutation_error": abs(float(cyclic_loss["total"] - correct_loss["total"])),
        "duplicate_missing_target_gradient": missing_gradient,
        "uniform_concentration": float(concentration[0, 0]),
        "sharp_concentration": float(concentration[0, 1]),
        "sharp_local_maxima": local_maxima,
        "extreme_finite": bool(torch.isfinite(log_mass).all() and torch.isfinite(bearing).all()),
        "normalization_error": float(torch.max(torch.abs(mass.sum(-1) - 1.0))),
    }


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    cardinality = summary["population"]["cardinality"]
    axes[0].bar(range(1, 5), [cardinality[str(value)] for value in range(1, 5)], color="#4e79a7")
    axes[0].set_xticks(range(1, 5)); axes[0].set(xlabel="exits", ylabel="observations", title="A  Variable-cardinality population")
    synthetic = summary["synthetic"]
    axes[1].bar((0, 1), (synthetic["correct_total"], synthetic["reversed_total"]), color=("#59a14f", "#e15759"))
    axes[1].set_xticks((0, 1), ("cyclic order", "reversed")); axes[1].set(ylabel="loss", title="B  Orientation-preserving transport")
    equivariance = summary["equivariance"]
    names = ("slot_mass_rotation", "kappa_rotation", "cyclic_slot_loss", "batch_permutation")
    axes[2].bar(range(4), [max(equivariance[name], 1e-12) for name in names], color="#76b7b2")
    axes[2].set_yscale("log"); axes[2].axhline(3e-5, color="#e15759", linestyle="--")
    axes[2].set_xticks(range(4), ("mass", "kappa", "cyclic slots", "batch"), rotation=15)
    axes[2].set(ylabel="maximum error", title="C  Symmetry contracts")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25); axis.set_axisbelow(True)
    figure.suptitle("GSE-Graph cyclic-ordered unimodal slot readiness")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_cyclic_ordered_unimodal_slot_transport_readiness_v1.{suffix}", dpi=220)
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
    teacher_paths = sorted(args.teacher_root.resolve().glob("*/*.zarr"))
    if len(teacher_paths) != 80:
        raise RuntimeError("COUST readiness requires 80 Teacher shards")
    rows, audit = _selected_rows(teacher_paths, args.source_root.resolve())
    scans, targets = _real_batch(rows, args.teacher_root.resolve(), args.source_root.resolve())
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.manual_seed(20260829)
    model = CyclicOrderedUnimodalSlotTransportNet()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    outputs = model(scans)
    losses = cyclic_ordered_unimodal_slot_loss(outputs, targets)
    losses["total"].backward()
    finite_outputs = all(bool(torch.isfinite(value).all()) for value in outputs.values() if torch.is_floating_point(value))
    finite_backward = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())

    model.eval()
    shift_columns, shift_bins, angle_deg = 40, 10, 20.0
    permutation = torch.tensor([7, 1, 5, 0, 6, 2, 4, 3])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        base = model(scans)
        repeat = model(scans)
        rotated = model(torch.roll(scans, shift_columns, -1))
        permuted = model(scans[permutation])
        history = scans.clone(); history[:, 0, 0] = torch.roll(history[:, 0, 0], 73, -1)
        changed = model(history)
    mass_rotation = _maximum_error(rotated["slot_mass"], torch.roll(base["slot_mass"], shift_bins, -1))
    raw_mass_rotation = _maximum_error(rotated["raw_slot_mass"], torch.roll(base["raw_slot_mass"], shift_bins, -1))
    bearing_rotation = _circular_error(rotated["slot_bearing_deg"], torch.remainder(base["slot_bearing_deg"] + angle_deg, 360.0))
    kappa_rotation = _maximum_error(rotated["slot_kappa"], base["slot_kappa"])
    concentration_rotation = _maximum_error(rotated["slot_concentration"], base["slot_concentration"])
    count_rotation = _maximum_error(rotated["exit_count_probability"], base["exit_count_probability"])
    slot_geometry_rotation = max(_maximum_error(rotated[name], base[name]) for name in ("slot_opening_width_m", "slot_vertical_profile_m", "slot_descriptor", "slot_uncertainty"))
    batch_permutation = max(_maximum_error(permuted[name][inverse], base[name]) for name in base)
    repeat_error = max(_maximum_error(repeat[name], base[name]) for name in base)
    history_sensitivity = _maximum_error(changed["raw_slot_logits"], base["raw_slot_logits"])
    cyclic_slot_loss = abs(float(cyclic_ordered_unimodal_slot_loss(_permute_branch(base), targets)["total"] - cyclic_ordered_unimodal_slot_loss(base, targets)["total"]))
    synthetic = _synthetic_contract()
    cardinality = {int(key): int(value) for key, value in audit["cardinality"].items()}
    checks = {
        "full_population_and_causal_join": audit["counts"].get("worlds") == 80 and audit["counts"].get("observations") == 188126 and audit["counts"].get("peaks") == 396913 and audit["join_mismatch_shards"] == 0 and audit["local_noncontiguous_rows"] == 0 and audit["global_noncontiguous_rows"] == 0,
        "exact_cardinality_population": cardinality == EXPECTED_CARDINALITY,
        "teacher_excludes_sensor_identity": audit["forbidden_teacher_arrays"] == 0,
        "typed_no_existence_query": CONTRACT.free_query_existence is False and not any("query" in name or "objectness" in name for name, _ in model.named_parameters()),
        "cyclic_not_reversed_assignment": synthetic["correct_total"] < synthetic["reversed_total"] and synthetic["cyclic_slot_permutation_error"] <= 1e-5,
        "duplicate_missing_mode_gradient": synthetic["duplicate_missing_target_gradient"] < 0,
        "proper_unimodal_concentration": synthetic["uniform_concentration"] < 1e-10 and synthetic["sharp_concentration"] > 0.999 and synthetic["sharp_local_maxima"] == 1 and synthetic["extreme_finite"] and synthetic["normalization_error"] <= 1e-12,
        "real_1_to_4_finite_backward": parameters == EXPECTED_PARAMETERS and finite_outputs and finite_backward and all(math.isfinite(float(value)) for value in losses.values()),
        "network_rotation_cyclic_and_batch": max(mass_rotation, raw_mass_rotation, bearing_rotation, kappa_rotation, concentration_rotation, count_rotation, slot_geometry_rotation, cyclic_slot_loss, batch_permutation) <= 3e-5 and repeat_error == 0,
        "five_frame_history_connected": history_sensitivity > 1e-6,
        "zero_training_test_graph": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_cyclic_ordered_unimodal_slot_transport_readiness_v1",
        "status": PASS if scientific_pass else FAIL, "scientific_pass": scientific_pass,
        "decision": "ALLOW_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT_THREE_SEED_DATA_CARD" if scientific_pass else "STOP_CYCLIC_ORDERED_UNIMODAL_SLOT_TRANSPORT",
        "population": {"counts": audit["counts"], "cardinality": {str(key): value for key, value in sorted(cardinality.items())}, "selected_real_rows": rows},
        "model": {"parameters": parameters, "contract": CONTRACT.__dict__},
        "synthetic": synthetic,
        "real_batch": {"rows": len(scans), "exit_counts": targets["presence"].sum(1).tolist(), "losses": {name: float(value.detach()) for name, value in losses.items()}, "finite_outputs": finite_outputs, "finite_backward": finite_backward},
        "equivariance": {"slot_mass_rotation": mass_rotation, "raw_mass_rotation": raw_mass_rotation, "bearing_rotation_deg": bearing_rotation, "kappa_rotation": kappa_rotation, "concentration_rotation": concentration_rotation, "count_rotation": count_rotation, "slot_geometry_rotation": slot_geometry_rotation, "cyclic_slot_loss": cyclic_slot_loss, "batch_permutation": batch_permutation, "repeat": repeat_error, "history_sensitivity": history_sensitivity},
        "checks": checks, "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "checkpoint_writes": 0, "threshold_selection_steps": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", summary)
    with (output / "selected_real_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("criterion", "parent_id", "row", "partition")); writer.writeheader(); writer.writerows(rows)
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
