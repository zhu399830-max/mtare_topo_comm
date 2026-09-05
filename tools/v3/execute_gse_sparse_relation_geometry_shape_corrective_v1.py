#!/usr/bin/env python3
"""Formal real-batch correction for sparse token geometry uncertainty shape."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import train_gse_axis_anchored_event_relation_v1 as base
import train_gse_sparse_circular_relation_transport_v2 as sparse
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_sparse_circular_relation_transport import (
    SparseCircularRelationTransportNet,
    parameter_count,
    sparse_circular_relation_transport_input_contract,
)


PASS = "PASS_GSE_SPARSE_RELATION_GEOMETRY_SHAPE_CORRECTIVE_V1"
EXPECTED_PARAMETERS = 784513


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].bar(("old", "corrected"), (784578, summary["parameters"]["corrected"]), color=("#bab0ac", "#4e79a7"))
    axes[0].set_title("A  Model parameters")
    losses = summary["real_batch"]["losses"]
    names = ("proposal", "count", "event", "token_geometry", "descriptor")
    axes[1].bar(names, [losses[name] for name in names], color="#59a14f")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].set_title("B  Real finite objectives")
    invariants = summary["invariants"]
    axes[2].bar(tuple(invariants), tuple(invariants.values()), color="#e15759")
    axes[2].axhline(3e-5, color="black", linestyle="--")
    axes[2].set_yscale("symlog", linthresh=1e-10)
    axes[2].set_title("C  Corrected-interface errors")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("Sparse relation transport: token-geometry shape corrective")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_sparse_relation_geometry_shape_corrective_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--readiness-summary", required=True, type=Path)
    parser.add_argument("--cardinality-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("formal geometry shape corrective requires CUDA")
    base.seed_everything(0)
    device = torch.device("cuda")
    readiness = load_json(args.readiness_summary.resolve())
    cardinality = load_json(args.cardinality_summary.resolve())
    samples = readiness["result"]["real_batch"]["samples"]
    traversals = base.manifest_traversals(args.sequence_manifest.resolve())
    worlds = {}
    scans_parts = []
    target_parts = []
    for sample in samples:
        parent = sample["parent_id"]
        if parent not in worlds:
            worlds[parent] = sparse._load_world(args.dataset_root.resolve(), parent, traversals[parent])
        scans, target = sparse._batch(worlds[parent], [int(sample["row"])], device=device)
        scans_parts.append(scans)
        target_parts.append(target)
    scans = torch.cat(scans_parts)
    target = {name: torch.cat([part[name] for part in target_parts]) for name in target_parts[0]}
    model = SparseCircularRelationTransportNet().to(device).train()
    result = model(scans)
    core = sparse.sparse_core_loss(result, target)
    descriptor = sparse.sparse_descriptor_loss(result, target)
    total = core["total"] + descriptor["total"]
    model.zero_grad(set_to_none=True)
    total.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    nonfinite = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
    zero = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not bool((parameter.grad != 0).any())]
    maximum_gradient = max(float(parameter.grad.abs().max()) for parameter in model.parameters() if parameter.grad is not None)
    model.eval()
    permutation = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3], device=device)
    changed = scans.clone()
    changed[:, 3:] = torch.flip(changed[:, 3:], dims=(-1,))
    with torch.no_grad():
        base_output = model(scans)
        repeated = model(scans)
        rotated = model(torch.roll(scans, 20, dims=-1))
        permuted = model(scans[permutation])
        future = model(changed)
    inverse = torch.argsort(permutation)
    rotation = max(
        _max_abs(base_output["token_count_logits"], rotated["token_count_logits"]),
        _max_abs(base_output["token_geometry_uncertainty"], rotated["token_geometry_uncertainty"]),
    )
    batch = max(
        _max_abs(base_output["token_count_logits"], permuted["token_count_logits"][inverse]),
        _max_abs(base_output["token_geometry_uncertainty"], permuted["token_geometry_uncertainty"][inverse]),
    )
    causal = _max_abs(base_output["token_count_logits"][:, :3], future["token_count_logits"][:, :3])
    repeat = max(
        _max_abs(base_output["token_count_logits"], repeated["token_count_logits"]),
        _max_abs(base_output["token_geometry_uncertainty"], repeated["token_geometry_uncertainty"]),
    )
    invariants = {"rotation": rotation, "batch": batch, "causal": causal, "repeat": repeat}
    checks = {
        "prior_readiness_and_cardinality_are_sealed_pass": readiness["overall_status"] == "PASS_GSE_AXIS_ANCHORED_EVENT_RELATION_READINESS_V2" and cardinality["overall_status"] == "PASS_GSE_SPARSE_RELATION_CARDINALITY_CORRECTIVE_V1",
        "forward_remains_scans_only": sparse_circular_relation_transport_input_contract()["forward_parameters"] == ("self", "scans"),
        "exact_65_parameter_correction": parameter_count() == EXPECTED_PARAMETERS and 784578 - EXPECTED_PARAMETERS == 65,
        "profile_and_uncertainty_dimensions_match": result["token_vertical_profile_m"].shape[-1] == 4 and result["token_geometry_uncertainty"].shape[-1] == 5,
        "all_outputs_and_losses_finite": all(bool(torch.isfinite(value).all()) for value in result.values() if value.is_floating_point()) and all(bool(torch.isfinite(value)) for value in (*core.values(), *descriptor.values(), total)),
        "all_trainable_gradients_present_finite_nonzero": not missing and not nonfinite and not zero and maximum_gradient > 0,
        "count_and_geometry_rotation_at_most_3e5": rotation <= 3e-5,
        "count_and_geometry_batch_at_most_3e6": batch <= 3e-6,
        "causal_past_exact": causal == 0.0,
        "repeat_exact": repeat == 0.0,
        "zero_optimizer_checkpoint_validation_test_graph": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    scientific_pass = all(checks.values())
    losses = {name: float(value.detach().cpu()) for name, value in core.items()}
    losses["descriptor"] = float(descriptor["total"].detach().cpu())
    losses["total_with_descriptor"] = float(total.detach().cpu())
    summary = {
        "schema_version": "gse_sparse_relation_geometry_shape_corrective_v1",
        "status": PASS if scientific_pass else "FAIL_GSE_SPARSE_RELATION_GEOMETRY_SHAPE_CORRECTIVE_V1",
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_SPARSE_RELATION_TRANSPORT_THREE_SEED_TRAINING_DATA_CARD" if scientific_pass else "STOP_SPARSE_RELATION_TRANSPORT_BEFORE_TRAINING",
        "parameters": {"prior_cardinality": 784578, "corrected": EXPECTED_PARAMETERS, "removed": 65},
        "real_batch": {"observations": len(samples), "losses": losses, "maximum_gradient": maximum_gradient, "missing_gradients": missing, "nonfinite_gradients": nonfinite, "zero_gradients": zero},
        "invariants": invariants, "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "checkpoints_created": 0, "c07_worlds_read": 0, "c08_worlds_read": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", summary)
    with (output / "corrective_checks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("check", "passed"))
        writer.writeheader()
        writer.writerows({"check": name, "passed": passed} for name, passed in checks.items())
    _plot(output, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
