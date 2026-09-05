#!/usr/bin/env python3
"""Formal circular proposal distinctness corrective for sparse relation V2."""
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
    SOFT_SUPPORT_RADIUS_BINS,
    SparseCircularRelationTransportNet,
    circular_nms_indices,
    parameter_count,
)


PASS = "PASS_GSE_SPARSE_RELATION_PROPOSAL_DISTINCTNESS_CORRECTIVE_V1"


def _minimum_distance(index: torch.Tensor) -> int:
    delta = torch.abs(index[..., :, None] - index[..., None, :])
    delta = torch.minimum(delta, 180 - delta)
    delta = delta + torch.eye(6, device=index.device, dtype=delta.dtype) * 180
    return int(delta.min().detach().cpu())


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    axes[0].bar(("required", "synthetic", "real"), (5, summary["synthetic"]["minimum_separation_bins"], summary["real_batch"]["minimum_separation_bins"]), color="#4e79a7")
    axes[0].set_title("A  Token separation")
    losses = summary["real_batch"]["losses"]
    axes[1].bar(("core", "descriptor"), (losses["core"], losses["descriptor"]), color="#59a14f")
    axes[1].set_title("B  Full real objectives")
    axes[2].bar(("rotation", "repeat"), (summary["invariants"]["rotation"], summary["invariants"]["repeat"]), color="#e15759")
    axes[2].axhline(3e-5, color="black", linestyle="--")
    axes[2].set_yscale("symlog", linthresh=1e-10)
    axes[2].set_title("C  Decoder errors")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("Sparse relation transport: circular proposal distinctness")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_sparse_relation_proposal_distinctness_corrective_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--readiness-summary", required=True, type=Path)
    parser.add_argument("--geometry-corrective-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("formal proposal distinctness requires CUDA")
    base.seed_everything(0)
    device = torch.device("cuda")
    readiness = load_json(args.readiness_summary.resolve())
    geometry = load_json(args.geometry_corrective_summary.resolve())
    samples = readiness["result"]["real_batch"]["samples"]
    traversals = base.manifest_traversals(args.sequence_manifest.resolve())
    worlds = {}
    scan_parts = []
    target_parts = []
    for sample in samples:
        parent = sample["parent_id"]
        if parent not in worlds:
            worlds[parent] = sparse._load_world(args.dataset_root.resolve(), parent, traversals[parent])
        scans, target = sparse._batch(worlds[parent], [int(sample["row"])], device=device)
        scan_parts.append(scans)
        target_parts.append(target)
    scans = torch.cat(scan_parts)
    target = {name: torch.cat([part[name] for part in target_parts]) for name in target_parts[0]}

    synthetic_logits = torch.full((2, 5, 180), -10.0, device=device)
    for rank, index in enumerate((179, 0, 1, 40, 80, 120, 150, 20)):
        synthetic_logits[..., index] = 10.0 - rank
    synthetic = circular_nms_indices(synthetic_logits)
    synthetic_shifted = circular_nms_indices(torch.roll(synthetic_logits, 5, dims=-1))
    synthetic_rotation_exact = bool(torch.equal(synthetic_shifted, torch.remainder(synthetic + 5, 180)))

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
    model.eval()
    with torch.no_grad():
        base_output = model(scans)
        repeated = model(scans)
        rotated = model(torch.roll(scans, 20, dims=-1))
    expected_rotated = torch.remainder(base_output["token_bin_index"] + 5, 180)
    real_rotation_exact = bool(torch.equal(rotated["token_bin_index"], expected_rotated))
    repeat_exact = bool(torch.equal(base_output["token_bin_index"], repeated["token_bin_index"]))
    checks = {
        "prior_geometry_corrective_pass": geometry["overall_status"] == "PASS_GSE_SPARSE_RELATION_GEOMETRY_SHAPE_CORRECTIVE_V1",
        "parameter_count_unchanged": parameter_count() == 784513,
        "frozen_support_radius_is_four": SOFT_SUPPORT_RADIUS_BINS == 4,
        "synthetic_tokens_are_distinct": _minimum_distance(synthetic) > SOFT_SUPPORT_RADIUS_BINS,
        "real_tokens_are_distinct": _minimum_distance(base_output["token_bin_index"]) > SOFT_SUPPORT_RADIUS_BINS,
        "synthetic_wrap_rotation_exact": synthetic_rotation_exact,
        "real_rotation_exact": real_rotation_exact,
        "deterministic_repeat_exact": repeat_exact,
        "full_real_loss_and_all_gradients_valid": bool(torch.isfinite(total)) and not missing and not nonfinite and not zero,
        "zero_optimizer_checkpoint_validation_test_graph": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "gse_sparse_relation_proposal_distinctness_corrective_v1",
        "status": PASS if scientific_pass else "FAIL_GSE_SPARSE_RELATION_PROPOSAL_DISTINCTNESS_CORRECTIVE_V1",
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_SPARSE_RELATION_TRANSPORT_THREE_SEED_TRAINING_DATA_CARD" if scientific_pass else "STOP_SPARSE_RELATION_TRANSPORT_BEFORE_TRAINING",
        "parameters": 784513,
        "synthetic": {"minimum_separation_bins": _minimum_distance(synthetic), "rotation_exact": synthetic_rotation_exact, "selected_bins": synthetic[0, 0].detach().cpu().tolist()},
        "real_batch": {"observations": len(samples), "minimum_separation_bins": _minimum_distance(base_output["token_bin_index"]), "losses": {"core": float(core["total"].detach()), "descriptor": float(descriptor["total"].detach()), "total": float(total.detach())}, "missing_gradients": missing, "nonfinite_gradients": nonfinite, "zero_gradients": zero},
        "invariants": {"rotation": 0.0 if real_rotation_exact and synthetic_rotation_exact else 1.0, "repeat": 0.0 if repeat_exact else 1.0},
        "checks": checks, "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "checkpoints_created": 0, "c07_worlds_read": 0, "c08_worlds_read": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "figure_source.json", summary)
    with (output_dir / "corrective_checks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("check", "passed"))
        writer.writeheader()
        writer.writerows({"check": name, "passed": passed} for name, passed in checks.items())
    _plot(output_dir, summary)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
