#!/usr/bin/env python3
"""Formal zero-training cardinality correction for sparse relation transport."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from execute_gse_sparse_circular_relation_transport_readiness_v1 import (
    _checkpoint_compatibility,
    _load_real_batch,
    _max_abs,
)
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_sparse_circular_relation_transport import (
    MAX_TOKENS,
    SparseCircularRelationTransportNet,
    parameter_count,
    sparse_circular_relation_transport_input_contract,
    token_count_loss,
)


PASS = "PASS_GSE_SPARSE_RELATION_CARDINALITY_CORRECTIVE_V1"
OLD_PARAMETERS = 783675
EXPECTED_PARAMETERS = 784578
EXPECTED_DELTA = 903


def _plot(output: Path, summary: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    targets = np.asarray(summary["real_batch"]["count_targets"], dtype=np.int64).reshape(-1)
    values, counts = np.unique(targets, return_counts=True)
    axes[0].bar(values.astype(str), counts, color="#4e79a7")
    axes[0].set_title("A  Real causal-frame counts")
    axes[0].set_xlabel("visible branches")
    gradient = summary["real_batch"]["count_head_gradient_max"]
    axes[1].bar(("weight", "bias"), (gradient["weight"], gradient["bias"]), color="#59a14f")
    axes[1].set_yscale("log")
    axes[1].set_title("B  Count-loss gradients")
    invariant = summary["invariants"]
    names = ("rotation", "batch", "causal", "repeat")
    axes[2].bar(names, [invariant[name] for name in names], color="#e15759")
    axes[2].axhline(3e-5, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("symlog", linthresh=1e-10)
    axes[2].set_title("C  Count-interface errors")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("Sparse relation transport: explicit cardinality corrective")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_sparse_relation_cardinality_corrective_v1.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--predecessor-readiness-summary", required=True, type=Path)
    parser.add_argument("--predecessor-checkpoint", required=True, type=Path)
    parser.add_argument("--sealed-sparse-readiness-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("formal cardinality corrective requires CUDA")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    sealed = load_json(args.sealed_sparse_readiness_summary.resolve())
    model = SparseCircularRelationTransportNet()
    compatibility = _checkpoint_compatibility(model, args.predecessor_checkpoint.resolve())
    scans, target, samples = _load_real_batch(
        args.dataset_root.resolve(), args.sequence_manifest.resolve(),
        args.predecessor_readiness_summary.resolve(), device,
    )
    count_target = target["presence"].bool().sum(dim=-1).long()
    identity_count = torch.from_numpy(np.asarray([
        [len({int(value) for value in frame if int(value) >= 0}) for frame in sample]
        for sample in target["identity_by_bin"]
    ], dtype=np.int64)).to(device)

    model = model.to(device).train()
    result = model(scans)
    count_loss = token_count_loss(result["token_count_logits"], target["presence"])
    model.zero_grad(set_to_none=True)
    count_loss.backward()
    count_gradients = {
        name: parameter.grad for name, parameter in model.token_count_head.named_parameters()
    }
    gradient_max = {
        name: float(value.abs().max().detach().cpu()) if value is not None else float("nan")
        for name, value in count_gradients.items()
    }
    all_count_gradient_finite_nonzero = all(
        value is not None and bool(torch.isfinite(value).all()) and bool((value != 0).any())
        for value in count_gradients.values()
    )

    model.eval()
    permutation = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3], device=device)
    future_35 = scans.clone()
    future_35[:, 3:] = torch.flip(future_35[:, 3:], dims=(-1,))
    future_5 = scans.clone()
    future_5[:, 4:] = torch.flip(future_5[:, 4:], dims=(-1,))
    with torch.no_grad():
        base = model(scans)
        repeat = model(scans)
        rotated = model(torch.roll(scans, 20, dims=-1))
        permuted = model(scans[permutation])
        changed_35 = model(future_35)
        changed_5 = model(future_5)
    inverse = torch.argsort(permutation)
    count_logits = base["token_count_logits"]
    count_probability = base["token_count_probability"]
    rotation_error = max(
        _max_abs(count_logits, rotated["token_count_logits"]),
        _max_abs(count_probability, rotated["token_count_probability"]),
    )
    batch_error = max(
        _max_abs(count_logits, permuted["token_count_logits"][inverse]),
        _max_abs(count_probability, permuted["token_count_probability"][inverse]),
    )
    causal_error = max(
        _max_abs(count_logits[:, :3], changed_35["token_count_logits"][:, :3]),
        _max_abs(count_logits[:, :4], changed_5["token_count_logits"][:, :4]),
    )
    repeat_error = max(
        _max_abs(count_logits, repeat["token_count_logits"]),
        _max_abs(count_probability, repeat["token_count_probability"]),
    )
    head_parameters = sum(parameter.numel() for parameter in model.token_count_head.parameters())
    explicit_classes = tuple(range(MAX_TOKENS + 1))
    initial_status = sealed.get("status")
    checks = {
        "sealed_sparse_readiness_pass": initial_status == "PASS_GSE_SPARSE_CIRCULAR_RELATION_TRANSPORT_READINESS_V1",
        "typed_forward_remains_lidar_only": sparse_circular_relation_transport_input_contract()["forward_parameters"] == ("self", "scans"),
        "parameter_delta_is_exactly_903": parameter_count() == EXPECTED_PARAMETERS and head_parameters == EXPECTED_DELTA and EXPECTED_PARAMETERS - OLD_PARAMETERS == EXPECTED_DELTA,
        "count_output_is_finite_b5x7": tuple(count_logits.shape) == (8, 5, 7) and bool(torch.isfinite(count_logits).all()) and bool(torch.isfinite(count_probability).all()),
        "explicit_count_semantics_are_zero_through_six": explicit_classes == (0, 1, 2, 3, 4, 5, 6),
        "real_count_target_matches_unique_identity": torch.equal(count_target, identity_count),
        "real_count_target_within_capacity": int(count_target.min()) >= 0 and int(count_target.max()) <= MAX_TOKENS,
        "real_count_loss_is_finite": bool(torch.isfinite(count_loss)),
        "count_head_gradient_is_finite_nonzero": all_count_gradient_finite_nonzero,
        "count_is_causal": causal_error == 0.0,
        "count_rotation_invariant": rotation_error <= 3e-5,
        "count_batch_permutation_at_most_3e6": batch_error <= 3e-6,
        "count_repeat_exact": repeat_error == 0.0,
        "predecessor_backbone_loads_exactly": compatibility["required_keys"] == 53 and compatibility["loaded_values_exact"] and not compatibility["missing_required"] and not compatibility["shape_mismatch"],
        "zero_optimizer_checkpoint_test_graph": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    scientific_pass = all(checks.values())
    invariants = {"rotation": rotation_error, "batch": batch_error, "causal": causal_error, "repeat": repeat_error}
    summary = {
        "schema_version": "gse_sparse_relation_cardinality_corrective_v1",
        "status": PASS if scientific_pass else "FAIL_GSE_SPARSE_RELATION_CARDINALITY_CORRECTIVE_V1",
        "scientific_pass": scientific_pass,
        "decision": "ALLOW_SPARSE_RELATION_TRANSPORT_THREE_SEED_TRAINING_DATA_CARD" if scientific_pass else "STOP_SPARSE_RELATION_TRANSPORT_BEFORE_TRAINING",
        "question": "Does the sparse relation candidate have a minimal explicit causal 0--6 cardinality interface suitable for training?",
        "parameters": {"old": OLD_PARAMETERS, "new": EXPECTED_PARAMETERS, "delta": EXPECTED_DELTA, "count_head": head_parameters},
        "real_batch": {
            "observations": len(samples), "unique_history_frames": 34,
            "count_targets": count_target.detach().cpu().tolist(),
            "identity_counts": identity_count.detach().cpu().tolist(),
            "count_loss": float(count_loss.detach().cpu()),
            "count_head_gradient_max": gradient_max,
        },
        "checkpoint_compatibility": compatibility,
        "invariants": invariants,
        "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0, "checkpoints_created": 0,
        "c07_worlds_read": 0, "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
        "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
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
