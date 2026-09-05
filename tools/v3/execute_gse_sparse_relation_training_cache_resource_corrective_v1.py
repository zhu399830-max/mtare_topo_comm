#!/usr/bin/env python3
"""Compare cache-qualified seed0 epoch against the sealed no-cache epoch0 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtare_topo.governance import load_json, write_json


PASS = "PASS_GSE_SPARSE_RELATION_TRAINING_CACHE_RESOURCE_CORRECTIVE_V1"
FAIL = "FAIL_GSE_SPARSE_RELATION_TRAINING_CACHE_RESOURCE_CORRECTIVE_V1"
LIMIT = 16 * 1024**3
TOLERANCE = 3e-6


def _numbers(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _numbers(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _numbers(item, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    probe = load_json(args.probe.resolve() / "summary.json")
    baseline_line = next(
        line for line in (args.baseline_run.resolve() / "logs/01_seed0_training.log").read_text().splitlines()
        if line.startswith('{"c07"')
    )
    baseline_record = json.loads(baseline_line)
    current_record = {"c07": probe["c07"], "train": probe["train"]}
    reference_record = {"c07": baseline_record["c07"], "train": baseline_record["train"]}
    current_numbers = dict(_numbers(current_record))
    reference_numbers = dict(_numbers(reference_record))
    if set(current_numbers) != set(reference_numbers):
        raise RuntimeError("training cache metric key drift")
    metric_errors = {name: abs(current_numbers[name] - reference_numbers[name]) for name in current_numbers}
    current_state = torch.load(args.probe.resolve() / "parity_state.pt", map_location="cpu", weights_only=True)
    baseline_checkpoint = torch.load(args.baseline_run.resolve() / "artifacts/models/seed0/best.pt", map_location="cpu", weights_only=False)
    reference_state = baseline_checkpoint["model"]
    if set(current_state) != set(reference_state):
        raise RuntimeError("training cache model-state key drift")
    tensor_errors = {}
    tensor_exact = {}
    for name in sorted(current_state):
        tensor_exact[name] = bool(torch.equal(current_state[name], reference_state[name]))
        tensor_errors[name] = float((current_state[name].to(torch.float64) - reference_state[name].to(torch.float64)).abs().max())
    per_world = probe["per_world"]
    memory_keys = ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "nvidia_process_memory_bytes")
    maximum = {key: max(row[key] for row in per_world) for key in memory_keys}
    checks = {
        "exact_population": probe["fit_worlds"] == 60 and probe["fit_observations"] == 142184 and probe["descriptor_rows"] == 6347,
        "exact_steps": probe["core_optimizer_steps"] == 1137 and probe["descriptor_optimizer_steps"] == 86 and probe["optimizer_steps"] == 1223,
        "all_model_tensors_exact": all(tensor_exact.values()),
        "all_epoch_metrics_equivalent": max(metric_errors.values()) <= TOLERANCE,
        "all_world_allocated_below_16gib": maximum["peak_cuda_allocated_bytes"] <= LIMIT,
        "all_world_reserved_below_16gib": maximum["peak_cuda_reserved_bytes"] <= LIMIT,
        "all_world_process_memory_below_16gib": maximum["nvidia_process_memory_bytes"] <= LIMIT,
        "cache_cleared_at_every_training_world_boundary": probe["cache_clear_calls"] >= 120,
        "zero_selection_checkpoint_forbidden_reads": probe["selection_checkpoints_written"] == 0 and all(probe[key] == 0 for key in ("c08_worlds_read", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read", "graph_replays", "planner_calls")),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    summary = {
        "schema_version": "gse_sparse_relation_training_cache_resource_corrective_v1",
        "status": PASS if passed else FAIL,
        "scientific_pass": passed,
        "decision": "ALLOW_TRAINING_WORLD_CACHE_CLEAR_THREE_SEED_RESTART" if passed else "STOP_BEFORE_THREE_SEED_RESTART",
        "memory_limit_bytes": LIMIT,
        "maximum_memory": maximum,
        "authoritative_max_metric_error": max(metric_errors.values()),
        "authoritative_max_tensor_error": max(tensor_errors.values()),
        "tensor_exact_count": sum(tensor_exact.values()),
        "tensor_count": len(tensor_exact),
        "per_metric_abs_error": metric_errors,
        "per_tensor_max_abs_error": tensor_errors,
        "per_world": per_world,
        "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 1223,
        "selection_checkpoints_written": 0,
        "c08_worlds_read": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "graph_replays": 0,
        "planner_calls": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "figure_source.json", summary)
    x = np.arange(len(per_world))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    for key, label in zip(memory_keys, ("allocated", "reserved", "nvidia process")):
        axes[0].plot(x, [row[key] / 1024**3 for row in per_world], label=label, linewidth=1.2)
    axes[0].axhline(16, color="red", linestyle="--")
    axes[0].set(title="Seed0 epoch0 memory across 60 training worlds", xlabel="training-world order", ylabel="GiB")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].bar(["model tensors", "epoch metrics"], [max(tensor_errors.values()), max(metric_errors.values())])
    axes[1].axhline(TOLERANCE, color="red", linestyle="--")
    axes[1].set(title="Cache release numerical parity", ylabel="maximum absolute error")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Sparse relation training-world CUDA cache qualification")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_sparse_relation_training_cache_resource_corrective_v1.{suffix}", dpi=220)
    plt.close(fig)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "maximum_memory": maximum, "max_metric_error": summary["authoritative_max_metric_error"], "max_tensor_error": summary["authoritative_max_tensor_error"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
