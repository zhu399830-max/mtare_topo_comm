#!/usr/bin/env python3
"""Compare isolated batch128/batch256 resource probes and render evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtare_topo.governance import load_json, write_json

PASS = "PASS_GSE_SPARSE_RELATION_EVALUATION_BATCH_RESOURCE_CORRECTIVE_V1"
FAIL = "FAIL_GSE_SPARSE_RELATION_EVALUATION_BATCH_RESOURCE_CORRECTIVE_V1"
LIMIT = 16 * 1024**3
TOLERANCE = 3e-6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe128", required=True, type=Path)
    parser.add_argument("--probe256", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    summaries = {size: load_json(path.resolve() / "summary.json") for size, path in ((128, args.probe128), (256, args.probe256))}
    arrays = {size: dict(np.load(path.resolve() / "outputs.npz")) for size, path in ((128, args.probe128), (256, args.probe256))}
    if set(arrays[128]) != set(arrays[256]):
        raise RuntimeError("probe output key drift")
    errors = {}
    for name in sorted(arrays[128]):
        left, right = arrays[128][name], arrays[256][name]
        if left.shape != right.shape:
            raise RuntimeError(f"probe output shape drift: {name}")
        errors[name] = 0.0 if np.issubdtype(left.dtype, np.integer) and np.array_equal(left, right) else float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    loss_error = {name: abs(summaries[128]["loss"][name] - summaries[256]["loss"][name]) for name in summaries[128]["loss"]}
    memory = {
        str(size): {name: summaries[size][name] for name in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "nvidia_process_memory_bytes")}
        for size in (128, 256)
    }
    checks = {
        "exact_probe_populations": all(summaries[size]["fit_backward_rows"] == 128 and summaries[size]["validation_rows"] == 256 for size in (128, 256)),
        "zero_optimizer_checkpoint_forbidden_reads": all(summaries[size]["optimizer_steps"] == 0 and summaries[size]["checkpoints_written"] == 0 and all(summaries[size][key] == 0 for key in ("c08_worlds_read", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read", "graph_replays", "planner_calls")) for size in (128, 256)),
        "integer_outputs_exact": all(errors[name] == 0 for name in errors if np.issubdtype(arrays[128][name].dtype, np.integer)),
        "floating_outputs_equivalent": max(errors.values()) <= TOLERANCE,
        "aggregate_losses_equivalent": max(loss_error.values()) <= TOLERANCE,
        "batch128_allocated_below_16gib": memory["128"]["peak_cuda_allocated_bytes"] <= LIMIT,
        "batch128_reserved_below_16gib": memory["128"]["peak_cuda_reserved_bytes"] <= LIMIT,
        "batch128_process_memory_below_16gib": memory["128"]["nvidia_process_memory_bytes"] <= LIMIT,
        "batch256_reproduces_resource_violation": max(memory["256"].values()) > LIMIT,
    }
    checks = {name: bool(value) for name, value in checks.items()}; passed = all(checks.values())
    summary = {
        "schema_version": "gse_sparse_relation_evaluation_batch_resource_corrective_v1",
        "status": PASS if passed else FAIL,
        "scientific_pass": passed,
        "decision": "ALLOW_EVALUATION_BATCH128_THREE_SEED_TRAINING" if passed else "STOP_BEFORE_TRAINING",
        "memory": memory,
        "per_output_max_abs_error": errors,
        "per_loss_abs_error": loss_error,
        "authoritative_max_output_error": max(errors.values()),
        "authoritative_max_loss_error": max(loss_error.values()),
        "memory_limit_bytes": LIMIT,
        "checks": checks,
        "duration_seconds": time.monotonic() - started,
        "optimizer_steps": 0,
        "checkpoints_written": 0,
        "c08_worlds_read": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "graph_replays": 0,
        "planner_calls": 0,
    }
    write_json(output / "summary.json", summary); write_json(output / "figure_source.json", summary)
    labels = ("allocated", "reserved", "nvidia process"); keys = ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "nvidia_process_memory_bytes")
    x = np.arange(3); fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for offset, size in ((-.18, 128), (.18, 256)):
        axes[0].bar(x + offset, [memory[str(size)][key] / 1024**3 for key in keys], .36, label=f"batch {size}")
    axes[0].axhline(16, color="red", ls="--", label="16 GiB contract"); axes[0].set(xticks=x, xticklabels=labels, ylabel="GiB", title="CUDA resource qualification"); axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    top = sorted(errors.items(), key=lambda item: item[1], reverse=True)[:8]
    axes[1].barh([name for name, _ in reversed(top)], [value for _, value in reversed(top)])
    axes[1].axvline(TOLERANCE, color="red", ls="--"); axes[1].set(title="Batch 128 vs 256 output parity", xlabel="maximum absolute error"); axes[1].grid(axis="x", alpha=.2)
    fig.suptitle("Sparse relation evaluation batch resource corrective")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_sparse_relation_evaluation_batch_resource_corrective_v1.{suffix}", dpi=220)
    plt.close(fig)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"], "memory": memory, "max_output_error": summary["authoritative_max_output_error"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
