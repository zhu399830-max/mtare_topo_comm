#!/usr/bin/env python3
"""Full-C07 batch256 validation probe with per-world CUDA cache clearing."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

import train_gse_axis_anchored_event_relation_v1 as base
import train_gse_sparse_circular_relation_transport_v2 as sparse
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_sparse_circular_relation_transport import SparseCircularRelationTransportNet
from probe_gse_sparse_relation_evaluation_batch_resource_v1 import DEPLOY_KEYS


def _nvidia_process_memory_bytes() -> int:
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    for line in output.splitlines():
        pid, memory = [part.strip() for part in line.split(",")]
        if int(pid) == os.getpid(): return int(memory) * 1024**2
    raise RuntimeError("cache probe process missing from nvidia-smi")


@torch.no_grad()
def _evaluate_rows(model, world, indices, *, device):
    values = defaultdict(list); loss_sums = defaultdict(float); rows = 0
    for start in range(0, len(indices), 256):
        selected = indices[start:start + 256]
        scans, target = sparse._batch(world, selected, device=device)
        predicted = model(scans); losses = sparse.sparse_core_loss(predicted, target)
        for name, value in losses.items(): loss_sums[name] += float(value) * len(selected)
        for name in DEPLOY_KEYS: values[name].append(predicted[name].cpu().numpy().astype(np.float32))
        values["token_bin_index"].append(predicted["token_bin_index"].cpu().numpy().astype(np.int16))
        rows += len(selected)
    return {name: np.concatenate(parts) for name, parts in values.items()}, {name: value / rows for name, value in loss_sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-root", required=True, type=Path); parser.add_argument("--sequence-manifest", required=True, type=Path); parser.add_argument("--predecessor-checkpoint", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path); args = parser.parse_args()
    started = time.monotonic(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    base.seed_everything(0)
    if not torch.cuda.is_available(): raise RuntimeError("cache resource probe requires CUDA")
    device = torch.device("cuda"); traversals = base.manifest_traversals(args.sequence_manifest.resolve()); dataset_root = args.dataset_root.resolve()
    fit_parent = "S01_flat_tree_small_C01"; fit = sparse._load_world(dataset_root, fit_parent, traversals[fit_parent])
    model = SparseCircularRelationTransportNet(); compatibility = sparse._load_predecessor_backbone(model, args.predecessor_checkpoint.resolve(), 0); model = model.to(device)
    model.train(); model.zero_grad(set_to_none=True); scans, target = sparse._batch(fit, np.arange(128), device=device); loss = sparse.sparse_core_loss(model(scans), target)["total"]; loss.backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()): raise RuntimeError("cache probe backward nonfinite")
    del scans, target, loss; model.eval()

    first_parent = "S01_flat_tree_small_C07"; first = sparse._load_world(dataset_root, first_parent, traversals[first_parent]); torch.cuda.empty_cache()
    parity_outputs, parity_loss = _evaluate_rows(model, first, np.arange(256), device=device)
    np.savez_compressed(output / "first256_outputs.npz", **parity_outputs); write_json(output / "first256_loss.json", parity_loss); del first, parity_outputs
    torch.cuda.empty_cache()

    parents = sorted(parent for parent in traversals if parent.endswith("_C07")); per_world = []; total_rows = 0; total_loss = defaultdict(float); cache_clear_calls = 1
    for parent in parents:
        world = sparse._load_world(dataset_root, parent, traversals[parent]); torch.cuda.empty_cache(); cache_clear_calls += 1; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        rows = len(world["event_index"]); world_loss = defaultdict(float)
        with torch.no_grad():
            for start in range(0, rows, 256):
                indices = np.arange(start, min(start + 256, rows)); scans, target = sparse._batch(world, indices, device=device); predicted = model(scans); losses = sparse.sparse_core_loss(predicted, target)
                for name, value in losses.items(): world_loss[name] += float(value) * len(indices)
                del scans, target, predicted, losses
        torch.cuda.synchronize(); memory = {"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()), "nvidia_process_memory_bytes": _nvidia_process_memory_bytes()}
        per_world.append({"parent_id": parent, "rows": rows, "loss": {name: value / rows for name, value in world_loss.items()}, **memory})
        for name, value in world_loss.items(): total_loss[name] += value
        total_rows += rows; del world; torch.cuda.empty_cache(); cache_clear_calls += 1
    summary = {"schema_version": "gse_sparse_relation_evaluation_cache_resource_probe_v1r", "fit_parent": fit_parent, "fit_backward_rows": 128, "optimizer_steps": 0, "predecessor_backbone": compatibility, "evaluation_batch_size": 256, "c07_worlds": len(parents), "c07_rows": total_rows, "cache_clear_calls": cache_clear_calls, "full_c07_loss": {name: value / total_rows for name, value in total_loss.items()}, "first256_loss": parity_loss, "per_world": per_world, "duration_seconds": time.monotonic() - started, "checkpoints_written": 0, "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0}
    write_json(output / "summary.json", summary); print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
