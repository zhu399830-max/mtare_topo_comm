#!/usr/bin/env python3
"""Isolated CUDA probe for sparse-relation evaluation batch memory/parity."""
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


DEPLOY_KEYS = (
    "event_logits", "proposal_logits", "token_count_logits", "token_count_probability",
    "token_bearing_deg", "token_existence_logits", "token_descriptor",
    "token_opening_width_m", "token_vertical_profile_m", "token_geometry_uncertainty",
    "transport_row_probability", "transport_reveal_probability", "local_axis",
    "place_descriptor", "observation_uncertainty",
)


def _nvidia_process_memory_bytes() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in output.splitlines():
        pid, memory = [part.strip() for part in line.split(",")]
        if int(pid) == os.getpid():
            return int(memory) * 1024**2
    raise RuntimeError("probe process missing from nvidia-smi")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--predecessor-checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation-batch-size", required=True, type=int, choices=(128, 256))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    base.seed_everything(0)
    if not torch.cuda.is_available():
        raise RuntimeError("resource probe requires CUDA")
    device = torch.device("cuda")
    traversals = base.manifest_traversals(args.sequence_manifest.resolve())
    fit_parent = "S01_flat_tree_small_C01"; validation_parent = "S01_flat_tree_small_C07"
    fit = sparse._load_world(args.dataset_root.resolve(), fit_parent, traversals[fit_parent])
    validation = sparse._load_world(args.dataset_root.resolve(), validation_parent, traversals[validation_parent])
    if len(fit["event_index"]) < 128 or len(validation["event_index"]) < 256:
        raise RuntimeError("resource probe population too small")
    model = SparseCircularRelationTransportNet()
    compatibility = sparse._load_predecessor_backbone(model, args.predecessor_checkpoint.resolve(), 0)
    model = model.to(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    model.train(); model.zero_grad(set_to_none=True)
    train_scans, train_target = sparse._batch(fit, np.arange(128), device=device)
    train_loss = sparse.sparse_core_loss(model(train_scans), train_target)["total"]
    train_loss.backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError("diagnostic backward produced nonfinite gradient")
    del train_scans, train_target, train_loss

    model.eval(); values = defaultdict(list); loss_sums = defaultdict(float); rows = 0
    with torch.no_grad():
        for start in range(0, 256, args.evaluation_batch_size):
            indices = np.arange(start, min(start + args.evaluation_batch_size, 256))
            scans, target = sparse._batch(validation, indices, device=device)
            predicted = model(scans); losses = sparse.sparse_core_loss(predicted, target)
            for name, value in losses.items():
                loss_sums[name] += float(value) * len(indices)
            for name in DEPLOY_KEYS:
                values[name].append(predicted[name].detach().cpu().numpy().astype(np.float32))
            values["token_bin_index"].append(predicted["token_bin_index"].detach().cpu().numpy().astype(np.int16))
            rows += len(indices)
            del scans, target, predicted, losses
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    process_memory = _nvidia_process_memory_bytes()
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}
    np.savez_compressed(output / "outputs.npz", **arrays)
    summary = {
        "schema_version": "gse_sparse_relation_evaluation_batch_probe_v1",
        "evaluation_batch_size": args.evaluation_batch_size,
        "fit_parent": fit_parent,
        "fit_backward_rows": 128,
        "optimizer_steps": 0,
        "validation_parent": validation_parent,
        "validation_rows": rows,
        "loss": {name: value / rows for name, value in loss_sums.items()},
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "nvidia_process_memory_bytes": process_memory,
        "predecessor_backbone": compatibility,
        "duration_seconds": time.monotonic() - started,
        "checkpoints_written": 0,
        "c08_worlds_read": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "graph_replays": 0,
        "planner_calls": 0,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
