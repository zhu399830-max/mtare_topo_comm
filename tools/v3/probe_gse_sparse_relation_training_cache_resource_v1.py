#!/usr/bin/env python3
"""One exact seed0 epoch with per-training-world CUDA cache release."""
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


def _nvidia_process_memory_bytes() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in output.splitlines():
        pid, memory = [part.strip() for part in line.split(",")]
        if int(pid) == os.getpid():
            return int(memory) * 1024**2
    raise RuntimeError("training cache probe process missing from nvidia-smi")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--predecessor-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base.seed_everything(0)
    if not torch.cuda.is_available():
        raise RuntimeError("training cache resource probe requires CUDA")
    device = torch.device("cuda")
    traversals = base.manifest_traversals(args.sequence_manifest.resolve())
    dataset_root = args.dataset_root.resolve()
    fit = sorted(parent for parent in traversals if int(parent.rsplit("_C", 1)[1]) <= 6)
    c07 = sorted(parent for parent in traversals if parent.endswith("_C07"))
    if (len(fit), len(c07)) != (60, 10):
        raise RuntimeError("training cache probe split drift")
    model = SparseCircularRelationTransportNet()
    compatibility = sparse._load_predecessor_backbone(model, args.predecessor_checkpoint.resolve(), 0)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = np.random.default_rng(0)
    order = [fit[index] for index in generator.permutation(len(fit))]
    model.train()
    sums = {"core": defaultdict(float), "descriptor": defaultdict(float)}
    row_counts = {"core": 0, "descriptor": 0}
    core_steps = descriptor_steps = 0
    event_population = np.zeros(5, dtype=np.int64)
    per_world = []
    cache_clear_calls = 0
    for parent in order:
        sparse._release_unused_cuda_cache(device)
        cache_clear_calls += 1
        torch.cuda.reset_peak_memory_stats()
        world = sparse._load_world(dataset_root, parent, traversals[parent])
        indices_order = generator.permutation(len(world["event_index"]))
        event_population += np.bincount(world["event_index"], minlength=5)
        world_core_steps = world_descriptor_steps = 0
        for batch_index, start in enumerate(range(0, len(indices_order), 128)):
            indices = indices_order[start:start + 128]
            shift = base._shift(0, 0, parent, batch_index, "sparse-core")
            scans, target = sparse._batch(world, indices, device=device, shift_columns=shift)
            losses = sparse.sparse_core_loss(model(scans), target)
            sparse._finite_step(model, optimizer, losses["total"])
            for name, value in losses.items():
                sums["core"][name] += float(value.detach()) * len(indices)
            row_counts["core"] += len(indices)
            core_steps += 1
            world_core_steps += 1
        records = None
        for batch_index, records in enumerate(base.descriptor_identity_batches(base._descriptor_records(world), seed=0, epoch=0)):
            indices = np.asarray([record.row for record in records], dtype=np.int64)
            shift = base._shift(0, 0, parent, batch_index, "sparse-descriptor")
            scans, target = sparse._batch(world, indices, device=device, shift_columns=shift)
            losses = sparse.sparse_descriptor_loss(model(scans), target)
            sparse._finite_step(model, optimizer, losses["total"])
            for name, value in losses.items():
                sums["descriptor"][name] += float(value.detach()) * len(indices)
            row_counts["descriptor"] += len(indices)
            descriptor_steps += 1
            world_descriptor_steps += 1
        torch.cuda.synchronize()
        per_world.append({
            "parent_id": parent,
            "core_steps": world_core_steps,
            "descriptor_steps": world_descriptor_steps,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "nvidia_process_memory_bytes": _nvidia_process_memory_bytes(),
        })
        del scans, target, losses, records, indices, indices_order, world
        sparse._release_unused_cuda_cache(device)
        cache_clear_calls += 1
    if core_steps != sparse.EXPECTED_CORE_STEPS_PER_EPOCH or descriptor_steps != sparse.EXPECTED_DESCRIPTOR_STEPS_PER_EPOCH:
        raise RuntimeError("training cache probe optimizer step drift")
    if row_counts != {"core": 142184, "descriptor": 6347}:
        raise RuntimeError("training cache probe row population drift")
    if not np.array_equal(event_population, sparse.EXPECTED_FIT_EVENT_COUNTS):
        raise RuntimeError("training cache probe event population drift")
    validation = sparse._validation(model, c07, traversals, dataset_root, device=device, batch_size=256)
    train = {
        stage: {name: value / row_counts[stage] for name, value in values.items()}
        for stage, values in sums.items()
    }
    parity_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(parity_state, output / "parity_state.pt")
    summary = {
        "schema_version": "gse_sparse_relation_training_cache_resource_probe_v1",
        "seed": 0,
        "epoch": 0,
        "parameters": sparse.EXPECTED_PARAMETERS,
        "predecessor_backbone": compatibility,
        "fit_worlds": 60,
        "fit_observations": row_counts["core"],
        "descriptor_rows": row_counts["descriptor"],
        "core_optimizer_steps": core_steps,
        "descriptor_optimizer_steps": descriptor_steps,
        "optimizer_steps": core_steps + descriptor_steps,
        "evaluation_batch_size": 256,
        "c07": validation,
        "train": train,
        "per_world": per_world,
        "cache_clear_calls": cache_clear_calls,
        "duration_seconds": time.monotonic() - started,
        "selection_checkpoints_written": 0,
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
