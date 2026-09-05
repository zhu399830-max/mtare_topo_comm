#!/usr/bin/env python3
"""Benchmark one real pair-aware GSE batch without an optimizer update."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.data.gse_training_sampler import PairAwareBatchSampler
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet, gse_multitask_loss
from train_gse_graph_v1 import _to_device, class_weights, collate_gse, seed_everything


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size != 64 or not torch.cuda.is_available():
        raise RuntimeError("the frozen real-batch smoke requires CUDA batch size 64")
    seed_everything(0)
    dataset = GSESequenceDataset(args.dataset_run, "train", augment_azimuth=True, augmentation_seed=0)
    sampler = PairAwareBatchSampler(dataset.association_labels(), batch_size=64, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_gse, num_workers=0, pin_memory=True)
    device = torch.device("cuda:0")
    weights, counts = class_weights(dataset.event_labels(), device)
    model = GeometrySemanticEventNet().to(device)
    before = _state_sha256(model)
    started = time.monotonic()
    batch = next(iter(loader))
    load_seconds = time.monotonic() - started
    student = batch["student"].to(device, non_blocking=True)
    targets = _to_device(batch["targets"], device)
    torch.cuda.reset_peak_memory_stats()
    compute_started = time.monotonic()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(student)
        losses = gse_multitask_loss(outputs, targets, event_class_weights=weights)
    losses["total"].backward()
    torch.cuda.synchronize()
    compute_seconds = time.monotonic() - compute_started
    after = _state_sha256(model)
    valid_identity = targets["association_valid_mask"].bool()
    identities = targets["association_identity"][valid_identity].cpu().numpy()
    unique, identity_counts = np.unique(identities, return_counts=True)
    result = {
        "schema_version": "gse_real_batch_no_update_smoke_v1",
        "overall_status": "PASS_GSE_REAL_BATCH_NO_UPDATE_SMOKE_V1"
        if before == after and all(torch.isfinite(value) for value in losses.values())
        else "FAIL_GSE_REAL_BATCH_NO_UPDATE_SMOKE_V1",
        "dataset_run": str(args.dataset_run.resolve().relative_to(PROJECT_ROOT)),
        "batch_size": len(student),
        "student_shape": list(student.shape),
        "target_exit_shape": list(targets["exit_mask"].shape),
        "event_counts": counts,
        "batch_valid_association_observations": int(valid_identity.sum()),
        "batch_positive_association_identities": int(np.sum(identity_counts >= 2)),
        "batch_unique_association_identities": int(len(unique)),
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
        "weights_unchanged": before == after,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "checkpoint_created": False,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "load_seconds": load_seconds,
        "forward_backward_seconds": compute_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["overall_status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
