#!/usr/bin/env python3
"""Replay the attributed JCGS circular-index failure without optimizer steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_gse_circular_peak_geometry_v1 as core
from mtare_topo.representation.gse_joint_cyclic_gap_simplex import (
    BEARING_BINS,
    BIN_WIDTH_DEG,
    JointCyclicGapSimplexNet,
    joint_cyclic_gap_simplex_loss,
    periodic_linear_coordinates,
)


ATTRIBUTED_PARENT = "S10_3d_complex_C01"
ATTRIBUTED_GLOBAL_SEQUENCE_INDEX = 180664
ATTRIBUTED_SEED = 1
ATTRIBUTED_EPOCH = 4
ATTRIBUTED_BATCH_INDEX = 9
ATTRIBUTED_SHIFT_COLUMNS = 360


def _attributed_batch(teacher_root: Path, source_root: Path, device: torch.device):
    paths = sorted(teacher_root.glob("fit/*.zarr"))
    generator = np.random.default_rng(ATTRIBUTED_SEED * 1000 + ATTRIBUTED_EPOCH)
    ordered = [paths[index] for index in generator.permutation(len(paths))]
    for path in ordered:
        world = core._load_world(path, source_root)
        order = generator.permutation(len(world["presence"]))
        if world["parent"] != ATTRIBUTED_PARENT:
            continue
        start = ATTRIBUTED_BATCH_INDEX * 128
        indices = order[start : start + 128]
        global_indices = np.asarray(world["global_sequence_index"])[indices]
        scans, targets = core._batch(
            world, indices, device=device, shift_columns=ATTRIBUTED_SHIFT_COLUMNS
        )
        return scans, targets, global_indices
    raise RuntimeError("attributed parent absent from deterministic epoch ordering")


def _coordinate_audit(targets: dict[str, torch.Tensor]) -> dict[str, object]:
    presence = targets["presence"].bool()
    count = presence.sum(dim=1)
    raw_lower_values, canonical_lower_values = [], []
    attributed_raw = attributed_canonical = None
    for cardinality in range(1, 5):
        rows = torch.nonzero(count == cardinality, as_tuple=False).flatten()
        if not len(rows):
            continue
        bins = torch.nonzero(presence[rows], as_tuple=False)[:, 1].reshape(len(rows), cardinality)
        residual = torch.gather(targets["heading_residual_deg"][rows], 1, bins).to(torch.float64)
        bearings = bins.to(torch.float64) * BIN_WIDTH_DEG + residual
        for shift in range(cardinality):
            first = torch.roll(bearings, shifts=-shift, dims=1)[:, 0]
            continuous = torch.remainder(first / BIN_WIDTH_DEG, BEARING_BINS)
            raw_lower = torch.floor(continuous).to(torch.long)
            _, canonical_lower, _ = periodic_linear_coordinates(first)
            raw_lower_values.append(raw_lower.cpu())
            canonical_lower_values.append(canonical_lower.cpu())
            boundary = raw_lower == BEARING_BINS
            if bool(boundary.any()):
                attributed_raw = int(raw_lower[boundary][0])
                attributed_canonical = int(canonical_lower[boundary][0])
    raw = torch.cat(raw_lower_values)
    canonical = torch.cat(canonical_lower_values)
    return {
        "raw_lower_min": int(raw.min()),
        "raw_lower_max": int(raw.max()),
        "raw_out_of_bounds": int(((raw < 0) | (raw >= BEARING_BINS)).sum()),
        "attributed_raw_lower": attributed_raw,
        "attributed_canonical_lower": attributed_canonical,
        "canonical_lower_min": int(canonical.min()),
        "canonical_lower_max": int(canonical.max()),
        "canonical_out_of_bounds": int(((canonical < 0) | (canonical >= BEARING_BINS)).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.manual_seed(ATTRIBUTED_SEED)
    torch.cuda.manual_seed_all(ATTRIBUTED_SEED)
    torch.use_deterministic_algorithms(True)

    _, cpu_targets, global_indices = _attributed_batch(
        args.teacher_root.resolve(), args.source_root.resolve(), torch.device("cpu")
    )
    coordinate = _coordinate_audit(cpu_targets)
    if ATTRIBUTED_GLOBAL_SEQUENCE_INDEX not in global_indices:
        raise RuntimeError("formal attributed row missing from reconstructed batch")
    if coordinate != {
        "raw_lower_min": 0,
        "raw_lower_max": 180,
        "raw_out_of_bounds": 1,
        "attributed_raw_lower": 180,
        "attributed_canonical_lower": 0,
        "canonical_lower_min": 0,
        "canonical_lower_max": 179,
        "canonical_out_of_bounds": 0,
    }:
        raise RuntimeError(f"attributed coordinate evidence drift: {coordinate}")
    if not torch.cuda.is_available():
        raise RuntimeError("boundary readiness requires CUDA replay")
    device = torch.device("cuda")
    scans, targets, replay_indices = _attributed_batch(
        args.teacher_root.resolve(), args.source_root.resolve(), device
    )
    if not np.array_equal(global_indices, replay_indices):
        raise RuntimeError("CPU/CUDA attributed batch drift")
    model = JointCyclicGapSimplexNet().to(device)
    losses = joint_cyclic_gap_simplex_loss(model(scans), targets)
    losses["total"].backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    summary = {
        "schema_version": "gse_joint_cyclic_gap_simplex_boundary_index_readiness_v1",
        "status": "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_BOUNDARY_INDEX_READINESS_V1",
        "scientific_pass": True,
        "parent": ATTRIBUTED_PARENT,
        "global_sequence_index": ATTRIBUTED_GLOBAL_SEQUENCE_INDEX,
        "batch_rows": int(len(global_indices)),
        "batch_global_sequence_index_min": int(global_indices.min()),
        "batch_global_sequence_index_max": int(global_indices.max()),
        "coordinate_audit": coordinate,
        "cuda_forward_backward_pass": True,
        "finite_gradients": finite_gradients,
        "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "c08_worlds_read": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "graph_replays": 0,
        "planner_calls": 0,
    }
    if not finite_gradients:
        summary["status"] = "FAIL_GSE_JOINT_CYCLIC_GAP_SIMPLEX_BOUNDARY_INDEX_READINESS_V1"
        summary["scientific_pass"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["scientific_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
