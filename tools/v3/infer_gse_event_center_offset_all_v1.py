#!/usr/bin/env python3
"""Infer deployment-safe event-center projections for every C01--C08 row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mtare_topo.data.gse_action_set_cache import ActionSetNodeDataset
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector
from mtare_topo.representation.gse_event_center_offset import EventCenterOffsetHead
from mtare_topo.teacher.gse_event_center_teacher import traversal_tangents


def _collate(dataset, start, stop):
    samples = [dataset[index] for index in range(start, stop)]
    return np.stack([value["tokens"] for value in samples]), np.stack([value["history_mask"] for value in samples])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--action-model-run", required=True, type=Path)
    parser.add_argument("--center-model-run", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("event-center inference output exists; overwrite is forbidden")
    import torch

    count = 188_126
    global_index = np.load(args.action_cache / "global_sequence_index.npy")
    traversal = np.load(args.action_cache / "traversal_id.npy").astype(str)
    sequence = np.load(args.action_cache / "sequence_index.npy")
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_index):
            raise RuntimeError("event-center inference pair-cache identity drift")
        xyz = archive["sensor_xyz_m"].astype(np.float64)
    if len(global_index) != count:
        raise RuntimeError("event-center inference population drift")
    dataset = ActionSetNodeDataset(args.action_cache, np.arange(count))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    for seed in range(3):
        action_checkpoint = torch.load(
            args.action_model_run / f"artifacts/models/seed{seed}/best.pt",
            map_location=device, weights_only=False,
        )
        center_checkpoint = torch.load(
            args.center_model_run / f"artifacts/models/seed{seed}/best.pt",
            map_location=device, weights_only=False,
        )
        if center_checkpoint.get("schema_version") != "gse_event_center_offset_checkpoint_v1":
            raise RuntimeError(f"event-center seed{seed} checkpoint drift")
        base = ActionSetNodeDetector().to(device); base.load_state_dict(action_checkpoint["model"], strict=True); base.eval()
        head = EventCenterOffsetHead().to(device); head.load_state_dict(center_checkpoint["head"], strict=True); head.eval()
        output = []
        with torch.inference_mode():
            for start in range(0, count, 1024):
                tokens, mask = _collate(dataset, start, min(count, start + 1024))
                context = base(torch.from_numpy(tokens).to(device), torch.from_numpy(mask).to(device))["causal_context"]
                output.append(head(context).cpu().numpy())
        predictions.append(np.concatenate(output))
    seed_offset = np.stack(predictions).astype(np.float32)
    offset = seed_offset.mean(axis=0)
    std = seed_offset.std(axis=0)
    tangent = traversal_tangents(traversal, sequence, xyz)
    projected = xyz + offset[:, None] * tangent
    np.savez_compressed(
        args.output, schema_version=np.asarray("gse_event_center_projection_v1"),
        global_sequence_index=global_index, seed_offset_m=seed_offset,
        predicted_offset_m=offset, offset_std_m=std,
        route_tangent_xyz=tangent, sensor_xyz_m=xyz.astype(np.float32),
        projected_center_xyz_m=projected.astype(np.float32),
    )
    print(json.dumps({
        "schema_version": "gse_event_center_projection_v1", "rows": count,
        "offset_min_m": float(offset.min()), "offset_max_m": float(offset.max()),
        "offset_std_mean_m": float(std.mean()), "offset_std_p95_m": float(np.percentile(std, 95)),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
