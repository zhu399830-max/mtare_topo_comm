#!/usr/bin/env python3
"""Infer one frozen spatial-longitudinal event-center seed on all C01--C08 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.gse_action_set_cache import ActionSetNodeDataset
from mtare_topo.data.gse_spatial_event_center_cache import SpatialEventCenterCache
from mtare_topo.evaluation.gse_spatial_center_projection import project_local_vectors
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector
from mtare_topo.representation.gse_event_center_offset import EventCenterOffsetHead
from mtare_topo.representation.gse_spatial_event_center import (
    SpatialEventCenterDecoder,
    SpatialLongitudinalCorrector,
)
from mtare_topo.teacher.gse_event_center_teacher import route_local_basis
from train_gse_spatial_event_center_v1 import _infer


EXPECTED = 188_126


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-cache", required=True, type=Path)
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--center-teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--scalar-checkpoint", required=True, type=Path)
    parser.add_argument("--spatial-checkpoint", required=True, type=Path)
    parser.add_argument("--corrective-checkpoint", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal spatial-center inference requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    global_index = np.load(args.action_cache.resolve() / "global_sequence_index.npy")
    if global_index.shape != (EXPECTED,) or len(np.unique(global_index)) != EXPECTED:
        raise RuntimeError("spatial-center inference observation identity drift")
    with np.load(args.center_teacher.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("spatial-center teacher identity drift")
        tangent = archive["route_tangent_xyz"].astype(np.float32)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_index):
            raise RuntimeError("spatial-center pair-cache identity drift")
        sensor = archive["sensor_xyz_m"].astype(np.float32)
        partition = archive["partition_code"].astype(np.uint8)
    if int(np.sum(partition == 0)) != 142_184 or int(np.sum(partition == 1)) != 45_942:
        raise RuntimeError("spatial-center inference split drift")
    tangent_norm = np.linalg.norm(tangent.astype(np.float64), axis=1)
    tangent_valid = tangent_norm > 0.5
    if int(np.sum(~tangent_valid)) != 2:
        raise RuntimeError("spatial-center degenerate route-tangent count drift")
    basis = route_local_basis(tangent, tangent_valid)

    action_checkpoint = torch.load(args.action_checkpoint.resolve(), map_location=device, weights_only=False)
    scalar_checkpoint = torch.load(args.scalar_checkpoint.resolve(), map_location=device, weights_only=False)
    spatial_checkpoint = torch.load(args.spatial_checkpoint.resolve(), map_location=device, weights_only=False)
    corrective_checkpoint = torch.load(args.corrective_checkpoint.resolve(), map_location=device, weights_only=False)
    if (
        action_checkpoint.get("schema_version") != "gse_action_set_node_checkpoint_v1"
        or scalar_checkpoint.get("schema_version") != "gse_event_center_dual_batch_corrective_checkpoint_v1"
        or spatial_checkpoint.get("schema_version") != "gse_spatial_event_center_checkpoint_v1"
        or corrective_checkpoint.get("schema_version") != "gse_spatial_longitudinal_corrective_checkpoint_v1"
        or any(value.get("seed") != args.seed for value in (
            action_checkpoint, scalar_checkpoint, spatial_checkpoint, corrective_checkpoint,
        ))
    ):
        raise RuntimeError("spatial-center inference checkpoint drift")
    if (
        corrective_checkpoint.get("action_checkpoint_sha256") != _sha(args.action_checkpoint.resolve())
        or corrective_checkpoint.get("scalar_checkpoint_sha256") != _sha(args.scalar_checkpoint.resolve())
        or corrective_checkpoint.get("spatial_checkpoint_sha256") != _sha(args.spatial_checkpoint.resolve())
    ):
        raise RuntimeError("spatial-center corrective provenance drift")

    base = ActionSetNodeDetector().to(device)
    base.load_state_dict(action_checkpoint["model"], strict=True)
    scalar = EventCenterOffsetHead().to(device)
    scalar.load_state_dict(scalar_checkpoint["head"], strict=True)
    spatial = SpatialEventCenterDecoder().to(device)
    spatial.load_state_dict(spatial_checkpoint["decoder"], strict=True)
    corrector = SpatialLongitudinalCorrector(spatial).to(device)
    corrector.longitudinal_residual.load_state_dict(
        corrective_checkpoint["longitudinal_residual"], strict=True,
    )
    corrector.freeze_spatial_decoder()
    for module in (base, scalar, corrector):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    rows = np.arange(EXPECTED, dtype=np.int64)
    vector = _infer(
        base, scalar, corrector,
        ActionSetNodeDataset(args.action_cache.resolve(), rows),
        SpatialEventCenterCache(args.spatial_cache.resolve()),
        rows, device, args.batch_size,
    ).astype(np.float32)
    if vector.shape != (EXPECTED, 3) or not np.all(np.isfinite(vector)):
        raise RuntimeError("spatial-center inference output drift")
    center = project_local_vectors(vector, sensor, basis)
    if not np.array_equal(center[~tangent_valid], sensor[~tangent_valid]):
        raise RuntimeError("degenerate-route spatial-center fallback drift")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray("gse_spatial_longitudinal_center_seed_all_v1"),
        seed=np.asarray(args.seed, dtype=np.int8),
        global_sequence_index=global_index.astype(np.int64),
        predicted_local_vector_m=vector,
        predicted_center_xyz_m=center,
        sensor_xyz_m=sensor,
        route_local_basis=basis,
        tangent_valid=tangent_valid,
    )
    summary = {
        "schema_version": "gse_spatial_longitudinal_center_seed_all_v1",
        "seed": args.seed,
        "rows": EXPECTED,
        "fit_rows": 142_184,
        "selection_rows": 45_942,
        "degenerate_route_rows": int(np.sum(~tangent_valid)),
        "output_sha256": _sha(args.output),
        "optimizer_steps": 0,
        "model_updates": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
