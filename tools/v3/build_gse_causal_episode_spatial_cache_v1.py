#!/usr/bin/env python3
"""Build one temporary, frozen-encoder cache for causal episode training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from mtare_topo.data.cano_sensor_smoke import MAX_RANGE_M, NEAR_RANGE_M
from mtare_topo.data.gse_causal_episode_cache import (
    compact_global_references,
    materialize_boundary_targets,
)
from mtare_topo.representation.gse_causal_episode_detector import (
    DIRECTIONAL_BINS,
    ENCODER_DIM,
    HISTORY_FRAMES,
    encode_spatial_scan_features,
    materialize_causal_episode_references,
)
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet


PARENT_PATTERN = re.compile(r"^S\d+_.+_C0[1-8]$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _save(path: Path, array: np.ndarray) -> None:
    np.save(path, array, allow_pickle=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--transition-timing", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--baseline-features", required=True, type=Path)
    parser.add_argument("--population-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    started = time.monotonic()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("spatial cache output already exists; overwrite is forbidden")
    output.mkdir(parents=True)

    import torch
    import zarr

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the requested cache build")
    device = torch.device(args.device)
    dataset_root = args.dataset_run.resolve() / "artifacts/dataset/train"
    shard_paths = sorted(dataset_root.glob("*.zarr"))
    if len(shard_paths) != 80 or any(not PARENT_PATTERN.match(path.stem) for path in shard_paths):
        raise RuntimeError("cache input must be exactly the 80 C01-C08 development worlds")
    teacher_rows = _read_jsonl(args.teacher.resolve())
    if len(teacher_rows) != 188126:
        raise RuntimeError("causal Teacher observation count drift")
    parent_ids = {str(row["parent_id"]) for row in teacher_rows}
    if parent_ids != {path.stem for path in shard_paths}:
        raise RuntimeError("Teacher and sensor shard world identities disagree")
    bank = materialize_causal_episode_references(teacher_rows)
    timing_rows = _read_csv(args.transition_timing.resolve())
    boundary_target, boundary_valid, boundary_audit = materialize_boundary_targets(
        teacher_rows, timing_rows
    )

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "gse_graph_checkpoint_v1":
        raise RuntimeError("frozen GSE checkpoint schema drift")
    model = GeometrySemanticEventNet()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    frame_chunks: list[np.ndarray] = []
    shard_metadata: list[dict] = []
    previous_global = -1
    for path in shard_paths:
        group = zarr.open_group(str(path), mode="r")
        frames = np.asarray(group["global_frame_index"][:], dtype=np.int64)
        if (
            group.attrs.get("parent_id") != path.stem
            or group.attrs.get("split") != "train"
            or len(frames) == 0
            or np.any(np.diff(frames) <= 0)
            or int(frames[0]) <= previous_global
        ):
            raise RuntimeError(f"source shard frame identity drift: {path.name}")
        previous_global = int(frames[-1])
        frame_chunks.append(frames)
        shard_metadata.append({
            "parent_id": path.stem,
            "unique_frames": len(frames),
            "first_global_frame_index": int(frames[0]),
            "last_global_frame_index": int(frames[-1]),
        })
    global_frames = np.concatenate(frame_chunks)
    if len(global_frames) != 252430 or len(np.unique(global_frames)) != len(global_frames):
        raise RuntimeError("unique C01-C08 frame count drift")
    compact = compact_global_references(
        bank.global_frame_references, bank.valid_history_mask, global_frames
    )

    with np.load(args.population_index.resolve(), allow_pickle=False) as archive:
        population_global = archive["compact_to_global_sequence_index"].astype(np.int64)
        population_parent = archive["parent_id"].astype(str)
        partition_code = archive["partition_code"].astype(np.uint8)
    teacher_global = np.asarray(
        [int(row["global_sequence_index"]) for row in teacher_rows], dtype=np.int64
    )
    teacher_parent = np.asarray([str(row["parent_id"]) for row in teacher_rows])
    if (
        not np.array_equal(population_global, teacher_global)
        or not np.array_equal(population_parent, teacher_parent)
        or partition_code.shape != (len(teacher_rows),)
        or set(partition_code.tolist()) != {0, 1}
        or int(np.sum(partition_code == 0)) != 142184
        or int(np.sum(partition_code == 1)) != 45942
    ):
        raise RuntimeError("frozen fit/selection population alignment drift")

    baseline = np.load(args.baseline_features.resolve(), mmap_mode="r")
    if baseline.shape != (len(teacher_rows), 146) or baseline.dtype != np.float32:
        raise RuntimeError("frozen baseline feature contract drift")

    pooled = np.lib.format.open_memmap(
        output / "pooled.npy", mode="w+", dtype=np.float16,
        shape=(len(global_frames), ENCODER_DIM),
    )
    directional = np.lib.format.open_memmap(
        output / "directional.npy", mode="w+", dtype=np.float16,
        shape=(len(global_frames), ENCODER_DIM, DIRECTIONAL_BINS),
    )
    write_offset = 0
    with torch.inference_mode():
        for path, metadata in zip(shard_paths, shard_metadata, strict=True):
            group = zarr.open_group(str(path), mode="r")
            count = int(metadata["unique_frames"])
            for start in range(0, count, args.batch_size):
                stop = min(count, start + args.batch_size)
                range_m = np.asarray(group["range_m"][start:stop], dtype=np.float32)
                valid = np.asarray(group["valid_mask"][start:stop], dtype=np.float32)
                if (
                    range_m.shape != (stop - start, 16, 720)
                    or valid.shape != range_m.shape
                    or not np.all(np.isfinite(range_m))
                    or np.any(range_m < NEAR_RANGE_M)
                    or np.any(range_m > MAX_RANGE_M)
                    or not np.all((valid == 0.0) | (valid == 1.0))
                ):
                    raise RuntimeError(f"sensor payload contract drift: {path.name}")
                scans = np.stack((range_m / MAX_RANGE_M, valid), axis=1)[:, None]
                tensor = torch.from_numpy(scans).to(device=device, dtype=torch.float32)
                features = encode_spatial_scan_features(model.encoder, tensor)
                destination = slice(write_offset + start, write_offset + stop)
                pooled[destination] = features["pooled"][:, 0].cpu().numpy().astype(np.float16)
                directional[destination] = features["directional"][:, 0].cpu().numpy().astype(np.float16)
            write_offset += count
    pooled.flush()
    directional.flush()
    del pooled, directional
    if write_offset != len(global_frames):
        raise RuntimeError("cache frame write count drift")

    _save(output / "global_frame_index.npy", global_frames.astype(np.int64, copy=False))
    _save(output / "compact_reference_index.npy", compact)
    _save(output / "valid_history_mask.npy", bank.valid_history_mask)
    _save(output / "episode_id.npy", bank.episode_id.astype(np.int64, copy=False))
    _save(output / "event_index.npy", bank.event_index.astype(np.int8, copy=False))
    _save(output / "global_sequence_index.npy", teacher_global)
    _save(output / "partition_code.npy", partition_code)
    _save(output / "boundary_offset_m.npy", boundary_target)
    _save(output / "boundary_valid.npy", boundary_valid)
    array_names = [
        "global_frame_index.npy", "pooled.npy", "directional.npy",
        "compact_reference_index.npy", "valid_history_mask.npy", "episode_id.npy",
        "event_index.npy", "global_sequence_index.npy", "partition_code.npy",
        "boundary_offset_m.npy", "boundary_valid.npy",
    ]
    manifest = {
        "schema_version": "gse_causal_episode_spatial_cache_v1",
        "seed": args.seed,
        "history_frames": HISTORY_FRAMES,
        "encoder_dim": ENCODER_DIM,
        "directional_bins": DIRECTIONAL_BINS,
        "worlds": len(shard_paths),
        "unique_frames": len(global_frames),
        "causal_observations": len(teacher_rows),
        "structural_episodes": int(np.max(bank.episode_id)) + 1,
        "valid_reference_cells": int(bank.valid_history_mask.sum()),
        "boundary_supervision": boundary_audit,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "baseline_features_sha256": _sha256(args.baseline_features.resolve()),
        "population_index_sha256": _sha256(args.population_index.resolve()),
        "teacher_sha256": _sha256(args.teacher.resolve()),
        "transition_timing_sha256": _sha256(args.transition_timing.resolve()),
        "array_sha256": {name: _sha256(output / name) for name in array_names},
        "cache_bytes": sum((output / name).stat().st_size for name in array_names),
        "model_inference_frames": len(global_frames),
        "optimizer_steps": 0,
        "strict_test_worlds_read": 0,
        "c09_worlds_read": 0,
        "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "shards": shard_metadata,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "schema_version", "seed", "worlds", "unique_frames", "causal_observations",
        "structural_episodes", "cache_bytes", "duration_seconds"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
