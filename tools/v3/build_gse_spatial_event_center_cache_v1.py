#!/usr/bin/env python3
"""Build one temporary height-aware frozen cache for event-center learning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from mtare_topo.data.cano_sensor_smoke import MAX_RANGE_M, NEAR_RANGE_M
from mtare_topo.data.gse_spatial_event_center_cache import compact_five_frame_references
from mtare_topo.representation.gse_causal_episode_detector import (
    encode_spatial_scan_features,
    materialize_past_only_references,
)
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet
from mtare_topo.representation.gse_spatial_event_center import (
    DIRECTIONAL_BINS,
    ELEVATION_BINS,
    ENCODER_DIM,
    HISTORY_FRAMES,
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("spatial event-center cache overwrite is forbidden")
    output.mkdir(parents=True)

    import torch
    import zarr

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA spatial cache build has no CUDA device")
    device = torch.device(args.device)
    shard_paths = sorted((args.dataset_run.resolve() / "artifacts/dataset/train").glob("*.zarr"))
    if len(shard_paths) != 80 or any(not PARENT_PATTERN.match(path.stem) for path in shard_paths):
        raise RuntimeError("spatial cache must read exactly the 80 C01-C08 worlds")
    teacher_rows = _read_jsonl(args.teacher.resolve())
    if len(teacher_rows) != 188126 or {str(row["parent_id"]) for row in teacher_rows} != {p.stem for p in shard_paths}:
        raise RuntimeError("spatial cache Teacher population drift")
    bank = materialize_past_only_references(teacher_rows)
    exact_references = bank.global_frame_references[:, -HISTORY_FRAMES:]
    if not bool(bank.valid_history_mask[:, -HISTORY_FRAMES:].all()):
        raise RuntimeError("one causal observation lacks its exact five LiDAR frames")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "gse_graph_checkpoint_v1" or checkpoint.get("seed") != args.seed:
        raise RuntimeError("frozen GSE spatial checkpoint drift")
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
            raise RuntimeError(f"source shard identity drift: {path.name}")
        previous_global = int(frames[-1])
        frame_chunks.append(frames)
        shard_metadata.append({
            "parent_id": path.stem, "unique_frames": len(frames),
            "first_global_frame_index": int(frames[0]),
            "last_global_frame_index": int(frames[-1]),
        })
    global_frames = np.concatenate(frame_chunks)
    if len(global_frames) != 252430 or len(np.unique(global_frames)) != len(global_frames):
        raise RuntimeError("spatial cache unique frame count drift")
    compact = compact_five_frame_references(exact_references, global_frames)

    pooled = np.lib.format.open_memmap(
        output / "pooled.npy", mode="w+", dtype=np.float16,
        shape=(len(global_frames), ENCODER_DIM),
    )
    azimuth = np.lib.format.open_memmap(
        output / "azimuth.npy", mode="w+", dtype=np.float16,
        shape=(len(global_frames), ENCODER_DIM, DIRECTIONAL_BINS),
    )
    elevation = np.lib.format.open_memmap(
        output / "elevation.npy", mode="w+", dtype=np.float16,
        shape=(len(global_frames), ENCODER_DIM, ELEVATION_BINS),
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
                    raise RuntimeError(f"sensor payload drift: {path.name}")
                scans = np.stack((range_m / MAX_RANGE_M, valid), axis=1)[:, None]
                features = encode_spatial_scan_features(
                    model.encoder, torch.from_numpy(scans).to(device=device, dtype=torch.float32),
                )
                destination = slice(write_offset + start, write_offset + stop)
                pooled[destination] = features["pooled"][:, 0].cpu().numpy().astype(np.float16)
                azimuth[destination] = features["directional"][:, 0].cpu().numpy().astype(np.float16)
                elevation[destination] = features["vertical"][:, 0].cpu().numpy().astype(np.float16)
            write_offset += count
    for array in (pooled, azimuth, elevation):
        array.flush()
    del pooled, azimuth, elevation
    if write_offset != len(global_frames):
        raise RuntimeError("spatial cache frame write count drift")
    np.save(output / "global_frame_index.npy", global_frames, allow_pickle=False)
    np.save(output / "compact_reference_index.npy", compact, allow_pickle=False)
    array_names = (
        "global_frame_index.npy", "compact_reference_index.npy", "pooled.npy",
        "azimuth.npy", "elevation.npy",
    )
    manifest = {
        "schema_version": "gse_spatial_event_center_cache_v1",
        "seed": args.seed,
        "history_frames": HISTORY_FRAMES,
        "encoder_dim": ENCODER_DIM,
        "directional_bins": DIRECTIONAL_BINS,
        "elevation_bins": ELEVATION_BINS,
        "worlds": 80,
        "unique_frames": len(global_frames),
        "causal_observations": len(teacher_rows),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "teacher_sha256": _sha256(args.teacher.resolve()),
        "array_sha256": {name: _sha256(output / name) for name in array_names},
        "cache_bytes": sum((output / name).stat().st_size for name in array_names),
        "model_inference_frames": len(global_frames),
        "optimizer_steps": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "shards": shard_metadata,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in (
        "schema_version", "seed", "unique_frames", "causal_observations",
        "cache_bytes", "duration_seconds",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
