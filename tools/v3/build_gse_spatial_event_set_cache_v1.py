#!/usr/bin/env python3
"""Cache one frozen encoder's exact causal context and 180-bin direction map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.cano_sensor_smoke import MAX_RANGE_M, NEAR_RANGE_M
from mtare_topo.data.gse_spatial_event_set_cache import (
    DIRECTIONAL_BINS,
    ENCODER_DIM,
    OBSERVATION_COUNT,
    load_spatial_event_teacher,
)
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet


PARENT_PATTERN = re.compile(r"^S\d+_.+_C0[1-8]$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--causal-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("spatial event feature cache overwrite is forbidden")
    output.mkdir(parents=True)

    import torch
    import zarr

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA feature cache has no CUDA device")
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    teacher = load_spatial_event_teacher(args.teacher_root.resolve())
    shard_paths = sorted(
        (args.dataset_run.resolve() / "artifacts/dataset/train").glob("*.zarr")
    )
    if len(shard_paths) != 80 or any(not PARENT_PATTERN.match(path.stem) for path in shard_paths):
        raise RuntimeError("feature cache must read exactly C01--C08")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "gse_graph_checkpoint_v1" or checkpoint.get("seed") != args.seed:
        raise RuntimeError("frozen GSE checkpoint drift")
    model = GeometrySemanticEventNet()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    global_index = np.lib.format.open_memmap(
        output / "global_sequence_index.npy", mode="w+", dtype=np.int64,
        shape=(OBSERVATION_COUNT,),
    )
    context = np.lib.format.open_memmap(
        output / "context.npy", mode="w+", dtype=np.float16,
        shape=(OBSERVATION_COUNT, ENCODER_DIM),
    )
    directional = np.lib.format.open_memmap(
        output / "directional.npy", mode="w+", dtype=np.float16,
        shape=(OBSERVATION_COUNT, ENCODER_DIM, DIRECTIONAL_BINS),
    )
    legacy_logits = np.lib.format.open_memmap(
        output / "legacy_event_logits.npy", mode="w+", dtype=np.float16,
        shape=(OBSERVATION_COUNT, 5),
    )
    write_offset = 0
    world_records = []
    with torch.inference_mode():
        for world_index, path in enumerate(shard_paths, start=1):
            group = zarr.open_group(str(path), mode="r")
            parent_id = path.stem
            references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
            sequence_global = np.asarray(group["global_sequence_index"][:], dtype=np.int64)
            unique_frames = len(group["global_frame_index"])
            observations = len(references)
            if (
                group.attrs.get("parent_id") != parent_id
                or group.attrs.get("split") != "train"
                or references.shape != (observations, 5)
                or np.any(references < 0)
                or np.any(references >= unique_frames)
                or observations == 0
            ):
                raise RuntimeError(f"source causal shard drift: {parent_id}")

            # A single world's exact encoder map is temporary host memory.  It
            # is discarded before the next world and never becomes retained evidence.
            pooled = np.empty((unique_frames, ENCODER_DIM), dtype=np.float32)
            azimuth = np.empty(
                (unique_frames, ENCODER_DIM, DIRECTIONAL_BINS), dtype=np.float32
            )
            for start in range(0, unique_frames, args.encoder_batch_size):
                stop = min(start + args.encoder_batch_size, unique_frames)
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
                    raise RuntimeError(f"source scan contract drift: {parent_id}")
                scans = np.stack((range_m / MAX_RANGE_M, valid), axis=1)
                encoded = model.encoder(
                    torch.from_numpy(scans).to(device=device, dtype=torch.float32)
                )
                if encoded.shape[1:] != (ENCODER_DIM, 2, DIRECTIONAL_BINS):
                    raise RuntimeError("frozen encoder spatial shape drift")
                pooled[start:stop] = encoded.mean(dim=(2, 3)).cpu().numpy()
                azimuth[start:stop] = encoded.mean(dim=2).cpu().numpy()

            destination_start = write_offset
            for start in range(0, observations, args.causal_batch_size):
                stop = min(start + args.causal_batch_size, observations)
                local = references[start:stop]
                pooled_sequence = torch.from_numpy(pooled[local]).to(device)
                azimuth_sequence = torch.from_numpy(azimuth[local]).to(device)
                temporal_values, _ = model.temporal(pooled_sequence)
                causal_context = temporal_values[:, -1]
                causal_directional = model.directional_temporal(
                    azimuth_sequence.reshape(
                        stop - start, 5 * ENCODER_DIM, DIRECTIONAL_BINS
                    )
                )
                destination = slice(write_offset + start, write_offset + stop)
                global_index[destination] = sequence_global[start:stop]
                context[destination] = causal_context.cpu().numpy().astype(np.float16)
                directional[destination] = causal_directional.cpu().numpy().astype(np.float16)
                legacy_logits[destination] = model.event_head(causal_context).cpu().numpy().astype(np.float16)
            write_offset += observations
            world_records.append(
                {
                    "parent_id": parent_id,
                    "world_index": world_index,
                    "unique_frames": unique_frames,
                    "observations": observations,
                    "first_output_row": destination_start,
                    "last_output_row": write_offset - 1,
                }
            )
            print(
                json.dumps(
                    {"world": world_index, "parent_id": parent_id, "observations": observations},
                    sort_keys=True,
                ),
                flush=True,
            )
            del pooled, azimuth

    for array in (global_index, context, directional, legacy_logits):
        array.flush()
    del global_index, context, directional, legacy_logits
    observed_global = np.load(output / "global_sequence_index.npy", mmap_mode="r")
    if write_offset != OBSERVATION_COUNT or not np.array_equal(
        observed_global, teacher.global_sequence_index
    ):
        raise RuntimeError("feature cache/Teacher global join drift")
    array_names = (
        "global_sequence_index.npy",
        "context.npy",
        "directional.npy",
        "legacy_event_logits.npy",
    )
    manifest = {
        "schema_version": "gse_spatial_event_set_feature_cache_v1",
        "seed": args.seed,
        "worlds": 80,
        "observations": OBSERVATION_COUNT,
        "encoder_dim": ENCODER_DIM,
        "directional_bins": DIRECTIONAL_BINS,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "array_sha256": {name: _sha256(output / name) for name in array_names},
        "cache_bytes": sum((output / name).stat().st_size for name in array_names),
        "model_inference_frames": sum(record["unique_frames"] for record in world_records),
        "causal_feature_rows": OBSERVATION_COUNT,
        "legacy_event_rows": OBSERVATION_COUNT,
        "optimizer_steps": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "world_records": world_records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "schema_version", "seed", "observations", "cache_bytes", "duration_seconds"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
