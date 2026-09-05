#!/usr/bin/env python3
"""Export one seed's frozen geometry-only state for Composer training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import zarr

import train_gse_sparse_circular_relation_transport_v2 as training
from mtare_topo.data.gse_explicit_composer_cache import (
    EXPLICIT_CACHE_FIELDS,
    validate_explicit_composer_world_cache,
)
from mtare_topo.representation.gse_sparse_circular_relation_transport import (
    SparseCircularRelationTransportNet,
    parameter_count,
)


EXPECTED_PARAMETERS = 784513
EXPECTED_SPLIT_ROWS = {"fit": 142184, "c07": 21548, "c08": 24394}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split(parent: str) -> str:
    index = int(parent.rsplit("_C", 1)[1])
    return "fit" if index <= 6 else "c07" if index == 7 else "c08"


def _load_scan_world(dataset_root: Path, parent: str, traversal: list[str]) -> dict[str, object]:
    """Read only deployed LiDAR and row identities; never materialize Teacher targets."""

    group = zarr.open_group(str(dataset_root / f"{parent}.zarr"), mode="r")
    global_index = np.asarray(group["global_sequence_index"][:], dtype=np.int64)
    if len(traversal) != len(global_index):
        raise RuntimeError(f"manifest/world observation drift: {parent}")
    return {
        "parent": parent,
        "references": np.asarray(group["local_frame_references"][:], dtype=np.int64),
        "range_m": np.asarray(group["range_m"][:], dtype=np.float32),
        "valid_mask": np.asarray(group["valid_mask"][:], dtype=np.uint8),
        "global_sequence_index": global_index,
    }


@torch.inference_mode()
def _infer_world(
    model: SparseCircularRelationTransportNet,
    world: dict[str, object],
    *, device: torch.device, batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    rows = len(world["global_sequence_index"])
    for start in range(0, rows, batch_size):
        indices = np.arange(start, min(start + batch_size, rows))
        references = world["references"][indices]
        ranges = world["range_m"][references] / training.base.MAX_RANGE_M
        valid = world["valid_mask"][references].astype(np.float32, copy=False)
        scans = torch.from_numpy(np.stack((ranges, valid), axis=2)).to(
            device=device, dtype=torch.float32,
        )
        output = model(scans)
        for name in (
            "token_bearing_deg", "token_existence_logits", "token_opening_width_m",
            "token_vertical_profile_m", "token_geometry_uncertainty",
            "token_count_probability", "transport_row_probability",
            "transport_reveal_probability", "observation_uncertainty",
        ):
            values[name].append(output[name].cpu().numpy().astype(np.float16))
        geometry = torch.stack((
            output["width_m"], output["height_m"], output["slope_deg"],
            output["curvature_per_m"],
        ), dim=-1)
        values["geometry"].append(geometry.cpu().numpy().astype(np.float32))
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}
    arrays["global_sequence_index"] = np.asarray(world["global_sequence_index"], dtype=np.int64)
    return {name: arrays[name] for name in EXPLICIT_CACHE_FIELDS}


def _development_parity(
    arrays: dict[str, np.ndarray], sealed_path: Path,
) -> dict[str, object]:
    if not sealed_path.is_file():
        raise RuntimeError(f"sealed development prediction missing: {sealed_path}")
    errors = {}
    with np.load(sealed_path, allow_pickle=False) as sealed:
        for name in EXPLICIT_CACHE_FIELDS:
            expected = np.asarray(sealed[name])
            actual = arrays[name]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError(f"sealed development shape/dtype drift: {sealed_path.name}/{name}")
            if np.issubdtype(actual.dtype, np.floating):
                error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            else:
                error = 0.0 if np.array_equal(actual, expected) else float("inf")
            errors[name] = error
            if error != 0.0:
                raise RuntimeError(f"frozen re-inference parity failed: {sealed_path.name}/{name}={error}")
    return {"all_fields_exact": True, "maximum_absolute_error": max(errors.values()), "by_field": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--development-prediction-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size != 256:
        raise ValueError("explicit cache export batch size is frozen at 256 for parity")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("explicit Composer cache seed directory already exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    training.base.seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal explicit state cache export requires CUDA")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if int(checkpoint["seed"]) != args.seed:
        raise RuntimeError("explicit cache checkpoint seed mismatch")
    model = SparseCircularRelationTransportNet()
    if parameter_count() != EXPECTED_PARAMETERS:
        raise RuntimeError("V2R5 parameter count drift")
    model.load_state_dict(checkpoint["model"], strict=True)
    if not all(torch.equal(model.state_dict()[name], value.cpu()) for name, value in checkpoint["model"].items()):
        raise RuntimeError("frozen V2R5 checkpoint did not load exactly")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device).eval()

    traversals = training.base.manifest_traversals(args.sequence_manifest.resolve())
    parents = sorted(traversals)
    if len(parents) != 80 or sum(_split(parent) == "fit" for parent in parents) != 60 or sum(
        _split(parent) == "c07" for parent in parents
    ) != 10 or sum(_split(parent) == "c08" for parent in parents) != 10:
        raise RuntimeError("explicit cache world split drift")
    split_rows = defaultdict(int)
    records = []
    maximum_parity_error = 0.0
    # Fail before the expensive 60-world fit export if frozen inference no
    # longer reproduces the already sealed development artifact exactly.
    preflight_parent = next(parent for parent in parents if _split(parent) == "c07")
    preflight_world = _load_scan_world(
        args.dataset_root.resolve(), preflight_parent, traversals[preflight_parent],
    )
    preflight_arrays = _infer_world(
        model, preflight_world, device=device, batch_size=args.batch_size,
    )
    validate_explicit_composer_world_cache(
        preflight_arrays, expected_rows=len(preflight_world["global_sequence_index"]),
    )
    _development_parity(
        preflight_arrays,
        args.development_prediction_root.resolve() / f"{preflight_parent}.npz",
    )
    del preflight_world
    training._release_unused_cuda_cache(device)
    for world_index, parent in enumerate(parents):
        training._release_unused_cuda_cache(device)
        world = _load_scan_world(args.dataset_root.resolve(), parent, traversals[parent])
        if parent == preflight_parent:
            arrays = preflight_arrays
            del preflight_arrays
        else:
            arrays = _infer_world(model, world, device=device, batch_size=args.batch_size)
        validation = validate_explicit_composer_world_cache(
            arrays, expected_rows=len(world["global_sequence_index"]),
        )
        split = _split(parent)
        parity = None
        if split in ("c07", "c08"):
            parity = _development_parity(
                arrays, args.development_prediction_root.resolve() / f"{parent}.npz",
            )
            maximum_parity_error = max(maximum_parity_error, float(parity["maximum_absolute_error"]))
        path = output / f"{parent}.npz"
        np.savez_compressed(path, **arrays)
        split_rows[split] += len(arrays["global_sequence_index"])
        records.append({
            "parent": parent, "split": split, **validation,
            "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path),
            "sealed_development_parity": parity,
        })
        del arrays, world
        training._release_unused_cuda_cache(device)
        print(json.dumps({
            "seed": args.seed, "world": parent, "progress": f"{world_index + 1}/80",
            "split_rows": dict(split_rows),
        }, sort_keys=True), flush=True)
    if dict(split_rows) != EXPECTED_SPLIT_ROWS:
        raise RuntimeError(f"explicit cache split row drift: {dict(split_rows)}")
    manifest = {
        "schema_version": "gse_explicit_composer_cache_v1",
        "seed": args.seed, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "parameters": EXPECTED_PARAMETERS, "fields": EXPLICIT_CACHE_FIELDS,
        "forbidden_fields": [
            "event_logits", "event_probability", "context", "hidden", "place_descriptor",
            "token_descriptor", "exit_descriptor", "pose", "world", "tng", "identity",
        ],
        "worlds": 80, "split_worlds": {"fit": 60, "c07": 10, "c08": 10},
        "split_rows": dict(split_rows), "records": records,
        "development_parity_worlds": 20,
        "maximum_development_parity_error": maximum_parity_error,
        "optimizer_steps": 0, "checkpoints_created": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "graph_replays": 0, "planner_calls": 0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "duration_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "seed": args.seed, "status": "PASS_GSE_EXPLICIT_COMPOSER_CACHE_EXPORT_SEED_V1",
        "split_rows": dict(split_rows), "maximum_development_parity_error": maximum_parity_error,
        "duration_seconds": manifest["duration_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
