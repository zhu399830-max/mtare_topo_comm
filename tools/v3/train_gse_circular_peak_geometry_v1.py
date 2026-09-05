#!/usr/bin/env python3
"""Train one circular peak-geometry seed on C01-C06 and select on C07."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import zarr

from mtare_topo.representation.gse_circular_peak_geometry_model import (
    CircularPeakGeometrySemanticNet,
    circular_peak_geometry_loss,
)


MAX_RANGE_M = 50.0
EXPECTED_PARAMETERS = 769268


def _clip_finite_gradients(model: torch.nn.Module, max_norm: float = 5.0) -> torch.Tensor:
    """Clip gradients and fail before an optimizer step if any are non-finite."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, error_if_nonfinite=True)


def _shift(seed: int, epoch: int, parent: str, batch_index: int) -> int:
    payload = f"gse-circular-peak-roll-v1:{seed}:{epoch}:{parent}:{batch_index}".encode()
    return 4 * (int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % 180)


def _load_world(teacher_path: Path, source_root: Path) -> dict[str, np.ndarray | str]:
    teacher = zarr.open_group(str(teacher_path), mode="r")
    parent = str(teacher.attrs["parent_id"])
    source = zarr.open_group(str(source_root / f"{parent}.zarr"), mode="r")
    global_sequence = np.asarray(teacher["global_sequence_index"][:], dtype=np.int64)
    if not np.array_equal(global_sequence, np.asarray(source["global_sequence_index"][:], dtype=np.int64)):
        raise RuntimeError(f"circular training Teacher/source join drift: {parent}")
    return {
        "parent": parent,
        "global_sequence_index": global_sequence,
        "references": np.asarray(teacher["local_frame_references"][:], dtype=np.int64),
        "range_m": np.asarray(source["range_m"][:], dtype=np.float32),
        "valid_mask": np.asarray(source["valid_mask"][:], dtype=np.uint8),
        "presence": np.asarray(teacher["presence"][:], dtype=np.uint8),
        "heading_residual_deg": np.asarray(teacher["heading_residual_deg"][:], dtype=np.float32),
        "opening_width_m": np.asarray(teacher["opening_width_m"][:], dtype=np.float32),
        "width_valid_mask": np.asarray(teacher["width_valid_mask"][:], dtype=np.uint8),
        "vertical_profile_m": np.asarray(teacher["vertical_profile_m"][:], dtype=np.float32),
        "local_axis": np.asarray(source["local_axis_robot"][:], dtype=np.float32),
        "geometry": np.asarray(source["geometry"][:], dtype=np.float32),
        "geometry_valid_mask": np.asarray(source["geometry_valid_mask"][:], dtype=np.uint8),
    }


def _batch(world: dict, indices: np.ndarray, *, device: torch.device, shift_columns: int = 0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    references = world["references"][indices]
    ranges = world["range_m"][references] / MAX_RANGE_M
    valid = world["valid_mask"][references].astype(np.float32, copy=False)
    scans = torch.from_numpy(np.stack((ranges, valid), axis=2)).to(device=device, dtype=torch.float32)
    targets = {
        name: torch.from_numpy(np.asarray(world[name][indices])).to(device=device)
        for name in (
            "presence", "heading_residual_deg", "opening_width_m", "width_valid_mask",
            "vertical_profile_m", "local_axis", "geometry", "geometry_valid_mask",
        )
    }
    if shift_columns:
        shift_bins = shift_columns // 4
        scans = torch.roll(scans, shift_columns, dims=-1)
        for name in ("presence", "heading_residual_deg", "opening_width_m", "width_valid_mask", "vertical_profile_m"):
            targets[name] = torch.roll(targets[name], shift_bins, dims=1)
        angle = 2.0 * math.pi * shift_bins / 180.0
        axis = targets["local_axis"]
        targets["local_axis"] = torch.stack(
            (
                math.cos(angle) * axis[:, 0] - math.sin(angle) * axis[:, 1],
                math.sin(angle) * axis[:, 0] + math.cos(angle) * axis[:, 1],
                axis[:, 2],
            ),
            dim=-1,
        )
    return scans, targets


def _validation_loss(model, paths: list[Path], source_root: Path, *, device: torch.device, batch_size: int) -> dict[str, float]:
    totals = defaultdict(float)
    rows = 0
    model.eval()
    with torch.inference_mode():
        for path in paths:
            world = _load_world(path, source_root)
            for start in range(0, len(world["presence"]), batch_size):
                indices = np.arange(start, min(start + batch_size, len(world["presence"])))
                scans, targets = _batch(world, indices, device=device)
                losses = circular_peak_geometry_loss(model(scans), targets)
                for name, value in losses.items():
                    totals[name] += float(value) * len(indices)
                rows += len(indices)
    return {name: value / rows for name, value in totals.items()}


def _infer_world(model, teacher_path: Path, source_root: Path, output: Path, *, device: torch.device, batch_size: int) -> int:
    world = _load_world(teacher_path, source_root)
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(world["presence"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(world["presence"])))
            scans, _ = _batch(world, indices, device=device)
            predicted = model(scans)
            values["confidence"].append(predicted["peak_confidence"].cpu().numpy().astype(np.float32))
            values["heading_residual_deg"].append(predicted["peak_heading_residual_deg"].cpu().numpy().astype(np.float16))
            values["opening_width_m"].append(predicted["peak_opening_width_m"].cpu().numpy().astype(np.float16))
            values["vertical_profile_m"].append(predicted["peak_vertical_profile_m"].cpu().numpy().astype(np.float16))
            values["local_axis"].append(predicted["local_axis"].cpu().numpy().astype(np.float32))
            values["geometry"].append(torch.stack((predicted["width_m"], predicted["height_m"], predicted["slope_deg"], predicted["curvature_per_m"]), dim=-1).cpu().numpy().astype(np.float32))
            values["observation_uncertainty"].append(predicted["observation_uncertainty"].cpu().numpy().astype(np.float32))
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}
    arrays["global_sequence_index"] = world["global_sequence_index"]
    np.savez_compressed(output / f"{world['parent']}.npz", **arrays)
    return len(world["presence"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise RuntimeError("circular peak seed output exists; overwrite is forbidden")
    args.output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("formal circular peak training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    fit_paths = sorted(args.teacher_root.glob("fit/*.zarr"))
    c07_paths = sorted(args.teacher_root.glob("selection/*_C07.zarr"))
    c08_paths = sorted(args.teacher_root.glob("selection/*_C08.zarr"))
    if (len(fit_paths), len(c07_paths), len(c08_paths)) != (60, 10, 10):
        raise RuntimeError("circular peak training split drift")
    model = CircularPeakGeometrySemanticNet().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("circular peak parameter drift")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    best_score = math.inf
    best_epoch = -1
    total_steps = 0
    for epoch in range(args.epochs):
        generator = np.random.default_rng(args.seed * 1000 + epoch)
        ordered_paths = [fit_paths[index] for index in generator.permutation(len(fit_paths))]
        model.train()
        totals = defaultdict(float)
        rows = 0
        steps = 0
        for teacher_path in ordered_paths:
            world = _load_world(teacher_path, args.source_root)
            order = generator.permutation(len(world["presence"]))
            for batch_index, start in enumerate(range(0, len(order), args.batch_size)):
                indices = order[start : start + args.batch_size]
                shift_columns = _shift(args.seed, epoch, str(world["parent"]), batch_index)
                if os.environ.get("GSE_DIAGNOSTIC_BATCH_TRACE") == "1":
                    presence = np.asarray(world["presence"][indices], dtype=np.uint8)
                    active = np.argwhere(presence > 0)
                    residual = np.asarray(world["heading_residual_deg"][indices], dtype=np.float32)
                    active_residual = residual[presence > 0]
                    print(json.dumps({
                        "diagnostic_batch": True, "seed": args.seed, "epoch": epoch,
                        "parent": str(world["parent"]), "batch_index": batch_index,
                        "batch_rows": int(len(indices)), "shift_columns": int(shift_columns),
                        "global_sequence_index_min": int(np.asarray(world["global_sequence_index"])[indices].min()),
                        "global_sequence_index_max": int(np.asarray(world["global_sequence_index"])[indices].max()),
                        "target_count_min": int(presence.sum(1).min()), "target_count_max": int(presence.sum(1).max()),
                        "active_bin_min": int(active[:, 1].min()), "active_bin_max": int(active[:, 1].max()),
                        "active_residual_min_deg": float(active_residual.min()), "active_residual_max_deg": float(active_residual.max()),
                    }, sort_keys=True), flush=True)
                scans, targets = _batch(world, indices, device=device, shift_columns=shift_columns)
                losses = circular_peak_geometry_loss(model(scans), targets)
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                _clip_finite_gradients(model)
                optimizer.step()
                for name, value in losses.items():
                    totals[name] += float(value.detach()) * len(indices)
                rows += len(indices)
                steps += 1
        c07 = _validation_loss(model, c07_paths, args.source_root, device=device, batch_size=args.evaluation_batch_size)
        record = {"epoch": epoch, "optimizer_steps": steps, "train": {name: value / rows for name, value in totals.items()}, "c07": c07}
        history.append(record)
        total_steps += steps
        print(json.dumps(record, sort_keys=True), flush=True)
        if c07["total"] < best_score:
            best_score = c07["total"]
            best_epoch = epoch
            torch.save({
                "schema_version": "gse_circular_peak_geometry_checkpoint_v1",
                "seed": args.seed, "epoch": epoch, "model": model.state_dict(),
                "c07_validation_loss": best_score, "parameters": EXPECTED_PARAMETERS,
                "fit_worlds": [path.stem for path in fit_paths],
                "c07_selection_worlds": [path.stem for path in c07_paths],
                "c08_checkpoint_observations": 0,
            }, args.output_dir / "best.pt")
    if os.environ.get("GSE_DIAGNOSTIC_STOP_AFTER_TRAIN") == "1":
        summary = {
            "schema_version": "gse_circular_peak_geometry_training_diagnostic_v1",
            "seed": args.seed, "parameters": EXPECTED_PARAMETERS, "epochs": args.epochs,
            "best_epoch": best_epoch, "best_c07_validation_loss": best_score,
            "optimizer_steps": total_steps, "fit_observations": 142184,
            "c07_checkpoint_observations": 21548, "c08_checkpoint_observations": 0,
            "development_output_observations": 0, "duration_seconds": time.monotonic() - started,
            "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        }
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    prediction_root = args.output_dir / "development_predictions"
    prediction_root.mkdir()
    development_rows = 0
    for path in (*c07_paths, *c08_paths):
        development_rows += _infer_world(model, path, args.source_root, prediction_root, device=device, batch_size=args.evaluation_batch_size)
    summary = {
        "schema_version": "gse_circular_peak_geometry_training_seed_v1",
        "seed": args.seed, "parameters": EXPECTED_PARAMETERS, "epochs": args.epochs,
        "best_epoch": best_epoch, "best_c07_validation_loss": best_score,
        "optimizer_steps": total_steps, "fit_observations": 142184,
        "c07_checkpoint_observations": 21548, "c08_checkpoint_observations": 0,
        "development_output_observations": development_rows,
        "final_inference_c07_observations": 21548, "final_inference_c08_observations": 24394,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
