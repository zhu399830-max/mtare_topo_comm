#!/usr/bin/env python3
"""Jointly train one geometry-anchored spatial event encoder seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

# Must be set before the representation module imports torch; otherwise
# deterministic CUDA attention fails before the first optimizer step.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.gse_observable_spatial_event_dataset import (
    load_observable_spatial_event_teacher,
    observable_fit_only_loss_weights,
)
from mtare_topo.evaluation.gse_spatial_event_set_metrics import select_fixed_grid_threshold
from mtare_topo.representation.gse_geometry_anchored_event import (
    GeometryAnchoredSpatialEventEncoder,
)
from mtare_topo.representation.gse_spatial_event_set import (
    SpatialEventSetLossWeights,
    spatial_event_set_loss,
)


THRESHOLD_GRID = tuple(round(value * 0.05, 2) for value in range(1, 20))
LOSS_WEIGHTS = SpatialEventSetLossWeights(
    presence=1.0,
    event_type=1.0,
    position=5.0,
    descriptor=0.1,
    uncertainty=0.01,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _roll_scans_and_targets(range_m, valid_mask, targets, shifts):
    import torch

    columns = range_m.shape[-1]
    column_shifts = shifts * 4
    source = (torch.arange(columns, device=range_m.device)[None] - column_shifts[:, None]) % columns
    gather = source[:, None, None].expand(-1, range_m.shape[1], range_m.shape[2], -1)
    rolled_range = torch.gather(range_m, 3, gather)
    rolled_valid = torch.gather(valid_mask, 3, gather)
    angle = shifts.to(dtype=range_m.dtype) * (2.0 * torch.pi / 180.0)
    cosine = torch.cos(angle)[:, None]
    sine = torch.sin(angle)[:, None]
    xyz = targets["event_relative_xyz_m"]
    rotated = xyz.clone()
    rotated[..., 0] = cosine * xyz[..., 0] - sine * xyz[..., 1]
    rotated[..., 1] = sine * xyz[..., 0] + cosine * xyz[..., 1]
    output_targets = dict(targets)
    output_targets["event_relative_xyz_m"] = rotated
    return rolled_range, rolled_valid, output_targets


def _tensor_targets(teacher, global_rows, device):
    import torch

    return {
        name: torch.from_numpy(values).to(device)
        for name, values in teacher.targets(global_rows).items()
    }


def _world_arrays(dataset_root: Path, parent: str):
    import zarr

    group = zarr.open_group(str(dataset_root / "train" / f"{parent}.zarr"), mode="r")
    return (
        np.asarray(group["global_sequence_index"][:], dtype=np.int64),
        np.asarray(group["local_frame_references"][:], dtype=np.int64),
        np.asarray(group["range_m"][:], dtype=np.float32),
        np.asarray(group["valid_mask"][:], dtype=np.uint8),
    )


def _forward_batch(
    model, teacher, global_rows, local_rows, references, ranges, valid, device, *, run_model=True
):
    import torch

    refs = references[local_rows]
    range_batch = torch.from_numpy(np.asarray(ranges[refs], dtype=np.float32)).to(device)
    valid_batch = torch.from_numpy(np.asarray(valid[refs], dtype=np.uint8)).to(device)
    targets = _tensor_targets(teacher, global_rows, device)
    outputs = model(range_batch, valid_batch) if run_model else None
    return outputs, targets, range_batch, valid_batch


def _evaluate_loss(model, teacher, dataset_root, parents, row_lookup, device, batch_size, loss_weights):
    import torch

    totals = {name: 0.0 for name in ("total", "presence", "event_type", "position", "descriptor", "uncertainty")}
    observed = 0
    model.eval()
    with torch.inference_mode():
        for parent in parents:
            gid, references, ranges, valid = _world_arrays(dataset_root, parent)
            global_rows = row_lookup[parent]
            if not np.array_equal(teacher.global_sequence_index[global_rows], gid):
                raise RuntimeError(f"selection Teacher/source join drift: {parent}")
            for start in range(0, len(gid), batch_size):
                local = np.arange(start, min(start + batch_size, len(gid)), dtype=np.int64)
                selected = global_rows[local]
                outputs, targets, _, _ = _forward_batch(model, teacher, selected, local, references, ranges, valid, device)
                losses = spatial_event_set_loss(
                    outputs,
                    targets,
                    weights=LOSS_WEIGHTS,
                    presence_positive_weight=loss_weights["presence_positive_weight"],
                    event_type_class_weights=torch.tensor(loss_weights["event_type_class_weights"], device=device),
                )
                for name in totals:
                    totals[name] += float(losses[name].cpu()) * len(local)
                observed += len(local)
    return {name: value / observed for name, value in totals.items()}


def _infer(model, teacher, dataset_root, parents, row_lookup, device, batch_size):
    import torch

    selection_rows = np.concatenate([row_lookup[parent] for parent in parents])
    selection_rows.sort()
    position = np.full(len(teacher.global_sequence_index), -1, dtype=np.int64)
    position[selection_rows] = np.arange(len(selection_rows), dtype=np.int64)
    confidence = np.empty((len(selection_rows), 16), dtype=np.float32)
    event_type = np.empty((len(selection_rows), 16), dtype=np.int8)
    relative = np.empty((len(selection_rows), 16, 3), dtype=np.float32)
    uncertainty = np.empty((len(selection_rows), 16), dtype=np.float32)
    descriptor = np.empty((len(selection_rows), 16, 64), dtype=np.float16)
    model.eval()
    with torch.inference_mode():
        for parent in parents:
            gid, references, ranges, valid = _world_arrays(dataset_root, parent)
            global_rows = row_lookup[parent]
            if not np.array_equal(teacher.global_sequence_index[global_rows], gid):
                raise RuntimeError(f"inference Teacher/source join drift: {parent}")
            for start in range(0, len(gid), batch_size):
                local = np.arange(start, min(start + batch_size, len(gid)), dtype=np.int64)
                selected = global_rows[local]
                outputs, _, _, _ = _forward_batch(model, teacher, selected, local, references, ranges, valid, device)
                destination = position[selected]
                confidence[destination] = outputs["event_confidence"].cpu().numpy().astype(np.float32)
                event_type[destination] = outputs["event_type_logits"].argmax(-1).cpu().numpy().astype(np.int8)
                relative[destination] = outputs["event_relative_xyz_m"].cpu().numpy().astype(np.float32)
                uncertainty[destination] = outputs["event_uncertainty_m"].cpu().numpy().astype(np.float32)
                descriptor[destination] = outputs["event_descriptor"].cpu().numpy().astype(np.float16)
    return {
        "confidence": confidence,
        "event_type": event_type,
        "relative_xyz_m": relative,
        "uncertainty_m": uncertainty,
        "descriptor": descriptor,
    }, selection_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--maximum-worlds", type=int, default=0, help="nonformal smoke only")
    parser.add_argument("--maximum-batches", type=int, default=0, help="nonformal smoke only")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal geometry-anchored joint training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    dataset_root = args.dataset_root.resolve()
    teacher = load_observable_spatial_event_teacher(args.teacher_root.resolve())
    loss_weight_values = observable_fit_only_loss_weights(teacher)
    parents = sorted(set(str(value) for value in teacher.parent_id))
    row_lookup = {parent: np.flatnonzero(teacher.parent_id == parent) for parent in parents}
    fit_parents = [parent for parent in parents if parent.endswith(tuple(f"C0{x}" for x in range(1, 7)))]
    selection_parents = [parent for parent in parents if parent.endswith(("C07", "C08"))]
    if len(fit_parents) != 60 or len(selection_parents) != 20:
        raise RuntimeError("geometry-anchored parent split drift")
    if args.maximum_worlds:
        fit_parents = fit_parents[: args.maximum_worlds]
        selection_parents = selection_parents[: min(args.maximum_worlds, len(selection_parents))]

    model = GeometryAnchoredSpatialEventEncoder().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 264134:
        raise RuntimeError("geometry-anchored parameter drift")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    best_selection_loss = float("inf")
    best_epoch = -1
    optimizer_steps = 0
    common = {
        "schema_version": "gse_geometry_anchored_spatial_event_checkpoint_v1",
        "seed": args.seed,
        "teacher_contract": "observable_fixed16_c01_c08_v2",
        "student_inputs": ["five_frame_range_m", "five_frame_valid_mask"],
        "trainable_parameters": parameter_count,
        "dataset_evidence_sha256": _sha256(PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0/artifacts/evidence_sha256.txt"),
        "teacher_evidence_sha256": _sha256(PROJECT_ROOT / "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0/artifacts/evidence_sha256.txt"),
    }
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed * 10000 + epoch)
        epoch_parents = [fit_parents[index] for index in rng.permutation(len(fit_parents))]
        totals = {name: 0.0 for name in ("total", "presence", "event_type", "position", "descriptor", "uncertainty")}
        observed = 0
        batch_counter = 0
        model.train()
        for parent in epoch_parents:
            gid, references, ranges, valid = _world_arrays(dataset_root, parent)
            global_rows = row_lookup[parent]
            if not np.array_equal(teacher.global_sequence_index[global_rows], gid):
                raise RuntimeError(f"fit Teacher/source join drift: {parent}")
            order = rng.permutation(len(gid))
            for start in range(0, len(order), args.batch_size):
                local = np.asarray(order[start : start + args.batch_size], dtype=np.int64)
                selected = global_rows[local]
                _, targets, range_batch, valid_batch = _forward_batch(
                    model,
                    teacher,
                    selected,
                    local,
                    references,
                    ranges,
                    valid,
                    device,
                    run_model=False,
                )
                shifts = torch.from_numpy(rng.integers(0, 180, size=len(local), dtype=np.int64)).to(device)
                range_batch, valid_batch, targets = _roll_scans_and_targets(range_batch, valid_batch, targets, shifts)
                outputs = model(range_batch, valid_batch)
                losses = spatial_event_set_loss(
                    outputs,
                    targets,
                    weights=LOSS_WEIGHTS,
                    presence_positive_weight=loss_weight_values["presence_positive_weight"],
                    event_type_class_weights=torch.tensor(loss_weight_values["event_type_class_weights"], device=device),
                )
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
                batch_counter += 1
                for name in totals:
                    totals[name] += float(losses[name].detach().cpu()) * len(local)
                observed += len(local)
                if args.maximum_batches and batch_counter >= args.maximum_batches:
                    break
            if args.maximum_batches and batch_counter >= args.maximum_batches:
                break
        selection_loss = _evaluate_loss(model, teacher, dataset_root, selection_parents, row_lookup, device, args.evaluation_batch_size, loss_weight_values)
        record = {"epoch": epoch, "optimizer_steps": optimizer_steps, "fit_observations": observed, "fit_loss": {name: value / observed for name, value in totals.items()}, "selection_loss": selection_loss}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if selection_loss["total"] < best_selection_loss:
            best_selection_loss = selection_loss["total"]
            best_epoch = epoch
            torch.save({**common, "epoch": epoch, "model": model.state_dict(), "selection_loss": selection_loss}, output / "best.pt")

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    predictions, selection_rows = _infer(
        model,
        teacher,
        dataset_root,
        selection_parents,
        row_lookup,
        device,
        args.evaluation_batch_size,
    )
    targets = teacher.targets(selection_rows)
    best_metrics, threshold_records = select_fixed_grid_threshold(predictions, targets, THRESHOLD_GRID, maximum_error_m=4.0)
    best_metrics["macro_f1"] = float(
        (
            best_metrics["per_type"]["terminal"]["f1"]
            + best_metrics["per_type"]["junction"]["f1"]
        )
        / 2.0
    )
    np.savez_compressed(
        output / "selection_outputs.npz",
        observation_row=selection_rows,
        global_sequence_index=teacher.global_sequence_index[selection_rows],
        confidence=predictions["confidence"],
        event_type=predictions["event_type"],
        relative_xyz_m=predictions["relative_xyz_m"],
        uncertainty_m=predictions["uncertainty_m"],
        descriptor=predictions["descriptor"],
    )
    summary = {
        "schema_version": "gse_geometry_anchored_spatial_event_seed_training_v1",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_selection_loss": best_selection_loss,
        "selected_threshold_metrics": best_metrics,
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_records": threshold_records,
        "fit_rows": 142184 if not args.maximum_worlds else observed,
        "selection_rows": 45942 if not args.maximum_worlds else len(selection_rows),
        "optimizer_steps": optimizer_steps,
        "trainable_parameters": parameter_count,
        "loss_weights": {"presence": LOSS_WEIGHTS.presence, "event_type": LOSS_WEIGHTS.event_type, "position": LOSS_WEIGHTS.position, "descriptor": LOSS_WEIGHTS.descriptor, "uncertainty": LOSS_WEIGHTS.uncertainty},
        "fit_only_frequency_weights": loss_weight_values,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "smoke_limited": bool(args.maximum_worlds or args.maximum_batches),
    }
    (output / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
