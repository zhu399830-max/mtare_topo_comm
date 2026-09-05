#!/usr/bin/env python3
"""Train one structured-polar multi-depth event seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.gse_observable_spatial_event_dataset import (
    load_observable_spatial_event_teacher,
    observable_fit_only_loss_weights,
)
from mtare_topo.evaluation.gse_spatial_event_set_metrics import select_fixed_grid_threshold
from mtare_topo.representation.gse_structured_polar_multidepth_event import (
    StructuredPolarMultiDepthEventEncoder,
)
from mtare_topo.representation.gse_structured_polar_multidepth_loss import (
    rasterize_structured_polar_targets_vectorized,
    structured_polar_direct_loss,
)


THRESHOLD_GRID = tuple(round(value * 0.05, 2) for value in range(1, 20))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _world_arrays(dataset_root: Path, parent: str):
    import zarr
    group = zarr.open_group(str(dataset_root / "train" / f"{parent}.zarr"), mode="r")
    return (
        np.asarray(group["global_sequence_index"][:], dtype=np.int64),
        np.asarray(group["local_frame_references"][:], dtype=np.int64),
        np.asarray(group["range_m"][:], dtype=np.float32),
        np.asarray(group["valid_mask"][:], dtype=np.uint8),
    )


def _tensor_targets(teacher, rows, device):
    import torch
    return {name: torch.from_numpy(value).to(device) for name, value in teacher.targets(rows).items()}


def _batch(teacher, selected, local, references, ranges, valid, device):
    import torch
    refs = references[local]
    return (
        torch.from_numpy(np.asarray(ranges[refs], dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(valid[refs], dtype=np.uint8)).to(device),
        _tensor_targets(teacher, selected, device),
    )


def _roll(range_batch, valid_batch, targets, shifts):
    import torch
    columns = range_batch.shape[-1]
    column_shifts = shifts * 4
    source = (torch.arange(columns, device=range_batch.device)[None] - column_shifts[:, None]) % columns
    gather = source[:, None, None].expand(-1, 5, 16, -1)
    ranges = torch.gather(range_batch, 3, gather)
    valid = torch.gather(valid_batch, 3, gather)
    angle = shifts.to(range_batch.dtype) * (2.0 * torch.pi / 180.0)
    cosine, sine = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
    xyz = targets["event_relative_xyz_m"]
    rotated = xyz.clone()
    rotated[..., 0] = cosine * xyz[..., 0] - sine * xyz[..., 1]
    rotated[..., 1] = sine * xyz[..., 0] + cosine * xyz[..., 1]
    return ranges, valid, {**targets, "event_relative_xyz_m": rotated}


def _loss(model, range_batch, valid_batch, targets, class_weights):
    outputs = model(range_batch, valid_batch)
    dense = rasterize_structured_polar_targets_vectorized(targets, outputs["free_range_profile_m"].detach())
    return outputs, structured_polar_direct_loss(outputs, dense, event_type_class_weights=class_weights)


def _evaluate_loss(model, teacher, dataset_root, parents, row_lookup, device, batch_size, class_weights):
    import torch
    totals = {name: 0.0 for name in ("total", "presence", "event_type", "position", "descriptor", "uncertainty")}
    observed = 0
    model.eval()
    with torch.inference_mode():
        for parent in parents:
            gid, references, ranges, valid = _world_arrays(dataset_root, parent)
            global_rows = row_lookup[parent]
            if not np.array_equal(teacher.global_sequence_index[global_rows], gid):
                raise RuntimeError(f"selection join drift: {parent}")
            for start in range(0, len(gid), batch_size):
                local = np.arange(start, min(start + batch_size, len(gid)), dtype=np.int64)
                rb, vb, targets = _batch(teacher, global_rows[local], local, references, ranges, valid, device)
                _, losses = _loss(model, rb, vb, targets, class_weights)
                for name in totals:
                    totals[name] += float(losses[name].cpu()) * len(local)
                observed += len(local)
    return {name: value / observed for name, value in totals.items()}


def _infer(model, teacher, dataset_root, parents, row_lookup, device, batch_size):
    import torch
    rows = np.concatenate([row_lookup[parent] for parent in parents]); rows.sort()
    destination = np.full(len(teacher.global_sequence_index), -1, dtype=np.int64); destination[rows] = np.arange(len(rows))
    confidence = np.empty((len(rows), 16), dtype=np.float32)
    event_type = np.empty((len(rows), 16), dtype=np.int8)
    relative = np.empty((len(rows), 16, 3), dtype=np.float32)
    uncertainty = np.empty((len(rows), 16), dtype=np.float32)
    descriptor = np.empty((len(rows), 16, 64), dtype=np.float16)
    azimuth_bin = np.empty((len(rows), 16), dtype=np.int16)
    depth_slot = np.empty((len(rows), 16), dtype=np.int8)
    model.eval()
    with torch.inference_mode():
        for parent in parents:
            gid, references, ranges, valid = _world_arrays(dataset_root, parent)
            global_rows = row_lookup[parent]
            if not np.array_equal(teacher.global_sequence_index[global_rows], gid):
                raise RuntimeError(f"inference join drift: {parent}")
            for start in range(0, len(gid), batch_size):
                local = np.arange(start, min(start + batch_size, len(gid)), dtype=np.int64)
                selected = global_rows[local]
                rb, vb, _ = _batch(teacher, selected, local, references, ranges, valid, device)
                outputs = model(rb, vb); dst = destination[selected]
                confidence[dst] = outputs["event_confidence"].cpu().numpy().astype(np.float32)
                event_type[dst] = outputs["event_type_logits"].argmax(-1).cpu().numpy().astype(np.int8)
                relative[dst] = outputs["event_relative_xyz_m"].cpu().numpy().astype(np.float32)
                uncertainty[dst] = outputs["event_uncertainty_m"].cpu().numpy().astype(np.float32)
                descriptor[dst] = outputs["event_descriptor"].cpu().numpy().astype(np.float16)
                azimuth_bin[dst] = outputs["event_azimuth_bin_index"].cpu().numpy().astype(np.int16)
                depth_slot[dst] = outputs["event_depth_slot_index"].cpu().numpy().astype(np.int8)
    return {"confidence": confidence, "event_type": event_type, "relative_xyz_m": relative, "uncertainty_m": uncertainty, "descriptor": descriptor, "azimuth_bin": azimuth_bin, "depth_slot": depth_slot}, rows


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
    parser.add_argument("--maximum-worlds", type=int, default=0)
    parser.add_argument("--maximum-batches", type=int, default=0)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("structured-polar training requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    teacher = load_observable_spatial_event_teacher(args.teacher_root.resolve())
    frequency = observable_fit_only_loss_weights(teacher)
    class_weights = torch.tensor(frequency["event_type_class_weights"], device=device)
    parents = sorted(set(str(value) for value in teacher.parent_id))
    row_lookup = {parent: np.flatnonzero(teacher.parent_id == parent) for parent in parents}
    fit = [parent for parent in parents if parent.endswith(tuple(f"C0{x}" for x in range(1, 7)))]
    selection = [parent for parent in parents if parent.endswith(("C07", "C08"))]
    if len(fit) != 60 or len(selection) != 20:
        raise RuntimeError("structured-polar split drift")
    if args.maximum_worlds:
        fit = fit[:args.maximum_worlds]; selection = selection[:min(args.maximum_worlds, len(selection))]
    model = StructuredPolarMultiDepthEventEncoder().to(device)
    if sum(p.numel() for p in model.parameters()) != 172430:
        raise RuntimeError("structured-polar parameter drift")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []; best_loss = float("inf"); best_epoch = -1; steps = 0
    common = {"schema_version": "gse_structured_polar_multidepth_checkpoint_v1", "seed": args.seed, "trainable_parameters": 172430, "dense_layout": [180, 2], "topk_events": 16, "dataset_evidence_sha256": _sha(PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0/artifacts/evidence_sha256.txt"), "teacher_evidence_sha256": _sha(PROJECT_ROOT / "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0/artifacts/evidence_sha256.txt")}
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed * 10000 + epoch); epoch_parents = [fit[index] for index in rng.permutation(len(fit))]
        totals = {name: 0.0 for name in ("total", "presence", "event_type", "position", "descriptor", "uncertainty")}; observed = 0; batches = 0
        model.train()
        for parent in epoch_parents:
            gid, references, ranges, valid = _world_arrays(args.dataset_root.resolve(), parent); global_rows = row_lookup[parent]
            if not np.array_equal(teacher.global_sequence_index[global_rows], gid): raise RuntimeError(f"fit join drift: {parent}")
            order = rng.permutation(len(gid))
            for start in range(0, len(order), args.batch_size):
                local = np.asarray(order[start:start + args.batch_size], dtype=np.int64); selected = global_rows[local]
                rb, vb, targets = _batch(teacher, selected, local, references, ranges, valid, device)
                shifts = torch.from_numpy(rng.integers(0, 180, size=len(local), dtype=np.int64)).to(device)
                rb, vb, targets = _roll(rb, vb, targets, shifts)
                _, losses = _loss(model, rb, vb, targets, class_weights)
                optimizer.zero_grad(set_to_none=True); losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                steps += 1; batches += 1; observed += len(local)
                for name in totals: totals[name] += float(losses[name].detach().cpu()) * len(local)
                if args.maximum_batches and batches >= args.maximum_batches: break
            if args.maximum_batches and batches >= args.maximum_batches: break
        selection_loss = _evaluate_loss(model, teacher, args.dataset_root.resolve(), selection, row_lookup, device, args.evaluation_batch_size, class_weights)
        record = {"epoch": epoch, "optimizer_steps": steps, "fit_observations": observed, "fit_loss": {name: value / observed for name, value in totals.items()}, "selection_loss": selection_loss}; history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        if selection_loss["total"] < best_loss:
            best_loss = selection_loss["total"]; best_epoch = epoch; torch.save({**common, "epoch": epoch, "model": model.state_dict(), "selection_loss": selection_loss}, output / "best.pt")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"], strict=True)
    predictions, rows = _infer(model, teacher, args.dataset_root.resolve(), selection, row_lookup, device, args.evaluation_batch_size)
    metrics, threshold_records = select_fixed_grid_threshold(predictions, teacher.targets(rows), THRESHOLD_GRID, maximum_error_m=4.0)
    metrics["macro_f1"] = float((metrics["per_type"]["terminal"]["f1"] + metrics["per_type"]["junction"]["f1"]) / 2.0)
    np.savez_compressed(output / "selection_outputs.npz", observation_row=rows, global_sequence_index=teacher.global_sequence_index[rows], **predictions)
    summary = {"schema_version": "gse_structured_polar_multidepth_seed_training_v1", "seed": args.seed, "epochs": args.epochs, "best_epoch": best_epoch, "best_selection_loss": best_loss, "selected_threshold_metrics": metrics, "threshold_grid": list(THRESHOLD_GRID), "threshold_records": threshold_records, "fit_rows": 142184 if not args.maximum_worlds else observed, "selection_rows": 45942 if not args.maximum_worlds else len(rows), "optimizer_steps": steps, "trainable_parameters": 172430, "dense_layout": [180, 2], "topk_events": 16, "presence_loss": "per_row_equal_positive_negative_mass; empty_rows_negative_only", "fit_only_type_weights": frequency["event_type_class_weights"], "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "duration_seconds": time.monotonic() - started, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "smoke_limited": bool(args.maximum_worlds or args.maximum_batches)}
    (output / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
