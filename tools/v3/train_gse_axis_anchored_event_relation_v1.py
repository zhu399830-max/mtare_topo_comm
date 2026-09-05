#!/usr/bin/env python3
"""Train one GSE axis-anchored multi-label event-relation seed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
import zarr

from mtare_topo.data.gse_axis_anchored_event_relation_training import (
    DescriptorRow,
    descriptor_identity_batches,
    materialize_world_relation_teacher,
)
from mtare_topo.representation.gse_axis_anchored_event_relation import (
    AxisAnchoredEventRelationNet,
    axis_anchored_descriptor_loss,
    axis_anchored_event_relation_core_loss,
    parameter_count,
)


MAX_RANGE_M = 50.0
EXPECTED_PARAMETERS = 819067
EVENT_CLASS_WEIGHT = np.asarray(
    [0.266182241, 0.607610566, 1.14589883, 2.21772549, 0.762582869], dtype=np.float32
)
RELATION_POSITIVE_RATE = np.asarray(
    [934760 / 80667360, 10583 / 80667360, 11008 / 80667360], dtype=np.float32
)
EXPECTED_FIT_EVENT_COUNTS = np.asarray([102874, 19743, 5551, 1482, 12534], dtype=np.int64)
EXPECTED_CORE_STEPS_PER_EPOCH = 1137
EXPECTED_DESCRIPTOR_STEPS_PER_EPOCH = 86
EXPECTED_STEPS_PER_EPOCH = 1223


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False


def manifest_traversals(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line); parent = str(record["parent_id"])
            match = re.search(r"_C(\d+)$", parent)
            if record.get("split") != "train" or match is None or int(match.group(1)) > 8:
                continue
            rows = result.setdefault(parent, [])
            if int(record["world_sequence_row"]) != len(rows):
                raise RuntimeError(f"non-contiguous sequence manifest: {parent}")
            rows.append(str(record["traversal_id"]))
    if len(result) != 80:
        raise RuntimeError(f"expected 80 C01-C08 parents, got {len(result)}")
    return result


def _shift(seed: int, epoch: int, parent: str, batch_index: int, stage: str) -> int:
    payload = f"gse-axis-relation-roll-v1:{stage}:{seed}:{epoch}:{parent}:{batch_index}".encode()
    return 4 * (int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % 180)


def _load_world(dataset_root: Path, parent: str, traversal: list[str]) -> dict:
    group = zarr.open_group(str(dataset_root / f"{parent}.zarr"), mode="r")
    observations = len(group["event_index"])
    if len(traversal) != observations:
        raise RuntimeError(f"manifest/world observation drift: {parent}")
    teacher = materialize_world_relation_teacher(
        local_frame_references=np.asarray(group["local_frame_references"][:], dtype=np.int64),
        traversal_id=traversal,
        exit_mask=np.asarray(group["exit_mask"][:], dtype=bool),
        exit_identity=np.asarray(group["exit_identity"][:], dtype=np.int64),
        exit_heading_unit=np.asarray(group["exit_heading_unit"][:], dtype=np.float32),
        exit_opening_width_m=np.asarray(group["exit_opening_width_m"][:], dtype=np.float32),
        exit_width_valid_mask=np.asarray(group["exit_width_valid_mask"][:], dtype=bool),
        exit_vertical_profile_m=np.asarray(group["exit_vertical_profile_m"][:], dtype=np.float32),
    )
    association_valid = np.asarray(group["association_valid_mask"][:], dtype=bool)
    association = np.asarray(group["association_identity"][:], dtype=np.int64)
    association = np.where(association_valid, association, -1)
    return {
        "parent": parent,
        "references": np.asarray(group["local_frame_references"][:], dtype=np.int64),
        "range_m": np.asarray(group["range_m"][:], dtype=np.float32),
        "valid_mask": np.asarray(group["valid_mask"][:], dtype=np.uint8),
        "event_index": np.asarray(group["event_index"][:], dtype=np.int64),
        "local_axis": np.asarray(group["local_axis_robot"][:], dtype=np.float32),
        "geometry": np.asarray(group["geometry"][:], dtype=np.float32),
        "geometry_valid_mask": np.asarray(group["geometry_valid_mask"][:], dtype=bool),
        "association_identity": association,
        "association_valid_mask": association_valid,
        "exit_invisible_count": np.asarray(group["exit_invisible_count"][:], dtype=np.int64),
        "global_sequence_index": np.asarray(group["global_sequence_index"][:], dtype=np.int64),
        "relation_index": teacher.relation_index,
        "branch_presence_mask": teacher.branch_presence_mask,
        "branch_heading_residual_deg": teacher.branch_heading_residual_deg,
        "branch_opening_width_m": teacher.branch_opening_width_m,
        "branch_width_valid_mask": teacher.branch_width_valid_mask,
        "branch_vertical_profile_m": teacher.branch_vertical_profile_m,
        "branch_identity": teacher.branch_identity,
    }


def _descriptor_records(world: dict) -> list[DescriptorRow]:
    valid = world["association_valid_mask"] & (world["exit_invisible_count"] == 0)
    result = []
    for row in np.flatnonzero(valid):
        branches = tuple(sorted(int(value) for value in world["branch_identity"][row] if value >= 0))
        result.append(DescriptorRow(int(world["association_identity"][row]), world["parent"], int(row), branches))
    return result


def _batch(
    world: dict, indices: np.ndarray, *, device: torch.device, shift_columns: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    indices = np.asarray(indices, dtype=np.int64)
    references = world["references"][indices]
    ranges = world["range_m"][references] / MAX_RANGE_M
    valid = world["valid_mask"][references].astype(np.float32, copy=False)
    scans = torch.from_numpy(np.stack((ranges, valid), axis=2)).to(device=device, dtype=torch.float32)
    names = (
        "event_index", "relation_index", "branch_presence_mask",
        "branch_heading_residual_deg", "branch_opening_width_m", "branch_width_valid_mask",
        "branch_vertical_profile_m", "branch_identity", "local_axis", "geometry",
        "geometry_valid_mask", "association_identity",
    )
    targets = {name: torch.from_numpy(np.asarray(world[name][indices])).to(device=device) for name in names}
    targets["event_class_weight"] = torch.from_numpy(EVENT_CLASS_WEIGHT).to(device)
    targets["relation_positive_rate"] = torch.from_numpy(RELATION_POSITIVE_RATE).to(device)
    if shift_columns:
        shift_bins = shift_columns // 4
        scans = torch.roll(scans, shift_columns, dims=-1)
        targets["relation_index"] = torch.roll(targets["relation_index"], shift_bins, dims=2)
        for name in (
            "branch_presence_mask", "branch_heading_residual_deg", "branch_opening_width_m",
            "branch_width_valid_mask", "branch_vertical_profile_m", "branch_identity",
        ):
            targets[name] = torch.roll(targets[name], shift_bins, dims=1)
        angle = 2.0 * math.pi * shift_bins / 180.0; axis = targets["local_axis"]
        targets["local_axis"] = torch.stack((
            math.cos(angle) * axis[:, 0] - math.sin(angle) * axis[:, 1],
            math.sin(angle) * axis[:, 0] + math.cos(angle) * axis[:, 1], axis[:, 2],
        ), dim=-1)
    return scans, targets


def _finite_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
    optimizer.step()


def _event_metrics(confusion: np.ndarray) -> dict:
    per_class = {}; values = []
    names = ("corridor", "junction", "terminal", "turn", "geometry_transition")
    for index, name in enumerate(names):
        tp = int(confusion[index, index]); predicted = int(confusion[:, index].sum()); actual = int(confusion[index].sum())
        precision = tp / predicted if predicted else 0.0; recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append(f1); per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return {"macro_f1": float(np.mean(values)), "accuracy": float(np.trace(confusion) / confusion.sum()), "per_class": per_class, "confusion": confusion.tolist()}


@torch.no_grad()
def _validation(model, parents: list[str], traversals: dict[str, list[str]], dataset_root: Path, *, device: torch.device, batch_size: int) -> dict:
    model.eval(); loss_sum = defaultdict(float); rows = 0; confusion = np.zeros((5, 5), dtype=np.int64)
    relation_tp = np.zeros(3, dtype=np.int64); relation_fp = np.zeros(3, dtype=np.int64); relation_fn = np.zeros(3, dtype=np.int64)
    axis_sum = 0.0; geometry_sum = np.zeros(4); geometry_count = np.zeros(4, dtype=np.int64)
    descriptor_sum = defaultdict(float); descriptor_rows = descriptor_batches = 0
    for parent in parents:
        world = _load_world(dataset_root, parent, traversals[parent])
        for start in range(0, len(world["event_index"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(world["event_index"])))
            scans, target = _batch(world, indices, device=device); output = model(scans)
            losses = axis_anchored_event_relation_core_loss(output, target)
            for name, value in losses.items(): loss_sum[name] += float(value) * len(indices)
            rows += len(indices); truth = target["event_index"].cpu().numpy(); predicted = output["event_logits"].argmax(1).cpu().numpy(); np.add.at(confusion, (truth, predicted), 1)
            relation_truth = target["relation_index"].cpu().numpy(); relation_pred = (output["relation_probability_sequence"].cpu().numpy() >= 0.5)
            for channel in range(3):
                valid = relation_truth[..., channel] >= 0; positive = relation_truth[..., channel] == 1; guess = relation_pred[..., channel]
                relation_tp[channel] += int((valid & positive & guess).sum()); relation_fp[channel] += int((valid & ~positive & guess).sum()); relation_fn[channel] += int((valid & positive & ~guess).sum())
            dot = (output["local_axis"] * target["local_axis"]).sum(1).clamp(-1, 1); axis_sum += float(torch.rad2deg(torch.acos(dot)).sum())
            pred_geometry = torch.stack((output["width_m"], output["height_m"], output["slope_deg"], output["curvature_per_m"]), dim=1)
            valid_geometry = target["geometry_valid_mask"].bool(); absolute = torch.abs(pred_geometry - target["geometry"])
            geometry_sum += (absolute * valid_geometry).sum(0).cpu().numpy(); geometry_count += valid_geometry.sum(0).cpu().numpy()
        for descriptor_batch in descriptor_identity_batches(_descriptor_records(world), seed=0, epoch=0):
            indices = np.asarray([record.row for record in descriptor_batch], dtype=np.int64)
            scans, target = _batch(world, indices, device=device); losses = axis_anchored_descriptor_loss(model(scans), target)
            for name, value in losses.items(): descriptor_sum[name] += float(value) * len(indices)
            descriptor_rows += len(indices); descriptor_batches += 1
    relation = {}
    for channel, name in enumerate(("persistent", "reveal", "withdraw")):
        precision = relation_tp[channel] / max(relation_tp[channel] + relation_fp[channel], 1); recall = relation_tp[channel] / max(relation_tp[channel] + relation_fn[channel], 1)
        relation[name] = {"precision_at_0p5": float(precision), "recall_at_0p5": float(recall), "f1_at_0p5": float(2 * precision * recall / max(precision + recall, 1e-12)), "true_positive": int(relation_tp[channel])}
    core = {name: value / rows for name, value in loss_sum.items()}; descriptor = {name: value / descriptor_rows for name, value in descriptor_sum.items()}
    return {
        "rows": rows, "core_loss": core, "descriptor_loss": descriptor,
        "selection_loss": core["total"] + descriptor["total"], "event": _event_metrics(confusion),
        "relation": relation, "axis_mean_error_deg": axis_sum / rows,
        "geometry_mae": {name: float(geometry_sum[index] / geometry_count[index]) for index, name in enumerate(("width_m", "height_m", "slope_deg", "curvature_per_m"))},
        "descriptor_rows": descriptor_rows, "descriptor_batches": descriptor_batches,
    }


@torch.no_grad()
def _infer_world(model, world: dict, output: Path, *, device: torch.device, batch_size: int) -> int:
    model.eval(); values = defaultdict(list)
    for start in range(0, len(world["event_index"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(world["event_index"])))
        scans, _ = _batch(world, indices, device=device); predicted = model(scans)
        for name in ("event_logits", "relation_probability_sequence", "branch_union_probability", "local_axis", "place_descriptor", "observation_uncertainty"):
            values[name].append(predicted[name].cpu().numpy().astype(np.float16))
        geometry = torch.stack((predicted["width_m"], predicted["height_m"], predicted["slope_deg"], predicted["curvature_per_m"]), dim=-1)
        values["geometry"].append(geometry.cpu().numpy().astype(np.float32))
        branch_bins = np.full((len(indices), 6), -1, dtype=np.int16)
        for local, row in enumerate(indices):
            active = np.flatnonzero(world["branch_presence_mask"][row]); branch_bins[local, :len(active)] = active
        safe = np.maximum(branch_bins, 0); row_index = torch.arange(len(indices), device=device)[:, None]; bin_index = torch.from_numpy(safe).to(device=device, dtype=torch.long)
        for source, name in (("branch_heading_residual_deg", "branch_heading_residual_deg"), ("branch_opening_width_m", "branch_opening_width_m"), ("branch_vertical_profile_m", "branch_vertical_profile_m"), ("branch_descriptor", "branch_descriptor"), ("branch_geometry_uncertainty", "branch_geometry_uncertainty")):
            gathered = predicted[source][row_index, bin_index].cpu().numpy(); gathered[branch_bins < 0] = 0
            values[name].append(gathered.astype(np.float16))
        values["branch_bin_index"].append(branch_bins)
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}
    arrays["global_sequence_index"] = world["global_sequence_index"]
    np.savez_compressed(output / f"{world['parent']}.npz", **arrays)
    return len(world["event_index"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path); parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args(); started = time.monotonic()
    if args.epochs != 10 or args.batch_size != 128 or args.evaluation_batch_size != 256 or args.learning_rate != 3e-4 or args.weight_decay != 1e-4:
        raise ValueError("formal axis-anchored optimization contract drift")
    output = args.output_dir.resolve()
    if output.exists(): raise RuntimeError("axis-anchored seed output exists; overwrite forbidden")
    output.mkdir(parents=True); seed_everything(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("formal axis-anchored training requires CUDA")
    device = torch.device("cuda"); dataset_root = args.dataset_root.resolve(); traversals = manifest_traversals(args.sequence_manifest.resolve())
    fit = sorted(parent for parent in traversals if int(parent.rsplit("_C", 1)[1]) <= 6); c07 = sorted(parent for parent in traversals if parent.endswith("_C07")); c08 = sorted(parent for parent in traversals if parent.endswith("_C08"))
    if (len(fit), len(c07), len(c08)) != (60, 10, 10): raise RuntimeError("axis-anchored split drift")
    model = AxisAnchoredEventRelationNet().to(device)
    if parameter_count() != EXPECTED_PARAMETERS: raise RuntimeError("axis-anchored parameter drift")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []; best_score = math.inf; best_epoch = -1; total_steps = core_steps_total = descriptor_steps_total = 0
    event_population = np.zeros(5, dtype=np.int64); descriptor_selected = None
    for epoch in range(args.epochs):
        generator = np.random.default_rng(args.seed * 1000 + epoch); order = [fit[index] for index in generator.permutation(len(fit))]
        model.train(); sums = {"core": defaultdict(float), "descriptor": defaultdict(float)}; row_counts = {"core": 0, "descriptor": 0}; core_steps = descriptor_steps = 0
        for parent in order:
            world = _load_world(dataset_root, parent, traversals[parent]); indices_order = generator.permutation(len(world["event_index"]))
            if epoch == 0: event_population += np.bincount(world["event_index"], minlength=5)
            for batch_index, start in enumerate(range(0, len(indices_order), args.batch_size)):
                indices = indices_order[start:start + args.batch_size]; shift = _shift(args.seed, epoch, parent, batch_index, "core")
                scans, target = _batch(world, indices, device=device, shift_columns=shift); losses = axis_anchored_event_relation_core_loss(model(scans), target); _finite_step(model, optimizer, losses["total"])
                for name, value in losses.items(): sums["core"][name] += float(value.detach()) * len(indices)
                row_counts["core"] += len(indices); core_steps += 1
            batches = descriptor_identity_batches(_descriptor_records(world), seed=args.seed, epoch=epoch)
            for batch_index, records in enumerate(batches):
                indices = np.asarray([record.row for record in records], dtype=np.int64); shift = _shift(args.seed, epoch, parent, batch_index, "descriptor")
                scans, target = _batch(world, indices, device=device, shift_columns=shift); losses = axis_anchored_descriptor_loss(model(scans), target); _finite_step(model, optimizer, losses["total"])
                for name, value in losses.items(): sums["descriptor"][name] += float(value.detach()) * len(indices)
                row_counts["descriptor"] += len(indices); descriptor_steps += 1
        if core_steps != EXPECTED_CORE_STEPS_PER_EPOCH or descriptor_steps != EXPECTED_DESCRIPTOR_STEPS_PER_EPOCH or row_counts != {"core": 142184, "descriptor": 6347}:
            raise RuntimeError(f"axis-anchored epoch population drift: {core_steps}/{descriptor_steps}/{row_counts}")
        if epoch == 0:
            if not np.array_equal(event_population, EXPECTED_FIT_EVENT_COUNTS): raise RuntimeError("fit event population drift")
            descriptor_selected = row_counts["descriptor"]
        validation = _validation(model, c07, traversals, dataset_root, device=device, batch_size=args.evaluation_batch_size)
        train = {stage: {name: value / row_counts[stage] for name, value in sums[stage].items()} for stage in sums}
        record = {"epoch": epoch, "optimizer_steps": core_steps + descriptor_steps, "core_steps": core_steps, "descriptor_steps": descriptor_steps, "train": train, "c07": validation}
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        total_steps += core_steps + descriptor_steps; core_steps_total += core_steps; descriptor_steps_total += descriptor_steps
        if validation["selection_loss"] < best_score:
            best_score = validation["selection_loss"]; best_epoch = epoch
            torch.save({"schema_version": "gse_axis_anchored_event_relation_checkpoint_v1", "seed": args.seed, "epoch": epoch, "model": model.state_dict(), "c07_selection_loss": best_score, "parameters": EXPECTED_PARAMETERS, "fit_worlds": fit, "c07_selection_worlds": c07, "c08_checkpoint_observations": 0}, output / "best.pt")
    if total_steps != args.epochs * EXPECTED_STEPS_PER_EPOCH: raise RuntimeError("axis-anchored total optimizer step drift")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"], strict=True)
    prediction_root = output / "development_predictions"; prediction_root.mkdir(); development_rows = 0
    for parent in (*c07, *c08): development_rows += _infer_world(model, _load_world(dataset_root, parent, traversals[parent]), prediction_root, device=device, batch_size=args.evaluation_batch_size)
    summary = {
        "schema_version": "gse_axis_anchored_event_relation_training_seed_v1", "seed": args.seed, "parameters": EXPECTED_PARAMETERS, "epochs": args.epochs,
        "best_epoch": best_epoch, "best_c07_selection_loss": best_score, "optimizer_steps": total_steps, "core_optimizer_steps": core_steps_total, "descriptor_optimizer_steps": descriptor_steps_total,
        "fit_observations": 142184, "fit_descriptor_rows_per_epoch": descriptor_selected, "c07_checkpoint_observations": 21548, "c08_checkpoint_observations": 0,
        "development_output_observations": development_rows, "final_inference_c07_observations": 21548, "final_inference_c08_observations": 24394,
        "duration_seconds": time.monotonic() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    }
    (output / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
