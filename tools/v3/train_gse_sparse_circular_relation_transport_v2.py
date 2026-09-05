#!/usr/bin/env python3
"""Train one sparse circular relation-transport seed on C01--C06."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

import train_gse_axis_anchored_event_relation_v1 as base
from mtare_topo.representation.gse_axis_anchored_event_relation import _identity_contrastive
from mtare_topo.representation.gse_circular_peak_geometry_model import (
    CURVATURE_SCALE_PER_M,
    METRIC_DISTANCE_SCALE_M,
    PROFILE_SCALE_M,
    SLOPE_SCALE_DEG,
)
from mtare_topo.representation.gse_sparse_circular_relation_transport import (
    BEARING_BINS,
    MAX_TOKENS,
    SparseCircularRelationTransportNet,
    parameter_count,
    sparse_relation_transport_loss,
    token_count_loss,
)


EXPECTED_PARAMETERS = 784513
EXPECTED_CORE_STEPS_PER_EPOCH = 1137
EXPECTED_DESCRIPTOR_STEPS_PER_EPOCH = 86
EXPECTED_STEPS_PER_EPOCH = 1223
EXPECTED_FIT_EVENT_COUNTS = np.asarray([102874, 19743, 5551, 1482, 12534], dtype=np.int64)
EVENT_CLASS_WEIGHT = torch.tensor([0.266182241, 0.607610566, 1.14589883, 2.21772549, 0.762582869])
WITHDRAW_WEIGHT = math.sqrt(934760.0 / 11008.0)
REVEAL_POSITIVE_WEIGHT = math.sqrt(934760.0 / 10583.0)
BACKBONE_PREFIXES = (
    "bearing_azimuth_rad", "encoder.", "temporal.", "directional_temporal.",
    "axis_azimuth_head.", "axis_vertical_head.", "geometry_head.",
    "place_head.", "observation_uncertainty_head.",
)


def _load_predecessor_backbone(model: SparseCircularRelationTransportNet, path: Path, expected_seed: int) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint["seed"]) != expected_seed:
        raise RuntimeError("predecessor seed mismatch")
    source = checkpoint["model"]
    target = model.state_dict()
    required = sorted(
        key for key in target
        if key == "bearing_azimuth_rad" or any(key.startswith(prefix) for prefix in BACKBONE_PREFIXES[1:])
    )
    missing = [key for key in required if key not in source]
    mismatch = [key for key in required if key in source and source[key].shape != target[key].shape]
    if missing or mismatch or len(required) != 53:
        raise RuntimeError(f"predecessor backbone drift: {missing}/{mismatch}/{len(required)}")
    model.load_state_dict({key: source[key] for key in required}, strict=False)
    if not all(torch.equal(model.state_dict()[key].cpu(), source[key].cpu()) for key in required):
        raise RuntimeError("predecessor backbone did not load exactly")
    return {"source_seed": int(checkpoint["seed"]), "source_epoch": int(checkpoint["epoch"]), "loaded_keys": len(required)}


def _history_rows(references: np.ndarray) -> np.ndarray:
    current = np.asarray(references[:, -1], dtype=np.int64)
    if len(np.unique(current)) != len(current):
        raise RuntimeError("current raw-frame references must be unique")
    lookup = {int(value): row for row, value in enumerate(current)}
    return np.asarray([[lookup.get(int(value), -1) for value in sequence] for sequence in references], dtype=np.int64)


def _load_world(dataset_root: Path, parent: str, traversal: list[str]) -> dict:
    world = base._load_world(dataset_root, parent, traversal)
    history = _history_rows(world["references"])
    pair_valid = (history[:, :-1] >= 0) & (history[:, 1:] >= 0)
    relation_valid = (world["relation_index"] >= 0).any(axis=(2, 3))
    if not np.array_equal(pair_valid, relation_valid):
        raise RuntimeError(f"history/relation validity drift: {parent}")
    world["history_observation_rows"] = history
    world["history_frame_valid_mask"] = history >= 0
    world["history_pair_valid_mask"] = pair_valid
    return world


def _sequence_array(world: dict, name: str, history: np.ndarray, fill_value) -> np.ndarray:
    safe = np.maximum(history, 0)
    value = np.asarray(world[name][safe]).copy()
    invalid = history < 0
    value[invalid] = fill_value
    return value


def _batch(
    world: dict, indices: np.ndarray, *, device: torch.device, shift_columns: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scans, target = base._batch(world, indices, device=device, shift_columns=shift_columns)
    history = world["history_observation_rows"][np.asarray(indices, dtype=np.int64)]
    sequence = {
        "sequence_presence": _sequence_array(world, "branch_presence_mask", history, False),
        "sequence_identity": _sequence_array(world, "branch_identity", history, -1),
        "sequence_width_m": _sequence_array(world, "branch_opening_width_m", history, 0.0),
        "sequence_width_valid": _sequence_array(world, "branch_width_valid_mask", history, False),
        "sequence_profile_m": _sequence_array(world, "branch_vertical_profile_m", history, 0.0),
        "sequence_frame_valid": history >= 0,
        "sequence_pair_valid": world["history_pair_valid_mask"][np.asarray(indices, dtype=np.int64)],
    }
    if shift_columns:
        shift_bins = shift_columns // 4
        for name in ("sequence_presence", "sequence_identity", "sequence_width_m", "sequence_width_valid", "sequence_profile_m"):
            sequence[name] = np.roll(sequence[name], shift_bins, axis=2)
    target.update({name: torch.from_numpy(np.asarray(value)).to(device=device) for name, value in sequence.items()})
    return scans, target


@lru_cache(maxsize=None)
def _permutations(count: int) -> torch.Tensor:
    return torch.tensor(tuple(itertools.permutations(range(MAX_TOKENS), count)), dtype=torch.long)


def align_teacher_to_tokens(
    predicted_bins: torch.Tensor, teacher_identity_by_bin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign each true branch injectively to the nearest one of six proposals."""
    if predicted_bins.ndim != 3 or predicted_bins.shape[-1] != MAX_TOKENS:
        raise ValueError("predicted token bins must be [B,5,6]")
    if teacher_identity_by_bin.shape != (*predicted_bins.shape[:2], BEARING_BINS):
        raise ValueError("teacher identity field shape drift")
    leading = predicted_bins.shape[:2]
    predicted = predicted_bins.reshape(-1, MAX_TOKENS).long()
    identity_field = teacher_identity_by_bin.reshape(-1, BEARING_BINS).long()
    valid_field = identity_field >= 0
    counts = valid_field.sum(dim=-1)
    if bool((counts > MAX_TOKENS).any()):
        raise ValueError("teacher exceeds six-token capacity")
    grid = torch.arange(BEARING_BINS, device=predicted.device)[None].expand(len(predicted), -1)
    teacher_bins = torch.sort(torch.where(valid_field, grid, BEARING_BINS), dim=-1).values[:, :MAX_TOKENS]
    safe_teacher = teacher_bins.clamp_max(BEARING_BINS - 1)
    teacher_identity = torch.gather(identity_field, 1, safe_teacher)
    aligned_identity = torch.full_like(predicted, -1)
    aligned_teacher_bin = torch.full_like(predicted, -1)
    for count in range(1, MAX_TOKENS + 1):
        rows = torch.nonzero(counts == count, as_tuple=False).squeeze(-1)
        if not len(rows):
            continue
        true_bin = teacher_bins[rows, :count]
        delta = torch.abs(predicted[rows, :, None] - true_bin[:, None, :])
        cost = torch.minimum(delta, BEARING_BINS - delta)
        permutations = _permutations(count).to(device=predicted.device)
        candidate = cost[:, permutations, torch.arange(count, device=predicted.device)].sum(dim=-1)
        selected = permutations[candidate.argmin(dim=-1)]
        identities = teacher_identity[rows, :count]
        aligned_identity[rows[:, None], selected] = identities
        aligned_teacher_bin[rows[:, None], selected] = true_bin
    return aligned_identity.reshape(*leading, MAX_TOKENS), aligned_teacher_bin.reshape(*leading, MAX_TOKENS)


def _token_geometry_loss(outputs: dict[str, torch.Tensor], target: dict[str, torch.Tensor], assigned_bin: torch.Tensor) -> torch.Tensor:
    safe = assigned_bin.clamp_min(0)
    width = torch.gather(target["sequence_width_m"], 2, safe)
    width_valid = torch.gather(target["sequence_width_valid"].bool(), 2, safe) & (assigned_bin >= 0)
    profile_index = safe[..., None].expand(-1, -1, -1, target["sequence_profile_m"].shape[-1])
    profile = torch.gather(target["sequence_profile_m"], 2, profile_index)
    present = assigned_bin >= 0
    if not bool(width_valid.any()) or not bool(present.any()):
        raise ValueError("token geometry batch lacks valid targets")
    width_error = (
        torch.log1p(outputs["token_opening_width_m"]) - torch.log1p(width.clamp_min(0))
    ) / math.log1p(60.0)
    profile_error = (outputs["token_vertical_profile_m"] - profile) / PROFILE_SCALE_M
    error = torch.cat((width_error[..., None], profile_error), dim=-1)
    valid = torch.cat((width_valid[..., None], present[..., None].expand_as(profile_error)), dim=-1)
    uncertainty = outputs["token_geometry_uncertainty"]
    element = F.smooth_l1_loss(error / uncertainty, torch.zeros_like(error), reduction="none") + torch.log(uncertainty)
    return torch.stack([element[..., index][valid[..., index]].mean() for index in range(element.shape[-1])]).mean()


def sparse_core_loss(outputs: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    aligned_identity, assigned_bin = align_teacher_to_tokens(
        outputs["token_bin_index"].detach(), target["sequence_identity"],
    )
    relation = sparse_relation_transport_loss(
        outputs, proposal_presence=target["sequence_presence"],
        aligned_token_identity=aligned_identity,
        proposal_valid_frame_mask=target["sequence_frame_valid"],
        relation_pair_valid_mask=target["sequence_pair_valid"],
        withdraw_weight=WITHDRAW_WEIGHT,
        reveal_positive_weight=REVEAL_POSITIVE_WEIGHT,
    )
    count = token_count_loss(
        outputs["token_count_logits"], target["sequence_presence"],
        valid_frame_mask=target["sequence_frame_valid"],
    )
    event = F.cross_entropy(
        outputs["event_logits"], target["event_index"].long(),
        weight=EVENT_CLASS_WEIGHT.to(device=outputs["event_logits"].device, dtype=outputs["event_logits"].dtype),
    )
    token_geometry = _token_geometry_loss(outputs, target, assigned_bin)
    target_axis = F.normalize(target["local_axis"].to(outputs["local_axis"].dtype), dim=-1)
    axis = (1.0 - (outputs["local_axis"] * target_axis).sum(dim=-1)).mean()
    geometry_target = target["geometry"].to(outputs["event_logits"].dtype)
    geometry_valid = target["geometry_valid_mask"].bool()
    prediction = torch.stack((
        outputs["width_m"] / METRIC_DISTANCE_SCALE_M,
        outputs["height_m"] / METRIC_DISTANCE_SCALE_M,
        outputs["slope_deg"] / SLOPE_SCALE_DEG,
        outputs["curvature_per_m"] / CURVATURE_SCALE_PER_M,
    ), dim=-1)
    normalized_target = torch.stack((
        geometry_target[:, 0] / METRIC_DISTANCE_SCALE_M,
        geometry_target[:, 1] / METRIC_DISTANCE_SCALE_M,
        geometry_target[:, 2] / SLOPE_SCALE_DEG,
        geometry_target[:, 3] / CURVATURE_SCALE_PER_M,
    ), dim=-1)
    global_element = F.smooth_l1_loss(prediction, normalized_target, reduction="none")
    global_geometry = torch.stack([
        global_element[:, index][geometry_valid[:, index]].mean() for index in range(4)
    ]).mean()
    truth_probability = torch.softmax(outputs["event_logits"], dim=-1).gather(1, target["event_index"].long()[:, None]).squeeze(1)
    uncertainty_target = (1.0 - truth_probability.detach()).clamp(0, 1)
    uncertainty = F.smooth_l1_loss(outputs["observation_uncertainty"], uncertainty_target)
    total = relation["total"] + count + event + token_geometry + axis + global_geometry + uncertainty
    return {
        "total": total, "proposal": relation["proposal"], "transport_row": relation["transport_row"],
        "reveal": relation["reveal"], "column_exclusivity": relation["column_exclusivity"],
        "count": count, "event": event, "token_geometry": token_geometry,
        "axis": axis, "global_geometry": global_geometry, "uncertainty": uncertainty,
    }


def sparse_descriptor_loss(outputs: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    aligned_identity, _ = align_teacher_to_tokens(
        outputs["token_bin_index"][:, -1].detach()[:, None],
        target["sequence_identity"][:, -1][:, None],
    )
    place = _identity_contrastive(outputs["place_descriptor"], target["association_identity"].long())
    identity = aligned_identity[:, 0]
    branch = _identity_contrastive(outputs["token_descriptor"][:, -1][identity >= 0], identity[identity >= 0])
    return {"total": place + branch, "place_association": place, "branch_association": branch}


def _finite_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
    optimizer.step()


def _release_unused_cuda_cache(device: torch.device) -> None:
    """Release allocator cache between independently loaded evaluation worlds."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _nvidia_process_memory_bytes() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in output.splitlines():
        pid, memory = [part.strip() for part in line.split(",")]
        if int(pid) == os.getpid():
            return int(memory) * 1024**2
    raise RuntimeError("formal training process missing from nvidia-smi")


@torch.no_grad()
def _validation(model, parents, traversals, dataset_root, *, device, batch_size) -> dict:
    model.eval()
    sums = defaultdict(float)
    rows = 0
    confusion = np.zeros((5, 5), dtype=np.int64)
    axis_sum = 0.0
    count_correct = count_valid = 0
    for parent in parents:
        _release_unused_cuda_cache(device)
        world = _load_world(dataset_root, parent, traversals[parent])
        for start in range(0, len(world["event_index"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(world["event_index"])))
            scans, target = _batch(world, indices, device=device)
            output = model(scans)
            losses = sparse_core_loss(output, target)
            for name, value in losses.items():
                sums[name] += float(value) * len(indices)
            truth = target["event_index"].cpu().numpy()
            predicted = output["event_logits"].argmax(1).cpu().numpy()
            np.add.at(confusion, (truth, predicted), 1)
            dot = (output["local_axis"] * target["local_axis"]).sum(1).clamp(-1, 1)
            axis_sum += float(torch.rad2deg(torch.acos(dot)).sum())
            valid = target["sequence_frame_valid"].bool()
            count_truth = target["sequence_presence"].sum(-1).long()
            count_pred = output["token_count_logits"].argmax(-1)
            count_correct += int((count_truth[valid] == count_pred[valid]).sum())
            count_valid += int(valid.sum())
            rows += len(indices)
        del scans, target, output, losses, world
        _release_unused_cuda_cache(device)
    core = {name: value / rows for name, value in sums.items()}
    return {
        "rows": rows, "core_loss": core, "selection_loss": core["total"],
        "event": base._event_metrics(confusion), "axis_mean_error_deg": axis_sum / rows,
        "causal_count_accuracy": count_correct / count_valid,
    }


@torch.no_grad()
def _infer_world(model, world: dict, output_dir: Path, *, device, batch_size) -> int:
    model.eval()
    values = defaultdict(list)
    for start in range(0, len(world["event_index"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(world["event_index"])))
        scans, _ = _batch(world, indices, device=device)
        predicted = model(scans)
        for name in (
            "event_logits", "proposal_logits", "token_count_logits", "token_count_probability",
            "token_bearing_deg", "token_existence_logits", "token_descriptor",
            "token_opening_width_m", "token_vertical_profile_m", "token_geometry_uncertainty",
            "transport_row_probability", "transport_reveal_probability", "local_axis",
            "place_descriptor", "observation_uncertainty",
        ):
            value = predicted[name].cpu().numpy()
            values[name].append(value.astype(np.float16))
        values["token_bin_index"].append(predicted["token_bin_index"].cpu().numpy().astype(np.int16))
        geometry = torch.stack((predicted["width_m"], predicted["height_m"], predicted["slope_deg"], predicted["curvature_per_m"]), dim=-1)
        values["geometry"].append(geometry.cpu().numpy().astype(np.float32))
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}
    arrays["global_sequence_index"] = world["global_sequence_index"]
    np.savez_compressed(output_dir / f"{world['parent']}.npz", **arrays)
    return len(world["event_index"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--predecessor-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (args.epochs, args.batch_size, args.evaluation_batch_size, args.learning_rate, args.weight_decay) != (10, 128, 256, 3e-4, 1e-4):
        raise ValueError("formal sparse relation training contract drift")
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("sparse relation seed output exists; overwrite forbidden")
    output.mkdir(parents=True)
    base.seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal sparse relation training requires CUDA")
    device = torch.device("cuda")
    dataset_root = args.dataset_root.resolve()
    traversals = base.manifest_traversals(args.sequence_manifest.resolve())
    fit = sorted(parent for parent in traversals if int(parent.rsplit("_C", 1)[1]) <= 6)
    c07 = sorted(parent for parent in traversals if parent.endswith("_C07"))
    c08 = sorted(parent for parent in traversals if parent.endswith("_C08"))
    if (len(fit), len(c07), len(c08)) != (60, 10, 10):
        raise RuntimeError("sparse relation split drift")
    model = SparseCircularRelationTransportNet()
    compatibility = _load_predecessor_backbone(model, args.predecessor_checkpoint.resolve(), args.seed)
    if parameter_count() != EXPECTED_PARAMETERS:
        raise RuntimeError("sparse relation parameter drift")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    best_score = math.inf
    best_epoch = -1
    total_steps = core_steps_total = descriptor_steps_total = 0
    peak_nvidia_process_memory_bytes = 0
    event_population = np.zeros(5, dtype=np.int64)
    for epoch in range(args.epochs):
        generator = np.random.default_rng(args.seed * 1000 + epoch)
        order = [fit[index] for index in generator.permutation(len(fit))]
        model.train()
        sums = {"core": defaultdict(float), "descriptor": defaultdict(float)}
        row_counts = {"core": 0, "descriptor": 0}
        core_steps = descriptor_steps = 0
        for parent in order:
            _release_unused_cuda_cache(device)
            world = _load_world(dataset_root, parent, traversals[parent])
            indices_order = generator.permutation(len(world["event_index"]))
            if epoch == 0:
                event_population += np.bincount(world["event_index"], minlength=5)
            for batch_index, start in enumerate(range(0, len(indices_order), args.batch_size)):
                indices = indices_order[start:start + args.batch_size]
                shift = base._shift(args.seed, epoch, parent, batch_index, "sparse-core")
                scans, target = _batch(world, indices, device=device, shift_columns=shift)
                losses = sparse_core_loss(model(scans), target)
                _finite_step(model, optimizer, losses["total"])
                for name, value in losses.items():
                    sums["core"][name] += float(value.detach()) * len(indices)
                row_counts["core"] += len(indices)
                core_steps += 1
            records = None
            for batch_index, records in enumerate(base.descriptor_identity_batches(base._descriptor_records(world), seed=args.seed, epoch=epoch)):
                indices = np.asarray([record.row for record in records], dtype=np.int64)
                shift = base._shift(args.seed, epoch, parent, batch_index, "sparse-descriptor")
                scans, target = _batch(world, indices, device=device, shift_columns=shift)
                losses = sparse_descriptor_loss(model(scans), target)
                _finite_step(model, optimizer, losses["total"])
                for name, value in losses.items():
                    sums["descriptor"][name] += float(value.detach()) * len(indices)
                row_counts["descriptor"] += len(indices)
                descriptor_steps += 1
            torch.cuda.synchronize()
            peak_nvidia_process_memory_bytes = max(peak_nvidia_process_memory_bytes, _nvidia_process_memory_bytes())
            del scans, target, losses, records, indices, indices_order, world
            _release_unused_cuda_cache(device)
        if core_steps != EXPECTED_CORE_STEPS_PER_EPOCH or descriptor_steps != EXPECTED_DESCRIPTOR_STEPS_PER_EPOCH or row_counts != {"core": 142184, "descriptor": 6347}:
            raise RuntimeError(f"sparse relation epoch population drift: {core_steps}/{descriptor_steps}/{row_counts}")
        if epoch == 0 and not np.array_equal(event_population, EXPECTED_FIT_EVENT_COUNTS):
            raise RuntimeError("sparse relation fit event population drift")
        validation = _validation(model, c07, traversals, dataset_root, device=device, batch_size=args.evaluation_batch_size)
        train = {stage: {name: value / row_counts[stage] for name, value in sums[stage].items()} for stage in sums}
        record = {"epoch": epoch, "optimizer_steps": core_steps + descriptor_steps, "core_steps": core_steps, "descriptor_steps": descriptor_steps, "train": train, "c07": validation}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        total_steps += core_steps + descriptor_steps
        core_steps_total += core_steps
        descriptor_steps_total += descriptor_steps
        if validation["selection_loss"] < best_score:
            best_score = validation["selection_loss"]
            best_epoch = epoch
            torch.save({
                "schema_version": "gse_sparse_circular_relation_transport_checkpoint_v2",
                "seed": args.seed, "epoch": epoch, "model": model.state_dict(),
                "c07_selection_loss": best_score, "parameters": EXPECTED_PARAMETERS,
                "fit_worlds": fit, "c07_selection_worlds": c07, "c08_checkpoint_observations": 0,
                "predecessor_backbone": compatibility,
            }, output / "best.pt")
    if total_steps != args.epochs * EXPECTED_STEPS_PER_EPOCH:
        raise RuntimeError("sparse relation total optimizer step drift")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    prediction_root = output / "development_predictions"
    prediction_root.mkdir()
    development_rows = 0
    for parent in (*c07, *c08):
        _release_unused_cuda_cache(device)
        development_rows += _infer_world(model, _load_world(dataset_root, parent, traversals[parent]), prediction_root, device=device, batch_size=args.evaluation_batch_size)
        _release_unused_cuda_cache(device)
    summary = {
        "schema_version": "gse_sparse_circular_relation_transport_training_seed_v2",
        "seed": args.seed, "parameters": EXPECTED_PARAMETERS, "epochs": args.epochs,
        "best_epoch": best_epoch, "best_c07_selection_loss": best_score,
        "optimizer_steps": total_steps, "core_optimizer_steps": core_steps_total,
        "descriptor_optimizer_steps": descriptor_steps_total, "fit_observations": 142184,
        "fit_descriptor_rows_per_epoch": 6347, "c07_checkpoint_observations": 21548,
        "c08_checkpoint_observations": 0, "development_output_observations": development_rows,
        "final_inference_c07_observations": 21548, "final_inference_c08_observations": 24394,
        "predecessor_backbone": compatibility,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_nvidia_process_memory_bytes": peak_nvidia_process_memory_bytes,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "graph_replays": 0, "planner_calls": 0,
    }
    (output / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
