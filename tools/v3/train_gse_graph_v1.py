#!/usr/bin/env python3
"""Train one frozen seed of the five-frame GSE-Graph perception model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.data.gse_training_sampler import PairAwareBatchSampler
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet, gse_multitask_loss
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


TARGET_KEYS = (
    "event_index",
    "local_axis",
    "width_m",
    "height_m",
    "slope_deg",
    "curvature_per_m",
    "geometry_valid_mask",
    "association_identity",
    "association_valid_mask",
    "exit_mask",
    "exit_heading_unit",
    "exit_opening_width_m",
    "exit_width_valid_mask",
    "exit_vertical_profile",
    "exit_identity",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def collate_gse(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty GSE batch")
    targets = {}
    for key in TARGET_KEYS:
        values = [sample["targets"][key] for sample in samples]
        first = values[0]
        if isinstance(first, np.ndarray):
            targets[key] = torch.from_numpy(np.stack(values))
        else:
            targets[key] = torch.as_tensor(np.asarray(values))
    return {
        "student": torch.from_numpy(np.stack([sample["student"] for sample in samples])),
        "targets": targets,
        "observation_id": [str(sample["observation_id"]) for sample in samples],
        "parent_id": [str(sample["parent_id"]) for sample in samples],
        "global_sequence_index": torch.tensor(
            [int(sample["global_sequence_index"]) for sample in samples], dtype=torch.int64
        ),
    }


def class_weights(labels: np.ndarray, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(EVENT_NAMES))
    if len(counts) != len(EVENT_NAMES) or np.any(counts <= 0):
        raise RuntimeError("all five training events must have positive support")
    weights = counts.sum() / (len(EVENT_NAMES) * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.tolist()


def _to_device(targets: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def _event_scores(matrix: np.ndarray) -> dict[str, Any]:
    per_class = {}
    f1_values = []
    for index, name in enumerate(EVENT_NAMES):
        true_positive = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        actual = int(matrix[index, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
        }
    return {
        "macro_f1": float(np.mean(f1_values)),
        "accuracy": float(np.trace(matrix) / max(matrix.sum(), 1)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def _association_retrieval(descriptors: np.ndarray, identities: np.ndarray) -> dict[str, Any]:
    if descriptors.ndim != 2 or identities.shape != (len(descriptors),):
        raise ValueError("association retrieval arrays are misaligned")
    unique, counts = np.unique(identities, return_counts=True)
    matchable_identity = {int(identity) for identity, count in zip(unique, counts, strict=True) if count >= 2}
    matchable = np.asarray([int(identity) in matchable_identity for identity in identities])
    correct = 0
    attempted = 0
    normalized = descriptors / np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12)
    for start in range(0, len(normalized), 512):
        stop = min(start + 512, len(normalized))
        similarity = normalized[start:stop] @ normalized.T
        rows = np.arange(stop - start)
        similarity[rows, np.arange(start, stop)] = -np.inf
        nearest = similarity.argmax(axis=1)
        active = matchable[start:stop]
        attempted += int(active.sum())
        correct += int(np.sum(identities[nearest[active]] == identities[start:stop][active]))
    return {
        "valid_observations": int(len(identities)),
        "matchable_observations": attempted,
        "identity_count": int(len(unique)),
        "top1_same_identity_precision": float(correct / attempted) if attempted else 0.0,
        "correct_top1": correct,
    }


@torch.no_grad()
def evaluate(
    model: GeometrySemanticEventNet,
    loader: DataLoader,
    device: torch.device,
    event_weights: torch.Tensor,
    *,
    preserve_arrays: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    loss_sums: dict[str, float] = {}
    count = 0
    event_matrix = np.zeros((len(EVENT_NAMES), len(EVENT_NAMES)), dtype=np.int64)
    axis_errors: list[np.ndarray] = []
    geometry_absolute = np.zeros(4, dtype=np.float64)
    geometry_count = np.zeros(4, dtype=np.int64)
    exit_count_absolute = 0.0
    exit_count_exact = 0
    association_descriptors: list[np.ndarray] = []
    association_identities: list[np.ndarray] = []
    arrays: dict[str, list[np.ndarray]] = {}
    observation_ids: list[str] = []
    parent_ids: list[str] = []
    for batch in loader:
        student = batch["student"].to(device, non_blocking=True)
        targets = _to_device(batch["targets"], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(student)
            losses = gse_multitask_loss(outputs, targets, event_class_weights=event_weights)
        size = len(student)
        count += size
        for key, value in losses.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu()) * size
        truth = targets["event_index"].long().cpu().numpy()
        predicted = outputs["event_logits"].argmax(dim=1).cpu().numpy()
        np.add.at(event_matrix, (truth, predicted), 1)
        dot = (outputs["local_axis"].float() * targets["local_axis"].float()).sum(dim=1)
        axis_errors.append(torch.rad2deg(torch.acos(dot.clamp(-1.0, 1.0))).cpu().numpy())
        predicted_geometry = torch.stack(
            (
                outputs["width_m"],
                outputs["height_m"],
                outputs["slope_deg"],
                outputs["curvature_per_m"],
            ),
            dim=1,
        ).float()
        target_geometry = torch.stack(
            (
                targets["width_m"],
                targets["height_m"],
                targets["slope_deg"],
                targets["curvature_per_m"],
            ),
            dim=1,
        ).float()
        valid_geometry = targets["geometry_valid_mask"].bool()
        absolute = torch.abs(predicted_geometry - target_geometry)
        geometry_absolute += (absolute * valid_geometry).sum(dim=0).cpu().numpy()
        geometry_count += valid_geometry.sum(dim=0).cpu().numpy()
        target_exit_count = targets["exit_mask"].sum(dim=1).cpu().numpy()
        predicted_exit_count = (outputs["exit_confidence"] >= 0.5).sum(dim=1).cpu().numpy()
        exit_count_absolute += float(np.abs(target_exit_count - predicted_exit_count).sum())
        exit_count_exact += int(np.sum(target_exit_count == predicted_exit_count))
        valid_association = targets["association_valid_mask"].bool()
        if bool(valid_association.any()):
            association_descriptors.append(outputs["place_descriptor"][valid_association].float().cpu().numpy())
            association_identities.append(targets["association_identity"][valid_association].cpu().numpy())
        if preserve_arrays:
            selected = {
                "event_logits": outputs["event_logits"],
                "local_axis": outputs["local_axis"],
                "width_m": outputs["width_m"],
                "height_m": outputs["height_m"],
                "slope_deg": outputs["slope_deg"],
                "curvature_per_m": outputs["curvature_per_m"],
                "place_descriptor": outputs["place_descriptor"],
                "uncertainty": outputs["uncertainty"],
                "exit_confidence": outputs["exit_confidence"],
                "exit_heading_unit": outputs["exit_heading_unit"],
                "exit_opening_width_m": outputs["exit_opening_width_m"],
                "exit_vertical_profile": outputs["exit_vertical_profile"],
                "exit_descriptor": outputs["exit_descriptor"],
            }
            for key, value in selected.items():
                arrays.setdefault(key, []).append(value.float().cpu().numpy().astype(np.float16))
            for key in TARGET_KEYS:
                arrays.setdefault(f"target_{key}", []).append(targets[key].cpu().numpy())
            arrays.setdefault("global_sequence_index", []).append(batch["global_sequence_index"].numpy())
            observation_ids.extend(batch["observation_id"])
            parent_ids.extend(batch["parent_id"])
    if count == 0 or np.any(geometry_count == 0):
        raise RuntimeError("validation produced no complete metric population")
    axis = np.concatenate(axis_errors)
    descriptors = np.concatenate(association_descriptors)
    identities = np.concatenate(association_identities)
    result = {
        "frames": count,
        "loss": {key: value / count for key, value in loss_sums.items()},
        "event": _event_scores(event_matrix),
        "axis": {
            "mean_angular_error_deg": float(axis.mean()),
            "p95_angular_error_deg": float(np.quantile(axis, 0.95)),
        },
        "geometry_mae": {
            name: float(geometry_absolute[index] / geometry_count[index])
            for index, name in enumerate(("width_m", "height_m", "slope_deg", "curvature_per_m"))
        },
        "geometry_valid_count": {
            name: int(geometry_count[index])
            for index, name in enumerate(("width_m", "height_m", "slope_deg", "curvature_per_m"))
        },
        "exit_count": {
            "mean_absolute_error": float(exit_count_absolute / count),
            "exact_accuracy": float(exit_count_exact / count),
        },
        "association_retrieval": _association_retrieval(descriptors, identities),
    }
    if not all(math.isfinite(float(value)) for value in result["loss"].values()):
        raise RuntimeError("validation loss is nonfinite")
    packed = None
    if preserve_arrays:
        packed = {key: np.concatenate(values) for key, values in arrays.items()}
        packed["observation_id"] = np.asarray(observation_ids, dtype="U128")
        packed["parent_id"] = np.asarray(parent_ids, dtype="U64")
    return result, packed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    if (
        args.seed not in (0, 1, 2)
        or args.epochs != 30
        or args.batch_size != 64
        or args.learning_rate != 3e-4
        or args.weight_decay != 1e-4
        or args.patience != 6
    ):
        raise ValueError("formal GSE V1 optimization contract drift")
    if not torch.cuda.is_available():
        raise RuntimeError("formal GSE V1 training requires the frozen CUDA environment")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    train = GSESequenceDataset(
        args.dataset_run,
        "train",
        augment_azimuth=True,
        augmentation_seed=args.seed,
    )
    validation = GSESequenceDataset(args.dataset_run, "validation", augment_azimuth=False)
    train_sampler = PairAwareBatchSampler(
        train.association_labels(), batch_size=args.batch_size, seed=args.seed
    )
    validation_sampler = PairAwareBatchSampler(
        validation.association_labels(), batch_size=args.batch_size, seed=1000
    )
    train_loader = DataLoader(
        train,
        batch_sampler=train_sampler,
        collate_fn=collate_gse,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation,
        batch_sampler=validation_sampler,
        collate_fn=collate_gse,
        num_workers=0,
        pin_memory=True,
    )
    event_weights, event_counts = class_weights(train.event_labels(), device)
    model = GeometrySemanticEventNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best_loss = math.inf
    stale = 0
    optimizer_steps = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        train.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        sums: dict[str, float] = {}
        seen = 0
        epoch_started = time.monotonic()
        for batch in train_loader:
            student = batch["student"].to(device, non_blocking=True)
            targets = _to_device(batch["targets"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(student)
                losses = gse_multitask_loss(
                    outputs,
                    targets,
                    event_class_weights=event_weights,
                )
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("nonfinite GSE gradient norm")
            optimizer.step()
            optimizer_steps += 1
            size = len(student)
            seen += size
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * size
        if seen != len(train):
            raise RuntimeError("training epoch did not visit every sequence exactly once")
        validation_metrics, _ = evaluate(
            model,
            validation_loader,
            device,
            event_weights,
            preserve_arrays=False,
        )
        record = {
            "epoch": epoch,
            "train_loss": {key: value / seen for key, value in sums.items()},
            "validation": validation_metrics,
            "duration_seconds": time.monotonic() - epoch_started,
            "optimizer_steps_total": optimizer_steps,
        }
        history.append(record)
        with (output_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        score = float(validation_metrics["loss"]["total"])
        if score < best_loss - 1e-8:
            best_loss = score
            stale = 0
            torch.save(
                {
                    "schema_version": "gse_graph_checkpoint_v1",
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "config": vars(args),
                    "event_counts": event_counts,
                    "event_class_weights": event_weights.detach().cpu(),
                    "selection_metric": "validation_total_loss",
                    "selection_value": score,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= args.patience:
            break
    torch.save(
        {
            "schema_version": "gse_graph_checkpoint_v1",
            "model": model.state_dict(),
            "epoch": history[-1]["epoch"],
            "seed": args.seed,
            "config": vars(args),
        },
        output_dir / "last.pt",
    )
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics, arrays = evaluate(
        model,
        validation_loader,
        device,
        event_weights,
        preserve_arrays=True,
    )
    if arrays is None:
        raise RuntimeError("final validation outputs were not preserved")
    np.savez_compressed(output_dir / "validation_outputs.npz", **arrays)
    (output_dir / "best_validation_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "gse_graph_training_seed_v1",
        "overall_status": "PASS_GSE_GRAPH_TRAINING_SEED_V1",
        "seed": args.seed,
        "epochs_completed": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": "validation_total_loss",
        "best_selection_value": float(checkpoint["selection_value"]),
        "train_sequences_per_epoch": len(train),
        "validation_sequences_per_evaluation": len(validation),
        "optimizer_steps": optimizer_steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "event_counts": event_counts,
        "event_class_weights": event_weights.detach().cpu().tolist(),
        "augmentation": {
            "scope": "train_only",
            "kind": "deterministic_circular_azimuth_roll",
            "validation_augmented": False,
        },
        "batching": {
            "batch_size": args.batch_size,
            "every_sample_once_per_epoch": True,
            "association_pair_units": True,
            "identity_balanced_association_loss": True,
        },
        "normalization": {
            "metric_distance_m": 30.0,
            "slope_deg": 45.0,
            "curvature_per_m": 0.1,
            "exit_vertical_profile_m": 10.0,
        },
        "best_validation": final_metrics,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
