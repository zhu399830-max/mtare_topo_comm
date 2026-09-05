#!/usr/bin/env python3
"""Adapt only M1D semantic heads on balanced Cano/AEE objective supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.data.aee_head_adaptation import (
    AEETeacherMultitaskDataset,
    BalancedDomainBatchSampler,
)
from mtare_topo.data.phase3_multitask_dataset import CanoV2RMultitaskDataset
from mtare_topo.evaluation.phase3_semantic_metrics import (
    confusion_matrix,
    decode_direction_components,
    match_headings,
    per_class_scores,
)
from mtare_topo.representation.phase3_structural_semantics import (
    StructuralSemanticNet,
    multitask_loss,
)
from mtare_topo.representation.ray_column_dropout import apply_ray_column_dropout
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline


TRAINABLE_PREFIXES = ("direction_head.", "count_head.", "role_head.")
FROZEN_PREFIXES = ("encoder.", "embedding_head.")
ELEVATION_DEG = np.arange(-15, 16, 2, dtype=np.float64)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def selected_state_hashes(
    model: StructuralSemanticNet,
    prefixes: tuple[str, ...],
) -> dict[str, str]:
    return {
        name: tensor_sha256(value)
        for name, value in sorted(model.state_dict().items())
        if name.startswith(prefixes)
    }


def freeze_representation(model: StructuralSemanticNet) -> dict[str, Any]:
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(TRAINABLE_PREFIXES)
        (trainable if parameter.requires_grad else frozen).append(name)
    if not trainable or any(not name.startswith(TRAINABLE_PREFIXES) for name in trainable):
        raise RuntimeError("head-only trainable parameter contract failed")
    if any(name.startswith(FROZEN_PREFIXES) and name not in frozen for name, _ in model.named_parameters()):
        raise RuntimeError("encoder/embedding freeze contract failed")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "student": torch.from_numpy(np.stack([item["student"] for item in samples])),
        "direction_target": torch.from_numpy(
            np.stack([item["direction_target"] for item in samples])
        ),
        "count_target": torch.tensor(
            [item["count_target"] for item in samples], dtype=torch.long
        ),
        "role_target": torch.tensor(
            [item["role_target"] for item in samples], dtype=torch.long
        ),
        "frame_id": [item["frame_id"] for item in samples],
        "headings_robot_deg": [item["headings_robot_deg"] for item in samples],
        "domain": [item.get("domain", "cano") for item in samples],
    }


@torch.no_grad()
def evaluate(
    model: StructuralSemanticNet,
    loader: DataLoader,
    device: torch.device,
    role_weights: torch.Tensor,
    *,
    preserve_arrays: bool = False,
) -> dict[str, Any]:
    model.eval()
    loss_sums = {key: 0.0 for key in ("total", "direction", "count", "role")}
    roles: list[int] = []
    role_predictions: list[int] = []
    counts: list[int] = []
    count_predictions: list[int] = []
    matched = predicted = truth = empty = samples = 0
    angular_errors: list[float] = []
    arrays: dict[str, list[Any]] = {
        "direction_logits": [],
        "direction_target": [],
        "role_logits": [],
        "count_logits": [],
        "role_target": [],
        "count_target": [],
        "frame_id": [],
    }
    for batch in loader:
        student = batch["student"].to(device, non_blocking=True)
        direction = batch["direction_target"].to(device, non_blocking=True)
        count = batch["count_target"].to(device, non_blocking=True)
        role = batch["role_target"].to(device, non_blocking=True)
        outputs = model(student)
        losses = multitask_loss(outputs, direction, count, role, role_weights)
        size = len(student)
        samples += size
        for key, value in losses.items():
            loss_sums[key] += float(value) * size
        role_cpu = role.cpu().numpy()
        count_cpu = count.cpu().numpy()
        roles.extend(role_cpu.tolist())
        counts.extend(count_cpu.tolist())
        role_predictions.extend(outputs["role_logits"].argmax(1).cpu().tolist())
        count_predictions.extend(outputs["count_logits"].argmax(1).cpu().tolist())
        for logits, headings in zip(
            outputs["direction_logits"].cpu().numpy(), batch["headings_robot_deg"]
        ):
            decoded = decode_direction_components(logits, 0.5)
            empty += int(not decoded)
            result = match_headings(decoded, headings, 20.0)
            matched += result[0]
            predicted += result[1]
            truth += result[2]
            angular_errors.extend(result[3])
        if preserve_arrays:
            arrays["direction_logits"].append(
                outputs["direction_logits"].cpu().numpy().astype(np.float16)
            )
            arrays["direction_target"].append(
                direction.cpu().numpy().astype(np.float16)
            )
            arrays["role_logits"].append(
                outputs["role_logits"].cpu().numpy().astype(np.float16)
            )
            arrays["count_logits"].append(
                outputs["count_logits"].cpu().numpy().astype(np.float16)
            )
            arrays["role_target"].extend(role_cpu.tolist())
            arrays["count_target"].extend(count_cpu.tolist())
            arrays["frame_id"].extend(batch["frame_id"])
    role_matrix = confusion_matrix(np.asarray(roles), np.asarray(role_predictions), 3)
    count_matrix = confusion_matrix(np.asarray(counts), np.asarray(count_predictions), 6)
    precision = matched / max(predicted, 1)
    recall = matched / max(truth, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    result = {
        "frames": samples,
        "loss": {key: value / max(samples, 1) for key, value in loss_sums.items()},
        "direction": {
            "threshold": 0.5,
            "matching_tolerance_deg": 20.0,
            "matched": matched,
            "predicted": predicted,
            "truth": truth,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "empty_frames": empty,
            "empty_rate": empty / max(samples, 1),
            "mean_matched_angular_error_deg": (
                float(np.mean(angular_errors)) if angular_errors else None
            ),
        },
        "count": {
            "confusion_matrix": count_matrix.tolist(),
            **per_class_scores(count_matrix),
        },
        "role": {
            "confusion_matrix": role_matrix.tolist(),
            **per_class_scores(role_matrix),
        },
    }
    if preserve_arrays:
        result["arrays"] = {
            "direction_logits": np.concatenate(arrays["direction_logits"]),
            "direction_target": np.concatenate(arrays["direction_target"]),
            "role_logits": np.concatenate(arrays["role_logits"]),
            "count_logits": np.concatenate(arrays["count_logits"]),
            "role_target": np.asarray(arrays["role_target"], dtype=np.int8),
            "count_target": np.asarray(arrays["count_target"], dtype=np.int8),
            "frame_id": np.asarray(arrays["frame_id"], dtype="U128"),
        }
    return result


def evaluate_b0(dataset: AEETeacherMultitaskDataset) -> dict[str, Any]:
    baseline = RangeExitBaseline()
    matched = predicted = truth = empty = 0
    errors: list[float] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        ranges = sample["student"][0] * 50.0
        valid = sample["student"][1]
        output = baseline.predict(ranges, valid, ELEVATION_DEG)
        headings = output["headings_robot_deg"]
        empty += int(not headings)
        result = match_headings(headings, sample["headings_robot_deg"], 20.0)
        matched += result[0]
        predicted += result[1]
        truth += result[2]
        errors.extend(result[3])
    precision = matched / max(predicted, 1)
    recall = matched / max(truth, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "configuration": baseline.config.to_dict(),
        "frames": len(dataset),
        "matched": matched,
        "predicted": predicted,
        "truth": truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "empty_frames": empty,
        "empty_rate": empty / max(len(dataset), 1),
        "mean_matched_angular_error_deg": float(np.mean(errors)) if errors else None,
    }


def role_counts(cano: CanoV2RMultitaskDataset, aee: AEETeacherMultitaskDataset) -> np.ndarray:
    mapping = {"interior": 0, "junction": 1, "terminal": 2}
    counts = np.bincount(
        [mapping[str(item["primary_role"])] for item in cano.records], minlength=3
    ).astype(np.int64)
    for record_index, row in aee._locations:
        _, teacher = aee._load(record_index)
        counts[int(teacher["role_target"][row])] += 1
    if np.any(counts == 0):
        raise RuntimeError(f"balanced training data lacks a role class: {counts.tolist()}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cano-dataset-run", required=True, type=Path)
    parser.add_argument("--aee-sensor-run", required=True, type=Path)
    parser.add_argument("--aee-teacher-run", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if (
        args.seed not in (0, 1, 2)
        or args.epochs != 10
        or args.batch_size != 128
        or not math.isclose(args.learning_rate, 1e-4)
        or not math.isclose(args.weight_decay, 1e-4)
        or args.workers != 0
    ):
        raise ValueError("AEE head-adaptation hyperparameter contract drift")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be frozen before process start")
    checkpoint_path = args.source_checkpoint.resolve()
    if sha256(checkpoint_path) != args.source_checkpoint_sha256:
        raise RuntimeError("source checkpoint identity drift")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("formal AEE head adaptation requires CUDA")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("mode") != "M1D" or int(payload.get("seed", -1)) != args.seed:
        raise RuntimeError("source checkpoint mode/seed identity mismatch")
    model = StructuralSemanticNet()
    model.load_state_dict(payload["model"], strict=True)
    freeze_contract = freeze_representation(model)
    frozen_before = selected_state_hashes(model, FROZEN_PREFIXES)
    trainable_before = selected_state_hashes(model, TRAINABLE_PREFIXES)
    model = model.to(device)

    cano_train = CanoV2RMultitaskDataset(args.cano_dataset_run, "train")
    cano_validation = CanoV2RMultitaskDataset(args.cano_dataset_run, "validation")
    aee_train = AEETeacherMultitaskDataset(
        args.aee_sensor_run, args.aee_teacher_run, "train"
    )
    aee_validation = AEETeacherMultitaskDataset(
        args.aee_sensor_run, args.aee_teacher_run, "validation"
    )
    if len(aee_train) != 3000 or len(aee_validation) != 3000:
        raise RuntimeError("AEE train/validation count drift")
    sampler = BalancedDomainBatchSampler(
        len(cano_train), len(aee_train), args.batch_size, args.seed
    )
    combined = ConcatDataset([cano_train, aee_train])
    train_loader = DataLoader(
        combined,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    cano_validation_loader = DataLoader(
        cano_validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    aee_validation_loader = DataLoader(
        aee_validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    counts = role_counts(cano_train, aee_train)
    role_weights = torch.tensor(
        counts.sum() / (3.0 * counts), dtype=torch.float32, device=device
    )
    probe_samples = [aee_validation[index] for index in range(32)]
    probe = torch.from_numpy(np.stack([item["student"] for item in probe_samples])).to(device)
    probe_frame_ids = [item["frame_id"] for item in probe_samples]
    model.eval()
    with torch.no_grad():
        probe_before = model(probe)["z_role"].detach().cpu().numpy()
    source_cano_validation = evaluate(
        model, cano_validation_loader, device, role_weights
    )
    b0_aee_validation = evaluate_b0(aee_validation)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    augmentation_generator = torch.Generator().manual_seed(200000 + args.seed)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, int] | None = None
    started = time.monotonic()
    optimizer_steps = 0
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        train_sums = {key: 0.0 for key in ("total", "direction", "count", "role")}
        domain_samples = {"cano": 0, "aee": 0}
        seen = 0
        augmentation_totals = {"samples": 0, "augmented_samples": 0, "phase_counts": [0] * 10}
        for batch in train_loader:
            cano_in_batch = batch["domain"].count("cano")
            aee_in_batch = batch["domain"].count("aee")
            if cano_in_batch != aee_in_batch:
                raise RuntimeError("domain-balanced batch contract failed")
            domain_samples["cano"] += cano_in_batch
            domain_samples["aee"] += aee_in_batch
            student, augmentation = apply_ray_column_dropout(
                batch["student"], augmentation_generator, probability=0.5, period=10
            )
            student = student.to(device, non_blocking=True)
            direction = batch["direction_target"].to(device, non_blocking=True)
            count = batch["count_target"].to(device, non_blocking=True)
            role = batch["role_target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(student)
            losses = multitask_loss(outputs, direction, count, role, role_weights)
            losses["total"].backward()
            if any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if name.startswith(FROZEN_PREFIXES)
            ):
                raise RuntimeError("frozen encoder/embedding received gradients")
            optimizer.step()
            optimizer_steps += 1
            size = len(student)
            seen += size
            for key, value in losses.items():
                train_sums[key] += float(value.detach()) * size
            augmentation_totals["samples"] += augmentation["samples"]
            augmentation_totals["augmented_samples"] += augmentation["augmented_samples"]
            augmentation_totals["phase_counts"] = [
                first + second
                for first, second in zip(
                    augmentation_totals["phase_counts"], augmentation["phase_counts"]
                )
            ]
        if domain_samples != {"cano": 3000, "aee": 3000} or seen != 6000:
            raise RuntimeError(f"epoch domain/sample count drift: {domain_samples}/{seen}")
        aee_metrics = evaluate(
            model, aee_validation_loader, device, role_weights
        )
        cano_metrics = evaluate(
            model, cano_validation_loader, device, role_weights
        )
        selection_key = (
            float(aee_metrics["direction"]["f1"]),
            float(cano_metrics["direction"]["f1"]),
            float(aee_metrics["role"]["macro_f1_present"]),
            -epoch,
        )
        record = {
            "epoch": epoch,
            "train_samples": seen,
            "domain_samples": domain_samples,
            "train_loss": {key: value / seen for key, value in train_sums.items()},
            "augmentation": augmentation_totals,
            "aee_validation": aee_metrics,
            "cano_validation": cano_metrics,
            "selection_key": list(selection_key),
        }
        history.append(record)
        with (output / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "mode": "M1D_AEE_HEAD_ADAPTED_V1",
                    "source_checkpoint_sha256": args.source_checkpoint_sha256,
                    "trainable_parameter_names": freeze_contract["trainable_parameters"],
                    "frozen_parameter_hashes": frozen_before,
                    "config": {
                        "epochs": 10,
                        "batch_size": 128,
                        "learning_rate": 1e-4,
                        "weight_decay": 1e-4,
                        "balanced_cano_aee_per_epoch": [3000, 3000],
                        "augmentation": "original_train_only_ray_column_dropout_p0.5_period10",
                    },
                },
                output / "best.pt",
            )
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    final_aee = evaluate(
        model, aee_validation_loader, device, role_weights, preserve_arrays=True
    )
    validation_arrays = final_aee.pop("arrays")
    final_cano = evaluate(model, cano_validation_loader, device, role_weights)
    model.eval()
    with torch.no_grad():
        probe_after = model(probe)["z_role"].detach().cpu().numpy()
    probe_max_difference = float(np.max(np.abs(probe_before - probe_after)))
    model_cpu = model.to("cpu")
    frozen_after = selected_state_hashes(model_cpu, FROZEN_PREFIXES)
    trainable_after = selected_state_hashes(model_cpu, TRAINABLE_PREFIXES)
    frozen_identity = frozen_before == frozen_after
    trainable_changed = any(
        trainable_before[name] != trainable_after[name] for name in trainable_before
    )
    gate = {
        "aee_direction_vs_b0": final_aee["direction"]["f1"] >= b0_aee_validation["f1"],
        "aee_empty_rate": final_aee["direction"]["empty_rate"] <= 0.05,
        "aee_count_macro_f1": final_aee["count"]["macro_f1_present"] >= 0.70,
        "aee_role_macro_f1": final_aee["role"]["macro_f1_present"] >= 0.70,
        "cano_direction_retention": final_cano["direction"]["f1"] >= source_cano_validation["direction"]["f1"] - 0.02,
        "frozen_tensor_identity": frozen_identity,
        "fixed_probe_z_role_identity": probe_max_difference == 0.0,
        "semantic_heads_changed": trainable_changed,
    }
    status = "PASS_AEE_HEAD_ADAPTATION_SEED_V1" if all(gate.values()) else "FAIL_AEE_HEAD_ADAPTATION_SEED_V1"
    np.savez_compressed(output / "aee_validation_outputs.npz", **validation_arrays)
    np.savez_compressed(
        output / "frozen_representation_probe.npz",
        frame_id=np.asarray(probe_frame_ids, dtype="U128"),
        z_role_before=probe_before.astype(np.float32),
        z_role_after=probe_after.astype(np.float32),
    )
    (output / "frozen_tensor_hashes.json").write_text(
        json.dumps({"before": frozen_before, "after": frozen_after}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "aee_head_adaptation_seed_summary_v1",
        "status": status,
        "seed": args.seed,
        "source_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "best_epoch": int(best["epoch"]),
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "train_samples_per_epoch": {"cano": 3000, "aee": 3000},
        "aee_validation_frames": len(aee_validation),
        "cano_validation_frames": len(cano_validation),
        "source_cano_validation": source_cano_validation,
        "adapted_cano_validation": final_cano,
        "b0_aee_validation": b0_aee_validation,
        "adapted_aee_validation": final_aee,
        "freeze_contract": freeze_contract,
        "frozen_tensor_identity": frozen_identity,
        "fixed_probe_z_role_max_absolute_difference": probe_max_difference,
        "semantic_heads_changed": trainable_changed,
        "gate": gate,
        "best_checkpoint_sha256": sha256(output / "best.pt"),
        "duration_seconds": time.monotonic() - started,
        "strict_test_frames_read": 0,
        "cano_c09_validation_frames_read": len(cano_validation),
        "c10_frames_read": 0,
        "later_sealed_world_frames_read": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
