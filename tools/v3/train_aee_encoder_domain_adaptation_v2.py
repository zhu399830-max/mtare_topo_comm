#!/usr/bin/env python3
"""Adapt the full M1D model with canonical AEE targets and mask-matched Cano views."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.data.aee_head_adaptation import (
    AEETeacherMultitaskDataset,
    BalancedDomainBatchSampler,
)
from mtare_topo.data.phase3_multitask_dataset import CanoV2RMultitaskDataset
from mtare_topo.representation.aee_mask_matched_views import expand_domain_adaptation_views
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet

from train_aee_head_adaptation_v1 import (
    collate,
    evaluate,
    evaluate_b0,
    role_counts,
    seed_everything,
    selected_state_hashes,
)


MODEL_PREFIXES = (
    "encoder.",
    "embedding_head.",
    "direction_head.",
    "count_head.",
    "role_head.",
)
ENCODER_PREFIXES = ("encoder.",)
EMBEDDING_PREFIXES = ("embedding_head.",)
SEMANTIC_HEAD_PREFIXES = ("direction_head.", "count_head.", "role_head.")


def make_full_model_trainable(model: StructuralSemanticNet) -> dict[str, Any]:
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if not name.startswith(MODEL_PREFIXES):
            raise RuntimeError(f"unexpected model parameter outside declared contract: {name}")
        parameter.requires_grad = True
        trainable.append(name)
    if not trainable or any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("full-model trainable parameter contract failed")
    represented = {prefix for prefix in MODEL_PREFIXES if any(name.startswith(prefix) for name in trainable)}
    if represented != set(MODEL_PREFIXES):
        raise RuntimeError("declared full-model parameter group is empty")
    return {"trainable_parameters": trainable, "frozen_parameters": []}


class DomainAdaptationCollator:
    """Epoch-addressed deterministic expansion of balanced source batches."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("epoch must be positive")
        self.epoch = int(epoch)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.epoch < 1:
            raise RuntimeError("set_epoch must be called before collation")
        expanded, provenance = expand_domain_adaptation_views(
            samples, seed=300000 + 1000 * self.seed + self.epoch
        )
        batch = collate(expanded)
        batch["loss_weight"] = torch.tensor(
            [float(item["loss_weight"]) for item in expanded], dtype=torch.float32
        )
        batch["view_id"] = [str(item["view_id"]) for item in expanded]
        batch["mask_pairing"] = provenance
        return batch


def weighted_multitask_loss(
    outputs: dict[str, torch.Tensor],
    direction_target: torch.Tensor,
    count_target: torch.Tensor,
    role_target: torch.Tensor,
    loss_weight: torch.Tensor,
    role_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if loss_weight.ndim != 1 or len(loss_weight) != len(direction_target):
        raise ValueError("loss weights must contain one value per view")
    if not torch.isfinite(loss_weight).all() or torch.any(loss_weight <= 0):
        raise ValueError("loss weights must be positive and finite")
    normalizer = loss_weight.sum()
    direction_per = F.binary_cross_entropy_with_logits(
        outputs["direction_logits"], direction_target, reduction="none"
    ).mean(dim=1)
    count_per = F.cross_entropy(outputs["count_logits"], count_target, reduction="none")
    role_per = F.cross_entropy(
        outputs["role_logits"], role_target, weight=role_weights, reduction="none"
    )
    direction = torch.sum(direction_per * loss_weight) / normalizer
    count = torch.sum(count_per * loss_weight) / normalizer
    role_normalizer = (
        normalizer
        if role_weights is None
        else torch.sum(loss_weight * role_weights[role_target])
    )
    role = torch.sum(role_per * loss_weight) / role_normalizer
    total = direction + 0.25 * count + 0.25 * role
    return {"total": total, "direction": direction, "count": count, "role": role}


def attach_count_contract(metrics: dict[str, Any]) -> dict[str, Any]:
    """Attach PLAN-defined common-class Gate and rare-class diagnostics."""

    count = metrics["count"]
    f1 = np.asarray(count["f1"], dtype=np.float64)
    support = np.asarray(count["support"], dtype=np.int64)
    common_present = support[:4] > 0
    count["gate_branch_counts"] = [1, 2, 3, 4]
    count["macro_f1_count_1_to_4"] = (
        float(f1[:4][common_present].mean()) if np.any(common_present) else 0.0
    )
    count["rare_branch_counts_diagnostic_only"] = {
        "5": {"support": int(support[4]), "f1": float(f1[4])},
        "6": {"support": int(support[5]), "f1": float(f1[5])},
    }
    return metrics


def all_state_tensors_finite(model: StructuralSemanticNet) -> bool:
    return all(torch.isfinite(value).all().item() for value in model.state_dict().values())


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
        raise ValueError("AEE encoder-domain-adaptation hyperparameter contract drift")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be frozen before process start")
    checkpoint_path = args.source_checkpoint.resolve()
    if sha256(checkpoint_path) != args.source_checkpoint_sha256:
        raise RuntimeError("source checkpoint identity drift")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal AEE encoder adaptation requires CUDA")
    device = torch.device("cuda")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("mode") != "M1D" or int(payload.get("seed", -1)) != args.seed:
        raise RuntimeError("source checkpoint mode/seed identity mismatch")
    model = StructuralSemanticNet()
    model.load_state_dict(payload["model"], strict=True)
    trainable_contract = make_full_model_trainable(model)
    before = selected_state_hashes(model, MODEL_PREFIXES)
    encoder_before = selected_state_hashes(model, ENCODER_PREFIXES)
    embedding_before = selected_state_hashes(model, EMBEDDING_PREFIXES)
    heads_before = selected_state_hashes(model, SEMANTIC_HEAD_PREFIXES)
    model = model.to(device)

    cano_train = CanoV2RMultitaskDataset(args.cano_dataset_run, "train")
    cano_validation = CanoV2RMultitaskDataset(args.cano_dataset_run, "validation")
    aee_train = AEETeacherMultitaskDataset(
        args.aee_sensor_run,
        args.aee_teacher_run,
        "train",
        direction_encoding="cano_gaussian_component_centers",
    )
    aee_validation = AEETeacherMultitaskDataset(
        args.aee_sensor_run,
        args.aee_teacher_run,
        "validation",
        direction_encoding="cano_gaussian_component_centers",
    )
    if len(aee_train) != 3000 or len(aee_validation) != 3000:
        raise RuntimeError("AEE train/validation count drift")
    sampler = BalancedDomainBatchSampler(len(cano_train), len(aee_train), args.batch_size, args.seed)
    collator = DomainAdaptationCollator(args.seed)
    train_loader = DataLoader(
        ConcatDataset([cano_train, aee_train]),
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )
    validation_collate = collate
    cano_validation_loader = DataLoader(
        cano_validation, batch_size=128, shuffle=False, num_workers=0,
        collate_fn=validation_collate, pin_memory=True,
    )
    aee_validation_loader = DataLoader(
        aee_validation, batch_size=128, shuffle=False, num_workers=0,
        collate_fn=validation_collate, pin_memory=True,
    )
    counts = role_counts(cano_train, aee_train)
    role_weights = torch.tensor(counts.sum() / (3.0 * counts), dtype=torch.float32, device=device)
    source_cano = attach_count_contract(evaluate(model, cano_validation_loader, device, role_weights))
    b0_aee = evaluate_b0(aee_validation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, int] | None = None
    optimizer_steps = 0
    started = time.monotonic()
    for epoch in range(1, 11):
        sampler.set_epoch(epoch)
        collator.set_epoch(epoch)
        model.train()
        weighted_seen = 0.0
        view_counts = {"cano_dense": 0, "cano_mask_matched": 0, "aee": 0}
        train_sums = {key: 0.0 for key in ("total", "direction", "count", "role")}
        pairings: list[dict[str, str]] = []
        for batch in train_loader:
            for domain in batch["domain"]:
                view_counts[domain] += 1
            pairings.extend(batch["mask_pairing"])
            student = batch["student"].to(device, non_blocking=True)
            direction = batch["direction_target"].to(device, non_blocking=True)
            count = batch["count_target"].to(device, non_blocking=True)
            role = batch["role_target"].to(device, non_blocking=True)
            weights = batch["loss_weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = weighted_multitask_loss(
                model(student), direction, count, role, weights, role_weights
            )
            losses["total"].backward()
            optimizer.step()
            optimizer_steps += 1
            batch_weight = float(weights.sum())
            weighted_seen += batch_weight
            for key, value in losses.items():
                train_sums[key] += float(value.detach()) * batch_weight
        expected_views = {"cano_dense": 3000, "cano_mask_matched": 3000, "aee": 3000}
        if view_counts != expected_views or weighted_seen != 6000.0 or len(pairings) != 3000:
            raise RuntimeError(f"epoch source/view count drift: {view_counts}/{weighted_seen}/{len(pairings)}")
        pairing_path = output / f"mask_pairings_epoch_{epoch:02d}.json"
        pairing_path.write_text(
            json.dumps(
                {
                    "schema_version": "aee_mask_pairings_epoch_v1",
                    "seed": args.seed,
                    "epoch": epoch,
                    "pairing_seed": 300000 + 1000 * args.seed + epoch,
                    "pairs": pairings,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        aee_metrics = attach_count_contract(evaluate(model, aee_validation_loader, device, role_weights))
        cano_metrics = attach_count_contract(evaluate(model, cano_validation_loader, device, role_weights))
        selection_key = (
            float(aee_metrics["direction"]["f1"]),
            float(cano_metrics["direction"]["f1"]),
            float(aee_metrics["role"]["macro_f1_present"]),
            -epoch,
        )
        record = {
            "epoch": epoch,
            "independent_samples": {"cano": 3000, "aee": 3000},
            "training_views": view_counts,
            "effective_domain_weight": {"cano": 3000.0, "aee": 3000.0},
            "mask_pairing_file": pairing_path.name,
            "mask_pairing_sha256": sha256(pairing_path),
            "train_loss": {key: value / weighted_seen for key, value in train_sums.items()},
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
                    "mode": "M1D_AEE_ENCODER_DOMAIN_ADAPTED_V2",
                    "source_checkpoint_sha256": args.source_checkpoint_sha256,
                    "trainable_parameter_names": trainable_contract["trainable_parameters"],
                    "config": {
                        "epochs": 10,
                        "batch_size_independent": 128,
                        "learning_rate": 1e-4,
                        "weight_decay": 1e-4,
                        "aee_direction_encoding": "cano_gaussian_component_centers_sigma_3deg",
                        "independent_samples_per_epoch": {"cano": 3000, "aee": 3000},
                        "augmented_cano_views_per_epoch": 3000,
                        "view_weights": {"cano_dense": 0.5, "cano_mask_matched": 0.5, "aee": 1.0},
                    },
                },
                output / "best.pt",
            )
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    final_aee = attach_count_contract(
        evaluate(model, aee_validation_loader, device, role_weights, preserve_arrays=True)
    )
    validation_arrays = final_aee.pop("arrays")
    final_cano = attach_count_contract(evaluate(model, cano_validation_loader, device, role_weights))
    model_cpu = model.to("cpu")
    after = selected_state_hashes(model_cpu, MODEL_PREFIXES)
    encoder_changed = encoder_before != selected_state_hashes(model_cpu, ENCODER_PREFIXES)
    embedding_changed = embedding_before != selected_state_hashes(model_cpu, EMBEDDING_PREFIXES)
    heads_changed = heads_before != selected_state_hashes(model_cpu, SEMANTIC_HEAD_PREFIXES)
    finite = all_state_tensors_finite(model_cpu)
    gate = {
        "aee_direction_vs_b0": final_aee["direction"]["f1"] >= b0_aee["f1"],
        "aee_empty_rate": final_aee["direction"]["empty_rate"] <= 0.05,
        "aee_count_1_to_4_macro_f1": final_aee["count"]["macro_f1_count_1_to_4"] >= 0.70,
        "aee_role_macro_f1": final_aee["role"]["macro_f1_present"] >= 0.70,
        "cano_direction_retention": final_cano["direction"]["f1"] >= source_cano["direction"]["f1"] - 0.02,
        "all_model_tensors_finite": finite,
        "encoder_changed": encoder_changed,
        "embedding_changed": embedding_changed,
        "semantic_heads_changed": heads_changed,
    }
    status = (
        "PASS_AEE_ENCODER_DOMAIN_ADAPTATION_SEED_V2"
        if all(gate.values())
        else "FAIL_AEE_ENCODER_DOMAIN_ADAPTATION_SEED_V2"
    )
    np.savez_compressed(output / "aee_validation_outputs.npz", **validation_arrays)
    (output / "model_tensor_hashes.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "aee_encoder_domain_adaptation_seed_summary_v2",
        "status": status,
        "seed": args.seed,
        "source_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "best_epoch": int(best["epoch"]),
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "independent_train_samples_per_epoch": {"cano": 3000, "aee": 3000},
        "augmented_cano_views_per_epoch": 3000,
        "aee_validation_frames": len(aee_validation),
        "cano_validation_frames": len(cano_validation),
        "source_cano_validation": source_cano,
        "adapted_cano_validation": final_cano,
        "b0_aee_validation": b0_aee,
        "adapted_aee_validation": final_aee,
        "trainable_contract": trainable_contract,
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
