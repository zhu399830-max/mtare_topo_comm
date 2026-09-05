#!/usr/bin/env python3
"""Train one frozen seed on the V1R4 corrective Cano/AEE dataset."""

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
from mtare_topo.data.aee_corrective_training import (
    AllSamplesBatchSampler,
    CorrectiveAEEMultitaskDataset,
    CorrectiveCanoMultitaskDataset,
    MaskMatchedCanoValidationDataset,
    _mask_index,
    corrective_role_counts,
    mask_matched_cano_sample,
)
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet
from train_aee_encoder_domain_adaptation_v2 import (
    EMBEDDING_PREFIXES,
    ENCODER_PREFIXES,
    MODEL_PREFIXES,
    SEMANTIC_HEAD_PREFIXES,
    all_state_tensors_finite,
    attach_count_contract,
    make_full_model_trainable,
    weighted_multitask_loss,
)
from train_aee_head_adaptation_v1 import (
    collate,
    evaluate,
    evaluate_b0,
    seed_everything,
    selected_state_hashes,
)


VALIDATION_MASK_SEED = 20260822
AEE_EFFECTIVE_WEIGHT = 5.0


class CorrectiveTrainingCollator:
    """Expand every Cano sample and pair it with a stable train-only AEE mask."""

    def __init__(self, masks: CorrectiveAEEMultitaskDataset, seed: int):
        self.masks, self.seed, self.epoch = masks, int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("epoch must be positive")
        self.epoch = int(epoch)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.epoch < 1:
            raise RuntimeError("set_epoch must be called before collation")
        expanded: list[dict[str, Any]] = []
        provenance: list[dict[str, str]] = []
        pairing_seed = 300000 + 1000 * self.seed + self.epoch
        for item in samples:
            if item["domain"] == "cano":
                dense = dict(item)
                dense.update(domain="cano_dense", view_id=f"{item['frame_id']}:dense", loss_weight=np.float32(0.5))
                mask = self.masks[_mask_index(str(item["frame_id"]), pairing_seed, len(self.masks))]
                matched = mask_matched_cano_sample(item, mask)
                matched.update(view_id=f"{item['frame_id']}:aee_mask:{mask['frame_id']}", loss_weight=np.float32(0.5))
                expanded.extend((dense, matched))
                provenance.append({"cano_frame_id": str(item["frame_id"]), "aee_mask_frame_id": str(mask["frame_id"]), "view_id": str(matched["view_id"])})
            elif item["domain"] == "aee":
                raw = dict(item)
                raw.update(view_id=f"{item['frame_id']}:raw", loss_weight=np.float32(AEE_EFFECTIVE_WEIGHT))
                expanded.append(raw)
            else:
                raise ValueError(f"unsupported training domain: {item['domain']}")
        batch = collate(expanded)
        batch["loss_weight"] = torch.tensor([float(item["loss_weight"]) for item in expanded], dtype=torch.float32)
        batch["view_id"] = [str(item["view_id"]) for item in expanded]
        batch["mask_pairing"] = provenance
        return batch


def _gate_metrics(adapted_dense: dict[str, Any], adapted_sparse: dict[str, Any], source_dense: dict[str, Any], source_sparse: dict[str, Any], b0_sparse: dict[str, Any], finite: bool, encoder_changed: bool, embedding_changed: bool, heads_changed: bool) -> dict[str, bool]:
    return {
        "sparse_direction_improvement_0p05": adapted_sparse["direction"]["f1"] >= source_sparse["direction"]["f1"] + 0.05,
        "sparse_direction_vs_b0": adapted_sparse["direction"]["f1"] >= b0_sparse["f1"],
        "sparse_empty_rate": adapted_sparse["direction"]["empty_rate"] <= 0.05,
        "sparse_count_1_to_4_macro_f1": adapted_sparse["count"]["macro_f1_count_1_to_4"] >= 0.70,
        "sparse_role_macro_f1": adapted_sparse["role"]["macro_f1_present"] >= 0.70,
        "dense_direction_retention": adapted_dense["direction"]["f1"] >= source_dense["direction"]["f1"] - 0.02,
        "all_model_tensors_finite": finite,
        "encoder_changed": encoder_changed,
        "embedding_changed": embedding_changed,
        "semantic_heads_changed": heads_changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", required=True, type=Path)
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
    if args.seed not in (0, 1, 2) or args.epochs != 10 or args.batch_size != 128 or not math.isclose(args.learning_rate, 1e-4) or not math.isclose(args.weight_decay, 1e-4) or args.workers != 0:
        raise ValueError("corrective full-encoder hyperparameter contract drift")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be frozen before process start")
    checkpoint_path = args.source_checkpoint.resolve()
    if sha256(checkpoint_path) != args.source_checkpoint_sha256:
        raise RuntimeError("source checkpoint identity drift")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal corrective training requires CUDA")
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

    dataset_run = args.dataset_run.resolve()
    cano_train = CorrectiveCanoMultitaskDataset(dataset_run, "train")
    cano_validation = CorrectiveCanoMultitaskDataset(dataset_run, "validation")
    aee_train = CorrectiveAEEMultitaskDataset(dataset_run)
    sparse_validation = MaskMatchedCanoValidationDataset(cano_validation, aee_train, VALIDATION_MASK_SEED)
    if (len(cano_train), len(aee_train), len(cano_validation), len(sparse_validation)) != (5000, 1000, 5000, 5000):
        raise RuntimeError("corrective dataset count drift")
    train_source = ConcatDataset([cano_train, aee_train])
    sampler = AllSamplesBatchSampler(len(train_source), args.batch_size, args.seed)
    training_collator = CorrectiveTrainingCollator(aee_train, args.seed)
    train_loader = DataLoader(train_source, batch_sampler=sampler, num_workers=0, collate_fn=training_collator, pin_memory=True)
    dense_loader = DataLoader(cano_validation, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)
    sparse_loader = DataLoader(sparse_validation, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)
    aee_loader = DataLoader(aee_train, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)
    counts = corrective_role_counts(cano_train, aee_train)
    role_weights = torch.tensor(counts.sum() / (3.0 * counts), dtype=torch.float32, device=device)
    source_dense = attach_count_contract(evaluate(model, dense_loader, device, role_weights))
    source_sparse = attach_count_contract(evaluate(model, sparse_loader, device, role_weights))
    b0_sparse = evaluate_b0(sparse_validation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float, int] | None = None
    optimizer_steps = 0
    started = time.monotonic()
    for epoch in range(1, 11):
        sampler.set_epoch(epoch)
        training_collator.set_epoch(epoch)
        model.train()
        seen_ids: set[str] = set()
        view_counts = {"cano_dense": 0, "cano_mask_matched": 0, "aee": 0}
        weighted_seen = 0.0
        train_sums = {key: 0.0 for key in ("total", "direction", "count", "role")}
        pairings: list[dict[str, str]] = []
        for batch in train_loader:
            for frame_id, domain in zip(batch["frame_id"], batch["domain"]):
                view_counts[domain] += 1
                if domain != "cano_mask_matched":
                    seen_ids.add(str(frame_id))
            pairings.extend(batch["mask_pairing"])
            student = batch["student"].to(device, non_blocking=True)
            direction = batch["direction_target"].to(device, non_blocking=True)
            count = batch["count_target"].to(device, non_blocking=True)
            role = batch["role_target"].to(device, non_blocking=True)
            weights = batch["loss_weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = weighted_multitask_loss(model(student), direction, count, role, weights, role_weights)
            losses["total"].backward()
            optimizer.step()
            optimizer_steps += 1
            batch_weight = float(weights.sum())
            weighted_seen += batch_weight
            for key, value in losses.items():
                train_sums[key] += float(value.detach()) * batch_weight
        expected_views = {"cano_dense": 5000, "cano_mask_matched": 5000, "aee": 1000}
        if view_counts != expected_views or len(seen_ids) != 6000 or weighted_seen != 10000.0 or len(pairings) != 5000:
            raise RuntimeError(f"epoch sample/view contract drift: {view_counts}/{len(seen_ids)}/{weighted_seen}/{len(pairings)}")
        pairing_path = output / f"mask_pairings_epoch_{epoch:02d}.json"
        pairing_path.write_text(json.dumps({"schema_version": "aee_corrective_mask_pairings_epoch_v1", "seed": args.seed, "epoch": epoch, "pairs": pairings}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        dense = attach_count_contract(evaluate(model, dense_loader, device, role_weights))
        sparse = attach_count_contract(evaluate(model, sparse_loader, device, role_weights))
        selection_key = (float(sparse["direction"]["f1"]), float(sparse["count"]["macro_f1_count_1_to_4"]), float(sparse["role"]["macro_f1_present"]), float(dense["direction"]["f1"]), -epoch)
        record = {"epoch": epoch, "independent_samples": {"cano": 5000, "aee": 1000}, "training_views": view_counts, "effective_domain_weight": {"cano": 5000.0, "aee": 5000.0}, "mask_pairing_file": pairing_path.name, "mask_pairing_sha256": sha256(pairing_path), "train_loss": {key: value / weighted_seen for key, value in train_sums.items()}, "cano_dense_validation": dense, "cano_sparse_validation": sparse, "selection_key": list(selection_key)}
        history.append(record)
        with (output / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed, "mode": "M1D_AEE_CORRECTIVE_FULL_ENCODER_V3", "source_checkpoint_sha256": args.source_checkpoint_sha256, "trainable_parameter_names": trainable_contract["trainable_parameters"], "config": {"epochs": 10, "batch_size_independent": 128, "learning_rate": 1e-4, "weight_decay": 1e-4, "independent_samples_per_epoch": {"cano": 5000, "aee": 1000}, "view_weights": {"cano_dense": 0.5, "cano_mask_matched": 0.5, "aee": 5.0}, "validation_mask_seed": VALIDATION_MASK_SEED}}, output / "best.pt")
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    final_dense = attach_count_contract(evaluate(model, dense_loader, device, role_weights, preserve_arrays=True))
    dense_arrays = final_dense.pop("arrays")
    final_sparse = attach_count_contract(evaluate(model, sparse_loader, device, role_weights, preserve_arrays=True))
    sparse_arrays = final_sparse.pop("arrays")
    aee_train_diagnostic = attach_count_contract(evaluate(model, aee_loader, device, role_weights))
    model_cpu = model.to("cpu")
    after = selected_state_hashes(model_cpu, MODEL_PREFIXES)
    encoder_changed = encoder_before != selected_state_hashes(model_cpu, ENCODER_PREFIXES)
    embedding_changed = embedding_before != selected_state_hashes(model_cpu, EMBEDDING_PREFIXES)
    heads_changed = heads_before != selected_state_hashes(model_cpu, SEMANTIC_HEAD_PREFIXES)
    finite = all_state_tensors_finite(model_cpu)
    gate = _gate_metrics(final_dense, final_sparse, source_dense, source_sparse, b0_sparse, finite, encoder_changed, embedding_changed, heads_changed)
    np.savez_compressed(output / "cano_dense_validation_outputs.npz", **dense_arrays)
    np.savez_compressed(output / "cano_sparse_validation_outputs.npz", **sparse_arrays)
    (output / "model_tensor_hashes.json").write_text(json.dumps({"before": before, "after": after}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "aee_corrective_full_encoder_seed_summary_v3",
        "status": "COMPLETED_AEE_CORRECTIVE_FULL_ENCODER_SEED_V3",
        "scientific_gate_passed": all(gate.values()),
        "seed": args.seed, "source_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)), "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "best_epoch": int(best["epoch"]), "epochs_completed": len(history), "optimizer_steps": optimizer_steps,
        "independent_train_samples_per_epoch": {"cano": 5000, "aee": 1000}, "training_views_per_epoch": {"cano_dense": 5000, "cano_mask_matched": 5000, "aee": 1000},
        "cano_dense_validation_frames": 5000, "cano_sparse_validation_frames": 5000, "aee_train_diagnostic_frames": 1000,
        "source_cano_dense_validation": source_dense, "source_cano_sparse_validation": source_sparse, "b0_cano_sparse_validation": b0_sparse,
        "adapted_cano_dense_validation": final_dense, "adapted_cano_sparse_validation": final_sparse, "aee_train_fit_diagnostic_only": aee_train_diagnostic,
        "trainable_contract": trainable_contract, "gate": gate, "best_checkpoint_sha256": sha256(output / "best.pt"), "duration_seconds": time.monotonic() - started,
        "c09_frames_read": 0, "c10_frames_read": 0, "formal_benchmark_frames_read": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
