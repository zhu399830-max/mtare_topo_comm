#!/usr/bin/env python3
"""Run V1R4 corrective training with balanced direction loss and frozen representation."""

from __future__ import annotations

import json

import torch
from torch.nn import functional as F

from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.representation.corrective_direction_loss import balanced_soft_direction_bce
import train_aee_corrective_full_encoder_v3 as base


def make_heads_trainable(model):
    trainable, frozen = [], []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(base.SEMANTIC_HEAD_PREFIXES)
        (trainable if parameter.requires_grad else frozen).append(name)
    if not trainable or not frozen:
        raise RuntimeError("balanced-head trainable/frozen parameter contract failed")
    if any(not name.startswith(base.SEMANTIC_HEAD_PREFIXES) for name in trainable):
        raise RuntimeError("non-semantic parameter entered balanced-head training")
    if any(not name.startswith(base.ENCODER_PREFIXES + base.EMBEDDING_PREFIXES) for name in frozen):
        raise RuntimeError("unexpected frozen parameter in balanced-head training")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def balanced_weighted_multitask_loss(outputs, direction_target, count_target, role_target, loss_weight, role_weights=None):
    if loss_weight.ndim != 1 or len(loss_weight) != len(direction_target):
        raise ValueError("loss weights must contain one value per view")
    if not torch.isfinite(loss_weight).all() or torch.any(loss_weight <= 0):
        raise ValueError("loss weights must be positive and finite")
    positive_mass = direction_target.sum(dim=1)
    negative_mass = (1.0 - direction_target).sum(dim=1)
    if torch.any(positive_mass <= 0) or torch.any(negative_mass <= 0):
        raise ValueError("every direction target must contain positive and negative mass")
    positive = -(direction_target * F.logsigmoid(outputs["direction_logits"])).sum(dim=1) / positive_mass
    negative = -((1.0 - direction_target) * F.logsigmoid(-outputs["direction_logits"])).sum(dim=1) / negative_mass
    direction_per = 0.5 * (positive + negative)
    count_per = F.cross_entropy(outputs["count_logits"], count_target, reduction="none")
    role_per = F.cross_entropy(outputs["role_logits"], role_target, weight=role_weights, reduction="none")
    normalizer = loss_weight.sum()
    direction = torch.sum(direction_per * loss_weight) / normalizer
    count = torch.sum(count_per * loss_weight) / normalizer
    role_normalizer = normalizer if role_weights is None else torch.sum(loss_weight * role_weights[role_target])
    role = torch.sum(role_per * loss_weight) / role_normalizer
    total = direction + 0.25 * count + 0.25 * role
    return {"total": total, "direction": direction, "count": count, "role": role}


def balanced_head_gate(adapted_dense, adapted_sparse, source_dense, source_sparse, b0_sparse, finite, encoder_changed, embedding_changed, heads_changed):
    return {
        "sparse_direction_improvement_0p05": adapted_sparse["direction"]["f1"] >= source_sparse["direction"]["f1"] + 0.05,
        "sparse_direction_vs_b0": adapted_sparse["direction"]["f1"] >= b0_sparse["f1"],
        "sparse_empty_rate": adapted_sparse["direction"]["empty_rate"] <= 0.05,
        "sparse_count_1_to_4_macro_f1": adapted_sparse["count"]["macro_f1_count_1_to_4"] >= 0.70,
        "sparse_role_macro_f1": adapted_sparse["role"]["macro_f1_present"] >= 0.70,
        "dense_direction_retention": adapted_dense["direction"]["f1"] >= source_dense["direction"]["f1"] - 0.02,
        "all_model_tensors_finite": finite,
        "encoder_identity": not encoder_changed,
        "embedding_identity": not embedding_changed,
        "semantic_heads_changed": heads_changed,
    }


def main() -> int:
    base.make_full_model_trainable = make_heads_trainable
    base.weighted_multitask_loss = balanced_weighted_multitask_loss
    base._gate_metrics = balanced_head_gate
    code = base.main()
    # The base owns the full deterministic data/evaluation loop. Relabel only
    # the newly-created V4 artifact after it has completed successfully.
    import sys
    output = __import__("pathlib").Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
    checkpoint_path = output / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["mode"] = "M1D_AEE_CORRECTIVE_BALANCED_HEADS_V4"
    checkpoint["config"]["direction_loss"] = "per_frame_equal_positive_negative_soft_mass_v1"
    checkpoint["config"]["representation"] = "encoder_and_embedding_frozen_exact"
    torch.save(checkpoint, checkpoint_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = "aee_corrective_balanced_heads_seed_summary_v4"
    summary["status"] = "COMPLETED_AEE_CORRECTIVE_BALANCED_HEADS_SEED_V4"
    summary["best_checkpoint_sha256"] = sha256(checkpoint_path)
    summary["direction_loss"] = "per_frame_equal_positive_negative_soft_mass_v1"
    summary["representation_contract"] = "encoder_and_embedding_frozen_exact"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
