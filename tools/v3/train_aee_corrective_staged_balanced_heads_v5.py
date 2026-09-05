#!/usr/bin/env python3
"""V5 staged balanced heads: preserve epoch-1 direction, then fit count/role."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mtare_topo.data.aee_corrective_training import CorrectiveAEEMultitaskDataset, CorrectiveCanoMultitaskDataset
from mtare_topo.data.aee_domain_adaptation import sha256
import train_aee_corrective_full_encoder_v3 as base


COUNT_EFFECTIVE_MASS = np.asarray([1244, 6793, 1329, 504, 105, 25], dtype=np.int64)
ROLE_EFFECTIVE_MASS = np.asarray([6495, 2105, 1400], dtype=np.int64)
STEPS_PER_EPOCH = 47
_DIRECTION_PARAMETERS: list[torch.nn.Parameter] = []


def gate_class_weights(mass: np.ndarray, gate_classes: int) -> np.ndarray:
    value = np.asarray(mass, dtype=np.float64)
    if value.ndim != 1 or gate_classes < 1 or gate_classes > len(value) or np.any(value <= 0):
        raise ValueError("positive one-dimensional class mass and valid gate class count required")
    weights = np.ones_like(value)
    gate_total = float(value[:gate_classes].sum())
    weights[:gate_classes] = gate_total / (gate_classes * value[:gate_classes])
    return weights


COUNT_CLASS_WEIGHTS = gate_class_weights(COUNT_EFFECTIVE_MASS, 4)
ROLE_CLASS_WEIGHTS = gate_class_weights(ROLE_EFFECTIVE_MASS, 3)


def audit_effective_mass(dataset_run: Path) -> None:
    cano = CorrectiveCanoMultitaskDataset(dataset_run, "train")
    aee = CorrectiveAEEMultitaskDataset(dataset_run)
    cano_count = np.bincount([int(item["branch_count"]) - 1 for item in cano.records], minlength=6)
    cano_role = np.bincount([{"interior": 0, "junction": 1, "terminal": 2}[str(item["primary_role"])] for item in cano.records], minlength=3)
    aee_count = np.bincount([int(aee[index]["count_target"]) for index in range(len(aee))], minlength=6)
    aee_role = np.bincount([int(aee[index]["role_target"]) for index in range(len(aee))], minlength=3)
    if not np.array_equal(cano_count + 5 * aee_count, COUNT_EFFECTIVE_MASS):
        raise RuntimeError("V5 count effective-class-mass drift")
    if not np.array_equal(cano_role + 5 * aee_role, ROLE_EFFECTIVE_MASS):
        raise RuntimeError("V5 role effective-class-mass drift")


def make_staged_heads_trainable(model):
    global _DIRECTION_PARAMETERS
    trainable, frozen = [], []
    _DIRECTION_PARAMETERS = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(base.SEMANTIC_HEAD_PREFIXES)
        (trainable if parameter.requires_grad else frozen).append(name)
        if name.startswith("direction_head."):
            _DIRECTION_PARAMETERS.append(parameter)
    if not _DIRECTION_PARAMETERS or any(not name.startswith(base.SEMANTIC_HEAD_PREFIXES) for name in trainable):
        raise RuntimeError("V5 staged-head parameter contract failed")
    if any(not name.startswith(base.ENCODER_PREFIXES + base.EMBEDDING_PREFIXES) for name in frozen):
        raise RuntimeError("V5 representation freeze contract failed")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


class StagedAdamW(torch.optim.AdamW):
    """Update direction for epoch 1, then preserve it exactly for epochs 2--10."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract_step = 0

    def step(self, closure=None):
        if self.contract_step >= STEPS_PER_EPOCH:
            for parameter in _DIRECTION_PARAMETERS:
                parameter.grad = None
        result = super().step(closure)
        self.contract_step += 1
        return result


def staged_balanced_loss(outputs, direction_target, count_target, role_target, loss_weight, role_weights=None):
    positive_mass = direction_target.sum(dim=1)
    negative_mass = (1.0 - direction_target).sum(dim=1)
    if torch.any(positive_mass <= 0) or torch.any(negative_mass <= 0):
        raise ValueError("every direction target must contain positive and negative mass")
    positive = -(direction_target * F.logsigmoid(outputs["direction_logits"])).sum(dim=1) / positive_mass
    negative = -((1.0 - direction_target) * F.logsigmoid(-outputs["direction_logits"])).sum(dim=1) / negative_mass
    direction_per = 0.5 * (positive + negative)
    count_weights = torch.as_tensor(COUNT_CLASS_WEIGHTS, dtype=outputs["count_logits"].dtype, device=outputs["count_logits"].device)
    role_contract_weights = torch.as_tensor(ROLE_CLASS_WEIGHTS, dtype=outputs["role_logits"].dtype, device=outputs["role_logits"].device)
    count_per = F.cross_entropy(outputs["count_logits"], count_target, weight=count_weights, reduction="none")
    role_per = F.cross_entropy(outputs["role_logits"], role_target, weight=role_contract_weights, reduction="none")
    normalizer = loss_weight.sum()
    direction = torch.sum(direction_per * loss_weight) / normalizer
    count = torch.sum(count_per * loss_weight) / torch.sum(loss_weight * count_weights[count_target])
    role = torch.sum(role_per * loss_weight) / torch.sum(loss_weight * role_contract_weights[role_target])
    total = direction + 0.25 * count + 0.25 * role
    return {"total": total, "direction": direction, "count": count, "role": role}


def v5_gate(adapted_dense, adapted_sparse, source_dense, source_sparse, b0_sparse, finite, encoder_changed, embedding_changed, heads_changed):
    return {
        "sparse_direction_improvement_0p05": adapted_sparse["direction"]["f1"] >= source_sparse["direction"]["f1"] + 0.05,
        "sparse_direction_vs_b0": adapted_sparse["direction"]["f1"] >= b0_sparse["f1"],
        "sparse_empty_rate": adapted_sparse["direction"]["empty_rate"] <= 0.05,
        "sparse_count_1_to_4_macro_f1": adapted_sparse["count"]["macro_f1_count_1_to_4"] >= 0.70,
        "sparse_role_macro_f1": adapted_sparse["role"]["macro_f1_present"] >= 0.70,
        "dense_direction_retention": adapted_dense["direction"]["f1"] >= source_dense["direction"]["f1"] - 0.02,
        "all_model_tensors_finite": finite, "encoder_identity": not encoder_changed,
        "embedding_identity": not embedding_changed, "semantic_heads_changed": heads_changed,
    }


def main() -> int:
    dataset_run = Path(sys.argv[sys.argv.index("--dataset-run") + 1]).resolve()
    audit_effective_mass(dataset_run)
    base.make_full_model_trainable = make_staged_heads_trainable
    base.weighted_multitask_loss = staged_balanced_loss
    base._gate_metrics = v5_gate
    base.torch.optim.AdamW = StagedAdamW
    code = base.main()
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
    checkpoint_path = output / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["mode"] = "M1D_AEE_CORRECTIVE_STAGED_BALANCED_HEADS_V5"
    checkpoint["config"].update(direction_loss="equal_positive_negative_soft_mass_v1", count_class_weights=COUNT_CLASS_WEIGHTS.tolist(), role_class_weights=ROLE_CLASS_WEIGHTS.tolist(), direction_update_epochs=1, count_role_update_epochs=10)
    torch.save(checkpoint, checkpoint_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(schema_version="aee_corrective_staged_balanced_heads_seed_summary_v5", status="COMPLETED_AEE_CORRECTIVE_STAGED_BALANCED_HEADS_SEED_V5", best_checkpoint_sha256=sha256(checkpoint_path), direction_loss="equal_positive_negative_soft_mass_v1", count_effective_mass=COUNT_EFFECTIVE_MASS.tolist(), role_effective_mass=ROLE_EFFECTIVE_MASS.tolist(), count_class_weights=COUNT_CLASS_WEIGHTS.tolist(), role_class_weights=ROLE_CLASS_WEIGHTS.tolist(), staged_update={"direction_epochs": 1, "count_role_epochs": 10, "steps_per_epoch": STEPS_PER_EPOCH}, representation_contract="encoder_and_embedding_frozen_exact")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
