#!/usr/bin/env python3
"""V8: adapt the final encoder block under frozen-teacher direction retention."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import train_aee_corrective_full_encoder_v3 as base
import train_aee_corrective_staged_balanced_heads_v5 as v5
import train_aee_corrective_embedding_adaptation_v6 as v6
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


STEPS_PER_EPOCH = 47
_MODEL: StructuralSemanticNet | None = None
_DIRECTION_PARAMETERS: list[torch.nn.Parameter] = []
_STAGE2_PARAMETERS: list[torch.nn.Parameter] = []
_TEACHER_ENCODER: torch.nn.Module | None = None
_TEACHER_DIRECTION: torch.nn.Module | None = None
_DIRECTION_STAGE1_SHA256: str | None = None


class RetainedStructuralSemanticNet(StructuralSemanticNet):
    def forward(self, student: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(student)
        if _TEACHER_ENCODER is not None and _TEACHER_DIRECTION is not None:
            with torch.no_grad():
                feature_map = _TEACHER_ENCODER(student)
                direction_low = _TEACHER_DIRECTION(feature_map.mean(dim=2))
                outputs["teacher_direction_logits"] = direction_low.repeat_interleave(4, dim=-1).squeeze(1)
        return outputs


def make_encoder_retention_trainable(model):
    """Stage 1 opens heads; stage 2 also opens embedding and encoder.5."""
    global _MODEL, _DIRECTION_PARAMETERS, _STAGE2_PARAMETERS
    _MODEL = model
    _DIRECTION_PARAMETERS = []
    _STAGE2_PARAMETERS = []
    trainable, frozen = [], []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(base.SEMANTIC_HEAD_PREFIXES)
        (trainable if parameter.requires_grad else frozen).append(name)
        if name.startswith("direction_head."):
            _DIRECTION_PARAMETERS.append(parameter)
        if name.startswith(base.EMBEDDING_PREFIXES) or name.startswith("encoder.5."):
            _STAGE2_PARAMETERS.append(parameter)
    if not _DIRECTION_PARAMETERS or not _STAGE2_PARAMETERS:
        raise RuntimeError("V8 staged parameter contract is empty")
    if any(not name.startswith(base.SEMANTIC_HEAD_PREFIXES) for name in trainable):
        raise RuntimeError("V8 stage-1 trainable contract failed")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


class EncoderRetentionAdamW(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract_step = 0

    def step(self, closure=None):
        global _TEACHER_ENCODER, _TEACHER_DIRECTION, _DIRECTION_STAGE1_SHA256
        if self.contract_step >= STEPS_PER_EPOCH:
            for parameter in _DIRECTION_PARAMETERS:
                parameter.grad = None
        result = super().step(closure)
        self.contract_step += 1
        if self.contract_step == STEPS_PER_EPOCH:
            if _MODEL is None:
                raise RuntimeError("V8 model unavailable at stage boundary")
            _DIRECTION_STAGE1_SHA256 = v6._tensor_digest(_DIRECTION_PARAMETERS)
            _TEACHER_ENCODER = copy.deepcopy(_MODEL.encoder).eval()
            _TEACHER_DIRECTION = copy.deepcopy(_MODEL.direction_head).eval()
            for parameter in list(_TEACHER_ENCODER.parameters()) + list(_TEACHER_DIRECTION.parameters()):
                parameter.requires_grad = False
            for parameter in _STAGE2_PARAMETERS:
                parameter.requires_grad = True
        return result


def retained_balanced_loss(outputs, direction_target, count_target, role_target, loss_weight, role_weights=None):
    losses = v5.staged_balanced_loss(outputs, direction_target, count_target, role_target, loss_weight, role_weights)
    teacher = outputs.get("teacher_direction_logits")
    if teacher is not None:
        cano = loss_weight < 1.0
        if int(cano.sum()) == 0:
            raise RuntimeError("V8 batch has no Cano retention views")
        retention = F.mse_loss(outputs["direction_logits"][cano], teacher[cano].detach())
        losses["direction"] = losses["direction"] + retention
        losses["total"] = losses["total"] + retention
    return losses


def v8_gate(adapted_dense, adapted_sparse, source_dense, source_sparse, b0_sparse, finite, encoder_changed, embedding_changed, heads_changed):
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
    global _TEACHER_ENCODER, _TEACHER_DIRECTION, _DIRECTION_STAGE1_SHA256
    _TEACHER_ENCODER = _TEACHER_DIRECTION = None
    _DIRECTION_STAGE1_SHA256 = None
    dataset_run = Path(sys.argv[sys.argv.index("--dataset-run") + 1]).resolve()
    v5.audit_effective_mass(dataset_run)
    base.StructuralSemanticNet = RetainedStructuralSemanticNet
    base.make_full_model_trainable = make_encoder_retention_trainable
    base.weighted_multitask_loss = retained_balanced_loss
    base._gate_metrics = v8_gate
    base.torch.optim.AdamW = EncoderRetentionAdamW
    code = base.main()
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
    checkpoint_path = output / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    final_direction_sha256 = v6._checkpoint_direction_digest(checkpoint["model"])
    tensor_hashes = json.loads((output / "model_tensor_hashes.json").read_text(encoding="utf-8"))
    changed = sorted(name for name in tensor_hashes["before"] if tensor_hashes["before"][name] != tensor_hashes["after"][name])
    changed_encoder = [name for name in changed if name.startswith("encoder.")]
    early_encoder_identity = all(name.startswith("encoder.5.") for name in changed_encoder)
    last_encoder_block_changed = bool(changed_encoder) and all(name.startswith("encoder.5.") for name in changed_encoder)
    direction_head_stage1_identity = _DIRECTION_STAGE1_SHA256 is not None and final_direction_sha256 == _DIRECTION_STAGE1_SHA256
    checkpoint["mode"] = "M1D_AEE_CORRECTIVE_ENCODER_RETENTION_V8"
    checkpoint["config"].update(
        direction_loss="equal_positive_negative_soft_mass_v1",
        direction_retention="same_view_cano_logit_mse_weight_1",
        direction_update_epochs=1,
        final_encoder_embedding_count_role_update_epochs=9,
        direction_stage1_sha256=_DIRECTION_STAGE1_SHA256,
    )
    torch.save(checkpoint, checkpoint_path)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gate"].update(
        early_encoder_identity=early_encoder_identity,
        last_encoder_block_changed=last_encoder_block_changed,
        direction_head_stage1_identity=direction_head_stage1_identity,
    )
    summary["scientific_gate_passed"] = all(summary["gate"].values())
    summary.update(
        schema_version="aee_corrective_encoder_retention_seed_summary_v8",
        status="COMPLETED_AEE_CORRECTIVE_ENCODER_RETENTION_SEED_V8",
        best_checkpoint_sha256=sha256(checkpoint_path),
        staged_update={"direction_epochs": 1, "final_encoder_embedding_count_role_epochs": 9, "steps_per_epoch": STEPS_PER_EPOCH},
        representation_contract="encoder_0_to_4_frozen_encoder_5_adapted_same_view_direction_retention",
        direction_retention="same_view_cano_logit_mse_weight_1",
        direction_stage1_sha256=_DIRECTION_STAGE1_SHA256,
        final_direction_head_sha256=final_direction_sha256,
        changed_encoder_tensors=changed_encoder,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
