"""Contracts for V8 final-block adaptation with direction retention."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("v8", TOOLS / "train_aee_corrective_encoder_retention_v8.py")
assert spec and spec.loader
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)


def test_v8_stage_contract_opens_only_heads_then_final_block_and_embedding() -> None:
    model = v8.RetainedStructuralSemanticNet()
    contract = v8.make_encoder_retention_trainable(model)
    assert all(name.startswith(v8.base.SEMANTIC_HEAD_PREFIXES) for name in contract["trainable_parameters"])
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("encoder."))
    optimizer = v8.EncoderRetentionAdamW(model.parameters(), lr=0.0)
    for _ in range(v8.STEPS_PER_EPOCH):
        optimizer.zero_grad(); sum(parameter.sum() for parameter in model.parameters() if parameter.requires_grad).backward(); optimizer.step()
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("encoder.5.") or name.startswith("embedding_head."))
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("encoder.") and not name.startswith("encoder.5."))


def test_v8_teacher_is_frozen_snapshot_at_stage_boundary() -> None:
    assert v8._TEACHER_ENCODER is not None and v8._TEACHER_DIRECTION is not None
    assert all(not parameter.requires_grad for parameter in v8._TEACHER_ENCODER.parameters())
    assert all(not parameter.requires_grad for parameter in v8._TEACHER_DIRECTION.parameters())


def test_v8_retention_is_zero_at_identity_and_positive_after_encoder_change() -> None:
    model = v8._MODEL
    assert model is not None
    model.eval()
    student = torch.randn(2, 2, 16, 720)
    with torch.no_grad():
        outputs = model(student)
        identity = torch.nn.functional.mse_loss(outputs["direction_logits"], outputs["teacher_direction_logits"])
        next(model.encoder[5].parameters()).add_(0.01)
        changed = model(student)
        moved = torch.nn.functional.mse_loss(changed["direction_logits"], changed["teacher_direction_logits"])
    assert float(identity) == 0.0
    assert float(moved) > 0.0


def test_v8_loss_applies_retention_only_after_teacher_exists() -> None:
    outputs = {"direction_logits":torch.zeros((4,720),requires_grad=True),"count_logits":torch.zeros((4,6),requires_grad=True),"role_logits":torch.zeros((4,3),requires_grad=True),"teacher_direction_logits":torch.ones((4,720))}
    target=torch.zeros((4,720)); target[:,:32]=1
    weights=torch.tensor([0.5,0.5,5.0,5.0])
    losses=v8.retained_balanced_loss(outputs,target,torch.tensor([0,1,2,3]),torch.tensor([0,1,2,0]),weights)
    assert float(losses["direction"].detach()) > 1.0
    losses["total"].backward()
    assert outputs["direction_logits"].grad is not None
