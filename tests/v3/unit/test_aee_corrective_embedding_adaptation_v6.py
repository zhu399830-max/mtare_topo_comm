"""Contracts for V6 embedding-only count/role adaptation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("v6", TOOLS / "train_aee_corrective_embedding_adaptation_v6.py")
assert spec and spec.loader
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)


def test_v6_freezes_only_encoder_and_opens_embedding_and_heads() -> None:
    model = v6.base.StructuralSemanticNet()
    contract = v6.make_embedding_adaptation_trainable(model)
    assert contract["frozen_parameters"]
    assert all(name.startswith(v6.base.ENCODER_PREFIXES) for name in contract["frozen_parameters"])
    assert any(name.startswith(v6.base.EMBEDDING_PREFIXES) for name in contract["trainable_parameters"])
    assert all(name.startswith(v6.base.EMBEDDING_PREFIXES + v6.base.SEMANTIC_HEAD_PREFIXES) for name in contract["trainable_parameters"])


def test_v6_optimizer_preserves_direction_but_updates_embedding_after_boundary() -> None:
    direction = torch.nn.Parameter(torch.tensor([1.0]))
    embedding = torch.nn.Parameter(torch.tensor([1.0]))
    v6._DIRECTION_PARAMETERS = [direction]
    v6._DIRECTION_STAGE1_SHA256 = None
    optimizer = v6.EmbeddingStagedAdamW([direction, embedding], lr=0.01, weight_decay=0.0)
    for _ in range(v6.STEPS_PER_EPOCH):
        optimizer.zero_grad(); (direction + embedding).backward(); optimizer.step()
    frozen = direction.detach().clone()
    stage_digest = v6._DIRECTION_STAGE1_SHA256
    for _ in range(3):
        optimizer.zero_grad(); (direction + embedding).backward(); optimizer.step()
    assert torch.equal(direction.detach(), frozen)
    assert v6._tensor_digest([direction]) == stage_digest
    assert float(embedding.detach()) < float(frozen)


def test_v6_direction_output_is_independent_of_embedding_parameters() -> None:
    model = v6.base.StructuralSemanticNet().eval()
    student = torch.randn(2, 2, 16, 720)
    with torch.no_grad():
        before = model(student)["direction_logits"].clone()
        for parameter in model.embedding_head.parameters():
            parameter.add_(torch.randn_like(parameter))
        after = model(student)["direction_logits"]
    assert torch.equal(before, after)


def test_checkpoint_direction_digest_uses_model_registration_order() -> None:
    model = v6.base.StructuralSemanticNet()
    direct = [parameter for name, parameter in model.named_parameters() if name.startswith("direction_head.")]
    assert v6._checkpoint_direction_digest(model.state_dict()) == v6._tensor_digest(direct)


def test_v6_gate_requires_embedding_change_and_encoder_identity() -> None:
    metric = {"direction":{"f1":0.8,"empty_rate":0.0},"count":{"macro_f1_count_1_to_4":0.8},"role":{"macro_f1_present":0.8}}
    source = {"direction":{"f1":0.7}}
    assert all(v6.v6_gate(metric, metric, source, source, {"f1":0.75}, True, False, True, True).values())
    assert not v6.v6_gate(metric, metric, source, source, {"f1":0.75}, True, False, False, True)["embedding_changed"]
