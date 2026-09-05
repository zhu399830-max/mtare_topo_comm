"""Contracts for V4 balanced semantic-head adaptation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("v4", TOOLS / "train_aee_corrective_balanced_heads_v4.py")
assert spec is not None and spec.loader is not None
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)


def test_v4_freezes_encoder_and_embedding_exactly() -> None:
    model = v4.base.StructuralSemanticNet()
    contract = v4.make_heads_trainable(model)
    assert all(name.startswith(v4.base.SEMANTIC_HEAD_PREFIXES) for name in contract["trainable_parameters"])
    assert all(name.startswith(v4.base.ENCODER_PREFIXES + v4.base.EMBEDDING_PREFIXES) for name in contract["frozen_parameters"])


def test_v4_balanced_direction_gradient_escapes_empty_solution() -> None:
    target = torch.zeros((2, 720)); target[:, :32] = 1.0
    logits = torch.full_like(target, -3.07, requires_grad=True)
    outputs = {"direction_logits": logits, "count_logits": torch.zeros((2, 6), requires_grad=True), "role_logits": torch.zeros((2, 3), requires_grad=True)}
    loss = v4.balanced_weighted_multitask_loss(outputs, target, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), torch.ones(2))["total"]
    loss.backward()
    assert float(logits.grad.sum()) < -0.40


def test_v4_gate_requires_representation_identity() -> None:
    metric = {"direction":{"f1":0.8,"empty_rate":0.0},"count":{"macro_f1_count_1_to_4":0.8},"role":{"macro_f1_present":0.8}}
    source = {"direction":{"f1":0.7}}
    gate = v4.balanced_head_gate(metric, metric, source, source, {"f1":0.75}, True, False, False, True)
    assert all(gate.values())
    assert not v4.balanced_head_gate(metric, metric, source, source, {"f1":0.75}, True, True, False, True)["encoder_identity"]
