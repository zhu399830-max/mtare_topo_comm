from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
sys.path.insert(0, str(TOOLS)) if str(TOOLS) not in sys.path else None
spec = importlib.util.spec_from_file_location("v5", TOOLS / "train_aee_corrective_staged_balanced_heads_v5.py")
assert spec and spec.loader
v5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)


def test_gate_class_weights_balance_common_and_leave_rare_empirical() -> None:
    weights = v5.gate_class_weights(v5.COUNT_EFFECTIVE_MASS, 4)
    contribution = weights * v5.COUNT_EFFECTIVE_MASS
    assert np.allclose(contribution[:4], contribution[0])
    assert weights[4:].tolist() == [1.0, 1.0]


def test_exact_effective_mass_and_weights_are_frozen() -> None:
    assert v5.COUNT_EFFECTIVE_MASS.tolist() == [1244, 6793, 1329, 504, 105, 25]
    assert v5.ROLE_EFFECTIVE_MASS.tolist() == [6495, 2105, 1400]
    assert v5.COUNT_CLASS_WEIGHTS[:4].tolist() == pytest.approx([9870/(4*1244),9870/(4*6793),9870/(4*1329),9870/(4*504)])


def test_staged_optimizer_freezes_direction_after_exact_epoch_boundary() -> None:
    direction = torch.nn.Parameter(torch.tensor([1.0])); other = torch.nn.Parameter(torch.tensor([1.0]))
    v5._DIRECTION_PARAMETERS = [direction]
    optimizer = v5.StagedAdamW([direction, other], lr=0.01, weight_decay=0.0)
    for _ in range(v5.STEPS_PER_EPOCH):
        optimizer.zero_grad(); (direction + other).backward(); optimizer.step()
    frozen = direction.detach().clone()
    optimizer.zero_grad(); (direction + other).backward(); optimizer.step()
    assert torch.equal(direction.detach(), frozen)
    assert float(other.detach()) < float(frozen)


def test_staged_loss_is_finite_and_backpropagates_all_heads() -> None:
    outputs={"direction_logits":torch.zeros((4,720),requires_grad=True),"count_logits":torch.zeros((4,6),requires_grad=True),"role_logits":torch.zeros((4,3),requires_grad=True)}
    target=torch.zeros((4,720)); target[:,:32]=1
    loss=v5.staged_balanced_loss(outputs,target,torch.tensor([0,1,2,3]),torch.tensor([0,1,2,0]),torch.ones(4))["total"]
    loss.backward(); assert torch.isfinite(loss); assert all(value.grad is not None for value in outputs.values())
