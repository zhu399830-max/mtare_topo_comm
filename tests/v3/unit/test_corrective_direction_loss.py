"""Numerical proof contracts for the corrective direction objective."""

from __future__ import annotations

import math

import pytest
import torch

from mtare_topo.representation.corrective_direction_loss import (
    balanced_soft_direction_bce,
    source_logit_retention,
)


def _target() -> torch.Tensor:
    target = torch.zeros((2, 720), dtype=torch.float64)
    target[:, :32] = 1.0
    return target


def test_balanced_loss_pushes_ordinary_bce_empty_optimum_upward() -> None:
    target = _target()
    prevalence = float(target.mean())
    empty_optimum = math.log(prevalence / (1.0 - prevalence))
    logits = torch.full_like(target, empty_optimum, requires_grad=True)
    ordinary = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    ordinary.backward()
    ordinary_gradient = float(logits.grad.sum())
    logits.grad.zero_()
    balanced = balanced_soft_direction_bce(logits, target)
    balanced.backward()
    balanced_gradient = float(logits.grad.sum())
    assert ordinary_gradient == pytest.approx(0.0, abs=1e-12)
    assert balanced_gradient < -0.40


def test_balanced_loss_equalizes_total_positive_and_negative_mass() -> None:
    target = _target()
    logits = torch.zeros_like(target, requires_grad=True)
    balanced_soft_direction_bce(logits, target).backward()
    positive = float(logits.grad[target > 0].sum())
    negative = float(logits.grad[target == 0].sum())
    assert positive == pytest.approx(-0.25, abs=1e-12)
    assert negative == pytest.approx(0.25, abs=1e-12)


def test_source_retention_is_exactly_zero_with_zero_gradient_at_identity() -> None:
    source = torch.linspace(-4.0, 4.0, 720, dtype=torch.float64).reshape(1, -1)
    student = source.clone().requires_grad_(True)
    loss = source_logit_retention(student, source)
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.count_nonzero(student.grad) == 0


def test_loss_rejects_empty_or_all_positive_targets() -> None:
    logits = torch.zeros((1, 720))
    with pytest.raises(ValueError, match="positive and negative"):
        balanced_soft_direction_bce(logits, torch.zeros_like(logits))
    with pytest.raises(ValueError, match="positive and negative"):
        balanced_soft_direction_bce(logits, torch.ones_like(logits))
