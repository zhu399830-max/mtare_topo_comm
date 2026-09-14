"""Direction losses that avoid the sealed V3 empty-output failure mode."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def balanced_soft_direction_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Give positive and negative soft-label mass equal influence per frame."""

    if logits.shape != target.shape or logits.ndim != 2:
        raise ValueError("direction logits and targets must have equal [batch,bins] shape")
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError("direction logits and targets must be finite")
    if torch.any((target < 0) | (target > 1)):
        raise ValueError("direction target must lie in [0,1]")
    positive_mass = target.sum(dim=1)
    negative_mass = (1.0 - target).sum(dim=1)
    if torch.any(positive_mass <= 0) or torch.any(negative_mass <= 0):
        raise ValueError("every direction target must contain positive and negative mass")
    positive = -(target * F.logsigmoid(logits)).sum(dim=1) / positive_mass
    negative = -((1.0 - target) * F.logsigmoid(-logits)).sum(dim=1) / negative_mass
    return (0.5 * (positive + negative)).mean()


def source_logit_retention(student_logits: torch.Tensor, source_logits: torch.Tensor) -> torch.Tensor:
    """Zero-at-identity dense-Cano retention term with no learned target leakage."""

    if student_logits.shape != source_logits.shape:
        raise ValueError("student/source logit shapes differ")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(source_logits).all():
        raise ValueError("student/source logits must be finite")
    return F.mse_loss(student_logits, source_logits.detach())


__all__ = ["balanced_soft_direction_bce", "source_logit_retention"]
