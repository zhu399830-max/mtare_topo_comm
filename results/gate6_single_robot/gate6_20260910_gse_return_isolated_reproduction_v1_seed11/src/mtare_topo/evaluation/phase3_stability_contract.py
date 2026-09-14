"""Pure helpers for the Phase-3 representation stability contract."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def fixed_ray_column_mask(student: torch.Tensor, period: int = 10) -> torch.Tensor:
    """Apply the frozen v1r3 perturbation to every ``period``-th azimuth ray."""
    if student.ndim != 4 or tuple(student.shape[1:]) != (2, 16, 720):
        raise ValueError(f"expected [B,2,16,720], got {tuple(student.shape)}")
    if period <= 0:
        raise ValueError("period must be positive")
    masked = student.clone()
    masked[:, 0, :, ::period] = 1.0
    masked[:, 1, :, ::period] = 0.0
    return masked


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be non-empty and finite")
    return {
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "minimum": float(values.min()),
    }


def compare_outputs(
    base: dict[str, torch.Tensor],
    rotated: dict[str, torch.Tensor],
    masked: dict[str, torch.Tensor],
    shift_columns: int = 12,
) -> dict[str, np.ndarray]:
    expected = torch.roll(base["direction_logits"], shift_columns, dims=-1)
    direction_error = (expected - rotated["direction_logits"]).abs().detach().cpu().numpy().reshape(-1)
    rotation_cosine = F.cosine_similarity(base["z_role"], rotated["z_role"]).detach().cpu().numpy()
    masking_cosine = F.cosine_similarity(base["z_role"], masked["z_role"]).detach().cpu().numpy()
    return {
        "rotation_direction_absolute_logit_error": direction_error,
        "rotation_z_role_cosine": rotation_cosine,
        "masking_z_role_cosine": masking_cosine,
    }
