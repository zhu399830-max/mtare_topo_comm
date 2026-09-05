"""Circular range-image encoder and objective structural-semantic heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CircularAzimuthConv2d(nn.Module):
    """Conv2d with circular azimuth and zero-padded elevation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.pad(values, (self.pad, self.pad, 0, 0), mode="circular")
        values = F.pad(values, (0, 0, self.pad, self.pad), mode="constant", value=0.0)
        return self.conv(values)


class ResidualRangeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.conv1 = CircularAzimuthConv2d(in_channels, out_channels, stride=stride)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = CircularAzimuthConv2d(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels and stride == (1, 1) else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.GroupNorm(8, out_channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = F.silu(self.norm1(self.conv1(values)))
        return F.silu(self.norm2(self.conv2(values)) + residual)


class StructuralSemanticNet(nn.Module):
    """Return direction logits, branch count, structural role, and z_role."""

    def __init__(self, embedding_dim: int = 128, count_classes: int = 6, role_classes: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            CircularAzimuthConv2d(2, 32, stride=(1, 2)), nn.GroupNorm(8, 32), nn.SiLU(),
            ResidualRangeBlock(32, 64, (2, 2)),
            ResidualRangeBlock(64, 96, (2, 1)),
            ResidualRangeBlock(96, 128, (2, 1)),
        )
        self.direction_head = nn.Sequential(
            nn.Conv1d(128, 96, 1), nn.SiLU(), nn.Conv1d(96, 1, 1),
        )
        self.embedding_head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, embedding_dim))
        self.count_head = nn.Linear(embedding_dim, count_classes)
        self.role_head = nn.Linear(embedding_dim, role_classes)

    def forward(self, student: torch.Tensor) -> dict[str, torch.Tensor]:
        if student.ndim != 4 or tuple(student.shape[1:]) != (2, 16, 720):
            raise ValueError(f"expected [B,2,16,720], got {tuple(student.shape)}")
        feature_map = self.encoder(student)
        azimuth_feature = feature_map.mean(dim=2)
        direction_low = self.direction_head(azimuth_feature)
        if direction_low.shape[-1] != 180:
            raise RuntimeError(f"expected 180-column directional latent, got {direction_low.shape[-1]}")
        # Exact periodic expansion preserves circular shifts aligned with the
        # four-column encoder stride; no non-periodic interpolation boundary.
        direction_logits = direction_low.repeat_interleave(4, dim=-1).squeeze(1)
        pooled = feature_map.mean(dim=(2, 3))
        z_role = F.normalize(self.embedding_head(pooled), dim=1)
        return {
            "direction_logits": direction_logits,
            "count_logits": self.count_head(z_role),
            "role_logits": self.role_head(z_role),
            "z_role": z_role,
        }


def multitask_loss(outputs: dict[str, torch.Tensor], direction_target: torch.Tensor, count_target: torch.Tensor, role_target: torch.Tensor, role_weights: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    direction = F.binary_cross_entropy_with_logits(outputs["direction_logits"], direction_target)
    count = F.cross_entropy(outputs["count_logits"], count_target)
    role = F.cross_entropy(outputs["role_logits"], role_target, weight=role_weights)
    total = direction + 0.25 * count + 0.25 * role
    return {"total": total, "direction": direction, "count": count, "role": role}
