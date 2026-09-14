from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinySurfaceCompletionNet(nn.Module):
    """Small U-Net used only for dataset/model feasibility testing."""

    def __init__(self, base_channels: int = 16, latent_dim: int = 64, input_channels: int = 4) -> None:
        super().__init__()
        b = base_channels
        self.input_channels = input_channels
        self.enc1 = ConvBlock(input_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.bottleneck = ConvBlock(b * 2, latent_dim)
        self.dec2 = ConvBlock(latent_dim + b * 2, b * 2)
        self.dec1 = ConvBlock(b * 2 + b, b)
        self.head = nn.Conv2d(b, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = -2.0

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        z = self.bottleneck(F.avg_pool2d(e2, 2))
        return e1, e2, z

    def forward(self, x: torch.Tensor, return_features: bool = False):
        e1, e2, z = self.encode(x)
        d2 = F.interpolate(z, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        raw = self.head(self.dec1(torch.cat([d1, e1], dim=1)))
        # The residual prior makes the untrained surface output equal the input mask.
        mask_logits = raw[:, :1] + 4.0 * x[:, :1]
        attributes = torch.sigmoid(raw[:, 1:])
        prediction = torch.cat([torch.sigmoid(mask_logits), attributes], dim=1)
        if return_features:
            features = F.adaptive_avg_pool2d(z, 1).flatten(1)
            return prediction, features
        return prediction


def completion_loss(prediction: torch.Tensor, teacher: torch.Tensor, positive_weight: float = 4.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_mask = teacher[:, :1]
    pred_mask = prediction[:, :1].clamp(1e-5, 1.0 - 1e-5)
    weights = 1.0 + (positive_weight - 1.0) * target_mask
    bce = F.binary_cross_entropy(pred_mask, target_mask, weight=weights)
    intersection = (pred_mask * target_mask).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (pred_mask.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3)) + 1.0)).mean()
    attr_error = F.smooth_l1_loss(prediction[:, 1:], teacher[:, 1:], reduction="none")
    attr = (attr_error * target_mask).sum() / (target_mask.sum() * 3.0 + 1.0)
    total = bce + dice + 0.5 * attr
    return total, {"total": total.detach(), "bce": bce.detach(), "dice": dice.detach(), "attribute": attr.detach()}
