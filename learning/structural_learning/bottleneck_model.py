from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 4
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ForcedGlobalBottleneckNet(nn.Module):
    """Teacher completion model whose decoder receives only one global vector."""

    def __init__(self, input_channels: int = 3, latent_dim: int = 128, base_channels: int = 24) -> None:
        super().__init__()
        b = base_channels
        self.stem = Block(input_channels, b)
        self.down1 = nn.Sequential(nn.Conv2d(b, 32, 4, 2, 1), Block(32, 32))
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), Block(64, 64))
        self.down3 = nn.Sequential(nn.Conv2d(64, 96, 4, 2, 1), Block(96, 96))
        self.down4 = nn.Sequential(nn.Conv2d(96, 128, 4, 2, 1), Block(128, 128))
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, latent_dim), nn.LayerNorm(latent_dim))
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, 128 * 7 * 7), nn.SiLU())
        self.dec3 = Block(128, 96)
        self.dec2 = Block(96, 64)
        self.dec1 = Block(64, 32)
        self.dec0 = Block(32, b)
        self.head = nn.Conv2d(b, 4, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        with torch.no_grad(): self.head.bias[0] = -2.0
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x); x = self.down2(x); x = self.down3(x); x = self.down4(x)
        x = F.adaptive_avg_pool2d(x, (4, 4))
        return self.to_latent(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(latent).reshape(-1, 128, 7, 7)
        x = self.dec3(F.interpolate(x, size=(13, 13), mode="bilinear", align_corners=False))
        x = self.dec2(F.interpolate(x, size=(25, 25), mode="bilinear", align_corners=False))
        x = self.dec1(F.interpolate(x, size=(50, 50), mode="bilinear", align_corners=False))
        raw = self.head(self.dec0(F.interpolate(x, size=(100, 100), mode="bilinear", align_corners=False)))
        return torch.cat([torch.sigmoid(raw[:, :1]), torch.sigmoid(raw[:, 1:])], dim=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        latent = self.encode(x)
        prediction = self.decode(latent)
        return (prediction, latent) if return_features else prediction
