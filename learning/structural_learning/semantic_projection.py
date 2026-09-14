from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StructuralProjectionHead(nn.Module):
    """Maps the frozen completion bottleneck to a geometry-oriented vector."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(latent), dim=1)
