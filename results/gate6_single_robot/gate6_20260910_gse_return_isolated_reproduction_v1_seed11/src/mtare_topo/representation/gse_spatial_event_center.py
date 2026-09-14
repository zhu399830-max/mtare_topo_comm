"""Route-frame event-center decoder that preserves LiDAR spatial layout."""

from __future__ import annotations

import math

import torch
from torch import nn


HISTORY_FRAMES = 5
ENCODER_DIM = 128
DIRECTIONAL_BINS = 36
ELEVATION_BINS = 2
TRANSVERSE_SCALE_M = (5.0, 5.0)


def circular_coordinate_summary(features: torch.Tensor) -> torch.Tensor:
    """Pool a circular feature map without discarding signed bearing.

    The output concatenates ordinary mean/max statistics with cosine and sine
    moments.  Bin zero is robot-forward and positive sine is robot-left, which
    matches the frozen CPU LiDAR and route-local teacher contracts.
    """

    if features.ndim != 3 or features.shape[-1] < 8:
        raise ValueError("circular features must have shape [B,C,A] with A>=8")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("circular features must be finite")
    angle = torch.arange(
        features.shape[-1], device=features.device, dtype=features.dtype,
    ) * (2.0 * math.pi / features.shape[-1])
    cosine = torch.cos(angle)[None, None]
    sine = torch.sin(angle)[None, None]
    return torch.cat(
        (
            features.mean(dim=-1),
            features.amax(dim=-1),
            (features * cosine).mean(dim=-1),
            (features * sine).mean(dim=-1),
        ),
        dim=1,
    )


class SpatialEventCenterDecoder(nn.Module):
    """Predict lateral/up center offsets while freezing longitudinal input.

    The model consumes only frozen-encoder features from the current and four
    past scans.  The already-qualified signed longitudinal prediction is an
    explicit input and is returned byte-for-byte at initialization and for all
    later weights; this decoder can only learn the missing lateral/up terms.
    """

    def __init__(self) -> None:
        super().__init__()
        self.azimuth_encoder = nn.Sequential(
            nn.Conv1d(3 * ENCODER_DIM, ENCODER_DIM, 5, padding=2, padding_mode="circular"),
            nn.GroupNorm(8, ENCODER_DIM),
            nn.SiLU(),
            nn.Conv1d(ENCODER_DIM, 64, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.elevation_encoder = nn.Sequential(
            nn.Conv1d(3 * ENCODER_DIM, 64, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.pooled_encoder = nn.Sequential(
            nn.Linear(3 * ENCODER_DIM, ENCODER_DIM),
            nn.SiLU(),
        )
        # 4 azimuth statistics, 3 elevation statistics and one pooled context.
        self.transverse_head = nn.Sequential(
            nn.Linear(4 * 64 + 3 * 64 + ENCODER_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.transverse_head[-1].weight)
        nn.init.zeros_(self.transverse_head[-1].bias)
        self.register_buffer("transverse_scale_m", torch.tensor(TRANSVERSE_SCALE_M))

    @staticmethod
    def _validate(
        azimuth_sequence: torch.Tensor,
        elevation_sequence: torch.Tensor,
        pooled_sequence: torch.Tensor,
        longitudinal_offset_m: torch.Tensor,
    ) -> None:
        batch = azimuth_sequence.shape[0] if azimuth_sequence.ndim else -1
        if azimuth_sequence.shape != (batch, HISTORY_FRAMES, ENCODER_DIM, DIRECTIONAL_BINS):
            raise ValueError("azimuth_sequence must have shape [B,5,128,36]")
        if elevation_sequence.shape != (batch, HISTORY_FRAMES, ENCODER_DIM, ELEVATION_BINS):
            raise ValueError("elevation_sequence must have shape [B,5,128,2]")
        if pooled_sequence.shape != (batch, HISTORY_FRAMES, ENCODER_DIM):
            raise ValueError("pooled_sequence must have shape [B,5,128]")
        if longitudinal_offset_m.shape != (batch,):
            raise ValueError("longitudinal_offset_m must have shape [B]")
        if not all(bool(torch.isfinite(value).all()) for value in (
            azimuth_sequence, elevation_sequence, pooled_sequence, longitudinal_offset_m,
        )):
            raise ValueError("spatial event-center inputs must be finite")
        if bool((longitudinal_offset_m.abs() > 12.0 + 1e-6).any()):
            raise ValueError("longitudinal offset exceeds frozen support")

    def encode_spatial_context(
        self,
        azimuth_sequence: torch.Tensor,
        elevation_sequence: torch.Tensor,
        pooled_sequence: torch.Tensor,
        longitudinal_offset_m: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(
            azimuth_sequence, elevation_sequence, pooled_sequence,
            longitudinal_offset_m,
        )
        current_azimuth = azimuth_sequence[:, -1]
        past_azimuth = azimuth_sequence[:, :-1].mean(dim=1)
        azimuth = self.azimuth_encoder(torch.cat(
            (current_azimuth, past_azimuth, current_azimuth - past_azimuth), dim=1,
        ))
        azimuth_summary = circular_coordinate_summary(azimuth)

        current_elevation = elevation_sequence[:, -1]
        past_elevation = elevation_sequence[:, :-1].mean(dim=1)
        elevation = self.elevation_encoder(torch.cat(
            (current_elevation, past_elevation, current_elevation - past_elevation), dim=1,
        ))
        elevation_coordinate = elevation.new_tensor((-1.0, 1.0))[None, None]
        elevation_summary = torch.cat(
            (
                elevation.mean(dim=-1),
                elevation.amax(dim=-1),
                (elevation * elevation_coordinate).mean(dim=-1),
            ),
            dim=1,
        )

        current_pooled = pooled_sequence[:, -1]
        past_pooled = pooled_sequence[:, :-1].mean(dim=1)
        pooled = self.pooled_encoder(torch.cat(
            (current_pooled, past_pooled, current_pooled - past_pooled), dim=1,
        ))
        return torch.cat((azimuth_summary, elevation_summary, pooled), dim=1)

    def transverse_from_context(self, spatial_context: torch.Tensor) -> torch.Tensor:
        if spatial_context.ndim != 2 or spatial_context.shape[1] != 576:
            raise ValueError("spatial event-center context must have shape [B,576]")
        return self.transverse_scale_m * torch.tanh(self.transverse_head(spatial_context))

    def forward(
        self,
        azimuth_sequence: torch.Tensor,
        elevation_sequence: torch.Tensor,
        pooled_sequence: torch.Tensor,
        longitudinal_offset_m: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_spatial_context(
            azimuth_sequence, elevation_sequence, pooled_sequence, longitudinal_offset_m,
        )
        transverse = self.transverse_from_context(context)
        return torch.cat((longitudinal_offset_m[:, None], transverse), dim=1)


class SpatialLongitudinalCorrector(nn.Module):
    """Correct only forward distance from a frozen spatial center decoder."""

    def __init__(self, spatial_decoder: SpatialEventCenterDecoder | None = None) -> None:
        super().__init__()
        self.spatial_decoder = spatial_decoder or SpatialEventCenterDecoder()
        self.longitudinal_residual = nn.Sequential(
            nn.Linear(576, 128), nn.SiLU(), nn.Linear(128, 1),
        )
        nn.init.zeros_(self.longitudinal_residual[-1].weight)
        nn.init.zeros_(self.longitudinal_residual[-1].bias)

    def freeze_spatial_decoder(self) -> None:
        self.spatial_decoder.eval()
        for parameter in self.spatial_decoder.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        azimuth_sequence: torch.Tensor,
        elevation_sequence: torch.Tensor,
        pooled_sequence: torch.Tensor,
        longitudinal_offset_m: torch.Tensor,
    ) -> torch.Tensor:
        context = self.spatial_decoder.encode_spatial_context(
            azimuth_sequence, elevation_sequence, pooled_sequence, longitudinal_offset_m,
        )
        transverse = self.spatial_decoder.transverse_from_context(context)
        residual = 12.0 * torch.tanh(self.longitudinal_residual(context).squeeze(1))
        corrected = torch.clamp(longitudinal_offset_m + residual, -12.0, 12.0)
        return torch.cat((corrected[:, None], transverse), dim=1)


__all__ = [
    "DIRECTIONAL_BINS", "ELEVATION_BINS", "ENCODER_DIM", "HISTORY_FRAMES",
    "TRANSVERSE_SCALE_M", "SpatialEventCenterDecoder", "SpatialLongitudinalCorrector",
    "circular_coordinate_summary",
]
