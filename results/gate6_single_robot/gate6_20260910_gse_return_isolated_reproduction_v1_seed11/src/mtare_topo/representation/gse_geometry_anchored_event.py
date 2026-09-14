"""Pose-free polar encoder for geometry-anchored structure-event sets.

The encoder consumes only five causal organized LiDAR range images and their
valid masks.  It deliberately does not accept a world pose, route identity,
TNG identity, future frame, or teacher feature.  Azimuth is retained as a
circular axis and event range is a fraction of an observed free-range anchor,
so the decoder cannot invent an event behind the current scan support.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.gse_spatial_event_set import (
    EVENT_TYPE_NAMES,
    MAXIMUM_EVENT_RANGE_M,
    SpatialEventSetConfig,
)


HISTORY_FRAMES = 5
ELEVATION_ROWS = 16
AZIMUTH_COLUMNS = 720
POLAR_BINS = 180
ELEVATION_BANDS = 4
SENSOR_MAXIMUM_RANGE_M = 50.0
SENSOR_NEAR_RANGE_M = 0.3
ELEVATION_DEGREES = tuple(float(value) for value in range(-15, 16, 2))
LOS_SUPPORT_MARGIN_M = 0.25
LOCAL_SUPPORT_RADIUS_BINS = 1


@dataclass(frozen=True)
class GeometryAnchoredSpatialEventConfig:
    event_set: SpatialEventSetConfig = SpatialEventSetConfig()
    polar_bins: int = POLAR_BINS
    elevation_bands: int = ELEVATION_BANDS
    sensor_maximum_range_m: float = SENSOR_MAXIMUM_RANGE_M
    sensor_near_range_m: float = SENSOR_NEAR_RANGE_M

    def __post_init__(self) -> None:
        if (
            self.polar_bins != POLAR_BINS
            or self.elevation_bands != ELEVATION_BANDS
            or self.event_set.encoder_dim != 128
            or not math.isfinite(float(self.sensor_maximum_range_m))
            or not math.isfinite(float(self.sensor_near_range_m))
            or self.sensor_near_range_m <= 0.0
            or self.sensor_maximum_range_m <= self.sensor_near_range_m
        ):
            raise ValueError("invalid frozen geometry-anchored event config")


class GeometryAnchoredSpatialEventEncoder(nn.Module):
    """Learn circular geometry and emit free-range-bounded event tokens."""

    def __init__(self, config: GeometryAnchoredSpatialEventConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeometryAnchoredSpatialEventConfig()
        event_config = self.config.event_set
        dim = event_config.encoder_dim

        # Per causal frame and elevation band: horizontal range projection,
        # vertical range projection, and return-valid fraction.
        input_channels = HISTORY_FRAMES * ELEVATION_BANDS * 3
        self.circular_encoder = nn.Sequential(
            nn.Conv1d(
                input_channels,
                dim,
                kernel_size=5,
                padding=2,
                padding_mode="circular",
            ),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
        )
        self.queries = nn.Parameter(torch.empty(event_config.query_count, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads=event_config.attention_heads,
            batch_first=True,
        )
        # presence + type + radial fraction + elevation + descriptor + uncertainty
        output_size = 1 + len(EVENT_TYPE_NAMES) + 1 + 1 + event_config.descriptor_dim + 1
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, output_size),
        )

        # Each learned bin averages four 0.5-degree rays; its geometric
        # coordinate is therefore the 0.75-degree cell center.
        azimuth = (torch.arange(POLAR_BINS, dtype=torch.float32) + 0.375) * (
            2.0 * torch.pi / POLAR_BINS
        )
        elevation = torch.tensor(ELEVATION_DEGREES, dtype=torch.float32).reshape(
            ELEVATION_BANDS, ELEVATION_ROWS // ELEVATION_BANDS
        )
        elevation = torch.deg2rad(elevation).mean(dim=1)
        self.register_buffer("polar_azimuth_rad", azimuth, persistent=True)
        self.register_buffer("elevation_band_rad", elevation, persistent=True)

    def _validate_inputs(
        self,
        range_m: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (HISTORY_FRAMES, ELEVATION_ROWS, AZIMUTH_COLUMNS)
        if (
            range_m.ndim != 4
            or tuple(range_m.shape[1:]) != expected
            or valid_mask.shape != range_m.shape
            or not torch.is_floating_point(range_m)
            or not bool(torch.isfinite(range_m).all())
            or bool((range_m < self.config.sensor_near_range_m - 1e-4).any())
            or bool((range_m > self.config.sensor_maximum_range_m + 1e-4).any())
        ):
            raise ValueError("geometry-anchored LiDAR input contract drift")
        if torch.is_floating_point(valid_mask):
            if not bool(torch.isfinite(valid_mask).all()):
                raise ValueError("valid mask must be finite")
            if bool(((valid_mask != 0) & (valid_mask != 1)).any()):
                raise ValueError("valid mask must be binary")
        elif valid_mask.dtype != torch.bool and valid_mask.dtype != torch.uint8:
            raise ValueError("valid mask must be bool, uint8, or binary float")
        return valid_mask.bool()

    def polar_inputs(
        self,
        range_m: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return circular learned features and current free-range profile."""

        valid = self._validate_inputs(range_m, valid_mask)
        batch = range_m.shape[0]
        columns_per_bin = AZIMUTH_COLUMNS // POLAR_BINS
        rows_per_band = ELEVATION_ROWS // ELEVATION_BANDS
        ranges = range_m.reshape(
            batch,
            HISTORY_FRAMES,
            ELEVATION_BANDS,
            rows_per_band,
            POLAR_BINS,
            columns_per_bin,
        )
        valids = valid.reshape_as(ranges)
        clipped = torch.clamp(ranges, max=self.config.sensor_maximum_range_m)

        # Invalid rays mean no return before the sensor maximum.  Keeping the
        # maximum value and a separate validity fraction exposes both free
        # space and measurement support without leaking any teacher signal.
        observed = torch.where(valids, clipped, clipped.new_full((), self.config.sensor_maximum_range_m))
        band_range = observed.mean(dim=(3, 5)) / self.config.sensor_maximum_range_m
        band_valid = valids.to(range_m.dtype).mean(dim=(3, 5))
        elevation = self.elevation_band_rad.to(dtype=range_m.dtype)[None, None, :, None]
        horizontal = band_range * torch.cos(elevation)
        vertical = band_range * torch.sin(elevation)
        polar = torch.stack((horizontal, vertical, band_valid), dim=3).reshape(
            batch,
            HISTORY_FRAMES * ELEVATION_BANDS * 3,
            POLAR_BINS,
        )

        # Five-frame causal support is part of the legal student input.  The
        # fixed one-bin circular envelope bridges discrete two-degree cells;
        # the 0.25 m allowance is inherited from the sealed native-LOS Teacher,
        # not selected from model errors.
        causal_range = range_m.reshape(
            batch,
            HISTORY_FRAMES,
            ELEVATION_ROWS,
            POLAR_BINS,
            columns_per_bin,
        )
        causal_valid = valid.reshape_as(causal_range)
        causal_support = torch.where(
            causal_valid,
            torch.clamp(causal_range, max=self.config.event_set.maximum_range_m),
            causal_range.new_full((), self.config.event_set.maximum_range_m),
        )
        free_range_profile_m = causal_support.amax(dim=(1, 2, 4))
        free_range_profile_m = F.max_pool1d(
            F.pad(
                free_range_profile_m[:, None],
                (LOCAL_SUPPORT_RADIUS_BINS, LOCAL_SUPPORT_RADIUS_BINS),
                mode="circular",
            ),
            kernel_size=2 * LOCAL_SUPPORT_RADIUS_BINS + 1,
            stride=1,
        ).squeeze(1)
        free_range_profile_m = torch.clamp(
            free_range_profile_m + LOS_SUPPORT_MARGIN_M,
            max=self.config.event_set.maximum_range_m,
        )
        return polar, free_range_profile_m

    def forward(
        self,
        range_m: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        polar, free_range_profile_m = self.polar_inputs(range_m, valid_mask)
        directional = self.circular_encoder(polar)
        context = self.context_projection(
            torch.cat((directional.mean(dim=-1), directional.amax(dim=-1)), dim=1)
        )
        batch = range_m.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1) + context.unsqueeze(1)
        token_features, attention = self.attention(
            queries,
            directional.transpose(1, 2),
            directional.transpose(1, 2),
            need_weights=True,
            average_attn_weights=True,
        )
        raw = self.head(token_features)
        offset = 0
        presence_logits = raw[..., offset]
        offset += 1
        type_logits = raw[..., offset : offset + len(EVENT_TYPE_NAMES)]
        offset += len(EVENT_TYPE_NAMES)
        radial_fraction = torch.sigmoid(raw[..., offset])
        offset += 1
        elevation_rad = (torch.pi / 2.0) * torch.tanh(raw[..., offset])
        offset += 1
        descriptor = F.normalize(
            raw[..., offset : offset + self.config.event_set.descriptor_dim],
            dim=-1,
            eps=1e-8,
        )
        offset += self.config.event_set.descriptor_dim
        uncertainty_m = F.softplus(raw[..., offset]) + 1e-4

        weighted_support = attention * free_range_profile_m[:, None]
        free_range_anchor_m = weighted_support.sum(dim=-1)
        angle = self.polar_azimuth_rad.to(dtype=range_m.dtype)
        support_vector_forward_left = torch.stack(
            (
                (weighted_support * torch.cos(angle)).sum(dim=-1),
                (weighted_support * torch.sin(angle)).sum(dim=-1),
            ),
            dim=-1,
        )
        heading_concentration = torch.linalg.vector_norm(
            support_vector_forward_left, dim=-1
        ) / free_range_anchor_m.clamp_min(1e-8)
        heading_forward_left = F.normalize(
            support_vector_forward_left, dim=-1, eps=1e-8
        )
        horizontal_scale = radial_fraction * torch.cos(elevation_rad)
        relative_xyz_m = torch.stack(
            (
                horizontal_scale * support_vector_forward_left[..., 0],
                horizontal_scale * support_vector_forward_left[..., 1],
                radial_fraction * free_range_anchor_m * torch.sin(elevation_rad),
            ),
            dim=-1,
        )
        radial_distance_m = torch.linalg.vector_norm(relative_xyz_m, dim=-1)
        return {
            "event_presence_logits": presence_logits,
            "event_confidence": torch.sigmoid(presence_logits),
            "event_type_logits": type_logits,
            "event_type_probabilities": torch.softmax(type_logits, dim=-1),
            "event_relative_xyz_m": relative_xyz_m,
            "event_radial_distance_m": radial_distance_m,
            "event_radial_fraction": radial_fraction,
            "event_free_range_anchor_m": free_range_anchor_m,
            "event_heading_forward_left": heading_forward_left,
            "event_heading_concentration": heading_concentration,
            "event_elevation_rad": elevation_rad,
            "event_descriptor": descriptor,
            "event_uncertainty_m": uncertainty_m,
            "event_attention": attention,
            "free_range_profile_m": free_range_profile_m,
        }


def geometry_anchored_input_contract() -> Mapping[str, object]:
    return {
        "student_inputs": ("five_frame_range_m", "five_frame_valid_mask"),
        "forbidden_inputs": (
            "world_pose",
            "sensor_pose",
            "route_identity",
            "traversal_identity",
            "TNG_identity",
            "future_frame",
            "teacher_feature",
        ),
        "range_shape": (HISTORY_FRAMES, ELEVATION_ROWS, AZIMUTH_COLUMNS),
        "polar_bins": POLAR_BINS,
        "event_range_rule": "radial_fraction * attention_weighted_five_frame_local_free_range_envelope",
        "local_support_radius_bins": LOCAL_SUPPORT_RADIUS_BINS,
        "los_support_margin_m": LOS_SUPPORT_MARGIN_M,
    }


__all__ = [
    "AZIMUTH_COLUMNS",
    "ELEVATION_BANDS",
    "ELEVATION_DEGREES",
    "ELEVATION_ROWS",
    "GeometryAnchoredSpatialEventConfig",
    "GeometryAnchoredSpatialEventEncoder",
    "HISTORY_FRAMES",
    "LOCAL_SUPPORT_RADIUS_BINS",
    "LOS_SUPPORT_MARGIN_M",
    "POLAR_BINS",
    "SENSOR_MAXIMUM_RANGE_M",
    "SENSOR_NEAR_RANGE_M",
    "geometry_anchored_input_contract",
]
