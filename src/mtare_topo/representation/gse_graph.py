"""Five-frame causal geometry-semantic encoder for GSE-Graph."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.phase3_structural_semantics import (
    CircularAzimuthConv2d,
    ResidualRangeBlock,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


METRIC_DISTANCE_SCALE_M = 30.0
SLOPE_SCALE_DEG = 45.0
CURVATURE_SCALE_PER_M = 0.1
EXIT_PROFILE_SCALE_M = 10.0


@dataclass(frozen=True)
class GSEModelConfig:
    history_frames: int = 5
    encoder_dim: int = 128
    place_descriptor_dim: int = 128
    exit_descriptor_dim: int = 32
    vertical_profile_dim: int = 4
    maximum_exit_tokens: int = 6


@dataclass(frozen=True)
class GSELossWeights:
    event: float = 1.0
    axis: float = 1.0
    geometry: float = 1.0
    exit_presence: float = 1.0
    exit_geometry: float = 1.0
    association: float = 1.0
    exit_association: float = 1.0
    uncertainty: float = 0.1


class GeometrySemanticEventNet(nn.Module):
    """Encode only the current and four past organized LiDAR frames.

    Input is ``[batch, time=5, channels=2, elevation=16, azimuth=720]``.
    The first channel is normalized range and the second is the valid-return
    mask. No pose, topology identity, complete map, or future observation is an
    input to this module.
    """

    def __init__(self, config: GSEModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or GSEModelConfig()
        dim = self.config.encoder_dim
        if dim != 128:
            raise ValueError("the V1 encoder contract fixes encoder_dim=128")
        self.encoder = nn.Sequential(
            CircularAzimuthConv2d(2, 32, stride=(1, 2)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResidualRangeBlock(32, 64, (2, 2)),
            ResidualRangeBlock(64, 96, (2, 1)),
            ResidualRangeBlock(96, dim, (2, 1)),
        )
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.directional_temporal = nn.Sequential(
            nn.Conv1d(self.config.history_frames * dim, dim, 1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
        )
        self.event_head = nn.Linear(dim, len(EVENT_NAMES))
        self.axis_azimuth_head = nn.Sequential(
            nn.Conv1d(dim, 64, 1), nn.SiLU(), nn.Conv1d(64, 1, 1)
        )
        self.axis_vertical_head = nn.Linear(dim, 1)
        self.geometry_head = nn.Linear(dim, 4)
        self.place_head = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, self.config.place_descriptor_dim)
        )
        self.uncertainty_head = nn.Linear(dim, 1)

        self.exit_queries = nn.Parameter(torch.empty(self.config.maximum_exit_tokens, dim))
        nn.init.normal_(self.exit_queries, std=0.02)
        self.exit_attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        exit_size = (
            1
            + 1
            + self.config.vertical_profile_dim
            + self.config.exit_descriptor_dim
        )
        self.exit_head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, exit_size))

    def encode_causal_features(self, scans: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return frozen causal features without discarding azimuth layout.

        Existing checkpoints are unchanged: this method only exposes tensors
        that the original forward pass already computed.  In particular,
        ``azimuth_sequence`` retains the five-frame directional layout that the
        V1 event head averaged away.
        """

        expected = (self.config.history_frames, 2, 16, 720)
        if scans.ndim != 5 or tuple(scans.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected[0]},2,16,720], got {tuple(scans.shape)}")
        if not torch.isfinite(scans).all():
            raise ValueError("GSE input contains non-finite values")
        batch, history = scans.shape[:2]
        features = self.encoder(scans.reshape(batch * history, 2, 16, 720))
        pooled = features.mean(dim=(2, 3)).reshape(batch, history, -1)
        temporal_values, _ = self.temporal(pooled)
        context = temporal_values[:, -1]

        feature_sequence = features.reshape(
            batch, history, 128, features.shape[-2], features.shape[-1]
        )
        azimuth_sequence = feature_sequence.mean(dim=3)
        directional = self.directional_temporal(
            azimuth_sequence.reshape(batch, history * 128, features.shape[-1])
        )
        return {
            "context": context,
            "azimuth_sequence": azimuth_sequence,
            "directional": directional,
        }

    def forward(self, scans: torch.Tensor) -> dict[str, torch.Tensor]:
        causal = self.encode_causal_features(scans)
        context = causal["context"]
        azimuth_sequence = causal["azimuth_sequence"]
        directional = causal["directional"]
        batch = scans.shape[0]
        current_azimuth = directional.transpose(1, 2)
        queries = self.exit_queries.unsqueeze(0).expand(batch, -1, -1) + context.unsqueeze(1)
        token_features, exit_attention = self.exit_attention(
            queries,
            current_azimuth,
            current_azimuth,
            need_weights=True,
            average_attn_weights=True,
        )
        token_raw = self.exit_head(token_features)
        offset = 0
        presence_logits = token_raw[..., offset]
        offset += 1
        opening_width = F.softplus(token_raw[..., offset]) + 1e-4
        offset += 1
        vertical_profile = token_raw[..., offset : offset + self.config.vertical_profile_dim]
        offset += self.config.vertical_profile_dim
        exit_descriptor = F.normalize(token_raw[..., offset:], dim=-1, eps=1e-8)
        azimuth_radians = torch.arange(
            current_azimuth.shape[1],
            device=current_azimuth.device,
            dtype=current_azimuth.dtype,
        ) * (2.0 * torch.pi / current_azimuth.shape[1])
        sine = torch.sin(azimuth_radians)
        cosine = torch.cos(azimuth_radians)
        heading_unit = F.normalize(
            torch.stack(
                (
                    (exit_attention * sine).sum(dim=-1),
                    (exit_attention * cosine).sum(dim=-1),
                ),
                dim=-1,
            ),
            dim=-1,
            eps=1e-8,
        )
        heading_deg = torch.remainder(torch.rad2deg(torch.atan2(heading_unit[..., 0], heading_unit[..., 1])), 360.0)

        geometry = self.geometry_head(context)
        axis_probability = torch.softmax(self.axis_azimuth_head(directional).squeeze(1), dim=-1)
        horizontal_axis = F.normalize(
            torch.stack(
                (
                    (axis_probability * cosine).sum(dim=-1),
                    (axis_probability * sine).sum(dim=-1),
                ),
                dim=-1,
            ),
            dim=-1,
            eps=1e-8,
        )
        local_axis = F.normalize(
            torch.cat((horizontal_axis, torch.tanh(self.axis_vertical_head(context))), dim=-1),
            dim=-1,
            eps=1e-8,
        )
        return {
            "event_logits": self.event_head(context),
            "local_axis": local_axis,
            "width_m": F.softplus(geometry[:, 0]) + 1e-4,
            "height_m": F.softplus(geometry[:, 1]) + 1e-4,
            "slope_deg": 45.0 * torch.tanh(geometry[:, 2]),
            "curvature_per_m": F.softplus(geometry[:, 3]),
            "place_descriptor": F.normalize(self.place_head(context), dim=-1, eps=1e-8),
            "uncertainty": torch.sigmoid(self.uncertainty_head(context).squeeze(-1)),
            "exit_presence_logits": presence_logits,
            "exit_confidence": torch.sigmoid(presence_logits),
            "exit_heading_unit": heading_unit,
            "exit_heading_robot_deg": heading_deg,
            "exit_opening_width_m": opening_width,
            "exit_vertical_profile": vertical_profile,
            "exit_descriptor": exit_descriptor,
        }


def supervised_association_loss(
    descriptors: torch.Tensor,
    identity_labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss for node/place identity association."""

    if descriptors.ndim != 2 or identity_labels.ndim != 1 or descriptors.shape[0] != identity_labels.shape[0]:
        raise ValueError("association inputs must be [N,D] descriptors and [N] labels")
    if descriptors.shape[0] < 2:
        raise ValueError("association loss requires at least two observations")
    if not temperature > 0.0:
        raise ValueError("temperature must be positive")
    normalized = F.normalize(descriptors, dim=1)
    logits = normalized @ normalized.T / float(temperature)
    self_mask = torch.eye(len(normalized), dtype=torch.bool, device=normalized.device)
    positive_mask = identity_labels[:, None].eq(identity_labels[None, :]) & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        raise ValueError("association batch contains no positive identity pair")
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive = (log_probability * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    anchor_loss = -mean_positive[valid_anchor]
    anchor_identity = identity_labels[valid_anchor]
    identity_losses = [
        anchor_loss[anchor_identity == identity].mean()
        for identity in torch.unique(anchor_identity, sorted=True)
    ]
    return torch.stack(identity_losses).mean()


def _minimum_cost_assignment(cost: torch.Tensor) -> list[tuple[int, int]]:
    """Exact deterministic small-set assignment; target columns are all matched."""

    if cost.ndim != 2 or cost.shape[1] > cost.shape[0]:
        raise ValueError("assignment cost must have predictions >= targets")
    values = cost.detach().to(device="cpu", dtype=torch.float64).numpy()
    prediction_count, target_count = values.shape

    @lru_cache(maxsize=None)
    def solve(target_index: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if target_index == target_count:
            return 0.0, ()
        best: tuple[float, tuple[int, ...]] | None = None
        for prediction_index in range(prediction_count):
            if used_mask & (1 << prediction_index):
                continue
            tail_cost, tail = solve(target_index + 1, used_mask | (1 << prediction_index))
            candidate = (float(values[prediction_index, target_index]) + tail_cost, (prediction_index,) + tail)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("no complete token assignment")
        return best

    _, predictions = solve(0, 0)
    return [(prediction_index, target_index) for target_index, prediction_index in enumerate(predictions)]


def gse_multitask_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    weights: GSELossWeights | None = None,
    association_temperature: float = 0.1,
    event_class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Event, metric geometry, set-token, association and uncertainty loss."""

    loss_weights = weights or GSELossWeights()
    if event_class_weights is not None:
        event_class_weights = event_class_weights.to(
            device=outputs["event_logits"].device,
            dtype=outputs["event_logits"].dtype,
        )
        if (
            event_class_weights.shape != (len(EVENT_NAMES),)
            or not torch.isfinite(event_class_weights).all()
            or bool((event_class_weights <= 0.0).any())
        ):
            raise ValueError("event_class_weights must contain one finite positive value per event")
    event = F.cross_entropy(
        outputs["event_logits"],
        targets["event_index"].long(),
        weight=event_class_weights,
    )
    target_axis = F.normalize(targets["local_axis"].float(), dim=-1)
    axis = (1.0 - (outputs["local_axis"] * target_axis).sum(dim=-1)).mean()
    predicted_geometry = torch.stack(
        (
            outputs["width_m"] / METRIC_DISTANCE_SCALE_M,
            outputs["height_m"] / METRIC_DISTANCE_SCALE_M,
            outputs["slope_deg"] / SLOPE_SCALE_DEG,
            outputs["curvature_per_m"] / CURVATURE_SCALE_PER_M,
        ),
        dim=-1,
    )
    target_geometry = torch.stack(
        (
            targets["width_m"].float() / METRIC_DISTANCE_SCALE_M,
            targets["height_m"].float() / METRIC_DISTANCE_SCALE_M,
            targets["slope_deg"].float() / SLOPE_SCALE_DEG,
            targets["curvature_per_m"].float() / CURVATURE_SCALE_PER_M,
        ),
        dim=-1,
    )
    geometry_valid = targets.get("geometry_valid_mask")
    if geometry_valid is None:
        geometry_valid = torch.ones_like(target_geometry, dtype=torch.bool)
    else:
        geometry_valid = geometry_valid.bool()
        if geometry_valid.shape != target_geometry.shape:
            raise ValueError("geometry_valid_mask must have shape [B,4]")
    if not geometry_valid.any(dim=1).all():
        raise ValueError("every sample must retain at least one objective geometry target")
    safe_target_geometry = torch.where(geometry_valid, target_geometry, predicted_geometry.detach())
    element_geometry = F.smooth_l1_loss(
        predicted_geometry,
        safe_target_geometry,
        reduction="none",
    )
    per_sample_geometry = (
        (element_geometry * geometry_valid).sum(dim=-1)
        / geometry_valid.sum(dim=-1).clamp_min(1)
    )
    geometry = per_sample_geometry.mean()

    target_mask = targets["exit_mask"].bool()
    exit_width_valid = targets.get("exit_width_valid_mask")
    if exit_width_valid is None:
        exit_width_valid = target_mask
    else:
        exit_width_valid = exit_width_valid.bool()
        if exit_width_valid.shape != target_mask.shape:
            raise ValueError("exit_width_valid_mask must align with exit_mask")
        if bool((exit_width_valid & ~target_mask).any()):
            raise ValueError("exit width cannot be valid outside the visible target set")
    presence_target = torch.zeros_like(outputs["exit_presence_logits"])
    heading_terms: list[torch.Tensor] = []
    width_terms: list[torch.Tensor] = []
    profile_terms: list[torch.Tensor] = []
    matched_exit_descriptors: list[torch.Tensor] = []
    matched_exit_identities: list[torch.Tensor] = []
    target_exit_identity = targets.get("exit_identity")
    if target_exit_identity is not None and target_exit_identity.shape != target_mask.shape:
        raise ValueError("exit_identity must align with exit_mask")
    for batch_index in range(outputs["exit_presence_logits"].shape[0]):
        valid_targets = torch.nonzero(target_mask[batch_index], as_tuple=False).flatten()
        if len(valid_targets) == 0:
            continue
        predicted_heading = outputs["exit_heading_unit"][batch_index]
        target_heading = F.normalize(targets["exit_heading_unit"][batch_index, valid_targets].float(), dim=-1)
        predicted_width = outputs["exit_opening_width_m"][batch_index]
        target_width = targets["exit_opening_width_m"][batch_index, valid_targets].float()
        compact_width_valid = exit_width_valid[batch_index, valid_targets]
        matching_cost = 1.0 - predicted_heading @ target_heading.T
        width_cost = torch.abs(
            torch.log(predicted_width[:, None].clamp_min(1e-4) / target_width[None, :].clamp_min(1e-4))
        )
        matching_cost = matching_cost + width_cost * compact_width_valid[None, :]
        for prediction_index, compact_target_index in _minimum_cost_assignment(matching_cost):
            target_index = int(valid_targets[compact_target_index])
            presence_target[batch_index, prediction_index] = 1.0
            heading_terms.append(1.0 - (predicted_heading[prediction_index] * target_heading[compact_target_index]).sum())
            if bool(exit_width_valid[batch_index, target_index]):
                width_terms.append(
                    F.smooth_l1_loss(
                        predicted_width[prediction_index] / METRIC_DISTANCE_SCALE_M,
                        targets["exit_opening_width_m"][batch_index, target_index].float()
                        / METRIC_DISTANCE_SCALE_M,
                    )
                )
            profile_terms.append(
                F.smooth_l1_loss(
                    outputs["exit_vertical_profile"][batch_index, prediction_index]
                    / EXIT_PROFILE_SCALE_M,
                    targets["exit_vertical_profile"][batch_index, target_index].float()
                    / EXIT_PROFILE_SCALE_M,
                )
            )
            if target_exit_identity is not None:
                matched_exit_descriptors.append(outputs["exit_descriptor"][batch_index, prediction_index])
                matched_exit_identities.append(target_exit_identity[batch_index, target_index].long())
    exit_presence = F.binary_cross_entropy_with_logits(outputs["exit_presence_logits"], presence_target)
    zero = outputs["event_logits"].sum() * 0.0
    exit_geometry = zero
    if heading_terms:
        exit_geometry = torch.stack(heading_terms).mean() + torch.stack(profile_terms).mean()
        if width_terms:
            exit_geometry = exit_geometry + torch.stack(width_terms).mean()
    association_valid = targets.get("association_valid_mask")
    if association_valid is None:
        association_valid = torch.ones_like(targets["association_identity"], dtype=torch.bool)
    else:
        association_valid = association_valid.bool()
        if association_valid.shape != targets["association_identity"].shape:
            raise ValueError("association_valid_mask must align with association_identity")
    association = zero
    valid_descriptors = outputs["place_descriptor"][association_valid]
    valid_identities = targets["association_identity"][association_valid].long()
    if len(valid_identities) >= 2:
        _, identity_counts = torch.unique(valid_identities, return_counts=True)
        if bool((identity_counts >= 2).any()):
            association = supervised_association_loss(
                valid_descriptors,
                valid_identities,
                temperature=association_temperature,
            )
    exit_association = zero
    if matched_exit_descriptors:
        exit_descriptors = torch.stack(matched_exit_descriptors)
        exit_identities = torch.stack(matched_exit_identities)
        unique, counts = torch.unique(exit_identities, return_counts=True)
        if bool((counts >= 2).any()):
            exit_association = supervised_association_loss(
                exit_descriptors,
                exit_identities,
                temperature=association_temperature,
            )
    variance = outputs["uncertainty"].square().clamp_min(1e-4)
    uncertainty = (0.5 * (per_sample_geometry / variance + torch.log(variance))).mean()
    total = (
        loss_weights.event * event
        + loss_weights.axis * axis
        + loss_weights.geometry * geometry
        + loss_weights.exit_presence * exit_presence
        + loss_weights.exit_geometry * exit_geometry
        + loss_weights.association * association
        + loss_weights.exit_association * exit_association
        + loss_weights.uncertainty * uncertainty
    )
    return {
        "total": total,
        "event": event,
        "axis": axis,
        "geometry": geometry,
        "exit_presence": exit_presence,
        "exit_geometry": exit_geometry,
        "association": association,
        "exit_association": exit_association,
        "uncertainty": uncertainty,
    }


__all__ = [
    "CURVATURE_SCALE_PER_M",
    "EXIT_PROFILE_SCALE_M",
    "GSELossWeights",
    "GSEModelConfig",
    "GeometrySemanticEventNet",
    "METRIC_DISTANCE_SCALE_M",
    "SLOPE_SCALE_DEG",
    "gse_multitask_loss",
    "supervised_association_loss",
]
