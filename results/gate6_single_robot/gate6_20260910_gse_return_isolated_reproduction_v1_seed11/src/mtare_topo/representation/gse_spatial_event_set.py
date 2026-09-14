"""Circular-directional set prediction for robot-relative structure events."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.gse_graph import _minimum_cost_assignment


EVENT_TYPE_NAMES = ("terminal", "junction")
MAXIMUM_EVENT_RANGE_M = 50.0


@dataclass(frozen=True)
class SpatialEventSetConfig:
    encoder_dim: int = 128
    query_count: int = 16
    descriptor_dim: int = 64
    attention_heads: int = 8
    maximum_range_m: float = MAXIMUM_EVENT_RANGE_M

    def __post_init__(self) -> None:
        if (
            self.encoder_dim != 128
            or self.query_count != 16
            or self.descriptor_dim <= 0
            or self.attention_heads <= 0
            or self.encoder_dim % self.attention_heads
            or not math.isfinite(float(self.maximum_range_m))
            or self.maximum_range_m <= 0.0
        ):
            raise ValueError("invalid frozen spatial event set config")


@dataclass(frozen=True)
class SpatialEventSetLossWeights:
    presence: float = 1.0
    event_type: float = 1.0
    position: float = 1.0
    descriptor: float = 0.1
    uncertainty: float = 0.1

    def __post_init__(self) -> None:
        values = (
            self.presence,
            self.event_type,
            self.position,
            self.descriptor,
            self.uncertainty,
        )
        if any(not math.isfinite(float(value)) or value < 0.0 for value in values):
            raise ValueError("spatial event loss weights must be finite and nonnegative")


class SpatialEventSetDecoder(nn.Module):
    """Decode a variable spatial event set from causal circular features.

    The decoder consumes only tensors already produced by
    ``GeometrySemanticEventNet.encode_causal_features``: an invariant causal
    context and an azimuth-indexed directional feature map.  It receives no
    world pose, TNG identity, absolute coordinate or future observation.
    """

    def __init__(self, config: SpatialEventSetConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpatialEventSetConfig()
        dim = self.config.encoder_dim
        self.queries = nn.Parameter(torch.empty(self.config.query_count, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads=self.config.attention_heads,
            batch_first=True,
        )
        # presence + 2 event classes + radial range + elevation + descriptor + uncertainty
        output_size = 1 + len(EVENT_TYPE_NAMES) + 1 + 1 + self.config.descriptor_dim + 1
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, output_size),
        )

    def forward(self, causal_features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(("context", "directional")) - set(causal_features):
            raise ValueError("spatial event decoder requires context and directional features")
        context = causal_features["context"]
        directional = causal_features["directional"]
        if (
            context.ndim != 2
            or context.shape[1] != self.config.encoder_dim
            or directional.ndim != 3
            or directional.shape[:2] != context.shape
            or directional.shape[2] < 4
            or not torch.isfinite(context).all()
            or not torch.isfinite(directional).all()
        ):
            raise ValueError("spatial event causal feature contract drift")
        batch = context.shape[0]
        azimuth_features = directional.transpose(1, 2)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1) + context.unsqueeze(1)
        token_features, attention = self.attention(
            queries,
            azimuth_features,
            azimuth_features,
            need_weights=True,
            average_attn_weights=True,
        )
        raw = self.head(token_features)
        offset = 0
        presence_logits = raw[..., offset]
        offset += 1
        type_logits = raw[..., offset : offset + len(EVENT_TYPE_NAMES)]
        offset += len(EVENT_TYPE_NAMES)
        radial_distance_m = self.config.maximum_range_m * torch.sigmoid(raw[..., offset])
        offset += 1
        elevation_rad = (torch.pi / 2.0) * torch.tanh(raw[..., offset])
        offset += 1
        descriptor = F.normalize(
            raw[..., offset : offset + self.config.descriptor_dim], dim=-1, eps=1e-8
        )
        offset += self.config.descriptor_dim
        uncertainty_m = F.softplus(raw[..., offset]) + 1e-4

        azimuth_radians = torch.arange(
            directional.shape[2],
            device=directional.device,
            dtype=directional.dtype,
        ) * (2.0 * torch.pi / directional.shape[2])
        cosine = torch.cos(azimuth_radians)
        sine = torch.sin(azimuth_radians)
        heading_forward_left = F.normalize(
            torch.stack(
                (
                    (attention * cosine).sum(dim=-1),
                    (attention * sine).sum(dim=-1),
                ),
                dim=-1,
            ),
            dim=-1,
            eps=1e-8,
        )
        horizontal_m = radial_distance_m * torch.cos(elevation_rad)
        relative_xyz_m = torch.stack(
            (
                horizontal_m * heading_forward_left[..., 0],
                horizontal_m * heading_forward_left[..., 1],
                radial_distance_m * torch.sin(elevation_rad),
            ),
            dim=-1,
        )
        return {
            "event_presence_logits": presence_logits,
            "event_confidence": torch.sigmoid(presence_logits),
            "event_type_logits": type_logits,
            "event_type_probabilities": torch.softmax(type_logits, dim=-1),
            "event_relative_xyz_m": relative_xyz_m,
            "event_radial_distance_m": radial_distance_m,
            "event_heading_forward_left": heading_forward_left,
            "event_elevation_rad": elevation_rad,
            "event_descriptor": descriptor,
            "event_uncertainty_m": uncertainty_m,
            "event_attention": attention,
        }


def validate_spatial_event_targets(
    targets: Mapping[str, torch.Tensor],
    *,
    query_count: int = 16,
    maximum_range_m: float = MAXIMUM_EVENT_RANGE_M,
) -> None:
    required = {
        "event_type_index",
        "event_relative_xyz_m",
        "event_identity_index",
        "event_mask",
    }
    if required - set(targets):
        raise ValueError(f"missing spatial event targets: {sorted(required - set(targets))}")
    event_type = targets["event_type_index"]
    relative = targets["event_relative_xyz_m"]
    identity = targets["event_identity_index"]
    mask = targets["event_mask"].bool()
    if (
        event_type.ndim != 2
        or event_type.shape[1] != query_count
        or relative.shape != (*event_type.shape, 3)
        or identity.shape != event_type.shape
        or mask.shape != event_type.shape
        or not torch.isfinite(relative).all()
    ):
        raise ValueError("spatial event target shape/finite contract drift")
    if bool(((event_type[mask] < 0) | (event_type[mask] >= len(EVENT_TYPE_NAMES))).any()):
        raise ValueError("active spatial event type index drift")
    if bool((event_type[~mask] != -1).any()) or bool((identity[~mask] != -1).any()):
        raise ValueError("spatial event padding sentinel drift")
    if bool((identity[mask] < 0).any()):
        raise ValueError("active spatial event identity drift")
    distance = torch.linalg.vector_norm(relative[mask], dim=-1)
    if bool((distance > float(maximum_range_m) + 1e-3).any()):
        raise ValueError("spatial event target exceeds frozen range")
    for row in range(len(mask)):
        active_identities = identity[row, mask[row]]
        if len(active_identities) != len(torch.unique(active_identities)):
            raise ValueError("duplicate spatial event identity in one target set")


def spatial_event_set_assignments(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    maximum_range_m: float = MAXIMUM_EVENT_RANGE_M,
) -> list[list[tuple[int, int]]]:
    """Return deterministic prediction-to-active-target assignments per batch."""

    presence_logits = outputs["event_presence_logits"]
    type_logits = outputs["event_type_logits"]
    relative = outputs["event_relative_xyz_m"]
    if (
        presence_logits.ndim != 2
        or type_logits.shape != (*presence_logits.shape, len(EVENT_TYPE_NAMES))
        or relative.shape != (*presence_logits.shape, 3)
        or not torch.isfinite(presence_logits).all()
        or not torch.isfinite(type_logits).all()
        or not torch.isfinite(relative).all()
    ):
        raise ValueError("spatial event output shape/finite contract drift")
    validate_spatial_event_targets(
        targets,
        query_count=presence_logits.shape[1],
        maximum_range_m=maximum_range_m,
    )
    if targets["event_mask"].shape[0] != presence_logits.shape[0]:
        raise ValueError("spatial event output/target batch drift")
    assignments: list[list[tuple[int, int]]] = []
    type_log_probability = F.log_softmax(type_logits, dim=-1)
    presence_cost = F.softplus(-presence_logits)
    # Transfer canonical matching signatures once per batch.  The previous
    # per-row transfer was mathematically identical but forced one device
    # synchronization per observation during joint encoder training.
    signature_parts = [
        presence_logits[..., None],
        type_logits,
        relative,
    ]
    for optional in ("event_uncertainty_m", "event_descriptor"):
        if optional in outputs:
            value = outputs[optional]
            signature_parts.append(value[..., None] if value.ndim == 2 else value)
    signatures = (
        torch.cat(signature_parts, dim=2)
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    for batch_index in range(presence_logits.shape[0]):
        active = torch.nonzero(targets["event_mask"][batch_index].bool(), as_tuple=False).flatten()
        if len(active) == 0:
            assignments.append([])
            continue
        target_type = targets["event_type_index"][batch_index, active].long()
        target_position = targets["event_relative_xyz_m"][batch_index, active].float()
        type_cost = -type_log_probability[batch_index, :, target_type]
        position_cost = torch.cdist(
            relative[batch_index].float(), target_position, p=1
        ) / (3.0 * float(maximum_range_m))
        cost = type_cost + position_cost + presence_cost[batch_index, :, None]
        # The exact small-set solver uses row index as its final tie-break.
        # Canonicalize rows by prediction content first so that this tie-break
        # cannot depend on the arbitrary learnable-query enumeration.
        signature = signatures[batch_index]
        canonical_order = sorted(
            range(len(signature)), key=lambda index: tuple(float(value) for value in signature[index])
        )
        canonical_tensor = torch.as_tensor(canonical_order, device=cost.device, dtype=torch.long)
        local_assignment = _minimum_cost_assignment(cost[canonical_tensor])
        assignments.append(
            [
                (canonical_order[prediction_index], int(active[target_index]))
                for prediction_index, target_index in local_assignment
            ]
        )
    return assignments


def _descriptor_loss_if_supported(
    descriptors: torch.Tensor,
    identities: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if len(descriptors) < 2:
        return descriptors.sum() * 0.0
    normalized = F.normalize(descriptors, dim=-1)
    logits = normalized @ normalized.T / float(temperature)
    self_mask = torch.eye(len(descriptors), dtype=torch.bool, device=descriptors.device)
    positive = identities[:, None].eq(identities[None, :]) & ~self_mask
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return descriptors.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator = torch.exp(logits).masked_fill(self_mask, 0.0).sum(dim=1).clamp_min(1e-12)
    log_probability = logits - torch.log(denominator[:, None])
    anchor_loss = -(log_probability * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    valid_loss = anchor_loss[valid]
    valid_identity = identities[valid]
    identity_means = [
        valid_loss[valid_identity == identity].mean()
        for identity in torch.unique(valid_identity, sorted=True)
    ]
    return torch.stack(identity_means).mean()


def spatial_event_set_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    weights: SpatialEventSetLossWeights | None = None,
    maximum_range_m: float = MAXIMUM_EVENT_RANGE_M,
    descriptor_temperature: float = 0.1,
    presence_positive_weight: float = 1.0,
    event_type_class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Masked set loss with deterministic assignment and empty-set support."""

    if not math.isfinite(float(descriptor_temperature)) or descriptor_temperature <= 0.0:
        raise ValueError("descriptor temperature must be finite and positive")
    if not math.isfinite(float(presence_positive_weight)) or presence_positive_weight <= 0.0:
        raise ValueError("presence positive weight must be finite and positive")
    loss_weights = weights or SpatialEventSetLossWeights()
    assignments = spatial_event_set_assignments(
        outputs, targets, maximum_range_m=maximum_range_m
    )
    presence_logits = outputs["event_presence_logits"]
    presence_target = torch.zeros_like(presence_logits)
    matched_predictions: list[int] = []
    matched_batch: list[int] = []
    matched_targets: list[int] = []
    for batch_index, pairs in enumerate(assignments):
        for prediction_index, target_index in pairs:
            presence_target[batch_index, prediction_index] = 1.0
            matched_batch.append(batch_index)
            matched_predictions.append(prediction_index)
            matched_targets.append(target_index)
    presence = F.binary_cross_entropy_with_logits(
        presence_logits,
        presence_target,
        pos_weight=presence_logits.new_tensor(float(presence_positive_weight)),
    )
    zero = presence_logits.sum() * 0.0
    if matched_predictions:
        batch_index = torch.as_tensor(matched_batch, device=presence_logits.device, dtype=torch.long)
        prediction_index = torch.as_tensor(
            matched_predictions, device=presence_logits.device, dtype=torch.long
        )
        target_index = torch.as_tensor(matched_targets, device=presence_logits.device, dtype=torch.long)
        target_type = targets["event_type_index"][batch_index, target_index].long()
        class_weights = None
        if event_type_class_weights is not None:
            class_weights = event_type_class_weights.to(
                device=presence_logits.device, dtype=presence_logits.dtype
            )
            if (
                class_weights.shape != (len(EVENT_TYPE_NAMES),)
                or not torch.isfinite(class_weights).all()
                or bool((class_weights <= 0.0).any())
            ):
                raise ValueError("event type class weights must be two finite positive values")
        event_type = F.cross_entropy(
            outputs["event_type_logits"][batch_index, prediction_index],
            target_type,
            weight=class_weights,
        )
        predicted_position = outputs["event_relative_xyz_m"][batch_index, prediction_index]
        target_position = targets["event_relative_xyz_m"][batch_index, target_index].float()
        position = F.smooth_l1_loss(
            predicted_position / float(maximum_range_m),
            target_position / float(maximum_range_m),
        )
        position_error_m = torch.linalg.vector_norm(predicted_position - target_position, dim=-1)
        uncertainty = F.smooth_l1_loss(
            outputs["event_uncertainty_m"][batch_index, prediction_index],
            position_error_m.detach(),
        )
        descriptor = _descriptor_loss_if_supported(
            outputs["event_descriptor"][batch_index, prediction_index],
            targets["event_identity_index"][batch_index, target_index].long(),
            temperature=descriptor_temperature,
        )
    else:
        event_type = zero
        position = zero
        uncertainty = zero
        descriptor = zero
    total = (
        loss_weights.presence * presence
        + loss_weights.event_type * event_type
        + loss_weights.position * position
        + loss_weights.descriptor * descriptor
        + loss_weights.uncertainty * uncertainty
    )
    return {
        "total": total,
        "presence": presence,
        "event_type": event_type,
        "position": position,
        "descriptor": descriptor,
        "uncertainty": uncertainty,
        "matched_event_count": presence_logits.new_tensor(float(len(matched_predictions))),
    }


__all__ = [
    "EVENT_TYPE_NAMES",
    "MAXIMUM_EVENT_RANGE_M",
    "SpatialEventSetConfig",
    "SpatialEventSetDecoder",
    "SpatialEventSetLossWeights",
    "spatial_event_set_assignments",
    "spatial_event_set_loss",
    "validate_spatial_event_targets",
]
