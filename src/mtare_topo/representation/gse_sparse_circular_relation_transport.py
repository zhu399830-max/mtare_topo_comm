"""Sparse circular exit tokens with explicit causal relation transport.

This is the minimal V2 representation selected after the sealed dense-field
failure attribution.  Forward accepts only five organized LiDAR scans.  Exit
identity is a loss-side target and never a model input.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.gse_circular_peak_geometry_model import (
    AZIMUTH_COLUMNS,
    BEARING_BINS,
    ENCODER_DIM,
    ELEVATION_ROWS,
    HISTORY_FRAMES,
    METRIC_DISTANCE_SCALE_M,
    PROFILE_DIM,
    PROFILE_SCALE_M,
    SLOPE_SCALE_DEG,
    CURVATURE_SCALE_PER_M,
    CircularPeakGeometrySemanticNet,
)


MAX_TOKENS = 6
TOKEN_DIM = 64
TOKEN_DESCRIPTOR_DIM = 32
SOFT_SUPPORT_RADIUS_BINS = 4
EVENT_CLASSES = 5


@dataclass(frozen=True)
class SparseCircularRelationTransportConfig:
    history_frames: int = HISTORY_FRAMES
    bearing_bins: int = BEARING_BINS
    maximum_tokens: int = MAX_TOKENS
    soft_support_radius_bins: int = SOFT_SUPPORT_RADIUS_BINS

    def __post_init__(self) -> None:
        if (
            self.history_frames != HISTORY_FRAMES
            or self.bearing_bins != BEARING_BINS
            or self.maximum_tokens != MAX_TOKENS
            or self.soft_support_radius_bins != SOFT_SUPPORT_RADIUS_BINS
        ):
            raise ValueError("sparse relation-transport configuration is frozen")


def periodic_gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather circular directional features at integer bearing indices."""
    if values.ndim != 3 or index.ndim != 2 or values.shape[0] != index.shape[0]:
        raise ValueError("periodic gather expects [B,C,N] and [B,K]")
    wrapped = torch.remainder(index.long(), values.shape[-1])
    return torch.gather(values, 2, wrapped[:, None].expand(-1, values.shape[1], -1)).transpose(1, 2)


def circular_nms_indices(
    logits: torch.Tensor,
    *,
    token_count: int = MAX_TOKENS,
    radius: int = SOFT_SUPPORT_RADIUS_BINS,
) -> torch.Tensor:
    """Select distinct circular proposals using the frozen local support."""
    if logits.ndim < 2 or logits.shape[-1] != BEARING_BINS:
        raise ValueError("circular NMS expects [...,180] logits")
    if token_count != MAX_TOKENS or radius != SOFT_SUPPORT_RADIUS_BINS:
        raise ValueError("circular NMS configuration drift")
    flat = logits.reshape(-1, BEARING_BINS)
    working = flat.clone()
    selected = []
    offsets = torch.arange(-radius, radius + 1, device=logits.device)
    rows = torch.arange(len(flat), device=logits.device)[:, None]
    for _ in range(token_count):
        index = working.argmax(dim=-1)
        selected.append(index)
        suppressed = torch.remainder(index[:, None] + offsets[None], BEARING_BINS)
        working[rows, suppressed] = -torch.inf
    return torch.stack(selected, dim=-1).reshape(*logits.shape[:-1], token_count)


def circular_soft_token_features(
    directional: torch.Tensor,
    logits: torch.Tensor,
    token_index: torch.Tensor,
    *,
    radius: int = SOFT_SUPPORT_RADIUS_BINS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode token features and continuous bearings in a fixed local support."""
    if directional.ndim != 3 or logits.shape != (len(directional), directional.shape[-1]) or token_index.ndim != 2:
        raise ValueError("soft token decoder shape drift")
    if directional.shape[-1] != BEARING_BINS or radius != SOFT_SUPPORT_RADIUS_BINS:
        raise ValueError("soft token decoder support drift")
    offsets = torch.arange(-radius, radius + 1, device=directional.device)
    support = torch.remainder(token_index[..., None] + offsets, BEARING_BINS)
    support_logits = torch.gather(logits[:, None].expand(-1, token_index.shape[1], -1), 2, support)
    weight = torch.softmax(support_logits, dim=-1)
    expanded = directional[:, None].expand(-1, token_index.shape[1], -1, -1)
    feature_index = support[:, :, None].expand(-1, -1, directional.shape[1], -1)
    local_feature = torch.gather(expanded, 3, feature_index)
    feature = (local_feature * weight[:, :, None]).sum(dim=-1)
    angle = support.to(directional.dtype) * (2.0 * torch.pi / BEARING_BINS)
    x = (weight * torch.cos(angle)).sum(dim=-1); y = (weight * torch.sin(angle)).sum(dim=-1)
    bearing_deg = torch.remainder(torch.rad2deg(torch.atan2(y, x)), 360.0)
    return feature, bearing_deg


def relation_targets_from_token_identities(token_identity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build previous-to-current rows and current reveal labels from token IDs.

    ``row_target[...,i]`` is a current-token column or ``K`` (dustbin).
    ``reveal_target[...,j]`` is 1 for a new current identity, 0 for a matched
    identity and -1 for an absent current token.
    """
    if token_identity.ndim != 3 or tuple(token_identity.shape[1:]) != (HISTORY_FRAMES, MAX_TOKENS):
        raise ValueError("token identities must be [B,5,6]")
    identity = token_identity.long()
    row_target = torch.full((len(identity), HISTORY_FRAMES - 1, MAX_TOKENS), -1, dtype=torch.long, device=identity.device)
    reveal_target = torch.full_like(row_target, -1)
    for batch in range(len(identity)):
        for step in range(HISTORY_FRAMES - 1):
            previous = identity[batch, step]; current = identity[batch, step + 1]
            for values in (previous, current):
                valid = values[values >= 0]
                if len(valid) != len(torch.unique(valid)):
                    raise ValueError("one frame cannot contain duplicate physical token identity")
            current_lookup = {int(value): index for index, value in enumerate(current.tolist()) if value >= 0}
            previous_set = {int(value) for value in previous.tolist() if value >= 0}
            for index, value in enumerate(previous.tolist()):
                if value >= 0:
                    row_target[batch, step, index] = current_lookup.get(int(value), MAX_TOKENS)
            for index, value in enumerate(current.tolist()):
                if value >= 0:
                    reveal_target[batch, step, index] = int(value not in previous_set)
    return row_target, reveal_target


def reverse_transport_logits(
    row_logits: torch.Tensor, reveal_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse one or more pairwise transports and swap reveal/withdraw."""
    if row_logits.shape[-2:] != (MAX_TOKENS, MAX_TOKENS + 1) or reveal_logits.shape != row_logits.shape[:-1]:
        raise ValueError("transport reverse shape drift")
    match = row_logits[..., :MAX_TOKENS]
    withdraw = row_logits[..., MAX_TOKENS]
    reverse_row = torch.cat((match.transpose(-2, -1), reveal_logits[..., None]), dim=-1)
    return reverse_row, withdraw


class SparseCircularRelationTransportNet(CircularPeakGeometrySemanticNet):
    """Causal circular geometry backbone plus sparse token transport."""

    def __init__(self, config: SparseCircularRelationTransportConfig | None = None) -> None:
        super().__init__()
        self.transport_config = config or SparseCircularRelationTransportConfig()
        del self.peak_head
        self.event_head = nn.Linear(ENCODER_DIM, EVENT_CLASSES)
        self.token_count_head = nn.Linear(ENCODER_DIM, MAX_TOKENS + 1)
        self.proposal_head = nn.Sequential(
            nn.Conv1d(ENCODER_DIM, ENCODER_DIM, 3, padding=1, padding_mode="circular"),
            nn.SiLU(), nn.Conv1d(ENCODER_DIM, 1, 1),
        )
        self.token_projection = nn.Sequential(nn.Linear(ENCODER_DIM, TOKEN_DIM), nn.SiLU())
        self.token_descriptor_head = nn.Linear(TOKEN_DIM, TOKEN_DESCRIPTOR_DIM)
        self.token_geometry_head = nn.Linear(TOKEN_DIM, 2 + 2 * PROFILE_DIM)
        self.match_query = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False)
        self.match_key = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False)
        self.match_bearing = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))
        self.withdraw_head = nn.Linear(TOKEN_DIM, 1)
        self.reveal_head = nn.Linear(TOKEN_DIM, 1)

    def encode_causal_directional(self, scans: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_scans(scans)
        batch = len(scans)
        encoded = self.encoder(scans.reshape(batch * HISTORY_FRAMES, 2, ELEVATION_ROWS, AZIMUTH_COLUMNS))
        if encoded.shape[-1] != BEARING_BINS:
            raise RuntimeError("sparse transport bearing resolution drift")
        pooled = encoded.mean(dim=(2, 3)).reshape(batch, HISTORY_FRAMES, ENCODER_DIM)
        temporal, _ = self.temporal(pooled); context = temporal[:, -1]
        sequence = encoded.reshape(batch, HISTORY_FRAMES, ENCODER_DIM, encoded.shape[-2], BEARING_BINS).mean(dim=3)
        causal = []
        for index in range(HISTORY_FRAMES):
            masked = torch.zeros_like(sequence); masked[:, : index + 1] = sequence[:, : index + 1]
            causal.append(self.directional_temporal(masked.reshape(batch, HISTORY_FRAMES * ENCODER_DIM, BEARING_BINS)))
        causal_directional = torch.stack(causal, dim=1)
        return context, causal_directional[:, -1], causal_directional, temporal

    def forward(self, scans: torch.Tensor) -> dict[str, torch.Tensor]:
        context, current_directional, causal_directional, frame_context = self.encode_causal_directional(scans)
        batch = len(scans)
        proposal_logits = self.proposal_head(causal_directional.reshape(batch * HISTORY_FRAMES, ENCODER_DIM, BEARING_BINS)).reshape(batch, HISTORY_FRAMES, BEARING_BINS)
        token_index = circular_nms_indices(proposal_logits)
        token_features = []; token_bearings = []
        for frame in range(HISTORY_FRAMES):
            feature, bearing = circular_soft_token_features(causal_directional[:, frame], proposal_logits[:, frame], token_index[:, frame])
            token_features.append(self.token_projection(feature)); token_bearings.append(bearing)
        token_feature = torch.stack(token_features, dim=1); token_bearing_deg = torch.stack(token_bearings, dim=1)
        token_existence_logits = torch.gather(proposal_logits, 2, token_index)
        token_descriptor = F.normalize(self.token_descriptor_head(token_feature), dim=-1, eps=1e-8)
        raw_geometry = self.token_geometry_head(token_feature)
        token_opening_width_m = F.softplus(raw_geometry[..., 0]) + 1e-4
        token_vertical_profile_m = PROFILE_SCALE_M * torch.tanh(raw_geometry[..., 1 : 1 + PROFILE_DIM])
        token_geometry_uncertainty = F.softplus(raw_geometry[..., 1 + PROFILE_DIM :]) + 0.05
        if token_geometry_uncertainty.shape[-1] != 1 + PROFILE_DIM:
            raise RuntimeError("token geometry uncertainty dimension drift")
        row_logits = []; reveal_logits = []
        for step in range(HISTORY_FRAMES - 1):
            previous = token_feature[:, step]; current = token_feature[:, step + 1]
            match = self.match_query(previous) @ self.match_key(current).transpose(1, 2) / math.sqrt(TOKEN_DIM)
            delta = torch.deg2rad(token_bearing_deg[:, step, :, None] - token_bearing_deg[:, step + 1, None, :])
            bearing_feature = torch.stack((torch.cos(delta), torch.sin(delta)), dim=-1)
            match = match + self.match_bearing(bearing_feature).squeeze(-1)
            withdraw = self.withdraw_head(previous).squeeze(-1)
            row_logits.append(torch.cat((match, withdraw[..., None]), dim=-1))
            reveal_logits.append(self.reveal_head(current).squeeze(-1))
        transport_row_logits = torch.stack(row_logits, dim=1)
        transport_reveal_logits = torch.stack(reveal_logits, dim=1)
        token_count_logits = self.token_count_head(frame_context)
        azimuth = self.bearing_azimuth_rad.to(dtype=scans.dtype)
        axis_probability = torch.softmax(self.axis_azimuth_head(current_directional).squeeze(1), dim=-1)
        horizontal_axis = F.normalize(torch.stack(((axis_probability * torch.cos(azimuth)).sum(-1), (axis_probability * torch.sin(azimuth)).sum(-1)), dim=-1), dim=-1, eps=1e-8)
        local_axis = F.normalize(torch.cat((horizontal_axis, torch.tanh(self.axis_vertical_head(context))), dim=-1), dim=-1, eps=1e-8)
        geometry = self.geometry_head(context); event_logits = self.event_head(context)
        return {
            "event_logits": event_logits, "event_probability": torch.softmax(event_logits, dim=-1),
            "proposal_logits": proposal_logits, "token_bin_index": token_index,
            "token_count_logits": token_count_logits,
            "token_count_probability": torch.softmax(token_count_logits, dim=-1),
            "token_bearing_deg": token_bearing_deg, "token_existence_logits": token_existence_logits,
            "token_descriptor": token_descriptor, "token_opening_width_m": token_opening_width_m,
            "token_vertical_profile_m": token_vertical_profile_m,
            "token_geometry_uncertainty": token_geometry_uncertainty,
            "transport_row_logits": transport_row_logits,
            "transport_row_probability": torch.softmax(transport_row_logits, dim=-1),
            "transport_reveal_logits": transport_reveal_logits,
            "transport_reveal_probability": torch.sigmoid(transport_reveal_logits),
            "local_axis": local_axis,
            "width_m": F.softplus(geometry[:, 0]) + 1e-4,
            "height_m": F.softplus(geometry[:, 1]) + 1e-4,
            "slope_deg": SLOPE_SCALE_DEG * torch.tanh(geometry[:, 2]),
            "curvature_per_m": F.softplus(geometry[:, 3]),
            "place_descriptor": F.normalize(self.place_head(context), dim=-1, eps=1e-8),
            "observation_uncertainty": torch.sigmoid(self.observation_uncertainty_head(context).squeeze(-1)),
        }


def token_count_loss(
    logits: torch.Tensor,
    presence: torch.Tensor,
    *,
    valid_frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise the explicit 0--6 token cardinality for each causal frame."""
    if logits.ndim != 3 or logits.shape[-1] != MAX_TOKENS + 1:
        raise ValueError("count loss expects logits [B,5,7]")
    if presence.shape != (*logits.shape[:2], BEARING_BINS):
        raise ValueError("count loss presence shape drift")
    if valid_frame_mask is None:
        valid_frame_mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
    if valid_frame_mask.shape != logits.shape[:2]:
        raise ValueError("count loss valid-frame mask shape drift")
    valid = valid_frame_mask.bool()
    if not bool(valid.any()):
        raise ValueError("count loss requires at least one valid frame")
    target = presence.bool().sum(dim=-1).long()
    if bool((target > MAX_TOKENS).any()):
        raise ValueError("count target exceeds frozen six-token capacity")
    return F.cross_entropy(logits[valid], target[valid])


def soft_circular_proposal_loss(
    logits: torch.Tensor,
    presence: torch.Tensor,
    *,
    valid_frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft circular support plus fixed-count hard-negative suppression."""
    if logits.ndim != 3 or logits.shape[-1] != BEARING_BINS or presence.shape != logits.shape:
        raise ValueError("proposal loss expects aligned [B,5,180]")
    if valid_frame_mask is None:
        valid_frame_mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
    if valid_frame_mask.shape != logits.shape[:2]:
        raise ValueError("proposal valid-frame mask shape drift")
    valid = valid_frame_mask.bool()
    if not bool(valid.any()):
        raise ValueError("proposal loss requires at least one valid frame")
    truth = presence.bool(); index = torch.arange(BEARING_BINS, device=logits.device)
    distance = torch.abs(index[None, None, :, None] - index[None, None, None, :])
    distance = torch.minimum(distance, BEARING_BINS - distance)
    support = torch.exp(-0.5 * (distance.to(logits.dtype) / 2.0).square())
    target = (truth[..., None, :] * support).amax(dim=-1)
    focal = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (0.25 + torch.abs(torch.sigmoid(logits) - target).square())
    core = focal[valid].mean()
    hard_terms = []
    for row in range(len(logits)):
        for frame in range(HISTORY_FRAMES):
            if not bool(valid[row, frame]):
                continue
            positive = truth[row, frame]
            if bool(positive.any()):
                protected = (distance[0, 0, :, positive] <= SOFT_SUPPORT_RADIUS_BINS).any(dim=-1)
                negative_logits = logits[row, frame][~protected]
                count = min(int(positive.sum()), len(negative_logits))
                if count:
                    hard_terms.append(F.softplus(torch.topk(negative_logits, count).values).mean())
    hard = torch.stack(hard_terms).mean() if hard_terms else logits.sum() * 0.0
    return core + hard


def sparse_relation_transport_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    proposal_presence: torch.Tensor,
    aligned_token_identity: torch.Tensor,
    proposal_valid_frame_mask: torch.Tensor | None = None,
    relation_pair_valid_mask: torch.Tensor | None = None,
    withdraw_weight: float = 1.0,
    reveal_positive_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Loss for dense proposal support and sparse identity transport."""
    if withdraw_weight < 1.0 or reveal_positive_weight < 1.0:
        raise ValueError("rare relation weights cannot down-weight positives")
    proposal = soft_circular_proposal_loss(
        outputs["proposal_logits"], proposal_presence,
        valid_frame_mask=proposal_valid_frame_mask,
    )
    row_target, reveal_target = relation_targets_from_token_identities(aligned_token_identity)
    row_valid = row_target >= 0; reveal_valid = reveal_target >= 0
    if relation_pair_valid_mask is not None:
        if relation_pair_valid_mask.shape != row_target.shape[:2]:
            raise ValueError("relation-pair valid mask shape drift")
        pair_valid = relation_pair_valid_mask.bool()[..., None]
        row_valid &= pair_valid
        reveal_valid &= pair_valid
    if not bool(row_valid.any()) or not bool(reveal_valid.any()):
        raise ValueError("transport loss requires valid sparse identities")
    row_element = F.cross_entropy(
        outputs["transport_row_logits"][row_valid], row_target[row_valid], reduction="none",
    )
    row_weight = torch.where(
        row_target[row_valid] == MAX_TOKENS,
        torch.as_tensor(withdraw_weight, dtype=row_element.dtype, device=row_element.device),
        torch.ones_like(row_element),
    )
    row = (row_element * row_weight).sum() / row_weight.sum()
    reveal = F.binary_cross_entropy_with_logits(
        outputs["transport_reveal_logits"][reveal_valid],
        reveal_target[reveal_valid].to(outputs["transport_reveal_logits"].dtype),
        pos_weight=torch.as_tensor(
            reveal_positive_weight,
            dtype=outputs["transport_reveal_logits"].dtype,
            device=outputs["transport_reveal_logits"].device,
        ),
    )
    match_probability = torch.softmax(outputs["transport_row_logits"], dim=-1)[..., :MAX_TOKENS]
    exclusivity = F.relu(match_probability.sum(dim=-2) - 1.0).square().mean()
    total = proposal + row + reveal + exclusivity
    return {"total": total, "proposal": proposal, "transport_row": row, "reveal": reveal, "column_exclusivity": exclusivity}


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in SparseCircularRelationTransportNet().parameters())


def sparse_circular_relation_transport_input_contract() -> dict[str, object]:
    signature = inspect.signature(SparseCircularRelationTransportNet.forward)
    return {
        "forward_parameters": tuple(signature.parameters),
        "input": "five past/current organized LiDAR range+valid scans [B,5,2,16,720]",
        "forbidden_forward_inputs": ("pose", "world", "traversal", "TNG", "exit_identity", "future", "graph", "planner"),
        "maximum_tokens": MAX_TOKENS,
        "relation_semantics": {"real_to_real": "persistent", "real_to_dustbin": "withdraw", "new_current": "reveal"},
        "axis_source": "five-frame causal directional-temporal fusion",
    }


__all__ = [
    "MAX_TOKENS", "SOFT_SUPPORT_RADIUS_BINS", "SparseCircularRelationTransportConfig",
    "SparseCircularRelationTransportNet", "circular_nms_indices", "circular_soft_token_features", "parameter_count",
    "periodic_gather", "relation_targets_from_token_identities", "reverse_transport_logits",
    "soft_circular_proposal_loss", "sparse_circular_relation_transport_input_contract",
    "sparse_relation_transport_loss", "token_count_loss",
]
