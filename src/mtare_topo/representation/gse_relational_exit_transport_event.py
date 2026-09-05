"""Causal relational exit-token transport model for GSE-Graph node events."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HISTORY = 5
SEEDS = 3
TOKENS = 6
RAW_TOKEN_DIM = 40
GEOMETRY_CONTEXT_DIM = 8
TOKEN_EMBED_DIM = 64
EVENT_QUERIES = 3


class RelationalExitTokenTransportEventModel(nn.Module):
    """Preserve individual exits through explicit adjacent-frame transport.

    Inputs contain no Teacher identity, pose, world or future frame.  Token
    slots and the three frozen perception seeds are treated as unordered sets.
    """

    def __init__(
        self,
        *,
        token_normalization_mean: np.ndarray | None = None,
        token_normalization_scale: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        mean = np.zeros(RAW_TOKEN_DIM, dtype=np.float32) if token_normalization_mean is None else np.asarray(token_normalization_mean, dtype=np.float32)
        scale = np.ones(RAW_TOKEN_DIM, dtype=np.float32) if token_normalization_scale is None else np.asarray(token_normalization_scale, dtype=np.float32)
        if (
            mean.shape != (RAW_TOKEN_DIM,)
            or scale.shape != (RAW_TOKEN_DIM,)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("token normalization must contain 40 finite dimensions")
        self.register_buffer("token_normalization_mean", torch.from_numpy(mean.copy()))
        self.register_buffer("token_normalization_scale", torch.from_numpy(scale.copy()))
        self.token_encoder = nn.Sequential(
            nn.Linear(RAW_TOKEN_DIM, 64), nn.SiLU(), nn.Linear(64, TOKEN_EMBED_DIM), nn.SiLU(),
        )
        self.affinity = nn.Sequential(nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 1))
        self.relational_token = nn.Sequential(
            nn.Linear(3 * TOKEN_EMBED_DIM + 2, 96), nn.SiLU(), nn.Linear(96, TOKEN_EMBED_DIM), nn.SiLU(),
        )
        self.event_queries = nn.Parameter(torch.empty(EVENT_QUERIES, TOKEN_EMBED_DIM))
        nn.init.normal_(self.event_queries, mean=0.0, std=0.02)
        self.set_attention = nn.MultiheadAttention(TOKEN_EMBED_DIM, 4, batch_first=True)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(GEOMETRY_CONTEXT_DIM, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(),
        )
        per_frame = 2 * EVENT_QUERIES * TOKEN_EMBED_DIM + 2 * 32
        self.frame_encoder = nn.Sequential(nn.Linear(per_frame, 128), nn.SiLU())
        self.temporal = nn.GRU(128, 128, batch_first=True)
        self.context = nn.Sequential(nn.Linear(256, 128), nn.SiLU())
        self.event_head = nn.Linear(128, 3)
        self.commit_head = nn.Linear(128, 1)

    @staticmethod
    def _pair_features(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        previous_descriptor = F.normalize(previous[..., 8:40], dim=-1, eps=1e-8)
        current_descriptor = F.normalize(current[..., 8:40], dim=-1, eps=1e-8)
        descriptor_cosine = torch.einsum("bsqd,bskd->bsqk", previous_descriptor, current_descriptor)
        heading_cosine = torch.einsum("bsqd,bskd->bsqk", previous[..., 1:3], current[..., 1:3])
        width_delta = torch.abs(
            torch.log(previous[..., 3].clamp_min(1e-4)[..., None])
            - torch.log(current[..., 3].clamp_min(1e-4)[..., None, :])
        )
        profile_delta = torch.mean(
            torch.abs(previous[..., 4:8][..., :, None, :] - current[..., 4:8][..., None, :, :]),
            dim=-1,
        ) / 10.0
        previous_confidence = previous[..., 0][..., :, None].expand_as(descriptor_cosine)
        current_confidence = current[..., 0][..., None, :].expand_as(descriptor_cosine)
        return torch.stack((
            descriptor_cosine,
            heading_cosine,
            width_delta,
            profile_delta,
            previous_confidence,
            current_confidence,
        ), dim=-1)

    def _relational_frame(
        self,
        previous_raw: torch.Tensor,
        current_raw: torch.Tensor,
        previous_encoded: torch.Tensor,
        current_encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        affinity = self.affinity(self._pair_features(previous_raw, current_raw)).squeeze(-1)
        # For each current token, normalize over all previous tokens.  A second
        # normalization makes the reported confidence mutual without imposing
        # a hard assignment or token order.
        transport = torch.softmax(affinity, dim=-2)
        reverse = torch.softmax(affinity, dim=-1)
        mutual = torch.sqrt(
            transport.clamp_min(1e-8) * reverse.clamp_min(1e-8)
        )
        transported = torch.einsum("bsqk,bsqd->bskd", transport, previous_encoded)
        match_confidence = mutual.amax(dim=-2)
        relational = self.relational_token(torch.cat((
            current_encoded,
            transported,
            current_encoded - transported,
            match_confidence[..., None],
            current_raw[..., 0:1],
        ), dim=-1))
        batch, seeds = relational.shape[:2]
        values = relational.reshape(batch * seeds, TOKENS, TOKEN_EMBED_DIM)
        query = self.event_queries[None].expand(batch * seeds, -1, -1)
        pooled, _ = self.set_attention(query, values, values, need_weights=False)
        return pooled.reshape(batch, seeds, EVENT_QUERIES * TOKEN_EMBED_DIM), transport, affinity

    def forward(
        self,
        tokens: torch.Tensor,
        geometry_context: torch.Tensor,
        valid_history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 5 or tuple(tokens.shape[1:]) != (HISTORY, SEEDS, TOKENS, RAW_TOKEN_DIM):
            raise ValueError("tokens must have shape [B,5,3,6,40]")
        batch = len(tokens)
        if geometry_context.shape != (batch, HISTORY, SEEDS, GEOMETRY_CONTEXT_DIM):
            raise ValueError("geometry_context must have shape [B,5,3,8]")
        if valid_history_mask.shape != (batch, HISTORY) or valid_history_mask.dtype != torch.bool:
            raise ValueError("valid_history_mask must be bool [B,5]")
        if (
            not bool(valid_history_mask[:, -1].all())
            or bool((~valid_history_mask[:, 1:] & valid_history_mask[:, :-1]).any())
            or not bool(torch.isfinite(tokens).all())
            or not bool(torch.isfinite(geometry_context).all())
            or bool(((tokens[..., 0] < 0.0) | (tokens[..., 0] > 1.0)).any())
            or bool((tokens[..., 3] <= 0.0).any())
        ):
            raise ValueError("relational event inputs violate the finite causal token contract")
        normalized = (tokens - self.token_normalization_mean) / self.token_normalization_scale
        encoded = self.token_encoder(normalized)
        geometry = self.geometry_encoder(geometry_context)
        frame_values = []
        transports = []
        affinities = []
        for index in range(HISTORY):
            current_raw = tokens[:, index]
            current_encoded = encoded[:, index]
            if index == 0:
                previous_raw = current_raw
                previous_encoded = current_encoded
            else:
                pair_valid = (valid_history_mask[:, index - 1] & valid_history_mask[:, index])[:, None, None, None]
                previous_raw = torch.where(pair_valid, tokens[:, index - 1], current_raw)
                previous_encoded = torch.where(pair_valid, encoded[:, index - 1], current_encoded)
            set_context, transport, affinity = self._relational_frame(
                previous_raw, current_raw, previous_encoded, current_encoded
            )
            if index:
                transports.append(transport)
                affinities.append(affinity)
            seed_set = torch.cat((set_context.mean(dim=1), set_context.std(dim=1)), dim=1)
            seed_geometry = torch.cat((geometry[:, index].mean(dim=1), geometry[:, index].std(dim=1)), dim=1)
            value = self.frame_encoder(torch.cat((seed_set, seed_geometry), dim=1))
            value = value * valid_history_mask[:, index:index + 1].to(dtype=value.dtype)
            frame_values.append(value)
        sequence_values = torch.stack(frame_values, dim=1)
        lengths = valid_history_mask.sum(dim=1).to(dtype=torch.int64)
        compact = sequence_values.new_zeros(sequence_values.shape)
        for row, length in enumerate(lengths.tolist()):
            compact[row, :length] = sequence_values[row, HISTORY - length:]
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.temporal(packed)
        current_frame = sequence_values[:, -1]
        context = self.context(torch.cat((hidden[-1], current_frame), dim=1))
        event_logits = self.event_head(context)
        commit_logit = self.commit_head(context).squeeze(1)
        event_probability = torch.softmax(event_logits, dim=1)
        commit_probability = torch.sigmoid(commit_logit)
        event_entropy = -(
            event_probability * torch.log(event_probability.clamp_min(1e-8))
        ).sum(dim=1) / math.log(3.0)
        binary_entropy = -(
            commit_probability * torch.log(commit_probability.clamp_min(1e-8))
            + (1.0 - commit_probability) * torch.log((1.0 - commit_probability).clamp_min(1e-8))
        ) / math.log(2.0)
        return {
            "event_logits": event_logits,
            "event_probability": event_probability,
            "commit_logit": commit_logit,
            "commit_probability": commit_probability,
            "provisional_probability": 1.0 - commit_probability,
            "uncertainty": 0.5 * (event_entropy + binary_entropy),
            "causal_context": context,
            "transport": torch.stack(transports, dim=1),
            "transport_affinity": torch.stack(affinities, dim=1),
        }


def relational_event_episode_loss(
    outputs: Mapping[str, torch.Tensor],
    decision_target: torch.Tensor,
    episode_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Episode MIL: a structural episode needs one safe correct commit."""

    logits = outputs["event_logits"]
    commit = outputs["commit_logit"]
    target = decision_target.long()
    episode = episode_id.long()
    if (
        logits.shape != (len(target), 3)
        or commit.shape != target.shape
        or episode.shape != target.shape
        or torch.any((target < 0) | (target > 2))
        or torch.any((target == 0) != (episode < 0))
    ):
        raise ValueError("relational event loss target/episode contract drift")
    negative = target == 0
    if not bool(negative.any()) or not bool((episode >= 0).any()):
        raise ValueError("relational event loss requires corridor and event episodes")
    negative_commit = F.binary_cross_entropy_with_logits(
        commit[negative], torch.zeros_like(commit[negative])
    )
    negative_event = F.cross_entropy(logits[negative], target[negative])
    positive_terms = []
    for identity in torch.unique(episode[episode >= 0], sorted=True):
        rows = episode == identity
        classes = torch.unique(target[rows])
        if len(classes) != 1 or int(classes[0]) not in (1, 2):
            raise ValueError("one relational event episode must have one decision class")
        joint = F.logsigmoid(commit[rows]) + F.log_softmax(logits[rows], dim=1)[:, int(classes[0])]
        positive_terms.append(-torch.amax(joint))
    positive = torch.stack(positive_terms).mean()
    total = negative_commit + negative_event + positive
    return {
        "negative_commit": negative_commit,
        "negative_event": negative_event,
        "positive_episode_joint": positive,
        "total": total,
    }


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in RelationalExitTokenTransportEventModel().parameters())


__all__ = [
    "GEOMETRY_CONTEXT_DIM", "HISTORY", "RAW_TOKEN_DIM", "SEEDS", "TOKENS",
    "RelationalExitTokenTransportEventModel", "parameter_count",
    "relational_event_episode_loss",
]
