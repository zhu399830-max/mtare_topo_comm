"""Twelve-scan past-only structural-event detector for GSE-Graph."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.evaluation.gse_causal_event_supervision import contiguous_structural_episodes
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


HISTORY_FRAMES = 12
ENCODER_DIM = 128
DIRECTIONAL_BINS = 36
EVENT_COUNT = len(EVENT_NAMES)
STRUCTURAL_EVENT_COUNT = EVENT_COUNT - 1


@dataclass(frozen=True)
class PastOnlyReferenceBank:
    """Deployment references built without event labels or place identity."""

    global_frame_references: np.ndarray
    valid_history_mask: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.global_frame_references)
        if self.global_frame_references.shape != (count, HISTORY_FRAMES):
            raise ValueError("past-only references must have shape [N,12]")
        if self.valid_history_mask.shape != (count, HISTORY_FRAMES):
            raise ValueError("past-only history mask must have shape [N,12]")
        if np.any(self.valid_history_mask & (self.global_frame_references < 0)):
            raise ValueError("valid past-only references must be nonnegative")
        if np.any(~self.valid_history_mask & (self.global_frame_references != -1)):
            raise ValueError("masked past-only references must use -1")


@dataclass(frozen=True)
class CausalEpisodeReferenceBank:
    """References and Teacher episode membership aligned to observation rows."""

    global_frame_references: np.ndarray
    valid_history_mask: np.ndarray
    episode_id: np.ndarray
    event_index: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.episode_id)
        if self.global_frame_references.shape != (count, HISTORY_FRAMES):
            raise ValueError("episode references must have shape [N,12]")
        if self.valid_history_mask.shape != (count, HISTORY_FRAMES):
            raise ValueError("episode history mask must have shape [N,12]")
        if self.event_index.shape != (count,):
            raise ValueError("episode event index must align with rows")
        if np.any(self.valid_history_mask & (self.global_frame_references < 0)):
            raise ValueError("valid episode references must be nonnegative")
        if np.any(~self.valid_history_mask & (self.global_frame_references != -1)):
            raise ValueError("masked episode references must use -1")
        if np.any((self.event_index == 0) != (self.episode_id < 0)):
            raise ValueError("corridor/episode membership contract drift")


def materialize_past_only_references(
    rows: Sequence[Mapping[str, object]],
) -> PastOnlyReferenceBank:
    """Build exact deployment history without reading Teacher semantics.

    Every input row needs only traversal/frame identity and its already sealed
    five causal frame references.  Event labels, episode ids and place identity
    are deliberately neither required nor inspected.
    """

    count = len(rows)
    references = np.full((count, HISTORY_FRAMES), -1, dtype=np.int64)
    mask = np.zeros((count, HISTORY_FRAMES), dtype=np.bool_)
    traversal_base: dict[str, int] = {}
    previous_sequence: dict[str, int] = {}

    for row_index, row in enumerate(rows):
        traversal = str(row["traversal_id"])
        sequence = int(row["sequence_index"])
        frame = int(row["frame_index"])
        legacy = np.asarray(row["global_frame_references"], dtype=np.int64)
        if legacy.shape != (5,) or frame < 4 or np.any(np.diff(legacy) != 1):
            raise ValueError("legacy five-frame reference contract drift")
        current = int(legacy[-1])
        base = current - frame
        previous_base = traversal_base.setdefault(traversal, base)
        if previous_base != base or not np.array_equal(legacy, np.arange(current - 4, current + 1)):
            raise ValueError("global frame references cross or drift within a traversal")
        if traversal in previous_sequence and sequence != previous_sequence[traversal] + 1:
            raise ValueError("past-only observations must be contiguous within each traversal")
        if traversal not in previous_sequence and sequence != 0:
            raise ValueError("past-only traversal observations must start at sequence zero")
        previous_sequence[traversal] = sequence
        start_frame = max(0, frame - HISTORY_FRAMES + 1)
        values = np.arange(base + start_frame, current + 1, dtype=np.int64)
        references[row_index, -len(values) :] = values
        mask[row_index, -len(values) :] = True

    return PastOnlyReferenceBank(references, mask)


def materialize_causal_episode_references(
    rows: Sequence[Mapping[str, object]],
) -> CausalEpisodeReferenceBank:
    """Add evaluation-only Teacher episodes to exact past-only references."""

    deployment = materialize_past_only_references(rows)
    count = len(rows)
    event_index = np.empty(count, dtype=np.int8)
    episode_id = np.full(count, -1, dtype=np.int64)
    event_lookup = {name: index for index, name in enumerate(EVENT_NAMES)}
    row_by_key: dict[tuple[str, int], int] = {}
    for row_index, row in enumerate(rows):
        key = (str(row["traversal_id"]), int(row["sequence_index"]))
        if key in row_by_key:
            raise ValueError("duplicate traversal/sequence observation")
        row_by_key[key] = row_index
        try:
            event_index[row_index] = event_lookup[str(row["event"])]
        except KeyError as exc:
            raise ValueError("unknown structural event") from exc

    episodes = contiguous_structural_episodes(rows)
    for compact_id, episode in enumerate(episodes):
        expected_event = event_lookup[episode.event]
        for sequence in range(episode.start_sequence_index, episode.end_sequence_index + 1):
            key = (episode.traversal_id, sequence)
            if key not in row_by_key:
                raise ValueError("structural episode contains a missing observation")
            row_index = row_by_key[key]
            if episode_id[row_index] >= 0 or int(event_index[row_index]) != expected_event:
                raise ValueError("structural episode membership is ambiguous")
            episode_id[row_index] = compact_id
    if np.any((event_index == 0) != (episode_id < 0)):
        raise RuntimeError("not every structural Teacher row received one episode")
    return CausalEpisodeReferenceBank(
        deployment.global_frame_references,
        deployment.valid_history_mask,
        episode_id,
        event_index,
    )


class CausalEpisodeDetector(nn.Module):
    """Confirm one structural event from twelve frozen spatial scan embeddings.

    The frozen spatial encoder is executed outside this module.  This detector
    receives only LiDAR-derived embeddings, a past-only mask and the frozen
    five-frame event logits used as a residual baseline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.GRU(ENCODER_DIM, ENCODER_DIM, batch_first=True)
        self.directional_change = nn.Sequential(
            nn.Conv1d(3 * ENCODER_DIM, ENCODER_DIM, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, ENCODER_DIM),
            nn.SiLU(),
            nn.Conv1d(ENCODER_DIM, ENCODER_DIM, kernel_size=3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.summary = nn.Sequential(
            nn.Linear(5 * ENCODER_DIM, ENCODER_DIM),
            nn.SiLU(),
        )
        self.structural_residual = nn.Linear(ENCODER_DIM, 1)
        self.class_residual = nn.Linear(ENCODER_DIM, STRUCTURAL_EVENT_COUNT)
        self.boundary_offset = nn.Linear(ENCODER_DIM, 1)
        nn.init.zeros_(self.structural_residual.weight)
        nn.init.zeros_(self.structural_residual.bias)
        nn.init.zeros_(self.class_residual.weight)
        nn.init.zeros_(self.class_residual.bias)

    def forward(
        self,
        spatial_embeddings: torch.Tensor,
        directional_embeddings: torch.Tensor,
        valid_history_mask: torch.Tensor,
        baseline_event_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if spatial_embeddings.ndim != 3 or tuple(spatial_embeddings.shape[1:]) != (HISTORY_FRAMES, ENCODER_DIM):
            raise ValueError("spatial_embeddings must have shape [B,12,128]")
        batch = len(spatial_embeddings)
        if directional_embeddings.shape != (batch, HISTORY_FRAMES, ENCODER_DIM, DIRECTIONAL_BINS):
            raise ValueError("directional_embeddings must have shape [B,12,128,36]")
        if valid_history_mask.shape != (batch, HISTORY_FRAMES) or valid_history_mask.dtype != torch.bool:
            raise ValueError("valid_history_mask must be bool [B,12]")
        if baseline_event_logits.shape != (batch, EVENT_COUNT):
            raise ValueError("baseline_event_logits must have shape [B,5]")
        if not bool(valid_history_mask[:, -1].all()) or bool((~valid_history_mask[:, 1:] & valid_history_mask[:, :-1]).any()):
            raise ValueError("history mask must be a nonempty left-padded causal suffix")
        if not torch.isfinite(spatial_embeddings).all() or not torch.isfinite(directional_embeddings).all() or not torch.isfinite(baseline_event_logits).all():
            raise ValueError("episode detector inputs must be finite")
        lengths = valid_history_mask.sum(dim=1).to(dtype=torch.int64)
        # Move each valid suffix to the left for packed recurrent execution.
        compact = spatial_embeddings.new_zeros(spatial_embeddings.shape)
        for index, length in enumerate(lengths.tolist()):
            compact[index, :length] = spatial_embeddings[index, -length:]
        packed = nn.utils.rnn.pack_padded_sequence(compact, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.temporal(packed)
        current = spatial_embeddings[:, -1]
        first_index = HISTORY_FRAMES - lengths
        first = spatial_embeddings[torch.arange(batch, device=spatial_embeddings.device), first_index]
        history_weight = valid_history_mask.to(dtype=directional_embeddings.dtype)[:, :, None, None]
        past_weight = history_weight.clone()
        past_weight[:, -1] = 0.0
        past = (directional_embeddings * past_weight).sum(dim=1) / past_weight.sum(dim=1).clamp_min(1.0)
        current_directional = directional_embeddings[:, -1]
        directional = self.directional_change(
            torch.cat((current_directional, past, current_directional - past), dim=1)
        )
        context = self.summary(
            torch.cat(
                (
                    hidden[-1], current, current - first,
                    directional.mean(dim=-1), directional.amax(dim=-1),
                ),
                dim=1,
            )
        )

        baseline_structural_logit = torch.logsumexp(baseline_event_logits[:, 1:], dim=1) - baseline_event_logits[:, 0]
        structural_logit = baseline_structural_logit + self.structural_residual(context).squeeze(1)
        conditional_logits = baseline_event_logits[:, 1:] + self.class_residual(context)
        structural_probability = torch.sigmoid(structural_logit)
        conditional_probability = torch.softmax(conditional_logits, dim=1)
        event_probability = torch.cat(
            ((1.0 - structural_probability).unsqueeze(1), structural_probability.unsqueeze(1) * conditional_probability),
            dim=1,
        )
        uncertainty = -(
            event_probability * torch.log(event_probability.clamp_min(1e-8))
        ).sum(dim=1) / math.log(EVENT_COUNT)
        return {
            "structural_logit": structural_logit,
            "conditional_event_logits": conditional_logits,
            "event_probability": event_probability,
            "boundary_offset_m": (HISTORY_FRAMES - 1) * torch.sigmoid(self.boundary_offset(context).squeeze(1)),
            "uncertainty": uncertainty,
        }


def encode_spatial_scan_embeddings(
    spatial_encoder: nn.Module,
    scans: torch.Tensor,
) -> torch.Tensor:
    """Apply a frozen GSE spatial encoder independently to exact scan frames."""

    if scans.ndim != 5 or tuple(scans.shape[2:]) != (2, 16, 720):
        raise ValueError("scans must have shape [B,T,2,16,720]")
    if scans.shape[1] < 1 or not torch.isfinite(scans).all():
        raise ValueError("scan sequence must be nonempty and finite")
    batch, history = scans.shape[:2]
    encoded = spatial_encoder(scans.reshape(batch * history, 2, 16, 720))
    if encoded.ndim != 4 or encoded.shape[0] != batch * history or encoded.shape[1] != ENCODER_DIM:
        raise ValueError("spatial encoder output violates the frozen 128D contract")
    return encoded.mean(dim=(2, 3)).reshape(batch, history, ENCODER_DIM)


def encode_spatial_scan_features(
    spatial_encoder: nn.Module,
    scans: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return pooled and separable spatial features for exact scan frames.

    ``directional`` keeps the robot-frame azimuth layout at ten-degree
    resolution.  ``vertical`` keeps the encoder's two elevation rows instead
    of averaging them away.  Together they are a compact separable projection
    of the frozen ``[128,2,180]`` feature map; no pose or Teacher value enters
    either projection.
    """

    if scans.ndim != 5 or tuple(scans.shape[2:]) != (2, 16, 720):
        raise ValueError("scans must have shape [B,T,2,16,720]")
    if scans.shape[1] < 1 or not torch.isfinite(scans).all():
        raise ValueError("scan sequence must be nonempty and finite")
    batch, history = scans.shape[:2]
    encoded = spatial_encoder(scans.reshape(batch * history, 2, 16, 720))
    if encoded.ndim != 4 or encoded.shape[0] != batch * history or encoded.shape[1] != ENCODER_DIM:
        raise ValueError("spatial encoder output violates the frozen 128D contract")
    azimuth = encoded.mean(dim=2)
    directional = F.adaptive_avg_pool1d(azimuth, DIRECTIONAL_BINS)
    vertical = encoded.mean(dim=3)
    return {
        "pooled": encoded.mean(dim=(2, 3)).reshape(batch, history, ENCODER_DIM),
        "directional": directional.reshape(batch, history, ENCODER_DIM, DIRECTIONAL_BINS),
        "vertical": vertical.reshape(batch, history, ENCODER_DIM, encoded.shape[2]),
    }


def causal_episode_multiple_instance_loss(
    outputs: Mapping[str, torch.Tensor],
    event_target: torch.Tensor,
    episode_id: torch.Tensor,
    *,
    boundary_offset_target_m: torch.Tensor | None = None,
    boundary_offset_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Require one jointly correct trigger per positive episode, not per frame."""

    structural = outputs["structural_logit"]
    conditional = outputs["conditional_event_logits"]
    if event_target.ndim != 1 or structural.shape != event_target.shape or episode_id.shape != event_target.shape:
        raise ValueError("episode loss vectors must align")
    if conditional.shape != (len(event_target), STRUCTURAL_EVENT_COUNT):
        raise ValueError("conditional logits must have shape [N,4]")
    target = event_target.long()
    bags = episode_id.long()
    if torch.any((target < 0) | (target >= EVENT_COUNT)) or torch.any((target == 0) != (bags < 0)):
        raise ValueError("event/episode target contract drift")
    corridor = target == 0
    if not bool(corridor.any()):
        raise ValueError("episode loss requires corridor hard negatives")
    negative = F.binary_cross_entropy_with_logits(structural[corridor], torch.zeros_like(structural[corridor]))

    joint_terms: list[torch.Tensor] = []
    for bag in torch.unique(bags[bags >= 0], sorted=True):
        rows = bags == bag
        classes = torch.unique(target[rows])
        if len(classes) != 1 or int(classes[0]) == 0:
            raise ValueError("one structural episode must have exactly one non-corridor class")
        class_index = int(classes[0]) - 1
        joint_log_probability = F.logsigmoid(structural[rows]) + F.log_softmax(conditional[rows], dim=1)[:, class_index]
        joint_terms.append(-torch.amax(joint_log_probability))
    if not joint_terms:
        raise ValueError("episode loss requires at least one structural episode")
    positive_joint = torch.stack(joint_terms).mean()

    boundary = structural.sum() * 0.0
    if boundary_offset_target_m is not None or boundary_offset_valid is not None:
        if boundary_offset_target_m is None or boundary_offset_valid is None:
            raise ValueError("boundary target and validity mask must be supplied together")
        if boundary_offset_target_m.shape != target.shape or boundary_offset_valid.shape != target.shape:
            raise ValueError("boundary supervision vectors must align")
        valid = boundary_offset_valid.bool()
        if bool(valid.any()):
            values = boundary_offset_target_m[valid].to(dtype=structural.dtype)
            if torch.any(~torch.isfinite(values)) or torch.any((values < 0.0) | (values > HISTORY_FRAMES - 1)):
                raise ValueError("boundary offsets violate the 0--11 m history")
            boundary = F.smooth_l1_loss(
                outputs["boundary_offset_m"][valid] / (HISTORY_FRAMES - 1),
                values / (HISTORY_FRAMES - 1),
            )
    return {
        "corridor_negative": negative,
        "positive_episode_joint": positive_joint,
        "boundary_offset": boundary,
        "total": negative + positive_joint + boundary,
    }


__all__ = [
    "CausalEpisodeDetector",
    "CausalEpisodeReferenceBank",
    "PastOnlyReferenceBank",
    "DIRECTIONAL_BINS",
    "HISTORY_FRAMES",
    "causal_episode_multiple_instance_loss",
    "encode_spatial_scan_embeddings",
    "encode_spatial_scan_features",
    "materialize_causal_episode_references",
    "materialize_past_only_references",
]
