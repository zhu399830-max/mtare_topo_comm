"""Directional five-frame structural-event corrective for GSE-Graph."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


HISTORY_FRAMES = 5
ENCODER_DIM = 128
EVENT_COUNT = len(EVENT_NAMES)
STRUCTURAL_EVENT_COUNT = EVENT_COUNT - 1


def directional_identity_balanced_weights(
    event: np.ndarray,
    identity: np.ndarray,
    baseline_structural_score: np.ndarray,
) -> np.ndarray:
    """Balance binary node evidence, rare classes, identities and hard corridors.

    Half of the sampling mass is assigned to corridor negatives.  Within that
    half, uniform and frozen-baseline hardness sampling contribute equally.
    The remaining mass is split equally between the four structural classes,
    then equally between identities of each class.
    """

    event = np.asarray(event, dtype=np.int64)
    identity = np.asarray(identity, dtype=np.int64)
    score = np.asarray(baseline_structural_score, dtype=np.float64)
    if event.ndim != 1 or identity.shape != event.shape or score.shape != event.shape or len(event) == 0:
        raise ValueError("directional sampling arrays must be aligned non-empty vectors")
    if np.any((event < 0) | (event >= EVENT_COUNT)):
        raise ValueError("event labels violate the five-class contract")
    if np.any((event == 0) != (identity < 0)):
        raise ValueError("corridor/structural identity contract drift")
    if np.any(~np.isfinite(score)) or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("baseline structural scores must lie in [0,1]")
    weights = np.empty(len(event), dtype=np.float64)
    corridor = np.where(event == 0)[0]
    if len(corridor) == 0:
        raise ValueError("directional sampling requires corridor negatives")
    hardness = score[corridor]
    hardness = hardness + np.finfo(np.float64).eps
    weights[corridor] = 0.25 / len(corridor) + 0.25 * hardness / hardness.sum()
    class_mass = 0.5 / STRUCTURAL_EVENT_COUNT
    for class_index in range(1, EVENT_COUNT):
        rows = np.where(event == class_index)[0]
        if len(rows) == 0:
            raise ValueError("every structural event class requires fit support")
        counts = Counter(identity[rows].tolist())
        identity_mass = class_mass / len(counts)
        weights[rows] = np.asarray(
            [identity_mass / counts[int(identity[row])] for row in rows], dtype=np.float64
        )
    weights /= weights.sum()
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RuntimeError("directional sampling weights are invalid")
    return weights


class DirectionalStructuralEventHead(nn.Module):
    """Recover structural events from frozen azimuth-preserving features.

    The head deliberately separates two questions:

    1. is there enough evidence to create a structural node; and
    2. conditional on structural evidence, which event class is present.

    Residual outputs are zero initialized, so a new head exactly reproduces the
    frozen five-class event distribution before any optimizer step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.change_encoder = nn.Sequential(
            nn.Conv1d(3 * ENCODER_DIM, ENCODER_DIM, kernel_size=5, padding=2, padding_mode="circular"),
            nn.GroupNorm(8, ENCODER_DIM),
            nn.SiLU(),
            nn.Conv1d(ENCODER_DIM, ENCODER_DIM, kernel_size=3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        summary_dim = 3 * ENCODER_DIM
        self.structural_residual = nn.Sequential(
            nn.Linear(summary_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.class_residual = nn.Sequential(
            nn.Linear(summary_dim, 128),
            nn.SiLU(),
            nn.Linear(128, STRUCTURAL_EVENT_COUNT),
        )
        nn.init.zeros_(self.structural_residual[-1].weight)
        nn.init.zeros_(self.structural_residual[-1].bias)
        nn.init.zeros_(self.class_residual[-1].weight)
        nn.init.zeros_(self.class_residual[-1].bias)

    @staticmethod
    def _validate(
        azimuth_sequence: torch.Tensor,
        context: torch.Tensor,
        baseline_event_logits: torch.Tensor,
    ) -> None:
        if azimuth_sequence.ndim != 4 or tuple(azimuth_sequence.shape[1:3]) != (
            HISTORY_FRAMES,
            ENCODER_DIM,
        ):
            raise ValueError("azimuth_sequence must have shape [B,5,128,A]")
        batch = azimuth_sequence.shape[0]
        if azimuth_sequence.shape[-1] < 8:
            raise ValueError("azimuth_sequence must retain a nontrivial circular layout")
        if context.shape != (batch, ENCODER_DIM):
            raise ValueError("context must have shape [B,128]")
        if baseline_event_logits.shape != (batch, EVENT_COUNT):
            raise ValueError("baseline_event_logits must have shape [B,5]")
        if not (
            torch.isfinite(azimuth_sequence).all()
            and torch.isfinite(context).all()
            and torch.isfinite(baseline_event_logits).all()
        ):
            raise ValueError("directional event inputs must be finite")

    def forward(
        self,
        azimuth_sequence: torch.Tensor,
        context: torch.Tensor,
        baseline_event_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate(azimuth_sequence, context, baseline_event_logits)
        current = azimuth_sequence[:, -1]
        past = azimuth_sequence[:, :-1].mean(dim=1)
        change = current - past
        directional_change = self.change_encoder(torch.cat((current, past, change), dim=1))
        summary = torch.cat(
            (
                directional_change.mean(dim=-1),
                directional_change.amax(dim=-1),
                context,
            ),
            dim=1,
        )

        baseline_structural_logit = (
            torch.logsumexp(baseline_event_logits[:, 1:], dim=1)
            - baseline_event_logits[:, 0]
        )
        structural_logit = baseline_structural_logit + self.structural_residual(summary).squeeze(1)
        conditional_logits = baseline_event_logits[:, 1:] + self.class_residual(summary)
        structural_probability = torch.sigmoid(structural_logit)
        conditional_probability = torch.softmax(conditional_logits, dim=1)
        event_probability = torch.cat(
            (
                (1.0 - structural_probability).unsqueeze(1),
                structural_probability.unsqueeze(1) * conditional_probability,
            ),
            dim=1,
        )
        return {
            "structural_logit": structural_logit,
            "conditional_event_logits": conditional_logits,
            "event_probability": event_probability,
            "directional_change": directional_change,
        }


def directional_structural_event_loss(
    outputs: Mapping[str, torch.Tensor],
    event_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return separate node-evidence and conditional-class losses."""

    structural_logit = outputs["structural_logit"]
    conditional_logits = outputs["conditional_event_logits"]
    if event_target.ndim != 1 or structural_logit.shape != event_target.shape:
        raise ValueError("event_target and structural_logit must be aligned vectors")
    if conditional_logits.shape != (len(event_target), STRUCTURAL_EVENT_COUNT):
        raise ValueError("conditional_event_logits must have shape [B,4]")
    target = event_target.long()
    if torch.any((target < 0) | (target >= EVENT_COUNT)):
        raise ValueError("event_target violates the five-class contract")
    structural_target = (target != 0).to(dtype=structural_logit.dtype)
    structural = F.binary_cross_entropy_with_logits(structural_logit, structural_target)
    mask = target != 0
    if not bool(mask.any()):
        conditional = conditional_logits.sum() * 0.0
    else:
        conditional = F.cross_entropy(conditional_logits[mask], target[mask] - 1)
    return {
        "structural": structural,
        "conditional_event": conditional,
        "total": structural + conditional,
    }


__all__ = [
    "DirectionalStructuralEventHead",
    "directional_identity_balanced_weights",
    "directional_structural_event_loss",
]
