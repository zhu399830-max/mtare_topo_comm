"""Typed, causal structural-event Composers for GSE-Graph.

The modules in this file deliberately consume only explicit learned geometry.
They are separate from the historical event heads so a structural prediction
cannot bypass the representation that the paper claims to use.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


HISTORY_FRAMES = 5
MAX_EXIT_TOKENS = 6
VERTICAL_PROFILE_DIM = 4
COUNT_CLASSES = MAX_EXIT_TOKENS + 1
ACTION_EVENT_NAMES = ("corridor", "junction", "terminal")
METRIC_EVENT_NAMES = ("corridor", "turn", "geometry_transition")
METRIC_SCALE = (4.0, 2.0, 5.0, 0.05)


@dataclass(frozen=True)
class ActionSetComposerInput:
    """Deployment-visible exit geometry and adjacent-frame relations."""

    token_bearing_unit: torch.Tensor
    token_existence_probability: torch.Tensor
    token_opening_width_m: torch.Tensor
    token_vertical_profile_m: torch.Tensor
    token_geometry_uncertainty: torch.Tensor
    token_count_probability: torch.Tensor
    transport_row_probability: torch.Tensor
    transport_reveal_probability: torch.Tensor
    valid_history_mask: torch.Tensor


@dataclass(frozen=True)
class ActionSetComposerOutput:
    event_logits: torch.Tensor
    event_probability: torch.Tensor
    refusal_logit: torch.Tensor
    refusal_probability: torch.Tensor
    commit_probability: torch.Tensor
    uncertainty: torch.Tensor


@dataclass(frozen=True)
class MetricChangeComposerInput:
    """Past-only channel geometry sequence in chronological order."""

    geometry_sequence: torch.Tensor
    geometry_uncertainty: torch.Tensor
    valid_history_mask: torch.Tensor


@dataclass(frozen=True)
class MetricChangeComposerOutput:
    event_logits: torch.Tensor
    event_probability: torch.Tensor
    refusal_logit: torch.Tensor
    refusal_probability: torch.Tensor
    commit_probability: torch.Tensor
    backprojection_logits: torch.Tensor
    backprojection_probability: torch.Tensor
    expected_steps_ago: torch.Tensor
    uncertainty: torch.Tensor


def _validate_suffix_mask(mask: torch.Tensor, batch: int) -> None:
    if mask.shape != (batch, HISTORY_FRAMES) or mask.dtype != torch.bool:
        raise ValueError("valid_history_mask must be bool [B,5]")
    if not bool(mask[:, -1].all()):
        raise ValueError("the current frame must always be valid")
    if bool((mask[:, :-1] & ~mask[:, 1:]).any()):
        raise ValueError("history must be a left-padded causal suffix")


def _compact_suffix(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a left-padded valid suffix to the beginning for packed GRUs."""

    lengths = mask.sum(dim=1).to(dtype=torch.int64)
    compact = values.new_zeros(values.shape)
    for row, length in enumerate(lengths.tolist()):
        compact[row, :length] = values[row, -length:]
    return compact, lengths


class ActionSetRelationComposer(nn.Module):
    """Compose decision events from an unordered causal exit-action set."""

    def __init__(self, token_dim: int = 64, temporal_dim: int = 96) -> None:
        super().__init__()
        # Per-token features: confidence, width, 4 profile values, 5 geometry
        # uncertainties, three relative-layout moments and four relation terms.
        self.token_encoder = nn.Sequential(
            nn.Linear(18, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim), nn.SiLU()
        )
        self.temporal = nn.GRU(2 * token_dim + COUNT_CLASSES, temporal_dim, batch_first=True)
        self.event_head = nn.Sequential(
            nn.Linear(temporal_dim + 2 * token_dim + COUNT_CLASSES, temporal_dim),
            nn.SiLU(),
            nn.Linear(temporal_dim, len(ACTION_EVENT_NAMES)),
        )
        self.refusal_head = nn.Sequential(
            nn.Linear(temporal_dim + 2 * token_dim + COUNT_CLASSES, 32),
            nn.SiLU(), nn.Linear(32, 1),
        )

    @staticmethod
    def _validate(inputs: ActionSetComposerInput) -> int:
        if not isinstance(inputs, ActionSetComposerInput):
            raise TypeError("ActionSetRelationComposer requires ActionSetComposerInput")
        bearing = inputs.token_bearing_unit
        if bearing.ndim != 4 or tuple(bearing.shape[1:]) != (
            HISTORY_FRAMES, MAX_EXIT_TOKENS, 2,
        ):
            raise ValueError("token_bearing_unit must have shape [B,5,6,2]")
        batch = len(bearing)
        expected = {
            "token_existence_probability": (batch, HISTORY_FRAMES, MAX_EXIT_TOKENS),
            "token_opening_width_m": (batch, HISTORY_FRAMES, MAX_EXIT_TOKENS),
            "token_vertical_profile_m": (
                batch, HISTORY_FRAMES, MAX_EXIT_TOKENS, VERTICAL_PROFILE_DIM,
            ),
            "token_geometry_uncertainty": (
                batch, HISTORY_FRAMES, MAX_EXIT_TOKENS, 1 + VERTICAL_PROFILE_DIM,
            ),
            "token_count_probability": (batch, HISTORY_FRAMES, COUNT_CLASSES),
            "transport_row_probability": (
                batch, HISTORY_FRAMES - 1, MAX_EXIT_TOKENS, MAX_EXIT_TOKENS + 1,
            ),
            "transport_reveal_probability": (
                batch, HISTORY_FRAMES - 1, MAX_EXIT_TOKENS,
            ),
        }
        tensors = {field.name: getattr(inputs, field.name) for field in fields(inputs)}
        for name, shape in expected.items():
            value = tensors[name]
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        _validate_suffix_mask(inputs.valid_history_mask, batch)
        probability_names = (
            "token_existence_probability", "token_count_probability",
            "transport_row_probability", "transport_reveal_probability",
        )
        for name in probability_names:
            value = tensors[name]
            if bool(((value < 0.0) | (value > 1.0)).any()):
                raise ValueError(f"{name} must be a probability")
        if not bool(torch.allclose(
            bearing.norm(dim=-1), torch.ones_like(bearing[..., 0]), atol=2e-3, rtol=0.0,
        )):
            raise ValueError("token bearings must be unit vectors")
        if bool((inputs.token_opening_width_m <= 0.0).any()):
            raise ValueError("token opening widths must be positive")
        for name in ("token_count_probability", "transport_row_probability"):
            value = tensors[name]
            if not bool(torch.allclose(
                value.sum(dim=-1), torch.ones_like(value[..., 0]), atol=2e-3, rtol=0.0,
            )):
                raise ValueError(f"{name} must sum to one")
        return batch

    def forward(self, inputs: ActionSetComposerInput) -> ActionSetComposerOutput:
        batch = self._validate(inputs)
        existence = inputs.token_existence_probability
        bearing = inputs.token_bearing_unit

        # Pairwise direction differences retain relative exit layout while
        # removing arbitrary global yaw.  Aggregation makes token order irrelevant.
        cosine = torch.einsum("bhid,bhjd->bhij", bearing, bearing)
        sine = (
            bearing[..., 0].unsqueeze(-1) * bearing[..., 1].unsqueeze(-2)
            - bearing[..., 1].unsqueeze(-1) * bearing[..., 0].unsqueeze(-2)
        )
        neighbour_weight = existence.unsqueeze(-2)
        neighbour_mass = neighbour_weight.sum(dim=-1).clamp_min(1e-6)
        relative_cosine = (cosine * neighbour_weight).sum(dim=-1) / neighbour_mass
        relative_sine = (sine * neighbour_weight).sum(dim=-1) / neighbour_mass
        relative_spread = (
            (1.0 - cosine).clamp_min(0.0) * neighbour_weight
        ).sum(dim=-1) / neighbour_mass

        incoming = existence.new_zeros(existence.shape)
        outgoing = existence.new_zeros(existence.shape)
        withdraw = existence.new_zeros(existence.shape)
        reveal = existence.new_zeros(existence.shape)
        row = inputs.transport_row_probability
        incoming[:, 1:] = (
            row[..., :MAX_EXIT_TOKENS]
            * existence[:, :-1, :, None]
        ).sum(dim=2)
        outgoing[:, :-1] = row[..., :MAX_EXIT_TOKENS].sum(dim=-1)
        withdraw[:, :-1] = row[..., MAX_EXIT_TOKENS]
        reveal[:, 1:] = inputs.transport_reveal_probability

        token_feature = torch.cat((
            existence[..., None],
            inputs.token_opening_width_m[..., None],
            inputs.token_vertical_profile_m,
            inputs.token_geometry_uncertainty,
            relative_cosine[..., None], relative_sine[..., None], relative_spread[..., None],
            incoming[..., None], outgoing[..., None], withdraw[..., None], reveal[..., None],
        ), dim=-1)
        encoded = self.token_encoder(token_feature)
        weight = existence / existence.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mean = (encoded * weight[..., None]).sum(dim=2)
        maximum = (encoded * existence[..., None]).amax(dim=2)
        observation = torch.cat((mean, maximum, inputs.token_count_probability), dim=-1)
        observation = observation * inputs.valid_history_mask[..., None]
        compact, lengths = _compact_suffix(observation, inputs.valid_history_mask)
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, hidden = self.temporal(packed)
        current = observation[:, -1]
        context = torch.cat((hidden[-1], current), dim=-1)
        event_logits = self.event_head(context)
        refusal_logit = self.refusal_head(context).squeeze(-1)
        event_probability = torch.softmax(event_logits, dim=-1)
        refusal_probability = torch.sigmoid(refusal_logit)
        entropy = -(
            event_probability * torch.log(event_probability.clamp_min(1e-8))
        ).sum(dim=-1) / torch.log(event_probability.new_tensor(float(len(ACTION_EVENT_NAMES))))
        return ActionSetComposerOutput(
            event_logits=event_logits,
            event_probability=event_probability,
            refusal_logit=refusal_logit,
            refusal_probability=refusal_probability,
            commit_probability=1.0 - refusal_probability,
            uncertainty=entropy,
        )


class MetricChangeComposer(nn.Module):
    """Compose turn/change events and a causal backprojection distribution."""

    def __init__(self, temporal_dim: int = 64) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Linear(12, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
        )
        self.temporal = nn.GRU(64, temporal_dim, batch_first=True)
        self.event_head = nn.Sequential(
            nn.Linear(temporal_dim, 64), nn.SiLU(), nn.Linear(64, len(METRIC_EVENT_NAMES)),
        )
        self.refusal_head = nn.Sequential(
            nn.Linear(temporal_dim, 32), nn.SiLU(), nn.Linear(32, 1),
        )
        self.backprojection_head = nn.Linear(temporal_dim, 1)

    @staticmethod
    def _validate(inputs: MetricChangeComposerInput) -> int:
        if not isinstance(inputs, MetricChangeComposerInput):
            raise TypeError("MetricChangeComposer requires MetricChangeComposerInput")
        geometry = inputs.geometry_sequence
        if geometry.ndim != 3 or tuple(geometry.shape[1:]) != (HISTORY_FRAMES, 4):
            raise ValueError("geometry_sequence must have shape [B,5,4]")
        batch = len(geometry)
        if inputs.geometry_uncertainty.shape != geometry.shape:
            raise ValueError("geometry_uncertainty must align with geometry_sequence")
        if not bool(torch.isfinite(geometry).all()) or not bool(
            torch.isfinite(inputs.geometry_uncertainty).all()
        ):
            raise ValueError("metric geometry inputs must be finite")
        if bool((geometry[..., :2] <= 0.0).any()) or bool(
            (inputs.geometry_uncertainty < 0.0).any()
        ):
            raise ValueError("width/height must be positive and uncertainty nonnegative")
        _validate_suffix_mask(inputs.valid_history_mask, batch)
        return batch

    def forward(self, inputs: MetricChangeComposerInput) -> MetricChangeComposerOutput:
        self._validate(inputs)
        scale = inputs.geometry_sequence.new_tensor(METRIC_SCALE)
        normalized = inputs.geometry_sequence / scale
        delta = torch.zeros_like(normalized)
        delta[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        adjacent_valid = inputs.valid_history_mask[:, 1:] & inputs.valid_history_mask[:, :-1]
        delta[:, 1:] = delta[:, 1:] * adjacent_valid[..., None]
        normalized_uncertainty = inputs.geometry_uncertainty / scale
        frame_feature = torch.cat((normalized, delta, normalized_uncertainty), dim=-1)
        encoded = self.frame_encoder(frame_feature) * inputs.valid_history_mask[..., None]
        compact, lengths = _compact_suffix(encoded, inputs.valid_history_mask)
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_output, hidden = self.temporal(packed)
        compact_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=HISTORY_FRAMES,
        )
        chronological = compact_output.new_zeros(compact_output.shape)
        for row, length in enumerate(lengths.tolist()):
            chronological[row, -length:] = compact_output[row, :length]
        context = hidden[-1]
        event_logits = self.event_head(context)
        refusal_logit = self.refusal_head(context).squeeze(-1)
        backprojection_logits = self.backprojection_head(chronological).squeeze(-1)
        backprojection_logits = backprojection_logits.masked_fill(
            ~inputs.valid_history_mask, torch.finfo(backprojection_logits.dtype).min,
        )
        backprojection_probability = torch.softmax(backprojection_logits, dim=-1)
        steps_ago = backprojection_probability.new_tensor((4.0, 3.0, 2.0, 1.0, 0.0))
        expected_steps_ago = (backprojection_probability * steps_ago).sum(dim=-1)
        event_probability = torch.softmax(event_logits, dim=-1)
        refusal_probability = torch.sigmoid(refusal_logit)
        entropy = -(
            event_probability * torch.log(event_probability.clamp_min(1e-8))
        ).sum(dim=-1) / torch.log(event_probability.new_tensor(float(len(METRIC_EVENT_NAMES))))
        return MetricChangeComposerOutput(
            event_logits=event_logits,
            event_probability=event_probability,
            refusal_logit=refusal_logit,
            refusal_probability=refusal_probability,
            commit_probability=1.0 - refusal_probability,
            backprojection_logits=backprojection_logits,
            backprojection_probability=backprojection_probability,
            expected_steps_ago=expected_steps_ago,
            uncertainty=entropy,
        )


def action_set_composer_loss(
    outputs: ActionSetComposerOutput,
    event_target: torch.Tensor,
    commit_target: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    if event_target.shape != outputs.refusal_logit.shape or commit_target.shape != event_target.shape:
        raise ValueError("action Composer targets must be aligned [B]")
    event = F.cross_entropy(outputs.event_logits, event_target.long())
    refusal = F.binary_cross_entropy_with_logits(
        outputs.refusal_logit, 1.0 - commit_target.to(outputs.refusal_logit.dtype),
    )
    return {"event": event, "refusal": refusal, "total": event + refusal}


def metric_change_composer_loss(
    outputs: MetricChangeComposerOutput,
    event_target: torch.Tensor,
    commit_target: torch.Tensor,
    backprojection_target: torch.Tensor,
    backprojection_valid: torch.Tensor | None = None,
) -> Mapping[str, torch.Tensor]:
    if any(target.shape != outputs.refusal_logit.shape for target in (event_target, commit_target)):
        raise ValueError("metric Composer event targets must be aligned [B]")
    if backprojection_valid is None:
        backprojection_valid = torch.ones_like(event_target, dtype=torch.bool)
    if backprojection_valid.shape != event_target.shape or backprojection_valid.dtype != torch.bool:
        raise ValueError("backprojection validity must be bool [B]")
    event = F.cross_entropy(outputs.event_logits, event_target.long())
    refusal = F.binary_cross_entropy_with_logits(
        outputs.refusal_logit, 1.0 - commit_target.to(outputs.refusal_logit.dtype),
    )
    if backprojection_target.shape == event_target.shape:
        if bool(((backprojection_target < 0) | (backprojection_target >= HISTORY_FRAMES)).any()):
            raise ValueError("backprojection target must index the five-frame causal history")
        per_row = F.cross_entropy(
            outputs.backprojection_logits, backprojection_target.long(), reduction="none",
        )
    elif backprojection_target.shape == (len(event_target), HISTORY_FRAMES):
        if bool((backprojection_target < 0.0).any()) or not bool(torch.isfinite(backprojection_target).all()):
            raise ValueError("soft backprojection target must be finite and nonnegative")
        mass = backprojection_target.sum(dim=-1)
        if bool(backprojection_valid.any()) and not bool(torch.allclose(
            mass[backprojection_valid], torch.ones_like(mass[backprojection_valid]), atol=1e-6, rtol=0.0,
        )):
            raise ValueError("valid soft backprojection target must sum to one")
        per_row = -(backprojection_target * torch.log_softmax(outputs.backprojection_logits, dim=-1)).sum(dim=-1)
    else:
        raise ValueError("backprojection target must be [B] indices or [B,5] probabilities")
    backprojection = per_row[backprojection_valid].mean() if bool(backprojection_valid.any()) else per_row.sum() * 0.0
    return {
        "event": event, "refusal": refusal, "backprojection": backprojection,
        "total": event + refusal + backprojection,
    }


def typed_composer_input_contract() -> Mapping[str, object]:
    """Machine-readable boundary used by readiness and later def-use audits."""

    return {
        "history_frames": HISTORY_FRAMES,
        "maximum_exit_tokens": MAX_EXIT_TOKENS,
        "action_event_names": ACTION_EVENT_NAMES,
        "metric_event_names": METRIC_EVENT_NAMES,
        "action_fields": tuple(field.name for field in fields(ActionSetComposerInput)),
        "metric_fields": tuple(field.name for field in fields(MetricChangeComposerInput)),
        "forbidden_inputs": (
            "encoder_context", "event_logits", "place_descriptor", "exit_descriptor",
            "pose", "world", "tng", "future_frame", "teacher_identity",
        ),
        "backprojection_support_steps_ago": (4, 3, 2, 1, 0),
    }


__all__ = [
    "ACTION_EVENT_NAMES", "HISTORY_FRAMES", "MAX_EXIT_TOKENS", "METRIC_EVENT_NAMES",
    "ActionSetComposerInput", "ActionSetComposerOutput", "ActionSetRelationComposer",
    "MetricChangeComposerInput", "MetricChangeComposerOutput", "MetricChangeComposer",
    "action_set_composer_loss", "metric_change_composer_loss", "typed_composer_input_contract",
]
