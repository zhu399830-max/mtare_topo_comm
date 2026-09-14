"""Permutation-invariant learned exit-action-set representation for decision nodes."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.teacher.gse_factorized_association_teacher import (
    causal_history_row_references_unordered,
)


TOKEN_COUNT = 6
TOKEN_DESCRIPTOR_DIM = 32
SINGLE_SEED_SIGNATURE_DIM = 59
ENSEMBLE_SIGNATURE_DIM = 2 * SINGLE_SEED_SIGNATURE_DIM
CAUSAL_ACTION_FEATURE_DIM = 4 * ENSEMBLE_SIGNATURE_DIM + 1
RAW_TOKEN_DIM = 40
ACTION_HISTORY = 5
UNNORMALIZED_TOKEN_DIMENSIONS = (0, 1, 2)


def _seed_signature(outputs: Mapping[str, np.ndarray]) -> np.ndarray:
    confidence = np.asarray(outputs["exit_confidence"], dtype=np.float64)
    heading = np.asarray(outputs["exit_heading_unit"], dtype=np.float64)
    width = np.asarray(outputs["exit_opening_width_m"], dtype=np.float64)
    profile = np.asarray(outputs["exit_vertical_profile"], dtype=np.float64)
    descriptor = np.asarray(outputs["exit_descriptor"], dtype=np.float64)
    rows = len(confidence)
    if (
        confidence.shape != (rows, TOKEN_COUNT)
        or heading.shape != (rows, TOKEN_COUNT, 2)
        or width.shape != (rows, TOKEN_COUNT)
        or profile.shape != (rows, TOKEN_COUNT, 4)
        or descriptor.shape != (rows, TOKEN_COUNT, TOKEN_DESCRIPTOR_DIM)
        or not all(np.all(np.isfinite(value)) for value in (confidence, heading, width, profile, descriptor))
        or np.any((confidence < 0.0) | (confidence > 1.0))
        or np.any(width <= 0.0)
        or not np.allclose(np.linalg.norm(heading, axis=2), 1.0, atol=2e-3)
    ):
        raise ValueError("learned exit-token set contract drift")
    mass = confidence.sum(axis=1, keepdims=True).clip(min=1e-8)
    weight = confidence / mass
    sine = heading[:, :, 0]
    cosine = heading[:, :, 1]
    harmonics = []
    complex_heading = cosine + 1j * sine
    for order in (1, 2, 3):
        moment = np.sum(weight * complex_heading**order, axis=1)
        harmonics.extend((moment.real[:, None], moment.imag[:, None]))
    width_mean = np.sum(weight * width, axis=1, keepdims=True)
    width_std = np.sqrt(
        np.sum(weight * (width - width_mean) ** 2, axis=1, keepdims=True).clip(min=0.0)
    )
    weighted_width_peak = np.max(confidence * width, axis=1, keepdims=True)
    profile_mean = np.sum(weight[:, :, None] * profile, axis=1)
    profile_std = np.sqrt(
        np.sum(weight[:, :, None] * (profile - profile_mean[:, None]) ** 2, axis=1).clip(min=0.0)
    )
    descriptor_mean = np.sum(weight[:, :, None] * descriptor, axis=1)
    binary_entropy = -(
        confidence * np.log(confidence.clip(min=1e-8))
        + (1.0 - confidence) * np.log((1.0 - confidence).clip(min=1e-8))
    ).mean(axis=1, keepdims=True)
    sorted_confidence = np.sort(confidence, axis=1)[:, ::-1]
    signature = np.concatenate((
        confidence.sum(axis=1, keepdims=True),
        np.square(confidence).sum(axis=1, keepdims=True),
        np.power(confidence, 3).sum(axis=1, keepdims=True),
        *harmonics,
        width_mean, width_std, weighted_width_peak,
        profile_mean, profile_std, descriptor_mean,
        binary_entropy, sorted_confidence,
    ), axis=1)
    if signature.shape != (rows, SINGLE_SEED_SIGNATURE_DIM) or not np.all(np.isfinite(signature)):
        raise RuntimeError("exit action-set signature dimension drift")
    return signature.astype(np.float32)


def ensemble_action_set_signature(
    outputs_by_seed: Mapping[int, Mapping[str, np.ndarray]],
) -> np.ndarray:
    """Fuse three seed-specific token sets without relying on token slot order."""

    if tuple(sorted(outputs_by_seed)) != (0, 1, 2):
        raise ValueError("action-set representation requires seeds 0/1/2")
    seed = np.stack([_seed_signature(outputs_by_seed[index]) for index in (0, 1, 2)])
    result = np.concatenate((seed.mean(axis=0), seed.std(axis=0)), axis=1)
    if result.shape[1] != ENSEMBLE_SIGNATURE_DIM:
        raise RuntimeError("ensemble action-set signature dimension drift")
    return result.astype(np.float32)


def causal_action_set_features(
    signature: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: Sequence[int],
    *,
    history_observations: int = 5,
) -> np.ndarray:
    """Add strictly-past stability/change statistics without Teacher inputs."""

    values = np.asarray(signature, dtype=np.float32)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    traversal = np.asarray(traversal_id, dtype=str)
    if (
        values.ndim != 2 or values.shape[1] != ENSEMBLE_SIGNATURE_DIM
        or traversal.shape != (len(values),) or sequence.shape != (len(values),)
        or not np.all(np.isfinite(values)) or history_observations != 5
    ):
        raise ValueError("causal action-set feature input contract drift")
    references, mask = causal_history_row_references_unordered(
        traversal, sequence, maximum=history_observations
    )
    output = np.empty((len(values), CAUSAL_ACTION_FEATURE_DIM), dtype=np.float32)
    for row in range(len(values)):
        indices = references[row, mask[row]]
        history = values[indices]
        current = values[row]
        output[row] = np.concatenate((
            current, history.mean(axis=0), history.std(axis=0),
            current - history[0], np.asarray([len(indices) / history_observations], dtype=np.float32),
        ))
    if not np.all(np.isfinite(output)):
        raise RuntimeError("causal action-set features are nonfinite")
    return output


def stack_raw_action_tokens(
    outputs_by_seed: Mapping[int, Mapping[str, np.ndarray]],
) -> np.ndarray:
    """Materialize [N,3,6,40] raw learned tokens without Teacher fields."""

    if tuple(sorted(outputs_by_seed)) != (0, 1, 2):
        raise ValueError("raw action tokens require seeds 0/1/2")
    result = []
    keys = None
    for seed in (0, 1, 2):
        outputs = outputs_by_seed[seed]
        confidence = np.asarray(outputs["exit_confidence"], dtype=np.float32)
        heading = np.asarray(outputs["exit_heading_unit"], dtype=np.float32)
        width = np.asarray(outputs["exit_opening_width_m"], dtype=np.float32)[..., None]
        profile = np.asarray(outputs["exit_vertical_profile"], dtype=np.float32)
        descriptor = np.asarray(outputs["exit_descriptor"], dtype=np.float32)
        count = len(confidence)
        token = np.concatenate((confidence[..., None], heading, width, profile, descriptor), axis=2)
        if (
            token.shape != (count, TOKEN_COUNT, RAW_TOKEN_DIM)
            or not np.all(np.isfinite(token))
            or np.any((confidence < 0.0) | (confidence > 1.0))
        ):
            raise ValueError(f"seed{seed} raw action-token contract drift")
        current_keys = np.asarray(outputs.get("global_sequence_index"), dtype=np.int64)
        if current_keys.shape != (count,) or len(np.unique(current_keys)) != count:
            raise ValueError(f"seed{seed} action-token identities are invalid")
        if keys is None:
            keys = current_keys
        elif not np.array_equal(keys, current_keys):
            raise RuntimeError("three action-token seed identities drift")
        result.append(token)
    return np.stack(result, axis=1)


def raw_action_tokens_one_seed(outputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return one seed's typed token tensor without allocating a three-seed copy."""

    confidence = np.asarray(outputs["exit_confidence"], dtype=np.float32)
    heading = np.asarray(outputs["exit_heading_unit"], dtype=np.float32)
    width = np.asarray(outputs["exit_opening_width_m"], dtype=np.float32)[..., None]
    profile = np.asarray(outputs["exit_vertical_profile"], dtype=np.float32)
    descriptor = np.asarray(outputs["exit_descriptor"], dtype=np.float32)
    count = len(confidence)
    token = np.concatenate((confidence[..., None], heading, width, profile, descriptor), axis=2)
    if (
        token.shape != (count, TOKEN_COUNT, RAW_TOKEN_DIM)
        or not np.all(np.isfinite(token))
        or np.any((confidence < 0.0) | (confidence > 1.0))
        or not np.allclose(np.linalg.norm(heading, axis=2), 1.0, atol=2e-3)
        or np.any(width <= 0.0)
    ):
        raise ValueError("single-seed raw action-token contract drift")
    return token


def fit_action_token_normalization(
    tokens: np.ndarray,
    fit_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit token normalization on C01--C06 only, preserving confidence/heading."""

    values = np.asarray(tokens)
    rows = np.asarray(fit_rows, dtype=np.int64)
    if (
        values.ndim != 4 or values.shape[1:] != (3, TOKEN_COUNT, RAW_TOKEN_DIM)
        or rows.ndim != 1 or len(rows) == 0 or np.any(rows < 0) or np.any(rows >= len(values))
        or len(np.unique(rows)) != len(rows)
    ):
        raise ValueError("action-token normalization input contract drift")
    total = np.zeros(RAW_TOKEN_DIM, dtype=np.float64)
    square = np.zeros(RAW_TOKEN_DIM, dtype=np.float64)
    population = 0
    for start in range(0, len(rows), 4096):
        selected = np.asarray(values[rows[start : start + 4096]], dtype=np.float64).reshape(
            -1, RAW_TOKEN_DIM
        )
        if not np.all(np.isfinite(selected)):
            raise ValueError("action-token normalization input is nonfinite")
        total += selected.sum(axis=0)
        square += np.square(selected).sum(axis=0)
        population += len(selected)
    mean = total / population
    scale = np.sqrt(np.maximum(square / population - np.square(mean), 0.0))
    mean[list(UNNORMALIZED_TOKEN_DIMENSIONS)] = 0.0
    scale[list(UNNORMALIZED_TOKEN_DIMENSIONS)] = 1.0
    scale[scale <= 1e-8] = 1.0
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("action-token normalization is nonfinite")
    return mean.astype(np.float32), scale.astype(np.float32)


class ActionSetNodeDetector(nn.Module):
    """Permutation-invariant token-set encoder with past-only confirmation."""

    def __init__(self) -> None:
        super().__init__()
        self.token_encoder = nn.Sequential(
            nn.Linear(RAW_TOKEN_DIM, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
        )
        self.temporal = nn.GRU(256, 128, batch_first=True)
        self.summary = nn.Sequential(nn.Linear(640, 128), nn.SiLU())
        self.structural = nn.Linear(128, 1)
        self.decision_class = nn.Linear(128, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 5 or tuple(tokens.shape[1:]) != (
            ACTION_HISTORY, 3, TOKEN_COUNT, RAW_TOKEN_DIM,
        ):
            raise ValueError("action tokens must have shape [B,5,3,6,40]")
        batch = len(tokens)
        if valid_history_mask.shape != (batch, ACTION_HISTORY) or valid_history_mask.dtype != torch.bool:
            raise ValueError("action history mask must be bool [B,5]")
        if (
            not bool(valid_history_mask[:, -1].all())
            or bool((~valid_history_mask[:, 1:] & valid_history_mask[:, :-1]).any())
            or not bool(torch.isfinite(tokens).all())
        ):
            raise ValueError("action history must be finite and a nonempty causal suffix")
        confidence = tokens[..., 0].clamp(0.0, 1.0)
        encoded = self.token_encoder(tokens)
        weight = confidence / confidence.sum(dim=3, keepdim=True).clamp_min(1e-6)
        mean = (encoded * weight[..., None]).sum(dim=3)
        maximum = (encoded * confidence[..., None]).amax(dim=3)
        seed_set = torch.cat((mean, maximum), dim=-1)
        observation = torch.cat((seed_set.mean(dim=2), seed_set.std(dim=2)), dim=-1)
        lengths = valid_history_mask.sum(dim=1).to(dtype=torch.int64)
        compact = observation.new_zeros(observation.shape)
        for index, length in enumerate(lengths.tolist()):
            compact[index, :length] = observation[index, -length:]
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.temporal(packed)
        first_index = ACTION_HISTORY - lengths
        first = observation[torch.arange(batch, device=tokens.device), first_index]
        current = observation[:, -1]
        context = self.summary(torch.cat((hidden[-1], current, current - first), dim=1))
        structural_logit = self.structural(context).squeeze(1)
        class_logits = self.decision_class(context)
        structural_probability = torch.sigmoid(structural_logit)
        conditional_probability = torch.softmax(class_logits, dim=1)
        probability = torch.cat((
            (1.0 - structural_probability).unsqueeze(1),
            structural_probability.unsqueeze(1) * conditional_probability,
        ), dim=1)
        uncertainty = -(
            probability * torch.log(probability.clamp_min(1e-8))
        ).sum(dim=1) / np.log(3.0)
        return {
            "structural_logit": structural_logit,
            "conditional_decision_logits": class_logits,
            "decision_probability": probability,
            "uncertainty": uncertainty,
            "causal_context": context,
        }


def action_set_episode_loss(
    outputs: Mapping[str, torch.Tensor],
    decision_target: torch.Tensor,
    episode_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One correct junction/terminal confirmation per episode is sufficient."""

    structural = outputs["structural_logit"]
    conditional = outputs["conditional_decision_logits"]
    if (
        decision_target.ndim != 1
        or structural.shape != decision_target.shape
        or episode_id.shape != decision_target.shape
        or conditional.shape != (len(decision_target), 2)
    ):
        raise ValueError("action-set episode loss vectors must align")
    target = decision_target.long()
    bags = episode_id.long()
    if torch.any((target < 0) | (target > 2)) or torch.any((target == 0) != (bags < 0)):
        raise ValueError("action-set target/episode contract drift")
    negative_mask = target == 0
    if not bool(negative_mask.any()):
        raise ValueError("action-set loss requires negative rows")
    negative = F.binary_cross_entropy_with_logits(
        structural[negative_mask], torch.zeros_like(structural[negative_mask])
    )
    positive_terms = []
    for bag in torch.unique(bags[bags >= 0], sorted=True):
        rows = bags == bag
        classes = torch.unique(target[rows])
        if len(classes) != 1 or int(classes[0]) not in (1, 2):
            raise ValueError("action-set episode must contain one decision class")
        class_index = int(classes[0]) - 1
        joint = F.logsigmoid(structural[rows]) + F.log_softmax(conditional[rows], dim=1)[:, class_index]
        positive_terms.append(-torch.amax(joint))
    if not positive_terms:
        raise ValueError("action-set loss requires positive episodes")
    positive = torch.stack(positive_terms).mean()
    return {"negative": negative, "positive_episode_joint": positive, "total": negative + positive}


__all__ = [
    "ACTION_HISTORY", "ActionSetNodeDetector", "CAUSAL_ACTION_FEATURE_DIM",
    "ENSEMBLE_SIGNATURE_DIM", "RAW_TOKEN_DIM",
    "action_set_episode_loss",
    "SINGLE_SEED_SIGNATURE_DIM", "causal_action_set_features",
    "ensemble_action_set_signature", "stack_raw_action_tokens",
    "raw_action_tokens_one_seed", "fit_action_token_normalization",
]
