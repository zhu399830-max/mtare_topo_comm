"""Structured one-threshold decoding for local composition slots."""

from __future__ import annotations

import torch

from mtare_topo.representation.primitive_local_composition_slot_model import (
    COMPOSITION_SLOT_COUNT,
    DUSTBIN_INDEX,
    ENDPOINT_COUNT,
    LocalCompositionSlotModelPrediction,
    composition_slot_probabilities,
)


def structured_endpoint_slot_confidence(
    prediction: LocalCompositionSlotModelPrediction,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return best slot and a deployable, ambiguity-aware endpoint confidence."""

    probability = composition_slot_probabilities(prediction.composition)
    slot_probability = probability[..., :COMPOSITION_SLOT_COUNT]
    top = torch.topk(slot_probability, k=2, dim=-1)
    best_probability, best_slot = top.values[..., 0], top.indices[..., 0]
    competitor = torch.maximum(top.values[..., 1], probability[..., DUSTBIN_INDEX])
    margin = (best_probability - competitor).clamp_min(0.0)
    slot_presence = torch.sigmoid(prediction.composition.slot_presence_logits).gather(
        1, best_slot,
    )
    primitive_existence = torch.sigmoid(
        prediction.primitive.existence_logits,
    ).repeat_interleave(2, dim=1)
    endpoint_evidence = torch.sigmoid(
        prediction.primitive.endpoint_evidence_logits,
    ).reshape(-1, ENDPOINT_COUNT)
    certainty = (1.0 - prediction.composition.endpoint_uncertainty).clamp(0.0, 1.0)
    confidence = (
        best_probability
        * margin
        * slot_presence
        * primitive_existence
        * endpoint_evidence
        * certainty
    )
    if tuple(confidence.shape) != tuple(best_slot.shape) or not bool(
        torch.isfinite(confidence).all()
    ):
        raise ValueError("structured endpoint-slot confidence is invalid")
    if bool((confidence < 0.0).any()) or bool((confidence > 1.0 + 1e-6).any()):
        raise ValueError("structured endpoint-slot confidence lies outside [0,1]")
    return best_slot, confidence


def structured_same_slot_pair_score(
    prediction: LocalCompositionSlotModelPrediction,
) -> torch.Tensor:
    """Score pairs while preserving a transitive hard decode at every threshold."""

    best_slot, confidence = structured_endpoint_slot_confidence(prediction)
    same = best_slot[:, :, None] == best_slot[:, None, :]
    pair = torch.minimum(confidence[:, :, None], confidence[:, None, :])
    score = torch.where(same, pair, torch.zeros_like(pair))
    diagonal = torch.eye(ENDPOINT_COUNT, dtype=torch.bool, device=score.device)[None]
    score = score.masked_fill(diagonal, 0.0)
    score = 0.5 * (score + score.transpose(1, 2))
    if not torch.equal(score, score.transpose(1, 2)) or not bool(torch.isfinite(score).all()):
        raise ValueError("structured same-slot score must be finite and symmetric")
    return score


def decode_structured_local_composition(
    prediction: LocalCompositionSlotModelPrediction,
    *,
    confidence_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("structured local composition threshold must lie in [0,1]")
    best_slot, confidence = structured_endpoint_slot_confidence(prediction)
    accepted = confidence >= confidence_threshold
    labels = torch.where(accepted, best_slot, torch.full_like(best_slot, -1))
    attachment = (
        (labels[:, :, None] >= 0)
        & (labels[:, :, None] == labels[:, None, :])
    )
    diagonal = torch.eye(ENDPOINT_COUNT, dtype=torch.bool, device=labels.device)[None]
    attachment &= ~diagonal
    return labels, attachment


__all__ = [
    "decode_structured_local_composition",
    "structured_endpoint_slot_confidence",
    "structured_same_slot_pair_score",
]
