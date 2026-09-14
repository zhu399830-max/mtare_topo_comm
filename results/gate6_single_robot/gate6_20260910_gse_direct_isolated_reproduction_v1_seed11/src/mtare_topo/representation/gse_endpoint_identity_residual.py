"""Identity-balanced structural-logit corrective for rare topology endpoints."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class EndpointIdentityStructuralResidual(nn.Module):
    """A zero-initialized 129-parameter residual on frozen causal context."""

    def __init__(self, context_dim: int = 128) -> None:
        super().__init__()
        if context_dim != 128:
            raise ValueError("endpoint residual context dimension is frozen at 128")
        self.residual = nn.Linear(context_dim, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, causal_context: torch.Tensor) -> torch.Tensor:
        if (
            causal_context.ndim != 2 or causal_context.shape[1] != 128
            or not bool(torch.isfinite(causal_context).all())
        ):
            raise ValueError("endpoint residual requires finite [N,128] causal context")
        return self.residual(causal_context).squeeze(1)


def corrected_decision_distribution(
    base_outputs: Mapping[str, torch.Tensor], residual: EndpointIdentityStructuralResidual,
) -> dict[str, torch.Tensor]:
    """Change only structural mass; keep context and conditional class logits fixed."""

    structural = base_outputs["structural_logit"]
    conditional = base_outputs["conditional_decision_logits"]
    context = base_outputs["causal_context"]
    if structural.shape != (len(context),) or conditional.shape != (len(context), 2):
        raise ValueError("endpoint residual/base output alignment drift")
    corrected = structural + residual(context)
    structural_probability = torch.sigmoid(corrected)
    conditional_probability = torch.softmax(conditional, dim=1)
    probability = torch.cat((
        (1.0 - structural_probability).unsqueeze(1),
        structural_probability.unsqueeze(1) * conditional_probability,
    ), dim=1)
    uncertainty = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=1) / torch.log(
        probability.new_tensor(3.0)
    )
    return {
        "structural_logit": corrected,
        "conditional_decision_logits": conditional,
        "decision_probability": probability,
        "uncertainty": uncertainty,
        "causal_context": context,
    }


def identity_balanced_endpoint_mil_loss(
    *, structural_logit: torch.Tensor, conditional_decision_logits: torch.Tensor,
    decision_target: torch.Tensor, episode_id: torch.Tensor,
    endpoint_identity_by_episode: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Original negative/all-episode MIL plus an exact identity-mean term.

    ``episode_id`` is -1 for corridor rows and compact 0..E-1 for positive
    rows. ``endpoint_identity_by_episode`` is -1 for ordinary episodes and
    compact 0..I-1 for relation-endpoint episodes. Each identity is therefore
    weighted once regardless of whether it produced one or eight episodes.
    """

    structural = structural_logit
    conditional = conditional_decision_logits
    target = decision_target.long()
    episodes = episode_id.long()
    endpoint_identity = endpoint_identity_by_episode.long()
    if (
        structural.ndim != 1 or conditional.shape != (len(structural), 2)
        or target.shape != structural.shape or episodes.shape != structural.shape
        or endpoint_identity.ndim != 1 or not bool(torch.isfinite(structural).all())
        or not bool(torch.isfinite(conditional).all())
        or bool(torch.any((target < 0) | (target > 2)))
        or bool(torch.any((target == 0) != (episodes < 0)))
    ):
        raise ValueError("identity-balanced endpoint MIL input drift")
    negative_mask = target == 0
    positive_mask = ~negative_mask
    if not bool(negative_mask.any()) or not bool(positive_mask.any()):
        raise ValueError("identity-balanced endpoint MIL needs positive and negative rows")
    positive_episode = episodes[positive_mask]
    unique_episode = torch.unique(positive_episode, sorted=True)
    if not torch.equal(unique_episode, torch.arange(len(unique_episode), device=episodes.device)):
        raise ValueError("positive episode IDs must be compact")
    if endpoint_identity.shape != (len(unique_episode),):
        raise ValueError("endpoint identity vector must align to positive episodes")
    endpoint_codes = torch.unique(endpoint_identity[endpoint_identity >= 0], sorted=True)
    if len(endpoint_codes) == 0 or not torch.equal(
        endpoint_codes, torch.arange(len(endpoint_codes), device=episodes.device)
    ):
        raise ValueError("endpoint identity codes must be nonempty and compact")

    negative = F.binary_cross_entropy_with_logits(
        structural[negative_mask], torch.zeros_like(structural[negative_mask])
    )
    class_index = target[positive_mask] - 1
    joint = F.logsigmoid(structural[positive_mask]) + F.log_softmax(
        conditional[positive_mask], dim=1
    ).gather(1, class_index[:, None]).squeeze(1)
    episode_best = joint.new_full((len(unique_episode),), -torch.inf)
    episode_best.scatter_reduce_(0, positive_episode, joint, reduce="amax", include_self=True)
    if not bool(torch.isfinite(episode_best).all()):
        raise ValueError("an endpoint MIL episode has no positive row")
    episode_loss = -episode_best
    positive = episode_loss.mean()
    selected = endpoint_identity >= 0
    identity_sum = episode_loss.new_zeros(len(endpoint_codes))
    identity_count = episode_loss.new_zeros(len(endpoint_codes))
    identity_sum.scatter_add_(0, endpoint_identity[selected], episode_loss[selected])
    identity_count.scatter_add_(
        0, endpoint_identity[selected], torch.ones_like(episode_loss[selected])
    )
    if bool(torch.any(identity_count <= 0)):
        raise ValueError("endpoint identity has no episode")
    identity_balanced = (identity_sum / identity_count).mean()
    total = negative + positive + identity_balanced
    return {
        "negative": negative, "positive_episode_joint": positive,
        "endpoint_identity_mean": identity_balanced, "total": total,
    }


__all__ = [
    "EndpointIdentityStructuralResidual", "corrected_decision_distribution",
    "identity_balanced_endpoint_mil_loss",
]
