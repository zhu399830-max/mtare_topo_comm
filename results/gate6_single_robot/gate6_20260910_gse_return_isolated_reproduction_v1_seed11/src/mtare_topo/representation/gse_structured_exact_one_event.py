"""Structured likelihood for exactly one correct graph-node commit per episode."""

from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


def structured_exact_one_event_loss(
    outputs: Mapping[str, torch.Tensor],
    decision_target: torch.Tensor,
    episode_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Negative log-likelihood of zero or exactly one structural commit.

    Corridor rows require zero junction/terminal commit.  Every positive bag
    requires exactly one commit of its correct class and no structural commit
    on every other row in the same bag.  No target identity enters the model.
    """

    logits = outputs["event_logits"]
    commit_logit = outputs["commit_logit"]
    target = decision_target.long()
    episode = episode_id.long()
    if (
        logits.shape != (len(target), 3)
        or commit_logit.shape != target.shape
        or episode.shape != target.shape
        or torch.any((target < 0) | (target > 2))
        or torch.any((target == 0) != (episode < 0))
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(commit_logit).all())
    ):
        raise ValueError("structured exact-one target/output contract drift")
    negative = target == 0
    if not bool(negative.any()) or not bool((episode >= 0).any()):
        raise ValueError("structured exact-one loss requires corridor and event episodes")
    log_event = F.log_softmax(logits, dim=1)
    log_commit = F.logsigmoid(commit_logit)
    # No structural commit is the disjoint union of not committing at all and
    # committing the corridor class.  logaddexp avoids 1-p cancellation when
    # a float32 sigmoid/softmax is saturated.
    log_no_structural = torch.logaddexp(
        F.logsigmoid(-commit_logit),
        log_commit + log_event[:, 0],
    )
    negative_zero_commit = -log_no_structural[negative].mean()
    positive_terms = []
    for identity in torch.unique(episode[episode >= 0], sorted=True):
        rows = episode == identity
        classes = torch.unique(target[rows])
        if len(classes) != 1 or int(classes[0]) not in (1, 2):
            raise ValueError("one exact-one episode must have one structural class")
        event = int(classes[0])
        log_correct = log_commit[rows] + log_event[rows, event]
        log_exactly_one_correct = (
            log_no_structural[rows].sum()
            + torch.logsumexp(log_correct - log_no_structural[rows], dim=0)
        )
        positive_terms.append(-log_exactly_one_correct)
    positive_exactly_one = torch.stack(positive_terms).mean()
    total = negative_zero_commit + positive_exactly_one
    if not bool(torch.isfinite(total)):
        raise RuntimeError("structured exact-one loss is non-finite")
    return {
        "negative_zero_structural_commit": negative_zero_commit,
        "positive_exactly_one_correct_commit": positive_exactly_one,
        "total": total,
    }


__all__ = ["structured_exact_one_event_loss"]
