"""Read-only diagnostics for the endpoint-observable primitive relation model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mtare_topo.evaluation.primitive_relation_metrics import (
    _attachment_observability_validity,
    _attachment_upper_mask,
)
from mtare_topo.evaluation.primitive_relation_sparse_port_failure_attribution import (
    best_link_pair_mask,
    endpoint_pair_eligibility,
)


SCORE_NAMES = ("raw", "uncertainty_adjusted", "learned_safe")


@dataclass(frozen=True)
class ObservableAttachmentScoreSlice:
    scores: dict[str, np.ndarray]
    target: np.ndarray
    overlap_hard_negative: np.ndarray
    eligible_pairs: int
    eligible_true_pairs: int
    all_observable_target_pairs: int


def observable_attachment_score_slice(
    prediction,
    aligned: dict[str, torch.Tensor],
    primitive_mask: torch.Tensor,
    *,
    decoder: str = "independent",
    score_names: tuple[str, ...] = SCORE_NAMES,
) -> ObservableAttachmentScoreSlice:
    """Flatten observable relation scores under one evaluation-only candidate mask.

    Hidden matched endpoint pairs are excluded by the frozen observable metric
    contract. Unmatched predicted candidates remain eligible negatives.  The
    proposal-oracle mask therefore answers whether the unchanged relation head
    ranks connections when redundant slots are removed; it is never deployable.
    """

    unknown = set(score_names) - set(SCORE_NAMES)
    if unknown:
        raise ValueError(f"unknown observable score families: {sorted(unknown)}")
    batch = len(primitive_mask)
    raw = torch.sigmoid(prediction.endpoint_attachment_logits).reshape(batch, 64, 64)
    uncertainty = prediction.endpoint_attachment_uncertainty.reshape_as(raw)
    evidence = torch.sigmoid(prediction.endpoint_evidence_logits).reshape(batch, 64)
    evidence_pair = evidence[:, :, None] * evidence[:, None, :]
    validity = _attachment_observability_validity(aligned)
    upper = _attachment_upper_mask(raw.device)[None]
    target = aligned["attachment"].reshape_as(raw).bool() & validity & upper
    eligible = endpoint_pair_eligibility(primitive_mask) & validity
    if decoder == "best_link_union":
        eligible &= best_link_pair_mask(raw, eligible, mutual=False)
    elif decoder == "best_link_mutual":
        eligible &= best_link_pair_mask(raw, eligible, mutual=True)
    elif decoder != "independent":
        raise ValueError(f"unknown observable relation decoder: {decoder}")
    score_tensors = {
        "raw": raw,
        "uncertainty_adjusted": raw * (1.0 - uncertainty),
        "learned_safe": raw * (1.0 - uncertainty) * evidence_pair,
    }
    overlap = aligned["overlap"].bool()
    overlap_endpoint = overlap.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    selected_target = target[eligible].detach().cpu().numpy().astype(np.bool_, copy=False)
    return ObservableAttachmentScoreSlice(
        scores={
            name: score_tensors[name][eligible].detach().cpu().numpy().astype(np.float32, copy=False)
            for name in score_names
        },
        target=selected_target,
        overlap_hard_negative=(
            overlap_endpoint[eligible].detach().cpu().numpy().astype(np.bool_, copy=False)
        ),
        eligible_pairs=int(eligible.sum()),
        eligible_true_pairs=int(np.count_nonzero(selected_target)),
        all_observable_target_pairs=int(target.sum()),
    )


def observable_endpoint_evidence_slice(
    prediction,
    aligned: dict[str, torch.Tensor],
    *,
    matched_only: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return learned endpoint-evidence scores and observable targets."""

    score = torch.sigmoid(prediction.endpoint_evidence_logits).reshape(-1)
    target = aligned["endpoint_observed"].reshape(-1).bool()
    eligible = aligned["mask"].repeat_interleave(2, dim=1).reshape(-1).bool()
    if not matched_only:
        eligible = torch.ones_like(eligible)
    return (
        score[eligible].detach().cpu().numpy().astype(np.float32, copy=False),
        target[eligible].detach().cpu().numpy().astype(np.bool_, copy=False),
        int(target.sum()),
    )


def diagnose_observable_relation(
    condition_passing_seed_counts: dict[str, int],
) -> tuple[str, str]:
    """Resolve one pre-registered mechanism in strongest-evidence order."""

    count = condition_passing_seed_counts
    if count["deployed__independent__raw"] >= 2:
        if count["deployed__independent__learned_safe"] < 2:
            return (
                "LEARNED_ENDPOINT_EVIDENCE_OR_UNCERTAINTY_SUPPRESSES_VALID_RELATIONS",
                "ALLOW_SAFE_SCORE_CALIBRATION_FAILURE_ANALYSIS",
            )
        return (
            "REGISTERED_THRESHOLD_GRID_MISSED_EXACT_SAFE_PREFIX",
            "ALLOW_EVALUATION_CALIBRATION_CORRECTIVE_READINESS",
        )
    if (
        count["deployed__best_link_union__raw"] >= 2
        or count["teacher_cardinality__best_link_union__raw"] >= 2
    ):
        return (
            "PAIRWISE_SCORE_REQUIRES_STRUCTURE_CONSTRAINED_DECODING",
            "ALLOW_LEARNED_SCORE_GRAPH_MATCHING_READINESS",
        )
    if count["proposal_oracle__independent__raw"] >= 2:
        if count["proposal_oracle__independent__uncertainty_adjusted"] < 2:
            return (
                "RELATION_UNCERTAINTY_SUPPRESSES_ORACLE_SAFE_RELATIONS",
                "ALLOW_RELATION_UNCERTAINTY_CALIBRATION_READINESS",
            )
        if count["proposal_oracle__independent__learned_safe"] < 2:
            return (
                "ENDPOINT_EVIDENCE_SUPPRESSES_ORACLE_SAFE_RELATIONS",
                "ALLOW_ENDPOINT_EVIDENCE_CALIBRATION_READINESS",
            )
        return (
            "PRIMITIVE_PROPOSAL_DOMINATES_OBSERVABLE_RELATION_FAILURE",
            "ALLOW_PROPOSAL_RELATION_DECOUPLING_READINESS",
        )
    if count["proposal_oracle__best_link_union__raw"] >= 2:
        return (
            "RELATION_SCORE_CONTAINS_ONLY_LOCAL_BEST_LINK_RANKING",
            "ALLOW_LEARNED_SCORE_GRAPH_MATCHING_READINESS",
        )
    return (
        "RELATION_SCORE_FAILS_EVEN_WITH_OBSERVABLE_PROPOSAL_ORACLE",
        "STOP_DIRECT_PAIR_RELATION_HEAD_AND_REASSESS_ARCHITECTURE",
    )


__all__ = [
    "ObservableAttachmentScoreSlice",
    "SCORE_NAMES",
    "diagnose_observable_relation",
    "observable_attachment_score_slice",
    "observable_endpoint_evidence_slice",
]
