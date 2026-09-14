"""Exit-token-aware open-set association on frozen GSE observations.

The V1 corrective retained only five aggregate exit statistics.  This module
keeps the frozen per-exit descriptor and geometry, while constructing features
that are exactly symmetric, invariant to token permutation, and invariant to a
common robot-frame yaw rotation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from mtare_topo.representation.gse_open_set_association import (
    OBSERVATION_FEATURE_DIM,
    PAIR_FEATURE_DIM,
)


EXIT_TOKEN_COUNT = 6
EXIT_DESCRIPTOR_DIM = 32
EXIT_VERTICAL_PROFILE_DIM = 4
EXIT_TOKEN_PAIR_FEATURE_DIM = 180
TOKEN_AWARE_PAIR_FEATURE_DIM = PAIR_FEATURE_DIM + EXIT_TOKEN_PAIR_FEATURE_DIM


def _array(outputs: Mapping[str, Any], name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(outputs[name], dtype=np.float32)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value


def _token_arrays(outputs: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    confidence = np.asarray(outputs["exit_confidence"], dtype=np.float32)
    if confidence.ndim != 2 or confidence.shape[1] != EXIT_TOKEN_COUNT:
        raise ValueError("exit_confidence must be [N,6]")
    rows = confidence.shape[0]
    if not np.all(np.isfinite(confidence)) or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("exit_confidence must be finite in [0,1]")
    heading = _array(outputs, "exit_heading_unit", (rows, EXIT_TOKEN_COUNT, 2))
    heading_norm = np.linalg.norm(heading, axis=2)
    if np.any(np.abs(heading_norm - 1.0) > 1e-2):
        raise ValueError("exit_heading_unit rows must be unit length")
    width = _array(outputs, "exit_opening_width_m", (rows, EXIT_TOKEN_COUNT))
    if np.any(width <= 0.0):
        raise ValueError("exit opening widths must be positive")
    vertical = _array(
        outputs,
        "exit_vertical_profile",
        (rows, EXIT_TOKEN_COUNT, EXIT_VERTICAL_PROFILE_DIM),
    )
    descriptor = _array(
        outputs,
        "exit_descriptor",
        (rows, EXIT_TOKEN_COUNT, EXIT_DESCRIPTOR_DIM),
    )
    descriptor_norm = np.linalg.norm(descriptor, axis=2)
    if np.any(np.abs(descriptor_norm - 1.0) > 1e-2):
        raise ValueError("exit descriptors must be unit length")
    return confidence, heading, width, vertical, descriptor


def exit_token_pair_features(
    left_outputs: Mapping[str, Any],
    right_outputs: Mapping[str, Any],
) -> np.ndarray:
    """Build symmetric, token-permutation and common-yaw invariant features.

    Descriptor mutual-nearest correspondences preserve which exit geometry
    agrees across observations.  Within-set heading Gram signatures preserve
    the exit layout without using absolute robot yaw.  Every directional list
    is concatenated with its reverse direction and sorted, making the result
    exactly left/right symmetric.
    """

    lc, lh, lw, lv, ld = _token_arrays(left_outputs)
    rc, rh, rw, rv, rd = _token_arrays(right_outputs)
    if lc.shape[0] != rc.shape[0]:
        raise ValueError("left/right exit-token batches must have equal length")

    descriptor_cross = np.einsum("nqd,nkd->nqk", ld, rd, optimize=True)
    confidence_cross = np.sqrt(np.maximum(lc[:, :, None] * rc[:, None, :], 0.0))
    correspondence_score = descriptor_cross * confidence_cross
    left_to_right = correspondence_score.argmax(axis=2)
    right_to_left = correspondence_score.argmax(axis=1)

    def directional(
        source_conf: np.ndarray,
        source_width: np.ndarray,
        source_vertical: np.ndarray,
        target_conf: np.ndarray,
        target_width: np.ndarray,
        target_vertical: np.ndarray,
        cross: np.ndarray,
        target_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        index = target_index[:, :, None]
        cosine = np.take_along_axis(cross, index, axis=2).squeeze(2)
        matched_conf = np.take_along_axis(target_conf, target_index, axis=1)
        matched_width = np.take_along_axis(target_width, target_index, axis=1)
        matched_vertical = np.take_along_axis(
            target_vertical,
            np.repeat(index, EXIT_VERTICAL_PROFILE_DIM, axis=2),
            axis=1,
        )
        confidence = np.sqrt(np.maximum(source_conf * matched_conf, 0.0))
        width_delta = np.abs(np.log(np.maximum(source_width, 1e-4) / np.maximum(matched_width, 1e-4)))
        vertical_delta = np.mean(np.abs(source_vertical - matched_vertical), axis=2) / 10.0
        return cosine, confidence, width_delta, vertical_delta

    forward = directional(lc, lw, lv, rc, rw, rv, descriptor_cross, left_to_right)
    reverse = directional(
        rc,
        rw,
        rv,
        lc,
        lw,
        lv,
        descriptor_cross.transpose(0, 2, 1),
        right_to_left,
    )
    nearest_features = [
        np.sort(np.concatenate((a, b), axis=1), axis=1)
        for a, b in zip(forward, reverse, strict=True)
    ]

    upper = np.triu_indices(EXIT_TOKEN_COUNT, 1)

    def layout(confidence: np.ndarray, heading: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gram = np.einsum("nqd,nkd->nqk", heading, heading, optimize=True)[:, upper[0], upper[1]]
        pair_weight = confidence[:, upper[0]] * confidence[:, upper[1]]
        return np.sort(gram, axis=1), np.sort(gram * pair_weight, axis=1)

    left_layout = layout(lc, lh)
    right_layout = layout(rc, rh)
    layout_features = []
    for left_value, right_value in zip(left_layout, right_layout, strict=True):
        layout_features.extend((np.abs(left_value - right_value), left_value * right_value))

    distribution_features = []
    left_distribution = (
        np.sort(lc, axis=1),
        np.sort(lw / 30.0, axis=1),
        np.sort(np.linalg.norm(lv, axis=2) / 10.0, axis=1),
    )
    right_distribution = (
        np.sort(rc, axis=1),
        np.sort(rw / 30.0, axis=1),
        np.sort(np.linalg.norm(rv, axis=2) / 10.0, axis=1),
    )
    for left_value, right_value in zip(left_distribution, right_distribution, strict=True):
        distribution_features.extend((np.abs(left_value - right_value), left_value * right_value))

    descriptor_distribution = np.sort(descriptor_cross.reshape(len(lc), -1), axis=1)
    result = np.concatenate(
        (*nearest_features, *layout_features, *distribution_features, descriptor_distribution),
        axis=1,
    ).astype(np.float32)
    if result.shape != (len(lc), EXIT_TOKEN_PAIR_FEATURE_DIM) or not np.all(np.isfinite(result)):
        raise RuntimeError("exit-token pair feature contract drift")
    return result


class GSEExitTokenAssociationVerifier(nn.Module):
    """Lightweight verifier using frozen observation and exit-token features."""

    def __init__(self) -> None:
        super().__init__()
        self.matchability = nn.Sequential(
            nn.Linear(OBSERVATION_FEATURE_DIM, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.pair = nn.Sequential(
            nn.Linear(TOKEN_AWARE_PAIR_FEATURE_DIM, 192),
            nn.GELU(),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            left.ndim != 2
            or left.shape != right.shape
            or left.shape[1] != OBSERVATION_FEATURE_DIM
            or pair_features.shape != (len(left), TOKEN_AWARE_PAIR_FEATURE_DIM)
        ):
            raise ValueError("token-aware verifier tensor shapes violate the frozen contract")
        left_logit = self.matchability(left).squeeze(1)
        right_logit = self.matchability(right).squeeze(1)
        pair_logit = self.pair(pair_features).squeeze(1)
        score = torch.sigmoid(pair_logit) * torch.minimum(
            torch.sigmoid(left_logit), torch.sigmoid(right_logit)
        )
        return {
            "left_matchability_logit": left_logit,
            "right_matchability_logit": right_logit,
            "pair_logit": pair_logit,
            "association_score": score,
        }


def hard_negative_tail_loss(probability: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Equal-weight low positives and the same count of high negatives."""

    labels = labels.bool()
    positive = probability[labels]
    negative = probability[~labels]
    if len(positive) == 0 or len(negative) == 0:
        raise RuntimeError("tail-risk batch requires both classes")
    count = min(len(positive), len(negative))
    hard_negative = torch.topk(negative, k=count, largest=True, sorted=False).values
    selected_positive = torch.topk(positive, k=count, largest=False, sorted=False).values
    selected_positive = selected_positive.clamp(1e-6, 1.0 - 1e-6)
    hard_negative = hard_negative.clamp(1e-6, 1.0 - 1e-6)
    return -0.5 * (selected_positive.log().mean() + (1.0 - hard_negative).log().mean())


__all__ = [
    "EXIT_TOKEN_PAIR_FEATURE_DIM",
    "GSEExitTokenAssociationVerifier",
    "TOKEN_AWARE_PAIR_FEATURE_DIM",
    "exit_token_pair_features",
    "hard_negative_tail_loss",
]
