"""Route-conditioned, leakage-safe association for Factorized GSE-Graph."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import nn

from mtare_topo.representation.gse_exit_token_association import (
    EXIT_TOKEN_COUNT,
    EXIT_TOKEN_PAIR_FEATURE_DIM,
    exit_token_pair_features,
)


OBSERVATION_FEATURE_DIM = 146
EVENT_SLICE = slice(0, 5)
AXIS_SLICE = slice(5, 8)
GEOMETRY_SLICE = slice(8, 12)
PLACE_SLICE = slice(12, 140)
UNCERTAINTY_INDEX = 140
PROFILE_DIM = 9  # mean/std(width,height,slope,curvature) plus causal length / 5
BASE_RELATION_DIM = 19
PROFILE_RELATION_DIM = 2 * PROFILE_DIM
ROUTE_TOKEN_RELATION_DIM = 4 * EXIT_TOKEN_COUNT
FACTORIZED_PAIR_FEATURE_DIM = (
    BASE_RELATION_DIM
    + EXIT_TOKEN_PAIR_FEATURE_DIM
    + PROFILE_RELATION_DIM
    + ROUTE_TOKEN_RELATION_DIM
)
NO_ROUTE_PAIR_FEATURE_DIM = BASE_RELATION_DIM + EXIT_TOKEN_PAIR_FEATURE_DIM
VERTICAL_PROFILE_OFFSETS_M = np.asarray((2.5, 5.0, 7.5, 10.0), dtype=np.float32)


def learned_geometry_profiles(
    observation_features: np.ndarray,
    history_references: np.ndarray,
    history_mask: np.ndarray,
) -> np.ndarray:
    """Aggregate only causal learned geometry along the executed approach."""

    features = np.asarray(observation_features, dtype=np.float32)
    references = np.asarray(history_references, dtype=np.int64)
    mask = np.asarray(history_mask, dtype=np.bool_)
    if (
        features.ndim != 2 or features.shape[1] != OBSERVATION_FEATURE_DIM
        or references.shape != mask.shape or references.ndim != 2
        or references.shape[0] != len(features) or references.shape[1] != 5
        or not np.all(np.isfinite(features))
    ):
        raise ValueError("learned geometry profile inputs violate the frozen contract")
    profiles = np.empty((len(features), PROFILE_DIM), dtype=np.float32)
    geometry = features[:, GEOMETRY_SLICE]
    for row in range(len(features)):
        indices = references[row, mask[row]]
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(features)):
            raise ValueError("causal learned geometry profile is empty or out of bounds")
        values = geometry[indices]
        profiles[row, :4] = values.mean(axis=0)
        profiles[row, 4:8] = values.std(axis=0)
        profiles[row, 8] = len(indices) / 5.0
    if not np.all(np.isfinite(profiles)):
        raise RuntimeError("learned geometry profiles are nonfinite")
    return profiles


def _route_to_tokens(
    profile: np.ndarray,
    confidence: np.ndarray,
    opening_width_m: np.ndarray,
    vertical_profile_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare an executed incoming edge to every observed exit token."""

    # Geometry columns were normalized by (30 m, 30 m, 45 deg, 0.1/m)
    # when the frozen observation features were exported.
    incoming_width_m = np.maximum(profile[:, 0] * 30.0, 1e-4)
    incoming_slope_deg = np.clip(profile[:, 2] * 45.0, -44.9, 44.9)
    width_delta = np.abs(np.log(incoming_width_m[:, None] / np.maximum(opening_width_m, 1e-4)))
    expected_vertical = (
        np.tan(np.deg2rad(incoming_slope_deg))[:, None, None]
        * VERTICAL_PROFILE_OFFSETS_M[None, None, :]
    )
    vertical_delta = np.mean(np.abs(expected_vertical - vertical_profile_m), axis=2) / 10.0
    # Confidence is not used to remove tokens. Multiplication retains all six
    # learned hypotheses while encoding how much each comparison is trusted.
    return width_delta * confidence, vertical_delta * confidence


def factorized_pair_features(
    observation_features: np.ndarray,
    token_outputs: Mapping[str, np.ndarray],
    geometry_profiles: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    include_route_geometry: bool = True,
) -> np.ndarray:
    """Build symmetric deployment-only pair features.

    Identity, parent/world, Teacher event/geometry and absolute pose are absent
    by construction. Spatial distance remains an external <=16 m candidate
    gate and is deliberately not a classifier feature.
    """

    observation = np.asarray(observation_features, dtype=np.float32)
    profile = np.asarray(geometry_profiles, dtype=np.float32)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if (
        observation.ndim != 2 or observation.shape[1] != OBSERVATION_FEATURE_DIM
        or profile.shape != (len(observation), PROFILE_DIM)
        or left.ndim != 1 or right.shape != left.shape
        or np.any(left < 0) or np.any(right < 0)
        or np.any(left >= len(observation)) or np.any(right >= len(observation))
        or not np.all(np.isfinite(observation)) or not np.all(np.isfinite(profile))
    ):
        raise ValueError("factorized pair indices or frozen features are invalid")
    lo, ro = observation[left], observation[right]
    event = np.concatenate((np.abs(lo[:, EVENT_SLICE] - ro[:, EVENT_SLICE]),
                            lo[:, EVENT_SLICE] * ro[:, EVENT_SLICE]), axis=1)
    axis_dot = np.abs(np.sum(lo[:, AXIS_SLICE] * ro[:, AXIS_SLICE], axis=1, keepdims=True))
    axis_norm_delta = np.abs(
        np.linalg.norm(lo[:, AXIS_SLICE], axis=1, keepdims=True)
        - np.linalg.norm(ro[:, AXIS_SLICE], axis=1, keepdims=True)
    )
    ld, rd = lo[:, PLACE_SLICE], ro[:, PLACE_SLICE]
    descriptor = np.stack((
        np.sum(ld * rd, axis=1),
        np.mean(np.abs(ld - rd), axis=1),
        np.max(np.abs(ld - rd), axis=1),
    ), axis=1)
    lu, ru = lo[:, UNCERTAINTY_INDEX], ro[:, UNCERTAINTY_INDEX]
    uncertainty = np.stack((np.minimum(lu, ru), np.maximum(lu, ru), np.abs(lu - ru), lu * ru), axis=1)
    base = np.concatenate((event, axis_dot, axis_norm_delta, descriptor, uncertainty), axis=1)
    if base.shape[1] != BASE_RELATION_DIM:
        raise RuntimeError("base factorized relation dimension drift")

    token_keys = (
        "exit_confidence", "exit_heading_unit", "exit_opening_width_m",
        "exit_vertical_profile", "exit_descriptor",
    )
    token = {name: np.asarray(token_outputs[name], dtype=np.float32) for name in token_keys}
    if any(value.shape[0] != len(observation) or not np.all(np.isfinite(value)) for value in token.values()):
        raise ValueError("factorized exit-token population is invalid")
    left_tokens = {name: value[left] for name, value in token.items()}
    right_tokens = {name: value[right] for name, value in token.items()}
    token_relation = exit_token_pair_features(left_tokens, right_tokens)
    values = [base, token_relation]
    if include_route_geometry:
        lp, rp = profile[left], profile[right]
        profile_relation = np.concatenate((np.abs(lp - rp), lp * rp), axis=1)
        lw, lv = _route_to_tokens(
            lp, right_tokens["exit_confidence"], right_tokens["exit_opening_width_m"],
            right_tokens["exit_vertical_profile"],
        )
        rw, rv = _route_to_tokens(
            rp, left_tokens["exit_confidence"], left_tokens["exit_opening_width_m"],
            left_tokens["exit_vertical_profile"],
        )
        # Sorting the two directed six-token comparisons together makes the
        # final relation exactly left/right symmetric and token-order invariant.
        route_relation = np.concatenate((
            np.sort(np.concatenate((lw, rw), axis=1), axis=1),
            np.sort(np.concatenate((lv, rv), axis=1), axis=1),
        ), axis=1)
        values.extend((profile_relation, route_relation))
    result = np.concatenate(values, axis=1).astype(np.float32)
    expected = FACTORIZED_PAIR_FEATURE_DIM if include_route_geometry else NO_ROUTE_PAIR_FEATURE_DIM
    if result.shape != (len(left), expected) or not np.all(np.isfinite(result)):
        raise RuntimeError("factorized pair feature contract drift")
    return result


class FactorizedAssociationVerifier(nn.Module):
    """Small refusal-aware relation model; the GSE perception model is frozen."""

    def __init__(self, *, include_route_geometry: bool = True) -> None:
        super().__init__()
        self.include_route_geometry = bool(include_route_geometry)
        dimension = FACTORIZED_PAIR_FEATURE_DIM if self.include_route_geometry else NO_ROUTE_PAIR_FEATURE_DIM
        self.relation = nn.Sequential(
            nn.Linear(dimension, 32), nn.GELU(), nn.Linear(32, 8), nn.GELU(), nn.Linear(8, 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        dimension = FACTORIZED_PAIR_FEATURE_DIM if self.include_route_geometry else NO_ROUTE_PAIR_FEATURE_DIM
        if pair_features.ndim != 2 or pair_features.shape[1] != dimension:
            raise ValueError("factorized association tensor shape drift")
        return self.relation(pair_features).squeeze(1)


def parameter_count(include_route_geometry: bool = True) -> int:
    return sum(parameter.numel() for parameter in FactorizedAssociationVerifier(
        include_route_geometry=include_route_geometry
    ).parameters())


__all__ = [
    "FACTORIZED_PAIR_FEATURE_DIM", "NO_ROUTE_PAIR_FEATURE_DIM",
    "FactorizedAssociationVerifier", "factorized_pair_features",
    "learned_geometry_profiles", "parameter_count",
]
