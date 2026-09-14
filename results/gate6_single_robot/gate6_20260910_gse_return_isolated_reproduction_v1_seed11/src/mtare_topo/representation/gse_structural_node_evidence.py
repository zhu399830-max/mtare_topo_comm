"""Fast, training-free evidence for geometry-semantic structural nodes.

The functions in this module deliberately operate on a *set* of observable
primitive ports.  Primitive identity, node identity, world identity and
absolute position are not part of the descriptor.  The representation is an
upper-bound diagnostic: it asks whether local primitive composition contains
enough invariant information to justify training a node model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


MAX_NODE_PORTS = 4
PORT_FEATURE_DIM = 5
PAIR_CAPACITY = MAX_NODE_PORTS * (MAX_NODE_PORTS - 1) // 2
NODE_EVIDENCE_DIM = 1 + MAX_NODE_PORTS * PORT_FEATURE_DIM + 2 * PAIR_CAPACITY + 10 + 1


@dataclass(frozen=True)
class PrimitivePortEvidence:
    """Deployment-compatible geometry for one observed tunnel port."""

    away_direction_xyz: tuple[float, float, float]
    half_axes_m: tuple[float, float]
    shape_exponent: float
    curvature_per_m: float
    support_rays: int

    def validated(self) -> "PrimitivePortEvidence":
        direction = np.asarray(self.away_direction_xyz, dtype=np.float64)
        axes = np.asarray(self.half_axes_m, dtype=np.float64)
        if (
            direction.shape != (3,)
            or axes.shape != (2,)
            or not np.all(np.isfinite(direction))
            or not np.all(np.isfinite(axes))
            or not np.isfinite(self.shape_exponent)
            or not np.isfinite(self.curvature_per_m)
            or np.linalg.norm(direction) <= 1e-8
            or np.any(axes <= 0.0)
            or self.shape_exponent < 2.0
            or self.curvature_per_m < 0.0
            or self.support_rays < 0
        ):
            raise ValueError("primitive port evidence violates the frozen contract")
        return self


def _port_feature(port: PrimitivePortEvidence) -> np.ndarray:
    port.validated()
    a, b = sorted(port.half_axes_m)
    # Fixed physical scalings keep the audit deterministic and prevent fit/C07
    # statistics from leaking into the representation.
    return np.asarray(
        (
            np.log1p(np.sqrt(a * b)) / np.log(31.0),
            abs(np.log(a / b)) / np.log(10.0),
            np.clip((port.shape_exponent - 2.0) / 8.0, 0.0, 2.0),
            np.clip(port.curvature_per_m / 0.5, 0.0, 2.0),
            # Gravity is observable online and yaw invariant.  Ray support is
            # intentionally absent here: it expresses confidence, not place
            # identity, and is retained only in the final uncertainty field.
            abs(np.asarray(port.away_direction_xyz, dtype=np.float64)[2])
            / np.linalg.norm(np.asarray(port.away_direction_xyz, dtype=np.float64)),
        ),
        dtype=np.float64,
    )


def structural_node_evidence_descriptor(
    ports: Sequence[PrimitivePortEvidence],
) -> np.ndarray:
    """Encode a primitive star without pose, orientation or slot-order leakage.

    Absolute directions are removed.  Only pairwise angular relations and
    port-local shape attributes remain, so rotating the robot frame or
    permuting primitive slots produces the same descriptor.
    """

    if not 1 <= len(ports) <= MAX_NODE_PORTS:
        raise ValueError("a structural node must expose one to four ports")
    checked = tuple(port.validated() for port in ports)
    features = np.stack([_port_feature(port) for port in checked])
    # Sorting makes the per-port block independent of primitive slot order.
    order = np.lexsort(tuple(features[:, column] for column in reversed(range(PORT_FEATURE_DIM))))
    sorted_features = features[order]
    port_block = np.zeros((MAX_NODE_PORTS, PORT_FEATURE_DIM), dtype=np.float64)
    port_block[: len(checked)] = sorted_features

    directions = np.stack(
        [np.asarray(port.away_direction_xyz, dtype=np.float64) for port in checked]
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    angular = sorted(
        float(np.clip(np.dot(directions[left], directions[right]), -1.0, 1.0))
        for left, right in combinations(range(len(checked)), 2)
    )
    angular_block = np.zeros(PAIR_CAPACITY, dtype=np.float64)
    angular_block[: len(angular)] = angular

    shape_delta = sorted(
        float(np.linalg.norm(features[left, :4] - features[right, :4]))
        for left, right in combinations(range(len(checked)), 2)
    )
    shape_block = np.zeros(PAIR_CAPACITY, dtype=np.float64)
    shape_block[: len(shape_delta)] = shape_delta

    moments = np.concatenate((features.mean(axis=0), features.std(axis=0)))
    support = np.asarray([port.support_rays for port in checked], dtype=np.float64)
    uncertainty = 1.0 / np.sqrt(1.0 + float(np.sum(support)))
    descriptor = np.concatenate(
        (
            np.asarray([len(checked) / MAX_NODE_PORTS], dtype=np.float64),
            port_block.reshape(-1),
            angular_block,
            shape_block,
            moments,
            np.asarray([uncertainty], dtype=np.float64),
        )
    ).astype(np.float32)
    if descriptor.shape != (NODE_EVIDENCE_DIM,) or not np.all(np.isfinite(descriptor)):
        raise RuntimeError("structural node evidence descriptor contract drift")
    return descriptor


def pair_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    include_uncertainty: bool = False,
) -> np.ndarray:
    """Return row-wise RMS identity distance between node descriptors.

    Support-derived uncertainty is excluded by default.  It may gate whether
    a proposal is trusted, but cannot make the same geometry a different
    place merely because viewpoint-dependent ray count changed.
    """

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.ndim != 2
        or right.shape != left.shape
        or left.shape[1] != NODE_EVIDENCE_DIM
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("node descriptor pair contract drift")
    identity_left = left if include_uncertainty else left[:, :-1]
    identity_right = right if include_uncertainty else right[:, :-1]
    return np.sqrt(np.mean(np.square(identity_left - identity_right), axis=1))


def safest_nonempty_threshold(
    distance: np.ndarray,
    same_node: np.ndarray,
    *,
    minimum_precision: float = 0.98,
) -> dict[str, float | int | bool | None]:
    """Select the highest-recall tie-safe distance threshold on fit data."""

    distance = np.asarray(distance, dtype=np.float64)
    target = np.asarray(same_node, dtype=np.bool_)
    if (
        distance.ndim != 1
        or target.shape != distance.shape
        or len(distance) == 0
        or not np.all(np.isfinite(distance))
        or not 0.0 < minimum_precision <= 1.0
        or not np.any(target)
        or np.all(target)
    ):
        raise ValueError("threshold selection population is invalid")
    order = np.argsort(distance, kind="stable")
    sorted_distance = distance[order]
    sorted_target = target[order]
    cumulative_true = np.cumsum(sorted_target, dtype=np.int64)
    group_end = np.flatnonzero(np.r_[sorted_distance[1:] != sorted_distance[:-1], True])
    best: tuple[float, int, int, float, float] | None = None
    positives = int(np.sum(target))
    for end in group_end.tolist():
        predicted = end + 1
        true_positive = int(cumulative_true[end])
        precision = true_positive / predicted
        if true_positive == 0 or precision < minimum_precision:
            continue
        recall = true_positive / positives
        candidate = (recall, true_positive, -predicted, precision, float(sorted_distance[end]))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return {
            "found": False,
            "threshold": None,
            "precision": 0.0,
            "recall": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "predicted_positive": 0,
        }
    recall, true_positive, negative_predicted, precision, threshold = best
    predicted = -negative_predicted
    return {
        "found": True,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": predicted - true_positive,
        "predicted_positive": predicted,
    }


__all__ = [
    "MAX_NODE_PORTS",
    "NODE_EVIDENCE_DIM",
    "PrimitivePortEvidence",
    "pair_distance",
    "safest_nonempty_threshold",
    "structural_node_evidence_descriptor",
]
