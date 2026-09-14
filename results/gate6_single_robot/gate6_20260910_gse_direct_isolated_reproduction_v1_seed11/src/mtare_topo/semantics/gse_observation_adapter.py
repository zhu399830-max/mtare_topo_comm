"""Convert frozen network arrays and validation calibration into typed observations."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from mtare_topo.semantics.geometric_semantics import (
    EVENT_NAMES,
    ExitGeometryToken,
    GeometricSemanticObservation,
)


def calibrated_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (len(EVENT_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("one event-logit vector is required")
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("event temperature must be positive and finite")
    values = values / float(temperature)
    exponential = np.exp(values - values.max())
    return exponential / exponential.sum()


def geometric_semantic_observation_from_arrays(
    outputs: Mapping[str, np.ndarray],
    row: int,
    *,
    event_temperature: float,
    exit_presence_threshold: float,
) -> GeometricSemanticObservation:
    """Materialize exactly the learned interface consumed by the online graph."""

    if not 0.0 <= exit_presence_threshold <= 1.0:
        raise ValueError("exit presence threshold must be in [0,1]")
    logits = np.asarray(outputs["event_logits"])[row]
    probabilities = calibrated_softmax(logits, event_temperature)
    # Slice first, then cast one observation.  Casting the complete archive for
    # every row is mathematically identical but quadratic in materialization
    # traffic and made the 24,462-row deployment adapter unusably slow.
    confidence = np.asarray(outputs["exit_confidence"][row], dtype=np.float64)
    heading = np.asarray(outputs["exit_heading_unit"][row], dtype=np.float64)
    width = np.asarray(outputs["exit_opening_width_m"][row], dtype=np.float64)
    profile = np.asarray(outputs["exit_vertical_profile"][row], dtype=np.float64)
    descriptor = np.asarray(outputs["exit_descriptor"][row], dtype=np.float64)
    if (
        confidence.ndim != 1
        or heading.shape != (len(confidence), 2)
        or width.shape != confidence.shape
        or profile.shape[0] != len(confidence)
        or descriptor.shape[0] != len(confidence)
    ):
        raise ValueError("exit output arrays are misaligned")
    tokens = []
    for index in np.flatnonzero(confidence >= exit_presence_threshold):
        sine, cosine = heading[index]
        tokens.append(
            ExitGeometryToken(
                heading_robot_deg=math.degrees(math.atan2(float(sine), float(cosine))),
                opening_width_m=float(width[index]),
                vertical_profile=tuple(float(value) for value in profile[index]),
                descriptor=tuple(float(value) for value in descriptor[index]),
                confidence=float(confidence[index]),
            )
        )
    tokens.sort(key=lambda token: (token.heading_robot_deg, -token.confidence))
    return GeometricSemanticObservation(
        event_probabilities={name: float(probabilities[index]) for index, name in enumerate(EVENT_NAMES)},
        local_axis=tuple(float(value) for value in np.asarray(outputs["local_axis"])[row]),
        width_m=float(np.asarray(outputs["width_m"])[row]),
        height_m=float(np.asarray(outputs["height_m"])[row]),
        slope_deg=float(np.asarray(outputs["slope_deg"])[row]),
        curvature_per_m=float(np.asarray(outputs["curvature_per_m"])[row]),
        place_descriptor=tuple(float(value) for value in np.asarray(outputs["place_descriptor"])[row]),
        exit_tokens=tuple(tokens),
        uncertainty=float(np.asarray(outputs["uncertainty"])[row]),
    )


__all__ = ["calibrated_softmax", "geometric_semantic_observation_from_arrays"]
