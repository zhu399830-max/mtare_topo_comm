"""Typed observation adapter for the frozen exit-only M1D baseline."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components
from mtare_topo.semantics.geometric_semantics import (
    EVENT_NAMES,
    ExitGeometryToken,
    GeometricSemanticObservation,
)


def _unit(values: list[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if array.ndim != 1 or norm <= 1e-12 or not np.all(np.isfinite(array)):
        raise ValueError("exit-only descriptor is degenerate")
    return tuple(float(value) for value in array / norm)


def exit_only_observation_from_arrays(
    arrays: Mapping[str, np.ndarray],
    row: int,
) -> GeometricSemanticObservation:
    """Materialize M1D role/direction outputs without adding metric semantics."""

    required = ("event_probabilities", "direction_logits", "z_role")
    if any(name not in arrays for name in required):
        raise KeyError("exit-only archive lacks role, direction or descriptor output")
    probabilities = np.asarray(arrays["event_probabilities"][row], dtype=np.float64)
    if probabilities.shape != (len(EVENT_NAMES),) or not np.all(np.isfinite(probabilities)):
        raise ValueError("exit-only event probabilities are invalid")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("exit-only event probabilities have zero mass")
    probabilities /= total
    headings = decode_direction_components(
        np.asarray(arrays["direction_logits"][row], dtype=np.float64),
        0.5,
    )
    tokens = []
    for heading in headings:
        radians = math.radians(heading)
        descriptor = _unit([math.sin(radians), math.cos(radians), 1.0])
        tokens.append(
            ExitGeometryToken(
                heading_robot_deg=heading,
                opening_width_m=1.0,
                vertical_profile=(0.0, 0.0, 0.0, 0.0),
                descriptor=descriptor,
                confidence=1.0,
            )
        )
    place = np.asarray(arrays["z_role"][row], dtype=np.float64)
    if place.ndim != 1 or not np.all(np.isfinite(place)) or float(np.linalg.norm(place)) <= 1e-12:
        event_index = int(np.argmax(probabilities))
        place = np.zeros(len(EVENT_NAMES), dtype=np.float64)
        place[event_index] = 1.0
    return GeometricSemanticObservation(
        event_probabilities={name: float(probabilities[index]) for index, name in enumerate(EVENT_NAMES)},
        local_axis=(1.0, 0.0, 0.0),
        width_m=1.0,
        height_m=1.0,
        slope_deg=0.0,
        curvature_per_m=0.0,
        place_descriptor=_unit(place.tolist()),
        exit_tokens=tuple(tokens),
        uncertainty=0.0,
    )


__all__ = ["exit_only_observation_from_arrays"]
