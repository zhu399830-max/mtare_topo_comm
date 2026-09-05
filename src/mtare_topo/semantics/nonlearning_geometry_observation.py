"""Non-learning geometry-event observation used as the principal GSE baseline."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG
from mtare_topo.semantics.geometric_semantics import (
    EVENT_NAMES,
    ExitGeometryToken,
    GeometricSemanticObservation,
    StructuralEvent,
)
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
from mtare_topo.semantics.range_geometry_baseline import RangeGeometryBaseline


TURN_CURVATURE_THRESHOLD_PER_M = math.radians(15.0) / 10.0
GEOMETRY_TRANSITION_THRESHOLD_M = 1.0
MAXIMUM_EXIT_TOKENS = 6


def _unit(values: list[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12 or not np.all(np.isfinite(array)):
        raise RuntimeError("non-learning descriptor is degenerate")
    return tuple(float(value) for value in array / norm)


def nonlearning_geometry_observation(
    range_sequence_m: np.ndarray,
    valid_sequence: np.ndarray,
    *,
    geometry_baseline: RangeGeometryBaseline | None = None,
    exit_baseline: RangeExitBaseline | None = None,
) -> GeometricSemanticObservation:
    """Map five causal scans to the same typed interface without learned weights."""

    ranges = np.asarray(range_sequence_m, dtype=np.float64)
    valid = np.asarray(valid_sequence, dtype=np.bool_)
    if ranges.shape != (5, 16, 720) or valid.shape != ranges.shape:
        raise ValueError("non-learning GSE baseline requires five [16,720] scans")
    geometry_method = geometry_baseline or RangeGeometryBaseline()
    exit_method = exit_baseline or RangeExitBaseline()
    geometry = [geometry_method.predict(ranges[index], valid[index]) for index in range(5)]
    current = geometry[-1]
    exits = exit_method.predict(ranges[-1], valid[-1], ELEVATION_DEG)
    count = int(exits["branch_count"])
    width_before = float(np.median([record["width_m"] for record in geometry[:2]]))
    width_after = float(np.median([record["width_m"] for record in geometry[-2:]]))
    height_before = float(np.median([record["height_m"] for record in geometry[:2]]))
    height_after = float(np.median([record["height_m"] for record in geometry[-2:]]))
    if count >= 3:
        event = StructuralEvent.JUNCTION
    elif count <= 1:
        event = StructuralEvent.TERMINAL
    elif (
        abs(width_after - width_before) >= GEOMETRY_TRANSITION_THRESHOLD_M
        or abs(height_after - height_before) >= GEOMETRY_TRANSITION_THRESHOLD_M
    ):
        event = StructuralEvent.GEOMETRY_TRANSITION
    elif float(current["curvature_per_m"]) >= TURN_CURVATURE_THRESHOLD_PER_M:
        event = StructuralEvent.TURN
    else:
        event = StructuralEvent.CORRIDOR

    selected_sectors = sorted(
        exits["sectors"],
        key=lambda sector: (-float(sector["peak_range_m"]), float(sector["heading_robot_deg"])),
    )[:MAXIMUM_EXIT_TOKENS]
    height_profile = [float(record["height_m"] - current["height_m"]) for record in geometry[1:]]
    tokens = []
    for sector in selected_sectors:
        heading = float(sector["heading_robot_deg"])
        angular_width = math.radians(float(sector["angular_width_deg"]))
        opening_width = max(1e-4, 2.0 * float(sector["peak_range_m"]) * math.sin(angular_width / 2.0))
        radians = math.radians(heading)
        token_descriptor = _unit(
            [math.sin(radians), math.cos(radians), math.log1p(opening_width), count / MAXIMUM_EXIT_TOKENS]
        )
        tokens.append(
            ExitGeometryToken(
                heading_robot_deg=heading,
                opening_width_m=opening_width,
                vertical_profile=tuple(height_profile),
                descriptor=token_descriptor,
                confidence=1.0,
            )
        )
    tokens.sort(key=lambda token: token.heading_robot_deg)
    event_encoding = [1.0 if name == event.value else 0.0 for name in EVENT_NAMES]
    exit_encoding = []
    for token in tokens:
        radians = math.radians(token.heading_robot_deg)
        exit_encoding.extend((math.sin(radians), math.cos(radians)))
    exit_encoding.extend((0.0,) * (2 * (MAXIMUM_EXIT_TOKENS - len(tokens))))
    place_descriptor = _unit(
        event_encoding
        + list(current["local_axis"])
        + [
            float(current["width_m"]) / 30.0,
            float(current["height_m"]) / 30.0,
            float(current["slope_deg"]) / 45.0,
            float(current["curvature_per_m"]) / 0.1,
        ]
        + exit_encoding
    )
    probabilities = {name: 1.0 if name == event.value else 0.0 for name in EVENT_NAMES}
    return GeometricSemanticObservation(
        event_probabilities=probabilities,
        local_axis=tuple(float(value) for value in current["local_axis"]),
        width_m=float(current["width_m"]),
        height_m=float(current["height_m"]),
        slope_deg=float(current["slope_deg"]),
        curvature_per_m=float(current["curvature_per_m"]),
        place_descriptor=place_descriptor,
        exit_tokens=tuple(tokens),
        uncertainty=0.0,
    )


__all__ = [
    "GEOMETRY_TRANSITION_THRESHOLD_M",
    "MAXIMUM_EXIT_TOKENS",
    "TURN_CURVATURE_THRESHOLD_PER_M",
    "nonlearning_geometry_observation",
]
