"""Typed deployment contract for Geometry-Semantic Event observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


class StructuralEvent(str, Enum):
    CORRIDOR = "corridor"
    JUNCTION = "junction"
    TERMINAL = "terminal"
    TURN = "turn"
    GEOMETRY_TRANSITION = "geometry_transition"


EVENT_NAMES = tuple(event.value for event in StructuralEvent)


def _finite_tuple(values: Sequence[float], *, name: str, length: int | None = None) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class ExitGeometryToken:
    """One currently observed outgoing opening in the robot frame."""

    heading_robot_deg: float
    opening_width_m: float
    vertical_profile: tuple[float, ...]
    descriptor: tuple[float, ...]
    confidence: float

    def __post_init__(self) -> None:
        heading = float(self.heading_robot_deg)
        width = float(self.opening_width_m)
        confidence = float(self.confidence)
        if not math.isfinite(heading):
            raise ValueError("heading_robot_deg must be finite")
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("opening_width_m must be positive and finite")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        object.__setattr__(self, "heading_robot_deg", heading % 360.0)
        object.__setattr__(self, "opening_width_m", width)
        object.__setattr__(self, "vertical_profile", _finite_tuple(self.vertical_profile, name="vertical_profile"))
        object.__setattr__(self, "descriptor", _finite_tuple(self.descriptor, name="descriptor"))
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeometricSemanticObservation:
    """The only learned structure interface consumed by GSE graph updates."""

    event_probabilities: Mapping[str, float]
    local_axis: tuple[float, float, float]
    width_m: float
    height_m: float
    slope_deg: float
    curvature_per_m: float
    place_descriptor: tuple[float, ...]
    exit_tokens: tuple[ExitGeometryToken, ...]
    uncertainty: float

    def __post_init__(self) -> None:
        probabilities = {str(key): float(value) for key, value in self.event_probabilities.items()}
        if tuple(sorted(probabilities)) != tuple(sorted(EVENT_NAMES)):
            raise ValueError(f"event_probabilities must contain exactly {EVENT_NAMES}")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities.values()):
            raise ValueError("event probabilities must be finite and in [0,1]")
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("event probabilities must sum to one")
        axis = _finite_tuple(self.local_axis, name="local_axis", length=3)
        axis_norm = math.sqrt(sum(value * value for value in axis))
        if axis_norm <= 1e-8:
            raise ValueError("local_axis must be non-zero")
        width = float(self.width_m)
        height = float(self.height_m)
        slope = float(self.slope_deg)
        curvature = float(self.curvature_per_m)
        uncertainty = float(self.uncertainty)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("width_m must be positive and finite")
        if not math.isfinite(height) or height <= 0.0:
            raise ValueError("height_m must be positive and finite")
        if not math.isfinite(slope):
            raise ValueError("slope_deg must be finite")
        if not math.isfinite(curvature) or curvature < 0.0:
            raise ValueError("curvature_per_m must be non-negative and finite")
        if not math.isfinite(uncertainty) or not 0.0 <= uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0,1]")
        descriptor = _finite_tuple(self.place_descriptor, name="place_descriptor")
        tokens = tuple(self.exit_tokens)
        if not all(isinstance(token, ExitGeometryToken) for token in tokens):
            raise TypeError("exit_tokens must contain ExitGeometryToken values")
        object.__setattr__(self, "event_probabilities", probabilities)
        object.__setattr__(self, "local_axis", tuple(value / axis_norm for value in axis))
        object.__setattr__(self, "width_m", width)
        object.__setattr__(self, "height_m", height)
        object.__setattr__(self, "slope_deg", slope)
        object.__setattr__(self, "curvature_per_m", curvature)
        object.__setattr__(self, "place_descriptor", descriptor)
        object.__setattr__(self, "exit_tokens", tokens)
        object.__setattr__(self, "uncertainty", uncertainty)

    @property
    def event(self) -> StructuralEvent:
        return StructuralEvent(max(EVENT_NAMES, key=lambda name: self.event_probabilities[name]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_probabilities": dict(self.event_probabilities),
            "local_axis": list(self.local_axis),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "slope_deg": self.slope_deg,
            "curvature_per_m": self.curvature_per_m,
            "place_descriptor": list(self.place_descriptor),
            "exit_tokens": [token.to_dict() for token in self.exit_tokens],
            "uncertainty": self.uncertainty,
        }


__all__ = [
    "EVENT_NAMES",
    "ExitGeometryToken",
    "GeometricSemanticObservation",
    "StructuralEvent",
]
