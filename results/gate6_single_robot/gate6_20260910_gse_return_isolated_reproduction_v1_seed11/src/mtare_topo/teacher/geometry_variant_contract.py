"""Deterministic geometry-realization contract for P1 data generation.

Each topology parent receives three shape realizations.  Parameters are drawn
from a hash of the parent and edge identities, never from sensor outcomes.
All cross-sections preserve the area of the archived radius circle so the
variants change shape rather than systematically changing tunnel scale.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy.special import gamma

from mtare_topo.teacher.primitive_construction_supervisor import PrimitiveConstructionGraph, SweptPrimitive
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive


class GeometryRealization(str, Enum):
    ELLIPSE = "ellipse"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    C1_MIXED = "c1_mixed"


@dataclass(frozen=True)
class PrimitiveCapGuard:
    primitive_id: str
    start_extension_m: float
    end_extension_m: float


def superellipse_unit_area(exponent: float) -> float:
    if not math.isfinite(exponent) or exponent < 2.0:
        raise ValueError("exponent must be finite and >=2")
    return float(4.0 * gamma(1.0 + 1.0 / exponent) ** 2 / gamma(1.0 + 2.0 / exponent))


def _uniform(identity: str, channel: str) -> float:
    digest = hashlib.sha256(f"{identity}|{channel}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def _area_preserving_axes(radius_m: float, exponent: float, aspect: float) -> tuple[float, float]:
    scale = radius_m * math.sqrt(math.pi / superellipse_unit_area(exponent))
    root = math.sqrt(aspect)
    return scale * root, scale / root


def _endpoint_parameters(parent_id: str, primitive: SweptPrimitive, family: str, endpoint: int):
    identity = f"{parent_id}|{primitive.primitive_id}|{endpoint}"
    if family == "ellipse":
        exponent = 2.0
        aspect = 1.0 + 0.25 * _uniform(identity, "ellipse_aspect")
    elif family == "rounded":
        exponent = 6.0 + 4.0 * _uniform(identity, "rounded_exponent")
        aspect = 0.9 + 0.2 * _uniform(identity, "rounded_aspect")
    else:
        raise ValueError(f"unknown family {family!r}")
    return _area_preserving_axes(primitive.radius_m, exponent, aspect), exponent


def realize_primitive(parent_id: str, primitive: SweptPrimitive, realization: GeometryRealization) -> SweptSuperellipsePrimitive:
    if not parent_id:
        raise ValueError("parent_id must be nonempty")
    if realization is GeometryRealization.ELLIPSE:
        first = _endpoint_parameters(parent_id, primitive, "ellipse", 0)
        second = _endpoint_parameters(parent_id, primitive, "ellipse", 1)
    elif realization is GeometryRealization.ROUNDED_RECTANGLE:
        first = _endpoint_parameters(parent_id, primitive, "rounded", 0)
        second = _endpoint_parameters(parent_id, primitive, "rounded", 1)
    elif realization is GeometryRealization.C1_MIXED:
        flip = _uniform(f"{parent_id}|{primitive.primitive_id}", "mixed_direction") >= 0.5
        families = ("rounded", "ellipse") if flip else ("ellipse", "rounded")
        first = _endpoint_parameters(parent_id, primitive, families[0], 0)
        second = _endpoint_parameters(parent_id, primitive, families[1], 1)
    else:
        raise ValueError(f"unsupported realization {realization!r}")
    return SweptSuperellipsePrimitive(
        primitive_id=primitive.primitive_id,
        centerline_xyz_m=primitive.centerline_xyz_m,
        endpoint_half_axes_m=(first[0], second[0]),
        endpoint_shape_exponent=(first[1], second[1]),
    )


def realize_construction(parent_id: str, construction: PrimitiveConstructionGraph, realization: GeometryRealization) -> tuple[SweptSuperellipsePrimitive, ...]:
    return tuple(realize_primitive(parent_id, value, realization) for value in construction.primitives)


def guard_vertical_sensor_offset_at_caps(
    primitives: tuple[SweptSuperellipsePrimitive, ...],
    *,
    vertical_sensor_offset_m: float,
    discretization_guard_m: float = 0.025,
) -> tuple[tuple[SweptSuperellipsePrimitive, ...], tuple[PrimitiveCapGuard, ...]]:
    """Extend only caps where a world-vertical sensor offset exits a sloped sweep.

    The rule depends only on the frozen sensor offset and endpoint tangents. It
    keeps topology, primitive identity and every cross-section parameter fixed.
    A one-field-cell guard is added only when a positive physical extension is
    required; flat caps are not changed.
    """

    offset = float(vertical_sensor_offset_m); guard = float(discretization_guard_m)
    if not math.isfinite(offset) or not math.isfinite(guard) or guard <= 0:
        raise ValueError("sensor offset must be finite and discretization guard positive")
    realized = []; records = []
    for primitive in primitives:
        points = primitive.centerline_xyz_m.copy()
        start_tangent = points[1] - points[0]; start_tangent /= np.linalg.norm(start_tangent)
        end_tangent = points[-1] - points[-2]; end_tangent /= np.linalg.norm(end_tangent)
        start_required = max(0.0, -offset * float(start_tangent[2]))
        end_required = max(0.0, offset * float(end_tangent[2]))
        start_extension = start_required + guard if start_required > 1e-12 else 0.0
        end_extension = end_required + guard if end_required > 1e-12 else 0.0
        if start_extension:
            points[0] -= start_extension * start_tangent
        if end_extension:
            points[-1] += end_extension * end_tangent
        realized.append(SweptSuperellipsePrimitive(
            primitive_id=primitive.primitive_id,
            centerline_xyz_m=points,
            endpoint_half_axes_m=primitive.endpoint_half_axes_m,
            endpoint_shape_exponent=primitive.endpoint_shape_exponent,
        ))
        records.append(PrimitiveCapGuard(primitive.primitive_id, start_extension, end_extension))
    return tuple(realized), tuple(records)


def cross_section_area(half_axes_m: tuple[float, float], exponent: float) -> float:
    return superellipse_unit_area(exponent) * float(half_axes_m[0]) * float(half_axes_m[1])


__all__ = ["GeometryRealization", "PrimitiveCapGuard", "cross_section_area", "guard_vertical_sensor_offset_at_caps", "realize_construction", "realize_primitive", "superellipse_unit_area"]
