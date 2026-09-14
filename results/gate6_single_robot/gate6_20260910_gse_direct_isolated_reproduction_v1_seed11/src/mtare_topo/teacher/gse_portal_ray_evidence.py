"""Positive-only portal evidence from finite causal first-return segments.

Training/diagnostic geometry only: no map identity reaches model inference.
An aperture witness is a ray crossing the strict interior of a bounded portal
before its first return. It is NOT a swept-robot clearance certificate, proof
of all exits, a node event label, or permission to create a graph edge.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalRaySegments:
    origins_m: np.ndarray                 # R,3 in one common frame
    directions: np.ndarray                # R,3 unit vectors
    first_return_m: np.ndarray            # R; preserve source float dtype/ULP
    valid: np.ndarray                     # R bool; no-return is not free evidence
    source_frame_index: np.ndarray        # R int, order not necessarily sorted
    current_frame_index: int
    range_error_bound_m: float            # additional measurement/caster error

    def validate(self):
        count = len(self.origins_m)
        if (np.shape(self.origins_m) != (count, 3)
                or np.shape(self.directions) != (count, 3)
                or np.shape(self.first_return_m) != (count,)
                or self.first_return_m.dtype.kind != "f"
                or np.shape(self.valid) != (count,)
                or self.valid.dtype != np.bool_
                or np.shape(self.source_frame_index) != (count,)
                or self.source_frame_index.dtype.kind not in "iu"):
            raise ValueError("invalid ray shape/dtype")
        if (isinstance(self.current_frame_index, bool)
                or not isinstance(self.current_frame_index, (int, np.integer))
                or self.current_frame_index < 0
                or np.any(self.source_frame_index < 0)
                or np.any(self.source_frame_index > self.current_frame_index)):
            raise ValueError("future/invalid frame evidence")
        if (not np.isfinite(self.range_error_bound_m) or self.range_error_bound_m < 0
                or not np.isfinite(self.origins_m).all()
                or not np.isfinite(self.directions).all()):
            raise ValueError("invalid finite ray geometry/error bound")
        norm = np.linalg.norm(self.directions, axis=-1)
        if np.any(np.abs(norm[self.valid] - 1) > 1e-10):
            raise ValueError("valid ray directions must be unit length")
        ranges = self.first_return_m[self.valid]
        if np.any(~np.isfinite(ranges)) or np.any(ranges <= 0):
            raise ValueError("valid first return must be finite and positive")


@dataclass(frozen=True)
class PortalSections:
    centers_m: np.ndarray                 # P,3
    inward_normals: np.ndarray            # P,3; into the associated passage
    width_directions: np.ndarray          # P,3 unit, normal-orthogonal
    half_axes_m: np.ndarray               # P,2
    exponents: np.ndarray                 # P; superellipse |u/a|^p+|v/b|^p<1
    polygon_normalized_xy: np.ndarray | None = None  # P,K,2 exact mesh ring, CCW

    def validate(self):
        count = len(self.centers_m)
        if not 1 <= count <= 64:
            raise ValueError("portal population must be 1..64")
        for name, tail in (("centers_m", (3,)), ("inward_normals", (3,)),
                           ("width_directions", (3,)), ("half_axes_m", (2,)),
                           ("exponents", ())):
            value = getattr(self, name)
            if np.shape(value) != (count, *tail) or not np.isfinite(value).all():
                raise ValueError(f"invalid portal {name}")
        if np.any(self.half_axes_m <= 0) or np.any(self.exponents < 1):
            raise ValueError("portal section must be positive and convex")
        n, u = self.inward_normals, self.width_directions
        if (np.any(np.abs(np.linalg.norm(n, axis=-1) - 1) > 1e-10)
                or np.any(np.abs(np.linalg.norm(u, axis=-1) - 1) > 1e-10)
                or np.any(np.abs(np.sum(n * u, axis=-1)) > 1e-10)):
            raise ValueError("portal normal/width must be orthonormal")
        polygon = self.polygon_normalized_xy
        if polygon is not None:
            if (polygon.ndim != 3 or polygon.shape[0] != count or polygon.shape[-1] != 2
                    or not 3 <= polygon.shape[1] <= 512 or not np.isfinite(polygon).all()):
                raise ValueError("invalid portal polygon")
            edge = np.roll(polygon, -1, axis=1) - polygon
            following = np.roll(edge, -1, axis=1)
            turn = edge[..., 0] * following[..., 1] - edge[..., 1] * following[..., 0]
            if np.any(turn <= 0):
                raise ValueError("portal polygon must be strictly convex and CCW")


@dataclass(frozen=True)
class PortalRayEvidence:
    crossing_ray_count: np.ndarray        # P, all candidate free crossings
    exclusive_ray_count: np.ndarray       # P, no other supplied portal crossed
    witnessed: np.ndarray                # P bool; false means UNKNOWN
    witness_ray_index: np.ndarray         # P, first exclusive witness or -1
    ambiguous_ray_count: int
    supplied_portal_count: int
    inward_ray_count: np.ndarray          # P, candidate crossings into passage
    outward_ray_count: np.ndarray         # P, candidate crossings out of passage


@dataclass(frozen=True)
class ConstructionCapCandidates:
    sections: PortalSections
    teacher_keys: tuple[tuple[str, int], ...]  # provenance/scoring, NEVER model inputs


def construction_cap_candidates(primitives, *, mesh_axial_spacing_m: float,
                                angular_segments: int) -> ConstructionCapCandidates:
    """Use ALL supplied visible operands, not target-node incident selection.

    Reuse the actual source mesh's sampled cap geometry, including any frozen
    cap extension. These are physical cap sections, NOT automatically node
    portals: ray evidence must still establish an opening. A source adapter
    must bind the exact visible-operand inventory and mesh sampling contract.
    No file reads, target-node identity, or new candidate filtering occurs here.
    """
    from mtare_topo.teacher.swept_superellipse_field import _sample_operand
    primitives = tuple(primitives)
    if not 1 <= len(primitives) <= 32:
        raise ValueError("all supplied visible operands must fit 1..32 slots")
    if not np.isfinite(mesh_axial_spacing_m) or mesh_axial_spacing_m <= 0:
        raise ValueError("source mesh spacing must be finite and positive")
    if type(angular_segments) is not int or not 12 <= angular_segments <= 512 or angular_segments % 4:
        raise ValueError("source angular segments must match a bounded mesh ring")
    ids = [value.primitive_id for value in primitives]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate primitive inventory")
    centers, normals, widths, axes, exponents, keys, polygons = [], [], [], [], [], [], []
    theta = np.arange(angular_segments) * (2 * np.pi / angular_segments)
    cosine, sine = np.cos(theta), np.sin(theta)
    for primitive in primitives:
        operand = _sample_operand(primitive, mesh_axial_spacing_m)
        for endpoint, index in ((0, 0), (1, -1)):
            centers.append(operand.points[index])
            normals.append(operand.tangents[index] * (1 if endpoint == 0 else -1))
            widths.append(operand.lateral[index])
            axes.append(operand.half_axes[index])
            exponents.append(operand.exponent[index])
            keys.append((primitive.primitive_id, endpoint))
            exponent = operand.exponent[index]
            polygon = np.stack((np.sign(cosine) * np.abs(cosine) ** (2 / exponent),
                                np.sign(sine) * np.abs(sine) ** (2 / exponent)), axis=-1)
            if endpoint == 1:  # inward normal flips the sampled vertical basis
                polygon[:, 1] *= -1
                polygon = polygon[::-1]
            polygons.append(polygon)
    sections = PortalSections(*(np.asarray(v, dtype=np.float64)
                                for v in (centers, normals, widths, axes, exponents, polygons)))
    sections.validate()
    return ConstructionCapCandidates(sections, tuple(keys))


def aperture_ray_evidence(rays: CausalRaySegments, portals: PortalSections,
                          *, chunk_size: int = 2048) -> PortalRayEvidence:
    """Conservative positive ray evidence; never derive negatives from absence.

    Competition is over EVERY supplied portal, not a GT-incident subset. A ray
    crossing several portals cannot independently witness any of them. This
    avoids double-counting indistinguishable exits, but is deliberately not a
    complete visibility algorithm. The adapter must disclose its candidate
    population. No extrapolation past first returns, no interpolation between
    rays, and no tolerance widening of the aperture are performed.

    Both crossing directions count: observing a connection does not depend on
    which side the sensor occupies. Geometry uses float64. A fixed roundoff
    guard rejects grazing/on-plane starts, and a full storage ULP is subtracted
    from ranges in addition to the supplied measurement/caster error. The
    adapter must NOT upcast stored float32 ranges before this computation.
    """
    rays.validate()
    portals.validate()
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 2048:
        raise ValueError("invalid bounded chunk size")
    p = len(portals.centers_m)
    counts = np.zeros(p, dtype=np.int64)
    exclusive_counts = np.zeros(p, dtype=np.int64)
    first = np.full(p, -1, dtype=np.int64)
    inward_counts = np.zeros(p, dtype=np.int64)
    outward_counts = np.zeros(p, dtype=np.int64)
    ambiguous = 0
    normal = np.asarray(portals.inward_normals, dtype=np.float64)
    width = np.asarray(portals.width_directions, dtype=np.float64)
    height = np.cross(normal, width)
    center = np.asarray(portals.centers_m, dtype=np.float64)
    eps = 128 * np.finfo(np.float64).eps
    for start in range(0, len(rays.origins_m), chunk_size):
        stop = min(start + chunk_size, len(rays.origins_m))
        origin = np.asarray(rays.origins_m[start:stop], dtype=np.float64)
        direction = np.asarray(rays.directions[start:stop], dtype=np.float64)
        offset = center[None] - origin[:, None]
        numerator = np.einsum("rpc,pc->rp", offset, normal)
        denominator = direction @ normal.T
        scale = np.maximum(1., np.linalg.norm(offset, axis=-1))
        distance = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                             where=np.abs(denominator) > eps)
        # Boundary contacts are deliberately unknown, including exactly-on-
        # plane starts: no numerical tolerance may extrapolate backwards.
        eligible = ((np.abs(denominator) > eps) & (np.abs(numerator) > eps * scale)
                    & (distance > 0))
        point = origin[:, None] + distance[..., None] * direction[:, None]
        local = point - center[None]
        u = np.einsum("rpc,pc->rp", local, width) / portals.half_axes_m[:, 0]
        v = np.einsum("rpc,pc->rp", local, height) / portals.half_axes_m[:, 1]
        with np.errstate(over="ignore", invalid="ignore"):
            section_value = np.abs(u) ** portals.exponents + np.abs(v) ** portals.exponents
        stored_range = rays.first_return_m[start:stop]
        quantization_guard = np.abs(np.spacing(stored_range)).astype(np.float64)
        observed_limit = (stored_range.astype(np.float64) - quantization_guard
                          - rays.range_error_bound_m)[:, None]
        crossed = (eligible & (section_value < 1 - eps)
                   & (distance + eps * scale < observed_limit)
                   & rays.valid[start:stop, None])
        if portals.polygon_normalized_xy is not None:
            # Check only potential intersections. Never replace the source
            # polygon by its larger ideal superellipse near the mesh boundary.
            for port in np.flatnonzero(crossed.any(axis=0)):
                selected = np.flatnonzero(crossed[:, port])
                xy = np.stack((u[selected, port], v[selected, port]), axis=-1)
                polygon = portals.polygon_normalized_xy[port]
                edges = np.roll(polygon, -1, axis=0) - polygon
                delta = xy[:, None] - polygon[None]
                halfspace = edges[None, :, 0] * delta[..., 1] - edges[None, :, 1] * delta[..., 0]
                crossed[selected, port] &= (halfspace > eps).all(axis=1)
        multiplicity = crossed.sum(axis=1)
        exclusive = crossed & (multiplicity == 1)[:, None]
        counts += crossed.sum(axis=0)
        inward_counts += (crossed & (denominator > 0)).sum(axis=0)
        outward_counts += (crossed & (denominator < 0)).sum(axis=0)
        exclusive_counts += exclusive.sum(axis=0)
        ambiguous += int(np.sum(multiplicity > 1))
        for port in np.flatnonzero((first < 0) & exclusive.any(axis=0)):
            first[port] = start + int(np.flatnonzero(exclusive[:, port])[0])
    return PortalRayEvidence(counts, exclusive_counts, exclusive_counts > 0,
                             first, ambiguous, p, inward_counts, outward_counts)
