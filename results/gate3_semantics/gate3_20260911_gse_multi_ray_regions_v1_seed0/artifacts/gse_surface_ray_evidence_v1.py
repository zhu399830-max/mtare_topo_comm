"""Five-frame, first-return voxel evidence; NEVER physical traversability.

The fixed cube is [-10,10)^3, resolution .25 m, 80^3 cells. A FREE cell means
one observed ray crossed its interior, not that the whole cell/body is clear.
First-return cells are OCCUPIED, with precedence over free observations.
No-return/invalid rays contribute nothing. Outside-cube returns may contribute
interior free segments, but are never clipped into artificial surface hits.

Traversal splits a finite segment at every axis-aligned grid plane (at most
243 split parameters); this is equivalent to positive-length voxel DDA, not
fixed-distance ray marching. Face/edge/corner ambiguity never creates FREE.
Input ULP plus arithmetic bounds are used only to withhold numerical evidence,
not as a tuned physical sensor-noise/clearance threshold. Registration and the
actual frame identities are authenticated by the caller; 0..4 are frame slots.
"""
from dataclasses import dataclass
import hashlib
import itertools
import math

import numpy as np


RESOLUTION_M = .25
HALF_EXTENT_M = 10.
SHAPE = (80, 80, 80)
UNKNOWN, FREE, OCCUPIED = 0, 1, 2
EVIDENCE_ORDER = ("observed_free", "observed_occupied", "unknown")
_PLANES = np.linspace(-10., 10., 81, dtype=np.longdouble)


def _readonly(array):
    array.setflags(write=False)
    return array


def _xyz(value, name):
    if (type(value) is not np.ndarray or value.ndim != 2 or value.shape[1] != 3
            or value.dtype not in (np.dtype("float32"), np.dtype("float64"))):
        raise ValueError(f"{name} must be N,3 float32/float64")


def _numerical_bound(*arrays):
    bound = 0.
    for a in arrays:
        if a.size:
            scale = np.asarray(max(10., float(np.max(np.abs(a)))), dtype=a.dtype)
            with np.errstate(over="ignore"):
                ulp = float(np.spacing(scale))
            bound = max(bound, 8 * ulp, 64 * np.finfo(np.float64).eps * float(scale))
    if not math.isfinite(bound):
        raise ValueError("input magnitude prevents a finite numerical bound")
    return float(bound)


def _point_cells(point, tolerance):
    """Cells incident to a point, including numerical boundary alternatives."""
    result = []
    for x in point:
        if x < -10. - tolerance or x > 10. + tolerance:
            return ()
        position = (x + 10.) / RESOLUTION_M
        nearest = int(np.rint(position))
        if abs(x - (-10. + RESOLUTION_M * nearest)) <= tolerance:
            candidates = (nearest - 1, nearest)
        else:
            candidates = (int(np.floor(position)),)
        options = tuple(i for i in candidates if 0 <= i < 80)
        if not options:
            return ()
        result.append(options)
    return tuple((i * 80 + j) * 80 + k for i, j, k in itertools.product(*result))


def _segment_cells(origin, endpoint, tolerance):
    """Return distinct positive-length touched cells and ambiguous subset.

    A segment exactly on a voxel plane touches adjacent cells but does not
    clear either one. Zero length contributes no free interval. No stochastic
    joggle, fixed step, or endpoint clipping is used.
    """
    p = np.asarray(origin, dtype=np.longdouble)
    q = np.asarray(endpoint, dtype=np.longdouble)
    d = q - p
    if not np.isfinite(d).all():
        raise ValueError("segment arithmetic overflow")
    if not np.any(d):
        return set(), set()
    lower, upper = np.longdouble(0), np.longdouble(1)
    for axis in range(3):
        if d[axis] == 0:
            if p[axis] < -10. or p[axis] > 10.:
                return set(), set()
            continue
        enter, leave = sorted(((-10. - p[axis]) / d[axis], (10. - p[axis]) / d[axis]))
        lower, upper = max(lower, enter), min(upper, leave)
    if lower >= upper:
        return set(), set()
    breaks = [np.asarray([lower, upper], dtype=np.longdouble)]
    for axis in range(3):
        if d[axis] != 0:
            t = (_PLANES - p[axis]) / d[axis]
            breaks.append(t[(t > lower) & (t < upper)])
    times = np.unique(np.concatenate(breaks))
    middle = times[:-1] + (times[1:] - times[:-1]) / 2
    locations = p[None] + middle[:, None] * d[None]
    scaled = (locations + 10.) / RESOLUTION_M
    boundary = np.any(np.abs(locations - (-10. + RESOLUTION_M * np.rint(scaled))) <= tolerance, axis=1)
    index = np.floor(scaled[~boundary]).astype(np.int64)
    index = index[np.all((index >= 0) & (index < 80), axis=1)]
    cells = set(((index[:, 0] * 80 + index[:, 1]) * 80 + index[:, 2]).tolist())
    ambiguous = set()
    # Only numerical boundary cases need the conservative up-to-eight-cell
    # branch. All ordinary intervals are classified in one vector operation.
    for location in locations[boundary]:
        ambiguous.update(_point_cells(location, tolerance))
    cells.update(ambiguous)
    return cells, ambiguous


def _digest(*arrays):
    h = hashlib.sha256(b"gse_surface_ray_evidence_v1;cube[-10,10);step.25;slots5\0")
    for array in arrays:
        h.update(str(array.shape).encode("ascii")); h.update(array.dtype.str.encode("ascii"))
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ObservedRayGrid:
    state: np.ndarray                  # uint8[80,80,80]: unknown/free/occupied
    free_frame_bits: np.ndarray        # uint8; bit0..4, not independent samples
    occupied_frame_bits: np.ndarray
    source_geometry_sha256: str        # ordered numerical source, not world identity
    content_sha256: str
    numerical_bound_m: float
    first_return_count: int
    ignored_ray_count: int
    ambiguous_ray_count: int
    physical_connectivity: bool = False


def build_surface_ray_grid(origins_m, endpoints_m, first_return_valid, frame_index):
    """One already registered five-frame observation, up to 57,600 rays.

    Invalid endpoints/origins may be NaN and are ignored, never inferred as
    max-range/free rays. All frame slots must still be explicit integers0..4.
    Synthetic subsets/empty frames are legal; no missing frame is fabricated.
    """
    _xyz(origins_m, "origins"); _xyz(endpoints_m, "endpoints")
    n = len(origins_m)
    if origins_m.shape != endpoints_m.shape or n > 57600:
        raise ValueError("aligned rays with maximum57600 required")
    if (type(first_return_valid) is not np.ndarray or first_return_valid.shape != (n,)
            or first_return_valid.dtype != np.bool_ or type(frame_index) is not np.ndarray
            or frame_index.shape != (n,) or frame_index.dtype.kind not in "iu"
            or np.any(frame_index < 0) or np.any(frame_index > 4)):
        raise ValueError("explicit N bool first-return validity and integer frame slots0..4 required")
    valid = first_return_valid
    if not np.isfinite(origins_m[valid]).all() or not np.isfinite(endpoints_m[valid]).all():
        raise ValueError("valid rays require finite registered origins and first-return endpoints")
    tolerance = _numerical_bound(origins_m[valid], endpoints_m[valid])
    free = np.zeros(np.prod(SHAPE), dtype=np.uint8)
    occupied = np.zeros_like(free)
    ambiguous_count = 0
    for origin, endpoint, frame in zip(origins_m[valid], endpoints_m[valid], frame_index[valid]):
        # Exact outside points never become cropped/quantization-created hits.
        hit = _point_cells(endpoint, tolerance) if np.all(endpoint >= -10.) and np.all(endpoint < 10.) else ()
        cells, ambiguous = _segment_cells(origin, endpoint, tolerance)
        ambiguous_count += bool(ambiguous)
        clear = cells.difference(ambiguous).difference(hit)
        bit = np.uint8(1 << int(frame))
        if clear:
            free[np.fromiter(clear, dtype=np.int64)] |= bit
        if hit:
            occupied[np.asarray(hit, dtype=np.int64)] |= bit
    free = free.reshape(SHAPE); occupied = occupied.reshape(SHAPE)
    state = np.where(occupied != 0, OCCUPIED, np.where(free != 0, FREE, UNKNOWN)).astype(np.uint8)
    # Invalid payload bytes cannot create evidence, but validity/order are bound.
    canonical_origins = np.where(valid[:, None], origins_m, 0.).astype("<f8")
    canonical_endpoints = np.where(valid[:, None], endpoints_m, 0.).astype("<f8")
    source = _digest(canonical_origins, canonical_endpoints, valid, frame_index.astype("<i8"),
                     np.asarray([origins_m.dtype.str, endpoints_m.dtype.str], dtype="S3"))
    content = _digest(state, free, occupied, np.asarray([tolerance], dtype="<f8"))
    return ObservedRayGrid(_readonly(state), _readonly(free), _readonly(occupied), source, content,
                           tolerance, int(valid.sum()), int((~valid).sum()), int(ambiguous_count))


@dataclass(frozen=True)
class GapEvidence:
    counts: np.ndarray                 # M,8,3; observed free / occupied / unknown
    fractions: np.ndarray              # measured cell fractions, NOT probabilities
    valid: np.ndarray                 # message-neighbor mask, NOT connectivity
    gap_defined: np.ndarray            # some non-endpoint in-cube interval cells
    endpoint_cell_count: np.ndarray
    ambiguous_cell_count: np.ndarray
    content_sha256: str
    physical_connectivity: bool = False


def query_patch_gaps(grid, centers_m, neighbor_index, neighbor_valid):
    """Finite center-to-center gap evidence, excluding both endpoint cell sets.

    Excluded endpoint cells normally contain the patches' surfaces; their hits
    must not label the intervening gap blocked. Any *interior* occupied cell is
    counted occupied. This does NOT infer an opening from its center line, a
    robot swept volume, ground support or root reachability. Empty/undefined
    gaps have zero fractions AND gap_defined=False, never 'all free'.
    """
    if type(grid) is not ObservedRayGrid:
        raise ValueError("typed observed ray grid required")
    if (type(grid.numerical_bound_m) is not float or not math.isfinite(grid.numerical_bound_m)
            or grid.numerical_bound_m < 0 or grid.physical_connectivity is not False):
        raise ValueError("finite numerical bound and observation-only scope required")
    for a in (grid.state, grid.free_frame_bits, grid.occupied_frame_bits):
        if type(a) is not np.ndarray or a.shape != SHAPE or a.dtype != np.uint8:
            raise ValueError("fixed80cubed uint8 evidence required")
    expected = np.where(grid.occupied_frame_bits != 0, OCCUPIED,
                        np.where(grid.free_frame_bits != 0, FREE, UNKNOWN)).astype(np.uint8)
    if (np.any(grid.free_frame_bits > 31) or np.any(grid.occupied_frame_bits > 31)
            or not np.array_equal(expected, grid.state)
            or _digest(grid.state, grid.free_frame_bits, grid.occupied_frame_bits,
                       np.asarray([grid.numerical_bound_m], dtype="<f8")) != grid.content_sha256):
        raise ValueError("grid state/support bits/content seal inconsistent")
    _xyz(centers_m, "patch centers")
    m = len(centers_m)
    if m > 4096 or not np.isfinite(centers_m).all() or np.any(np.abs(centers_m) > 10.):
        raise ValueError("finite patch centers inside closed10m cube and capacity4096 required")
    if (type(neighbor_index) is not np.ndarray or neighbor_index.shape != (m, 8)
            or neighbor_index.dtype.kind not in "iu" or type(neighbor_valid) is not np.ndarray
            or neighbor_valid.shape != (m, 8) or neighbor_valid.dtype != np.bool_
            or np.any(neighbor_index < -1) or np.any(neighbor_index >= m)
            or not np.array_equal(neighbor_valid, neighbor_index >= 0)):
        raise ValueError("aligned M,8 integer neighbors/-1 and bool validity required")
    counts = np.zeros((m, 8, 3), dtype=np.int64)
    endpoints = np.zeros((m, 8), dtype=np.int64); ambiguous_counts = np.zeros_like(endpoints)
    tolerance = max(grid.numerical_bound_m, _numerical_bound(centers_m))
    flat = grid.state.reshape(-1)
    cache = {}
    for i in range(m):
        chosen = neighbor_index[i, neighbor_valid[i]]
        if len(np.unique(chosen)) != len(chosen) or i in chosen:
            raise ValueError("self/duplicate neighbors do not define distinct patch gaps")
        for slot in np.flatnonzero(neighbor_valid[i]):
            j = int(neighbor_index[i, slot]); key = (min(i, j), max(i, j))
            if key not in cache:
                cells, ambiguous = _segment_cells(centers_m[i], centers_m[j], tolerance)
                end_cells = set(_point_cells(centers_m[i], tolerance)) | set(_point_cells(centers_m[j], tolerance))
                gap = cells.difference(end_cells)
                summary = [0, 0, 0]
                for cell in gap:
                    state = flat[cell]
                    if state == OCCUPIED:
                        summary[1] += 1
                    elif state == FREE and cell not in ambiguous:
                        summary[0] += 1
                    else:
                        summary[2] += 1
                cache[key] = (summary, len(cells & end_cells), len(gap & ambiguous))
            counts[i, slot], endpoints[i, slot], ambiguous_counts[i, slot] = cache[key]
    total = counts.sum(-1)
    fractions = counts / np.maximum(total, 1)[..., None]
    return GapEvidence(_readonly(counts), _readonly(fractions), _readonly(neighbor_valid.copy()),
                       _readonly(total > 0), _readonly(endpoints), _readonly(ambiguous_counts), grid.content_sha256)
