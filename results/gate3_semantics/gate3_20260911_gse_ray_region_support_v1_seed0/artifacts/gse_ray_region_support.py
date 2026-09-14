"""Bind authenticated observed rays to regions, not to structural identities.

The full original observation is required to authenticate ray/frame indices.
Return surfaces stay surfaces; crop endpoints are never manufactured hits.
Common component support does NOT assign an opening to a junction.
"""
from dataclasses import dataclass
import numpy as np

from .gse_surface_ray_evidence_v1 import (
    FREE, OCCUPIED, SHAPE, _digest, _numerical_bound, _point_cells,
    _segment_cells, _xyz,
)


@dataclass(frozen=True)
class RayRegionSupport:
    ray_index: int
    frame_slot: int
    free_cells: tuple
    surface_cells: tuple
    components: tuple
    structural_membership: None = None
    training_qualified: bool = False


def bind_ray_support(grid, regions, origins, endpoints, valid, frames):
    """Observation-only provenance; never accepts reference centers/source IDs.

    Region labels must come from the frozen, authenticated grid. This function
    does not regenerate regions or fill unknown cells. Caller authenticates
    persisted region arrays against their saved artifact hash before loading.
    """
    _xyz(origins, 'origins'); _xyz(endpoints, 'endpoints')
    n = len(origins)
    if origins.shape != endpoints.shape or n > 57600:
        raise ValueError('aligned full observation required')
    if (type(valid) is not np.ndarray or valid.dtype != np.bool_ or valid.shape != (n,)
            or type(frames) is not np.ndarray or frames.shape != (n,)
            or frames.dtype.kind not in 'iu' or np.any(frames < 0) or np.any(frames > 4)):
        raise ValueError('original validity and frame slots required')
    if not np.isfinite(origins[valid]).all() or not np.isfinite(endpoints[valid]).all():
        raise ValueError('nonfinite valid ray')
    source = _digest(np.where(valid[:, None], origins, 0.).astype('<f8'),
                     np.where(valid[:, None], endpoints, 0.).astype('<f8'), valid,
                     frames.astype('<i8'),
                     np.asarray([origins.dtype.str, endpoints.dtype.str], dtype='S3'))
    if source != grid.source_geometry_sha256:
        raise ValueError('ray identity/order/frame drift')
    if (_digest(grid.state, grid.free_frame_bits, grid.occupied_frame_bits,
                np.asarray([grid.numerical_bound_m], dtype='<f8')) != grid.content_sha256
            or regions.grid_sha256 != grid.content_sha256
            or regions.labels.shape != SHAPE
            or np.any((regions.labels > 0) & (grid.state != FREE))):
        raise ValueError('grid or region binding mismatch')
    tolerance = _numerical_bound(origins[valid], endpoints[valid])
    if tolerance != grid.numerical_bound_m:
        raise ValueError('numerical evidence contract drift')
    records = []
    for index in np.flatnonzero(valid):
        origin, end = origins[index], endpoints[index]
        frame = int(frames[index]); bit = 1 << frame
        hit = (_point_cells(end, tolerance)
               if np.all(end >= -10.) and np.all(end < 10.) else ())
        cells, ambiguous = _segment_cells(origin, end, tolerance)
        free = tuple(sorted(c for c in cells.difference(ambiguous).difference(hit)
                            if regions.labels.flat[c] > 0
                            and int(grid.free_frame_bits.flat[c]) & bit))
        # Only actual in-sphere first returns, not a synthetic clipping surface.
        surface = tuple(sorted(c for c in hit
                               if float(np.dot(end, end)) <= 100.
                               and grid.state.flat[c] == OCCUPIED
                               and int(grid.occupied_frame_bits.flat[c]) & bit))
        components = tuple(sorted({int(regions.labels.flat[c]) for c in free}))
        records.append(RayRegionSupport(int(index), frame, free, surface, components))
    return tuple(records)


def compare_support(first, second):
    """A measured support overlap is not a structural positive or negative."""
    return {
        'common_components': sorted(set(first.components) & set(second.components)),
        'shared_free_cells': len(set(first.free_cells) & set(second.free_cells)),
        'structural_membership': None,
        'reason': 'REGION_CONNECTIVITY_DOES_NOT_IDENTIFY_A_STRUCTURE',
        'training_qualified': False,
    }
